"""FastAPI service: lifespan-loaded models + per-request latency telemetry (JSONL) and rolling percentiles."""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from src.api.schemas import SearchRequest, SearchResponse
from src.common.utils import load_config, resolve
from src.search_service import SearchService

STATE: dict = {}
LOCK = threading.Lock()  # torch/faiss/sparse retrieval are not re-entrant per request in this simple server
LAT = deque(maxlen=2000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    STATE["cfg"] = cfg
    STATE["svc"] = SearchService.load(cfg)
    STATE["log"] = open(resolve(cfg, "logs") / "inference.jsonl", "a", encoding="utf-8")
    STATE["started"] = time.time()
    yield
    STATE["log"].close()


app = FastAPI(
    title="Retail Search Funnel API",
    description="GPU-accelerated multi-stage retail search: query understanding -> hybrid dense/sparse retrieval -> "
                "learning-to-rank + multi-task (CTR/ATC/CVR/Return) scoring -> margin-aware business re-ranking.",
    version="1.0.0", lifespan=lifespan)


@app.get("/health", tags=["ops"])
def health():
    svc = STATE.get("svc")
    return {"status": "ok" if svc else "loading", "skus": svc.world.n if svc else 0,
            "uptime_s": round(time.time() - STATE.get("started", time.time()), 1)}


@app.get("/models", tags=["ops"])
def models():
    svc = STATE["svc"]
    return {"rankers": list(svc.models), "default_ranker": svc.cfg["ranking"]["default_ranker"],
            "fusion": svc.cfg["retrieval"]["fusion"], "alpha": svc.cfg["retrieval"]["alpha"],
            "encoder": svc.engine.encoder.name, "device": svc.engine.encoder.device,
            "stores": svc.world.stores[["store_id", "city", "zip_code"]].to_dict("records")}


@app.get("/metrics", tags=["ops"])
def metrics():
    """Rolling serving latency percentiles + offline evaluation reports."""
    a = np.array(LAT) if LAT else np.array([0.0])
    cfg = STATE["cfg"]
    off = {}
    for f in ["ranking_metrics", "retrieval_metrics", "rerank_ablation", "latency"]:
        p = resolve(cfg, "metrics") / f"{f}.json"
        if p.exists():
            off[f] = json.loads(p.read_text())
    return {"online_latency_ms": {"count": len(LAT), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
                                  "p99": float(np.percentile(a, 99))}, "offline": off}


@app.get("/users/sample", tags=["ops"])
def sample_users(n: int = 5):
    u = STATE["svc"].world.users.head(n)
    return u[["user_id", "price_tier", "preferred_brands", "home_store"]].to_dict("records")


@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest, request: Request):
    svc = STATE["svc"]
    if req.user_id and req.user_id not in svc.world.user_index:
        raise HTTPException(404, f"unknown user_id {req.user_id}")
    if req.ranker and req.ranker not in svc.models:
        raise HTTPException(422, f"unknown ranker {req.ranker}; choose from {list(svc.models)}")
    t0 = time.perf_counter()
    with LOCK:
        out = svc.search(req.query, req.user_id, req.store_id, req.zip_code, req.fulfillment, req.brand_allow,
                         req.brand_deny, req.min_price, req.max_price, req.ranker, req.fusion, req.alpha, req.top_k,
                         req.margin_weight, req.apply_rules, req.margin_opt, req.diversify)
    wall = (time.perf_counter() - t0) * 1e3
    out["latency_ms"]["wall_ms"] = round(wall, 2)
    LAT.append(out["latency_ms"]["total_ms"])
    STATE["log"].write(json.dumps({"ts": time.time(), "query": req.query, "user": req.user_id, "ranker": out["ranker"],
                                   "latency_ms": out["latency_ms"], "funnel": out["funnel"]}) + "\n")
    STATE["log"].flush()
    out["spell_edits"] = [list(e) for e in out["spell_edits"]]
    return out
