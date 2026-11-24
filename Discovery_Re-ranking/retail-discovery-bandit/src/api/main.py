"""FastAPI serving layer:  POST /discover, POST /feedback, GET /health, GET /metrics.

Run:  uvicorn src.api.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.system import load_system
from src.utils import gpu_info, load_config, output_dir


class DiscoverRequest(BaseModel):
    user_id: int = Field(..., ge=0, examples=[42])
    k: int = Field(5, ge=1, le=20)
    alpha: Optional[float] = Field(None, ge=0.0, description="LinUCB exploration weight override")
    lambda_: Optional[float] = Field(None, ge=0.0, le=1.0, alias="lambda", description="MMR relevance/diversity trade-off override")
    policy: str = Field("linucb_mmr", description="pure_ranker | epsilon_greedy | linucb_mmr | linucb_high_expl")
    model_config = {"populate_by_name": True}


class SlateItem(BaseModel):
    slot: int
    item_id: int
    category: int
    subcat: int
    price: float
    margin: float
    p_click: float
    p_buy: float
    base_utility: float
    learned_residual: float
    exploration_bonus: float
    mmr_penalty: float
    bandit_score: float
    propensity_score: float
    novelty_tier: str
    is_cold_start: bool


class DiscoverResponse(BaseModel):
    request_id: int
    user_id: int
    policy: str
    slate: List[SlateItem]
    intra_list_diversity: float
    n_candidates: int
    latency_ms: Dict[str, float]
    filter_drop_counts: Dict[str, int]
    guardrails: Dict[str, int]


class FeedbackRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    item_id: int = Field(..., ge=0)
    slot: int = Field(..., ge=0)
    reward: float = Field(..., ge=0.0, description="0 = no click, 0.3 = click, 1.3 = click + purchase")
    propensity_score: float = Field(..., ge=0.0, le=1.0)
    policy: str = "linucb_mmr"


class Counters:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.feedback = 0
        self.impressions = 0
        self.cold_impressions = 0
        self.discovery_impressions = 0
        self.clicks = 0
        self.purchases = 0
        self.discovery_purchases = 0
        self.cold_purchases = 0
        self.seen_items: set = set()
        self.item_tier: Dict = {}
        self.latencies: List[float] = []


C = Counters()
SYS: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    log_dir = output_dir(cfg, "logs")
    SYS.update(load_system(cfg, log_dir=log_dir, seed=cfg["seed"]))
    SYS["cfg"] = cfg
    SYS["started"] = time.time()
    pipe = SYS["pipeline"]
    pipe.bandits = pipe.bandits or pipe.new_bandits()
    yield
    if pipe.prop_log:
        pipe.prop_log.flush()


app = FastAPI(title="Retail Discovery Bandit Re-Ranker", version="1.0", lifespan=lifespan)


@app.post("/discover", response_model=DiscoverResponse)
def discover(req: DiscoverRequest):
    pipe = SYS["pipeline"]
    if req.user_id >= pipe.world.n_users:
        raise HTTPException(404, f"unknown user_id {req.user_id}")
    if req.policy not in pipe.policies:
        raise HTTPException(422, f"unknown policy {req.policy}; choose from {list(pipe.policies)}")
    out = pipe.recommend(req.user_id, req.policy, k=req.k, alpha=req.alpha, lam=req.lambda_)
    pipe.log_propensity(out, None, event="impression")
    with C.lock:
        C.requests += 1
        C.latencies.append(out["latency_ms"]["total"])
        for s in out["slate"]:
            C.impressions += 1
            C.cold_impressions += int(s["is_cold_start"])
            disc = s["novelty_tier_id"] >= 2
            C.discovery_impressions += int(disc)
            C.seen_items.add(s["item_id"])
            C.item_tier[(req.user_id, s["item_id"])] = (disc, s["is_cold_start"])
    f = out["stage_counts"]["filter"]
    return DiscoverResponse(
        request_id=out["request"], user_id=out["user_id"], policy=out["policy"], slate=[SlateItem(**{k: v for k, v in s.items() if k in SlateItem.model_fields}) for s in out["slate"]],
        intra_list_diversity=out["ild"], n_candidates=out["n_candidates"], latency_ms=out["latency_ms"],
        filter_drop_counts={k: int(v) for k, v in f.items()},
        guardrails={k: int(v) for k, v in out["stage_counts"]["guardrails"].items()})


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    pipe = SYS["pipeline"]
    if req.policy not in pipe.policies:
        raise HTTPException(422, f"unknown policy {req.policy}")
    res = pipe.feedback(req.user_id, req.item_id, req.reward, req.policy)
    if pipe.prop_log:
        pipe.prop_log.log({"ts": time.time(), "event": "feedback", "policy": req.policy, "user_id": req.user_id,
                           "item_id": req.item_id, "slot": req.slot, "propensity_score": req.propensity_score,
                           "reward": req.reward})
        pipe.prop_log.flush()
    with C.lock:
        C.feedback += 1
        clicked = req.reward > 0
        bought = req.reward >= pipe.rw["buy"]
        C.clicks += int(clicked)
        C.purchases += int(bought)
        disc, cold = C.item_tier.get((req.user_id, req.item_id), (False, False))
        C.discovery_purchases += int(bought and disc)
        C.cold_purchases += int(bought and cold)
    return res


@app.get("/health")
def health():
    pipe = SYS.get("pipeline")
    return {"status": "ok" if pipe is not None else "starting", "gpu": gpu_info(),
            "cuda_available": torch.cuda.is_available(), "uptime_s": round(time.time() - SYS.get("started", time.time()), 1),
            "n_items": SYS["world"].n_items if SYS else 0, "n_users": SYS["world"].n_users if SYS else 0,
            "bandit_updates": {k: b.n_updates for k, b in pipe.bandits.items()} if pipe else {}}


@app.get("/metrics")
def metrics():
    pipe = SYS["pipeline"]
    n_items = pipe.world.n_items
    with C.lock:
        lat = np.asarray(C.latencies) if C.latencies else np.zeros(1)
        return {
            "gpu": gpu_info(),
            "requests": C.requests, "feedback_events": C.feedback, "impressions": C.impressions,
            "catalog_coverage_pct": round(100.0 * len(C.seen_items) / n_items, 3),
            "cold_start_impression_share": round(C.cold_impressions / max(C.impressions, 1), 4),
            "discovery_impression_share": round(C.discovery_impressions / max(C.impressions, 1), 4),
            "ctr": round(C.clicks / max(C.impressions, 1), 4),
            "purchases": C.purchases,
            "discovery_purchases": C.discovery_purchases,
            "discovery_conversion_rate": round(C.discovery_purchases / max(C.discovery_impressions, 1), 4),
            "cold_start_purchases": C.cold_purchases,
            "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                           "p99": float(np.percentile(lat, 99))},
            "stage_latency_ms": pipe.timer.summary(),
            "filter_drop_totals": pipe.drop_totals, "guardrail_totals": pipe.guard.totals,
            "bandit_updates": {k: b.n_updates for k, b in pipe.bandits.items()},
        }


# ----------------------------------------------------------------------------------------
# Small HTML views (used for automated Playwright screenshots)
# ----------------------------------------------------------------------------------------
from fastapi.responses import HTMLResponse  # noqa: E402

_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;background:#f6f7f9;color:#1d2433;margin:0;padding:24px 32px}
h1{font-size:20px;margin:0 0 4px} .sub{color:#5b667a;margin-bottom:18px;font-size:13px}
.shelf{display:flex;gap:14px;flex-wrap:wrap}.card{background:#fff;border:1px solid #dfe3ea;border-radius:10px;padding:12px 14px;width:250px;box-shadow:0 1px 2px #0001}
.card h3{margin:0 0 6px;font-size:14px}.badge{display:inline-block;border-radius:999px;padding:1px 8px;font-size:11px;margin-right:4px;color:#fff}
.cold{background:#d9534f}.t3{background:#e6550d}.t2{background:#f0a040}.t1{background:#7aa7d6}.t0{background:#9aa3b2}
table{border-collapse:collapse;font-size:12px;margin-top:6px;width:100%}td{padding:2px 0;border-bottom:1px solid #eef0f4}td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.tile{background:#fff;border:1px solid #dfe3ea;border-radius:10px;padding:14px}
.tile .v{font-size:26px;font-weight:600}.tile .k{font-size:12px;color:#5b667a}pre{background:#fff;border:1px solid #dfe3ea;border-radius:10px;padding:12px;font-size:12px;overflow:auto}
"""


@app.get("/demo", response_class=HTMLResponse)
def demo(user_id: int = 7, policy: str = "linucb_mmr", k: int = 5):
    pipe = SYS["pipeline"]
    out = pipe.recommend(user_id, policy, k=k)
    u = pipe.world.users.iloc[user_id]
    cards = []
    for s in out["slate"]:
        badges = f'<span class="badge t{s["novelty_tier_id"]}">{s["novelty_tier"]}</span>' + ('<span class="badge cold">cold-start</span>' if s["is_cold_start"] else "")
        cards.append(f"""<div class="card"><h3>#{s['slot'] + 1} &middot; SKU {s['item_id']} &middot; ${s['price']:.2f}</h3>{badges}
        <table><tr><td>p(click)</td><td>{s['p_click']:.3f}</td></tr><tr><td>p(buy)</td><td>{s['p_buy']:.3f}</td></tr>
        <tr><td>base utility</td><td>{s['base_utility']:.3f}</td></tr><tr><td>learned residual</td><td>{s['learned_residual']:+.3f}</td></tr>
        <tr><td>exploration bonus</td><td>{s['exploration_bonus']:.3f}</td></tr><tr><td>MMR penalty</td><td>{s['mmr_penalty']:.3f}</td></tr>
        <tr><td>propensity</td><td>{s['propensity_score']:.4f}</td></tr><tr><td>margin</td><td>{s['margin']:.0%}</td></tr></table></div>""")
    lat = out["latency_ms"]
    return f"""<html><head><title>Discovery shelf</title><style>{_CSS}</style></head><body>
    <h1>Discovery slate &mdash; user {user_id} ({u['persona_name']})</h1>
    <div class="sub">policy <b>{policy}</b> &middot; ILD {out['ild']:.3f} &middot; {out['n_candidates']} candidates after guardrails &middot;
    total latency {lat['total']:.1f} ms (retrieval {lat.get('1_retrieval', 0):.1f}, ranking {lat.get('3_ranking', 0):.1f}, MMR {lat.get('4f_mmr', 0):.1f})</div>
    <div class="shelf">{''.join(cards)}</div></body></html>"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return f"""<html><head><title>Discovery metrics</title><style>{_CSS}</style></head><body>
    <h1>Live discovery metrics</h1><div class="sub">GET /metrics</div><div class="tiles" id="grid"></div><h1 style="margin-top:24px">Stage latency (ms)</h1><pre id="lat"></pre>
    <script>fetch('/metrics').then(r=>r.json()).then(m=>{{
      const t=[['Requests',m.requests],['Feedback events',m.feedback_events],['Catalog coverage %',m.catalog_coverage_pct],['CTR',m.ctr],
      ['Cold-start impression share',m.cold_start_impression_share],['Discovery impression share',m.discovery_impression_share],
      ['Discovery conversion',m.discovery_conversion_rate],['p99 latency ms',m.latency_ms.p99.toFixed(1)],
      ['GPU',m.gpu.device],['VRAM used MB',m.gpu.vram_used_mb],['Torch allocated MB',m.gpu.torch_allocated_mb],['Bandit updates',JSON.stringify(m.bandit_updates)]];
      document.getElementById('grid').innerHTML=t.map(x=>`<div class="tile"><div class="v">${{x[1]}}</div><div class="k">${{x[0]}}</div></div>`).join('');
      document.getElementById('lat').textContent=JSON.stringify(m.stage_latency_ms,null,1);}});</script></body></html>"""
