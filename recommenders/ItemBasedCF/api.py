"""FastAPI microservice serving IBCF recommendations. Run: uvicorn api:app --port 8000"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from recommender_engine import RecommenderEngine

ARTIFACT_DIR = Path(os.environ.get("IBCF_ARTIFACT_DIR", Path(__file__).resolve().parent / "output" / "artifacts"))


class Metrics:
    def __init__(self, window: int = 5000) -> None:
        self.lock = threading.Lock()
        self.lat = defaultdict(lambda: deque(maxlen=window))
        self.count = defaultdict(int)
        self.errors = defaultdict(int)
        self.strategies = defaultdict(int)

    def record(self, route: str, ms: float, status: int) -> None:
        with self.lock:
            self.lat[route].append(ms)
            self.count[route] += 1
            if status >= 400:
                self.errors[route] += 1

    @staticmethod
    def _pct(vals) -> dict:
        if not vals:
            return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}
        a = np.asarray(vals)
        return {"count": int(a.size), "p50_ms": round(float(np.percentile(a, 50)), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3), "p99_ms": round(float(np.percentile(a, 99)), 3)}

    def snapshot(self) -> dict:
        with self.lock:
            rec = {r: list(v) for r, v in self.lat.items() if r.startswith("/v1/")}
            allv = [x for v in rec.values() for x in v]
            return {"latency_ms_recommend_overall": self._pct(allv),
                    "latency_ms_by_route": {r: self._pct(v) for r, v in self.lat.items()},
                    "requests": dict(self.count), "errors": dict(self.errors),
                    "strategies": dict(self.strategies)}


metrics = Metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = RecommenderEngine()
    app.state.error = None
    try:
        engine.load(ARTIFACT_DIR)
    except Exception as exc:  # missing artifacts -> degraded, not crash
        app.state.error = f"{type(exc).__name__}: {exc}"
    app.state.engine = engine
    app.state.started = time.time()
    yield


app = FastAPI(title="IBCF Personalization API", version="1.0.0", lifespan=lifespan)


# ------------------------------------------------------------------- schemas
class UserRecRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=64)
    recent_item_ids: list[str] = Field(default_factory=list, max_length=100)
    exclude_item_ids: list[str] = Field(default_factory=list, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)


class RecItem(BaseModel):
    item_id: str
    score: float
    category: str
    sub_category: str
    price: float
    in_stock: bool
    reason: str


class RecResponse(BaseModel):
    strategy: str
    model_version: str
    items: list[RecItem]
    latency_ms: float


# ---------------------------------------------------------------- middleware
@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    metrics.record(path, ms, resp.status_code)
    resp.headers["X-Latency-Ms"] = f"{ms:.3f}"
    return resp


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    details = [{"loc": [str(x) for x in e["loc"]], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "validation_error", "details": details})


def _engine(request: Request) -> RecommenderEngine | None:
    eng = request.app.state.engine
    return eng if eng.loaded else None


def _unavailable(request: Request):
    return JSONResponse(status_code=503, content={"error": "model_unavailable", "detail": request.app.state.error})


# ----------------------------------------------------------------- endpoints
@app.post("/v1/recommend/user", response_model=RecResponse)
async def recommend_user(body: UserRecRequest, request: Request):
    eng = _engine(request)
    if eng is None:
        return _unavailable(request)
    t0 = time.perf_counter()
    r = eng.recommend_user(body.user_id, body.recent_item_ids, body.exclude_item_ids, body.limit)
    metrics.strategies[r["strategy"]] += 1
    return {"strategy": r["strategy"], "model_version": eng.version, "items": r["items"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}


@app.get("/v1/recommend/item/{item_id}", response_model=RecResponse)
async def recommend_item(item_id: str, request: Request, limit: int = Query(10, ge=1, le=50),
                         in_stock_only: bool = True):
    eng = _engine(request)
    if eng is None:
        return _unavailable(request)
    t0 = time.perf_counter()
    r = eng.recommend_item(item_id, limit, in_stock_only)
    metrics.strategies[r["strategy"]] += 1
    return {"strategy": r["strategy"], "model_version": eng.version, "items": r["items"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}


@app.get("/health")
async def health(request: Request):
    eng = request.app.state.engine
    ok = eng.loaded
    body = {"status": "ok" if ok else "degraded", "artifacts_loaded": ok, "model_version": eng.status()["model_version"],
            "error": request.app.state.error}
    return JSONResponse(status_code=200 if ok else 503, content=body)


@app.get("/metrics")
async def get_metrics(request: Request):
    eng = request.app.state.engine
    st = eng.status()
    return {"model_version": st["model_version"], "artifact_status": "loaded" if st["loaded"] else "missing",
            "n_items": st["n_items"], "n_users": st["n_users"], "catalog_coverage": st["catalog_coverage"],
            "params": st["params"], "uptime_s": round(time.time() - request.app.state.started, 1),
            **metrics.snapshot()}
