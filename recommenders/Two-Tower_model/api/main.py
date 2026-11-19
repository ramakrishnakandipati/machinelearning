"""FastAPI serving layer. Run from project root:  uvicorn api.main:app --port 8000"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.logger import get_logger

from .feature_store import FeatureStore
from .schemas import RecResponse, ReloadResponse, SimilarItemsRequest, UserRecRequest
from .serving import ModelBundle

log = get_logger("api", "05_api_server.log")
_swap_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    app.state.error = None
    app.state.bundle = None
    store = FeatureStore().connect()
    app.state.store = store
    try:
        store.load_snapshot()
        app.state.bundle = ModelBundle().load()
        b = app.state.bundle
        log.info("startup OK in %.2fs | device=%s model=%s items=%d faiss_ntotal=%d store=%s users=%d", time.perf_counter() - t0,
                 b.device, b.version, b.n_items, b.index.ntotal, store.backend, store.n_users)
    except Exception as exc:
        app.state.error = f"{type(exc).__name__}: {exc}"
        log.error("startup failed: %s", app.state.error)
    yield
    log.info("shutdown")


app = FastAPI(title="Two-Tower Retail Recommender", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    log.info("%s %s -> %d (%.2f ms)", request.method, request.url.path, resp.status_code, ms)
    return resp


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    details = [{"loc": [str(x) for x in e["loc"]], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "validation_error", "details": details})


def _bundle(request: Request) -> ModelBundle:
    b = request.app.state.bundle
    if b is None:
        raise HTTPException(503, detail=f"model not loaded: {request.app.state.error}")
    return b


@app.get("/health")
def health(request: Request):
    b, store = request.app.state.bundle, request.app.state.store
    gpu_name = None
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    ok = b is not None and store.ping()
    body = {"status": "ok" if ok else "degraded", "device": str(b.device) if b else None, "gpu_name": gpu_name,
            "user_tower_loaded": b is not None, "faiss_ready": bool(b and b.index.ntotal > 0),
            "faiss_ntotal": int(b.index.ntotal) if b else 0, "faiss_backend": "cpu-index (HNSW)",
            "redis": {"backend": store.backend, "connected": store.ping(), "users_loaded": store.n_users},
            "model_version": b.version if b else None, "error": request.app.state.error}
    return JSONResponse(body, status_code=200 if ok else 503)


@app.post("/recommend/user", response_model=RecResponse)
def recommend_user(req: UserRecRequest, request: Request):
    b = _bundle(request)
    return b.recommend_user(request.app.state.store, req.user_id, req.top_k, req.channel, req.exclude_purchased,
                            req.category_filter, req.in_stock_only)


@app.post("/recommend/similar-items", response_model=RecResponse)
def similar_items(req: SimilarItemsRequest, request: Request):
    b = _bundle(request)
    if req.item_id >= b.n_items:
        raise HTTPException(404, detail=f"unknown item_id {req.item_id}")
    return b.similar_items(req.item_id, req.top_k, req.category_filter, req.in_stock_only)


@app.post("/index/reload", response_model=ReloadResponse)
def reload_index(request: Request):
    """Hot-reload: build the new bundle off to the side, then atomically swap the reference."""
    t0 = time.perf_counter()
    try:
        new = ModelBundle().load()
    except Exception as exc:
        log.error("reload failed, keeping current model: %s", exc)
        raise HTTPException(500, detail=f"reload failed, previous model still serving: {exc}")
    with _swap_lock:
        old = request.app.state.bundle
        request.app.state.bundle = new
    ms = (time.perf_counter() - t0) * 1000
    log.info("hot-reload: %s -> %s (%.1f ms)", old.version if old else None, new.version, ms)
    return {"status": "reloaded", "model_version": new.version, "n_items": new.n_items, "reload_ms": round(ms, 2)}
