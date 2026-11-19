"""FastAPI microservice: /health, /recommend, /items/onboard, /feedback.

Run:  uvicorn src.serving.app:app --port 8000
Env:  COLDSTART_CONFIG (default config/config.yaml), COLDSTART_MODELS (default <outputs>/models)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pickle
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Request

from src.data.preprocessor import Preprocessor
from src.models.content_based import ItemIndex
from src.models.contextual_bandit import REWARD, ContextualBandit
from src.models.hybrid_two_tower import TwoTowerRecommender
from src.models.router import MetaRouter
from src.serving.schemas import (FeedbackRequest, FeedbackResponse, HealthResponse, OnboardItemRequest,
                                 OnboardItemResponse, RecommendedItem, RecommendRequest, RecommendResponse)
from src.utils.hardware import faiss_gpu_available, get_device

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- cache
class ResponseCache:
    """Redis cache with an automatic thread-safe in-memory LRU fallback."""

    PREFIX = "coldstart:rec:"

    def __init__(self, redis_url: Optional[str], timeout: float, ttl: int, capacity: int):
        self.ttl, self.capacity = ttl, capacity
        self._lru: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0
        self._redis = None
        self.redis_error: Optional[str] = None
        if redis_url:
            try:
                import redis

                r = redis.Redis.from_url(redis_url, socket_timeout=timeout, socket_connect_timeout=timeout)
                r.ping()
                self._redis = r
            except Exception as exc:
                self.redis_error = type(exc).__name__

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "in-memory-lru"

    def _fallback(self, exc: Exception) -> None:
        self.redis_error, self._redis = type(exc).__name__, None

    def get(self, key: str) -> Optional[str]:
        val = None
        if self._redis is not None:
            try:
                raw = self._redis.get(self.PREFIX + key)
                val = raw.decode() if raw else None
            except Exception as exc:
                self._fallback(exc)
        if self._redis is None:
            with self._lock:
                ent = self._lru.get(key)
                if ent and ent[0] > time.time():
                    self._lru.move_to_end(key)
                    val = ent[1]
                elif ent:
                    del self._lru[key]
        self.hits, self.misses = self.hits + (val is not None), self.misses + (val is None)
        return val

    def set(self, key: str, value: str) -> None:
        if self._redis is not None:
            try:
                self._redis.setex(self.PREFIX + key, self.ttl, value)
                return
            except Exception as exc:
                self._fallback(exc)
        with self._lock:
            self._lru[key] = (time.time() + self.ttl, value)
            self._lru.move_to_end(key)
            while len(self._lru) > self.capacity:
                self._lru.popitem(last=False)

    def clear(self) -> None:
        if self._redis is not None:
            try:
                for k in self._redis.scan_iter(self.PREFIX + "*"):
                    self._redis.delete(k)
            except Exception as exc:
                self._fallback(exc)
        with self._lock:
            self._lru.clear()


# ----------------------------------------------------------------------------- state
class ServiceState:
    """Everything loaded from ``outputs/models`` plus mutable live state (sessions)."""

    def __init__(self, cfg: Dict[str, Any], models_dir: Path):
        self.cfg, self.models_dir = cfg, models_dir
        self.device = get_device(cfg["hardware"]["use_gpu"])
        self.pre = Preprocessor.load(models_dir)
        self.index = ItemIndex.load(models_dir / "item_index.faiss", use_gpu=cfg["hardware"]["faiss_gpu"])
        self.tower = TwoTowerRecommender(self.pre, cfg, self.device).load(models_dir)
        self.bandit = ContextualBandit(self.pre, cfg, cfg["seed"]).load(models_dir)
        with open(models_dir / "serving_state.pkl", "rb") as f:
            st = pickle.load(f)
        self.pop = st["popularity"]
        self.users: Dict[str, Dict[str, Any]] = st["users"]
        self.hist: Dict[str, List[int]] = st["history"]
        self.router = MetaRouter(cfg, self.pre, st["items_df"], self.pop, self.index, self.tower, self.bandit, st["sales"])
        self.sessions: Dict[str, List[int]] = {}
        self.lock = threading.Lock()
        s = cfg["serving"]
        self.cache = ResponseCache(os.environ.get("COLDSTART_REDIS_URL", s["redis_url"]), float(s["redis_timeout_s"]),
                                   int(s["cache_ttl_s"]), int(s["lru_capacity"]))

    def user_row(self, user_id: Optional[str], ctx) -> Dict[str, Any]:
        base = dict(self.users.get(user_id, {})) if user_id else {}
        base.update({k: v for k, v in dict(region=ctx.region, device=ctx.device, acq_channel=ctx.acquisition_channel,
                                           age_band=ctx.age_band).items() if v})
        if ctx.onboarding_categories:
            base["onboarding_categories"] = list(ctx.onboarding_categories)
        if user_id:
            base["user_id"] = user_id
        return base

    @staticmethod
    def request_ctx(ctx) -> Dict[str, Any]:
        now = dt.datetime.now()
        season = ["winter", "winter", "spring", "spring", "spring", "summer", "summer", "summer", "autumn", "autumn",
                  "autumn", "winter"][now.month - 1]
        return dict(channel=ctx.channel or "web", season=ctx.season or season,
                    hour=ctx.hour if ctx.hour is not None else now.hour)


def get_state(request: Request) -> ServiceState:
    st = getattr(request.app.state, "svc", None)
    if st is None:
        raise HTTPException(503, "Models not loaded - run `python run_pipeline.py` first")
    return st


# ------------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg_path = Path(os.environ.get("COLDSTART_CONFIG", ROOT / "config" / "config.yaml"))
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    models_dir = Path(os.environ.get("COLDSTART_MODELS", ROOT / cfg["paths"]["models"]))
    app.state.svc, app.state.load_error = None, None
    try:
        app.state.svc = ServiceState(cfg, models_dir)
    except Exception as exc:  # keep /health available to report the problem
        app.state.load_error = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="Cold-Start Personalization API", version="1.0.0",
              description="Omnichannel retail recommender solving new-user, new-item and zero-history cold-start.",
              lifespan=lifespan)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    st: Optional[ServiceState] = getattr(request.app.state, "svc", None)
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else None
    if st is None:
        return HealthResponse(status=f"degraded: {getattr(request.app.state, 'load_error', 'not loaded')}",
                              device="cpu", gpu_name=gpu, cuda_available=cuda, amp_trained=False,
                              models_loaded={k: False for k in ("preprocessor", "two_tower", "faiss_index", "bandit", "popularity")},
                              faiss_backend="none", faiss_index_size=0, n_items=0, n_known_users=0, redis="n/a",
                              cache={}, embedding_backend="n/a")
    return HealthResponse(
        status="ok", device=str(st.device), gpu_name=gpu, cuda_available=cuda, amp_trained=st.tower.amp_used,
        models_loaded=dict(preprocessor=True, two_tower=True, faiss_index=True, bandit=True, popularity=True),
        faiss_backend=st.index.backend + (" (faiss-gpu available)" if faiss_gpu_available() else ""),
        faiss_index_size=st.index.size, n_items=st.router.n_items, n_known_users=len(st.users),
        redis=("connected" if st.cache.backend == "redis" else f"unavailable -> in-memory LRU fallback"
               + (f" ({st.cache.redis_error})" if st.cache.redis_error else "")),
        cache=dict(hits=st.cache.hits, misses=st.cache.misses), embedding_backend=st.pre.embedder.backend)


@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(req: RecommendRequest, request: Request) -> RecommendResponse:
    st = get_state(request)
    t0 = time.perf_counter()
    key = hashlib.sha1(json.dumps(req.model_dump(), sort_keys=True).encode()).hexdigest()
    cached = st.cache.get(key)
    if cached:
        resp = RecommendResponse.model_validate_json(cached)
        resp.cached, resp.latency_ms = True, round((time.perf_counter() - t0) * 1000, 3)
        return resp
    warnings: List[str] = []
    urow = st.user_row(req.user_id, req.user_context)
    if req.user_id and req.user_id not in st.users:
        warnings.append(f"user_id {req.user_id} is unknown - treated as a new user")
    history = list(st.hist.get(req.user_id, [])) if req.user_id else []
    if req.user_id:
        history += st.sessions.get(req.user_id, [])
    for sid in req.session_item_ids:
        p = st.pre.item_pos.get(sid)
        if p is None:
            warnings.append(f"session item {sid} not in catalog - ignored")
        else:
            history.append(p)
    ctx = st.request_ctx(req.user_context)
    res = st.router.recommend(urow, ctx, history, k=req.k, include_new_items_boost=req.include_new_items_boost, exclude=set(history),
                              record_impressions=True)
    items = []
    for r, it in enumerate(res.items, 1):
        m = st.router.meta[it.position]
        items.append(RecommendedItem(rank=r, item_id=it.item_id, title=str(m["title"]), brand=str(m["brand"]),
                                     dept=str(m["dept"]), subcat=str(m["subcat"]), price=float(m["price"]),
                                     score=round(it.score, 5), is_new_item=it.is_new_item, explain=it.tags))
    resp = RecommendResponse(strategy=res.strategy, n_interactions=res.n_interactions, weights=res.components,
                             items=items, latency_ms=round((time.perf_counter() - t0) * 1000, 3), cached=False,
                             warnings=warnings)
    st.cache.set(key, resp.model_dump_json())
    return resp


@app.post("/items/onboard", response_model=OnboardItemResponse, tags=["catalog"])
def onboard_item(req: OnboardItemRequest, request: Request) -> OnboardItemResponse:
    st = get_state(request)
    t0 = time.perf_counter()
    try:
        pos = st.router.onboard_item(req.model_dump())
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    st.cache.clear()
    return OnboardItemResponse(item_id=req.item_id, position=pos, index_size=st.index.size,
                               embed_and_inject_ms=round((time.perf_counter() - t0) * 1000, 3),
                               message="SKU embedded and injected into the live FAISS index (zero downtime)")


@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def feedback(req: FeedbackRequest, request: Request) -> FeedbackResponse:
    st = get_state(request)
    pos = st.pre.item_pos.get(req.item_id)
    if pos is None:
        raise HTTPException(404, f"item_id {req.item_id} not found")
    reward = REWARD[req.event]
    key = req.user_id or req.session_id
    urow = st.user_row(req.user_id, req.user_context)
    x = st.bandit.context_vector(urow, st.request_ctx(req.user_context))
    st.bandit.update(x, int(st.router.subcat_idx[pos]), reward, pos)
    if reward > 0:
        st.pop.observe(pos, reward)
        with st.lock:
            st.router.sales[pos] += 1 if req.event == "purchase" else 0.0
            if key:
                st.sessions.setdefault(key, []).append(pos)
    st.cache.clear()
    return FeedbackResponse(accepted=True, reward=reward, bandit_total_updates=st.bandit.total_updates,
                            session_length=len(st.sessions.get(key, [])) if key else 0)
