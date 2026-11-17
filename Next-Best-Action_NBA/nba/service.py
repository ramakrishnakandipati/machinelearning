"""Module 6 - Real-time NBA inference microservice (FastAPI).

Run:  uvicorn nba.service:app --port 8000
Env:  NBA_REDIS_URL (optional, e.g. redis://localhost:6379/0). Without it, or if
      Redis is unreachable at startup, an in-memory contact store is used.

Latency design (<40ms p99): models are loaded once at startup; the hot path is
one numpy row -> LightGBM final-stage boosters (no DataFrame, no EconML
validation) -> vectorised guardrails -> one pipelined Redis round trip.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .artifact_manager import DEFAULT_ROOT
from .config import ACTION_COSTS, ACTION_NAMES, COLD_START_MIN_SESSIONS, EXPLORATION_EPSILON, PROMOTIONAL_ACTIONS, TRIGGER_ACTIONS
from .exploration import EpsilonGreedyPolicy
from .guardrails import ContactStore, InMemoryContactStore, RedisContactStore, apply_guardrails
from .policy import choose_actions, net_value_matrix
from .schemas import DecisionLogRecord
from .uplift_model import UpliftBundle, load_bundle

MODELS_DIR = Path(os.environ.get("NBA_MODELS_DIR", DEFAULT_ROOT / "saved_models"))
DECISION_LOG = Path(os.environ.get("NBA_DECISION_LOG", DEFAULT_ROOT / "decision_log.jsonl"))


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class RecommendRequest(BaseModel):
    customer_id: int
    trigger: Literal["app_open", "browse_abandon", "post_purchase"]
    context_features: dict[str, float | None] = Field(default_factory=dict, description="Optional real-time overrides")


class RecommendResponse(BaseModel):
    request_id: str
    customer_id: int
    trigger: str
    action_id: int
    action_name: str
    net_value: float
    tau_margin: float
    tau_unsub: float
    propensity_score: float
    explored: bool
    cold_start: bool
    reason_codes: list[str]
    model_version: str
    latency_ms: float


# --------------------------------------------------------------------------- #
# Application state (loaded once on startup)
# --------------------------------------------------------------------------- #
class State:
    bundle: UpliftBundle
    manifest: dict[str, Any]
    lam: float
    features: dict[int, dict[str, float | None]]   # online feature store (demo: in-memory snapshot)
    contacts: ContactStore
    explorer: EpsilonGreedyPolicy
    contact_backend: str = "memory"


state = State()


def _load_feature_snapshot() -> dict[int, dict[str, float | None]]:
    """Demo feature store from the pipeline's snapshot (production: a Redis hash / feature platform)."""
    path = MODELS_DIR / "serving_feature_snapshot.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).astype(object).where(lambda d: d.notna(), None)
    return {int(r["customer_id"]): {k: v for k, v in r.items() if k != "customer_id"} for r in df.to_dict("records")}


async def _build_contact_store() -> tuple[ContactStore, str]:
    url = os.environ.get("NBA_REDIS_URL")
    if url:
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(url, socket_timeout=0.02, socket_connect_timeout=0.2)
            await client.ping()
            return RedisContactStore(client), "redis"
        except Exception as exc:  # noqa: BLE001 - any failure falls back safely
            print(f"[service] Redis unavailable ({exc!r}); using in-memory contact store")
    return InMemoryContactStore(), "memory"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load serialized models + manifest, build stores, warm up the model."""
    state.bundle, state.manifest = load_bundle(MODELS_DIR)
    state.lam = float(state.manifest["risk_aversion_lambda"])
    state.features = _load_feature_snapshot()
    state.contacts, state.contact_backend = await _build_contact_store()
    state.explorer = EpsilonGreedyPolicy(EXPLORATION_EPSILON)
    state.bundle.predict_tau(np.zeros((1, len(state.bundle.transformer.columns)), dtype=np.float32))  # warm-up
    yield


app = FastAPI(title="Retail Next Best Action", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Latency-ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
    return response


# --------------------------------------------------------------------------- #
# Decision logging (asynchronous, off the request path)
# --------------------------------------------------------------------------- #
def write_decision_log(record: DecisionLogRecord) -> None:
    """Append the decision + its logged propensity (production: Kafka / queue)."""
    DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_LOG.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "model_version": state.manifest["model_version"], "lambda": state.lam,
            "contact_backend": state.contact_backend}


@app.post("/v1/nba/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, background: BackgroundTasks) -> RecommendResponse:
    t0 = time.perf_counter()
    b = state.bundle
    cols = b.transformer.columns

    # 1) Features: store lookup + live overrides; missing history -> population priors.
    row: dict[str, float | None] = dict(state.features.get(req.customer_id, {}))
    row.update(req.context_features)
    known_customer = req.customer_id in state.features
    if not known_customer:  # unseen guest: treat as cold start
        row.setdefault("is_cold_start", 1.0)
    x = b.transformer.transform_dict(row)[None, :]
    ci = {c: i for i, c in enumerate(cols)}
    cold = bool(x[0, ci["is_cold_start"]] == 1) or float(row.get("n_sessions_total", 99) or 99) < COLD_START_MIN_SESSIONS

    # 2) CATE for all actions (fast path) and net value.
    tm, tu = b.predict_tau(x)
    nv = net_value_matrix(tm, tu, state.lam)

    # 3) Contact-fatigue state: one Redis round trip; fail safe on error.
    store_down = False
    try:
        promos = await state.contacts.count_promos(req.customer_id)
    except Exception:  # noqa: BLE001
        promos, store_down = 0, True

    # 4) Hard guardrails (same code as offline evaluation).
    gr = apply_guardrails(
        b.baseline_purchase_prob(x),
        x[:, ci["markdown_share"]].astype(np.float64),
        np.array([promos], dtype=np.float64),
        x[:, ci["size_availability_score"]].astype(np.float64),
        np.array([1.0 if cold else 0.0]),
        TRIGGER_ACTIONS[req.trigger],
        contact_store_unavailable=store_down,
    )
    greedy = choose_actions(nv, gr.eligible)

    # 5) Epsilon-greedy exploration over *eligible* actions, with logged propensity.
    action, prop, explored = state.explorer.select(greedy, gr.eligible)
    a, explored_flag = int(action[0]), bool(explored[0])

    reasons = [f"suppressed_a{k}:{r}" for k in range(1, 5) for r in gr.reasons_for(0, k)]
    if explored_flag:
        reasons.append("exploration")
    if cold:
        reasons.append("cold_start_shrinkage")
    if a == 0 and greedy[0] == 0:
        reasons.append("no_positive_net_value")

    # 6) Side effects off the hot path.
    if a in PROMOTIONAL_ACTIONS:
        background.add_task(state.contacts.record_promo, req.customer_id)
    rid = uuid.uuid4().hex
    background.add_task(write_decision_log, DecisionLogRecord(
        request_id=rid, customer_id=req.customer_id, timestamp=datetime.now(timezone.utc), trigger=req.trigger,
        action_id=a, propensity_score=float(prop[0]), explored=explored_flag, net_value=float(nv[0, a]),
        reason_codes=reasons, model_version=state.manifest["model_version"]))

    j = a - 1
    return RecommendResponse(
        request_id=rid, customer_id=req.customer_id, trigger=req.trigger, action_id=a, action_name=ACTION_NAMES[a],
        net_value=float(nv[0, a]), tau_margin=float(tm[0, j]) if a else 0.0, tau_unsub=float(tu[0, j]) if a else 0.0,
        propensity_score=float(prop[0]), explored=explored_flag, cold_start=cold, reason_codes=reasons,
        model_version=state.manifest["model_version"], latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )
