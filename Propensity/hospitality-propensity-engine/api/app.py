"""FastAPI propensity service.  Run:  uvicorn api.app:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.schemas import (HealthResponse, PredictBatchRequest, PredictBatchResponse, PredictGuestRequest,  # noqa: E402
                         PredictGuestResponse, ScenarioScore)
from src.predictor import PropensityEngine  # noqa: E402

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["engine"] = PropensityEngine()
    yield
    STATE.clear()


app = FastAPI(
    title="Hospitality Propensity Engine",
    description="Calibrated, GPU-served propensity scores for 8 hospitality lifecycle scenarios "
                "(OTA→Direct, abandonment, upgrade, ancillary, churn, property repeat, brand repeat, cancel/no-show) "
                "with decile tiers, uplift segmentation and next-best-action.",
    version="1.0.0", lifespan=lifespan)


def _engine() -> PropensityEngine:
    if "engine" not in STATE:
        raise HTTPException(503, "models not loaded")
    return STATE["engine"]


def _to_df(guests) -> pd.DataFrame:
    return pd.DataFrame([g.model_dump(exclude_none=True) for g in guests])


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """CUDA status, GPU device name, and the model serving each scenario."""
    return _engine().health()


@app.get("/leaderboard", tags=["system"])
def leaderboard():
    """Master leaderboard (test metrics for every scenario × model)."""
    return _engine().leaderboard().to_dict(orient="records")


@app.post("/predict/guest", response_model=PredictGuestResponse, tags=["inference"])
def predict_guest(req: PredictGuestRequest):
    """Real-time scoring of one guest across all 8 scenarios (calibrated probability, decile tier, next-best-action)."""
    t = time.perf_counter()
    eng = _engine()
    df = _to_df([req.guest])
    if df.empty:
        df = pd.DataFrame([{}])
    res = eng.predict_guests(df, explain=req.explain)[0]
    return PredictGuestResponse(guest_id=req.guest.guest_id, scores={k: ScenarioScore(**v) for k, v in res.items()},
                                latency_ms=round((time.perf_counter() - t) * 1000, 2))


@app.post("/predict/batch", response_model=PredictBatchResponse, tags=["inference"])
def predict_batch(req: PredictBatchRequest):
    """High-throughput batch scoring (vectorised; GPU inference)."""
    t = time.perf_counter()
    eng = _engine()
    df = _to_df(req.guests)
    if df.empty:
        df = pd.DataFrame([{} for _ in req.guests])
    out = eng.predict_guests(df)
    ms = (time.perf_counter() - t) * 1000
    results = [PredictGuestResponse(guest_id=g.guest_id, scores={k: ScenarioScore(**v) for k, v in r.items()}, latency_ms=0.0)
               for g, r in zip(req.guests, out)]
    return PredictBatchResponse(n=len(results), results=results, latency_ms=round(ms, 2),
                                guests_per_second=round(len(results) / max(ms / 1000, 1e-9), 1))
