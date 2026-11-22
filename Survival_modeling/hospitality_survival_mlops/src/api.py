"""FastAPI inference service: survival curves, risk score and next-best marketing action."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .base import median_from_curve
from .features import Preprocessor
from .registry import ARTIFACTS, load_model
from .utils import ROOT, get_device, get_logger, load_config

log = get_logger("api")
HORIZONS = [30, 60, 90, 180, 365]
STATE: dict = {}


class GuestProfile(BaseModel):
    loyalty_tier: Literal["Member", "Silver", "Gold", "Platinum"] = Field(..., description="Loyalty programme tier")
    adr_avg: float = Field(..., gt=0, le=2000, description="Average daily rate (USD)")
    length_of_stay_avg: float = Field(..., ge=1, le=60, description="Average length of stay (nights)")
    lead_time_days: float = Field(..., ge=0, le=730, description="Average booking lead time (days)")
    days_since_last_stay: float = Field(..., ge=0, le=2000, description="Average gap between prior stays (days)")
    total_prior_stays: int = Field(..., ge=0, le=500)
    ancillary_spend_ratio: float = Field(..., ge=0, le=1, description="Spa / dining share of spend")
    promo_email_open_rate: float = Field(..., ge=0, le=1)
    app_sessions_30d: int = Field(..., ge=0, le=1000)
    channel_direct: Literal[0, 1] = Field(..., description="1 = books through direct channel")
    trip_purpose_business: Literal[0, 1] = Field(..., description="1 = business traveller")
    seasonality_index: float = Field(..., ge=0.3, le=2.0)

    model_config = {"json_schema_extra": {"examples": [{
        "loyalty_tier": "Gold", "adr_avg": 210.0, "length_of_stay_avg": 3.0, "lead_time_days": 14.0,
        "days_since_last_stay": 45.0, "total_prior_stays": 9, "ancillary_spend_ratio": 0.22,
        "promo_email_open_rate": 0.45, "app_sessions_30d": 12, "channel_direct": 1,
        "trip_purpose_business": 1, "seasonality_index": 1.05}]}}


class SurvivalResponse(BaseModel):
    model_used: str
    device: str
    risk_score: float = Field(..., description="Percentile of rebooking risk vs. the training population (0-100)")
    raw_risk: float
    median_survival_days: float | None = Field(None, description="Median time to next booking; null if beyond 365 days")
    survival_probabilities: dict[str, float] = Field(..., description="P(guest has NOT rebooked by day t)")
    rebooking_probabilities: dict[str, float] = Field(..., description="1 - survival probability")
    risk_segment: str
    recommended_action: str


def recommend(surv: dict[int, float], tier: str) -> tuple[str, str]:
    """Rule-based next-best-action from the guest's survival curve."""
    s90, s180, s365 = surv[90], surv[180], surv[365]
    vip = tier in ("Gold", "Platinum")
    if s365 >= 0.60:
        return "Critical churn risk", ("Launch win-back: personalised 'we miss you' offer with bonus points / "
                                       "complimentary upgrade" + (" and a personal outreach from the loyalty manager."
                                                                  if vip else " via email and app push."))
    if s180 >= 0.55:
        return "High churn risk", "Send a time-limited rate offer with ancillary credit (spa / dining) within 30 days."
    if s90 >= 0.60:
        return "Moderate churn risk", "Nudge with a tier-progress reminder and a destination-themed email at ~day 60."
    return "Low churn risk / likely to rebook", ("Maintain cadence: standard loyalty newsletter and pre-arrival "
                                                 "upsell; avoid discounting.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    models_dir = ROOT / cfg["paths"]["models"]
    dev = get_device(cfg)
    try:
        meta = json.loads((models_dir / "best_model.json").read_text())
        STATE.update(
            preprocessor=Preprocessor.load(models_dir / "preprocessor.pkl"),
            meta=meta,
            device=dev,
            model=load_model(meta["name"], models_dir, device=dev.torch_device),
        )
        # Reference distribution of the raw risk score on the (unscaled) training population
        ref = pd.read_csv(ROOT / cfg["paths"]["data"]).sample(3000, random_state=0)
        STATE["ref_risk"] = np.sort(STATE["model"].risk(STATE["preprocessor"].transform(ref)))
        STATE["loaded"] = [meta["name"]]
        log.info("API ready | model=%s | device=%s", meta["name"], dev.torch_device)
    except Exception as exc:                      # surface a clear health status instead of crashing
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        log.error("Failed to load artefacts: %s", STATE["error"])
    yield
    STATE.clear()


app = FastAPI(title="Hospitality Guest Survival API",
              description="Time-to-next-booking / churn risk for loyalty members using the best survival model "
                          "selected by the training pipeline.",
              version="1.0.0", lifespan=lifespan)


@app.get("/health", tags=["system"])
def health():
    dev = STATE.get("device") or get_device()
    return {
        "status": "ok" if "model" in STATE else "degraded",
        "cuda_available": dev.cuda_available,
        "active_device": dev.torch_device,
        "gpu_name": dev.name,
        "loaded_models": STATE.get("loaded", []),
        "best_model_metrics": STATE.get("meta"),
        "available_artifacts": sorted(ARTIFACTS),
        "error": STATE.get("error"),
    }


@app.post("/predict/survival", response_model=SurvivalResponse, tags=["prediction"])
def predict_survival(guest: GuestProfile):
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {STATE.get('error', 'unknown error')}")
    model, pre = STATE["model"], STATE["preprocessor"]
    X = pre.transform(pd.DataFrame([guest.model_dump()]))

    grid = np.arange(1, 366, dtype=float)
    curve = model.survival(X, grid)
    probs = {h: float(model.survival(X, np.array([float(h)]))[0, 0]) for h in HORIZONS}
    median = median_from_curve(grid, curve)[0]

    raw = float(model.risk(X)[0])
    pct = float(100.0 * np.searchsorted(STATE["ref_risk"], raw) / len(STATE["ref_risk"]))
    segment, action = recommend(probs, guest.loyalty_tier)
    return SurvivalResponse(
        model_used=STATE["meta"]["name"], device=STATE["device"].torch_device,
        risk_score=round(pct, 2), raw_risk=round(raw, 5),
        median_survival_days=None if np.isnan(median) else round(float(median), 1),
        survival_probabilities={f"{h}d": round(p, 4) for h, p in probs.items()},
        rebooking_probabilities={f"{h}d": round(1 - p, 4) for h, p in probs.items()},
        risk_segment=segment, recommended_action=action,
    )
