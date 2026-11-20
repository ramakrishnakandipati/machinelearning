"""FastAPI service for guest persona prediction.  Run: uvicorn api:app --port 8000"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import ConfigDict, Field, create_model

import config as C

EXAMPLE = dict(recency_days=45, annual_stays=12, avg_length_of_stay=2.5, booking_lead_time_days=9,
               adr_usd=215.0, fb_spend_ratio=0.25, spa_golf_spend_usd=40.0, room_upgrade_rate=0.2,
               loyalty_tenure_years=6.0, direct_booking_ratio=0.65, cancellation_rate=0.1,
               weekend_stay_ratio=0.1, business_trip_ratio=0.85)

GuestFeatures = create_model(
    "GuestFeatures",
    __config__=ConfigDict(json_schema_extra={"example": EXAMPLE}),
    **{f: (float, Field(..., ge=lo, le=hi, description=f"Range [{lo}, {hi}]"))
       for f, (lo, hi) in C.RANGES.items()},
)

STATE: dict = {}


def _load_models() -> dict:
    from models_dl import load_dec, load_vae
    loaders = {"kmeans": lambda: joblib.load(C.MODELS_DIR / "kmeans.joblib"),
               "gmm": lambda: joblib.load(C.MODELS_DIR / "gmm.joblib"),
               "agglomerative": lambda: joblib.load(C.MODELS_DIR / "agglomerative.joblib"),
               "dbscan": lambda: joblib.load(C.MODELS_DIR / "dbscan.joblib"),
               "dec": load_dec, "vae_gmm": load_vae}
    out = {}
    for name, fn in loaders.items():
        try:
            out[name] = fn()
        except FileNotFoundError:
            pass
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["pre"] = joblib.load(C.PREPROCESSOR_FILE)
        STATE["metrics"] = json.loads(C.METRICS_FILE.read_text(encoding="utf-8"))
        STATE["personas"] = json.loads(C.PERSONA_FILE.read_text(encoding="utf-8"))
        STATE["models"] = _load_models()
        STATE["default"] = STATE["metrics"]["_best_model"]
    except FileNotFoundError:
        STATE.clear()  # endpoints return 503 until run_pipeline.py has produced artifacts
    yield
    STATE.clear()


app = FastAPI(title="Hospitality Guest Persona API", version="1.0.0", lifespan=lifespan,
              description="Assigns hotel guests to behavioural clusters/personas.")


def _require_ready():
    if "models" not in STATE or not STATE["models"]:
        raise HTTPException(503, "Models not trained yet. Run run_pipeline.py first.")


@app.get("/health")
def health():
    ready = bool(STATE.get("models"))
    return {"status": "ok" if ready else "degraded", "device": C.DEVICE, "cuml": C.HAS_CUML,
            "models_loaded": sorted(STATE.get("models", {})), "default_model": STATE.get("default")}


@app.get("/models/metrics")
def models_metrics():
    _require_ready()
    return STATE["metrics"]


@app.post("/predict")
def predict(guest: GuestFeatures,
            model: str | None = Query(None, description="Model name; defaults to best by metrics")):
    _require_ready()
    name = model or STATE["default"]
    if name not in STATE["models"]:
        raise HTTPException(404, f"Unknown model '{name}'. Available: {sorted(STATE['models'])}")
    t = time.perf_counter()
    X = STATE["pre"].transform(pd.DataFrame([guest.model_dump()])[C.FEATURES]).astype(np.float32)
    probs = np.asarray(STATE["models"][name].predict_proba(X))[0]
    cid = int(probs.argmax())
    return {"model": name, "cluster_id": cid,
            "persona": STATE["personas"].get(name, {}).get(str(cid), f"Cluster {cid}"),
            "confidence": round(float(probs[cid]), 4),
            "probabilities": {str(i): round(float(p), 4) for i, p in enumerate(probs)},
            "latency_ms": round((time.perf_counter() - t) * 1000, 3), "device": C.DEVICE}
