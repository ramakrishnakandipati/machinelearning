"""FastAPI real-time segmentation service.   Run: uvicorn src.api:app --port 8000"""
import json
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .models.classical_ml import GMMGPU, HDBSCANModel, KMeansGPU
from .models.deep_autoencoder import dec_predict, load_dec
from .models.deep_vae import load_vae, vae_latents
from .models.sequence_transformer import load_seq, seq_embeddings
from .utils import ROOT, get_logger, load_config, out_dir

CFG = load_config()
log = get_logger("api", "api_server.log", CFG)
START = time.time()
STATE = {}


# ------------------------------------------------------------------ schemas
class StayEvent(BaseModel):
    stay_gap_days: float = Field(0, ge=0)
    spend: float = Field(..., ge=0)
    property_type_id: int = Field(0, ge=0, le=3, description="0 Urban Business, 1 Resort/Leisure, 2 Boutique, 3 Extended Stay")
    lead_time: float = Field(0, ge=0)


class Guest(BaseModel):
    loyalty_tier: str = Field("Gold", pattern="^(Member|Silver|Gold|Platinum|Ambassador)$")
    tenure_months: float = Field(48, ge=0)
    home_region: str = "North America"
    age_group: str = "30-44"
    recency_days: float = Field(30, ge=0)
    frequency_stays_12m: float = Field(6, ge=0)
    monetary_total_spend: float = Field(9000, ge=0)
    avg_daily_rate: float = Field(220, ge=0)
    avg_length_of_stay: float = Field(2, ge=0)
    booking_lead_time_days: float = Field(8, ge=0)
    weekend_vs_weekday_ratio: float = Field(0.15, ge=0, le=1)
    cancellation_rate: float = Field(0.05, ge=0, le=1)
    last_minute_booking_ratio: float = Field(0.4, ge=0, le=1)
    room_tier_preference: str = Field("Deluxe", pattern="^(Standard|Deluxe|Suite|Villa)$")
    property_type_affinity: str = "Urban Business"
    spa_spend_ratio: float = Field(0.03, ge=0, le=1)
    dining_fb_spend_ratio: float = Field(0.15, ge=0, le=1)
    golf_recreation_spend_ratio: float = Field(0.02, ge=0, le=1)
    business_center_usage: float = Field(0.7, ge=0, le=1)
    direct_app_booking_ratio: float = Field(0.55, ge=0, le=1)
    ota_booking_ratio: float = Field(0.1, ge=0, le=1)
    email_open_rate: float = Field(0.5, ge=0, le=1)
    mobile_key_usage_rate: float = Field(0.7, ge=0, le=1)
    points_redemption_rate: float = Field(0.15, ge=0, le=1)
    total_stays: Optional[float] = Field(None, ge=1, description="Lifetime stays (defaults to max(12m stays, 1))")
    stay_history: Optional[List[StayEvent]] = Field(None, max_length=10, description="Last <=10 stays (oldest first); used by the sequence model")


class BatchRequest(BaseModel):
    guests: List[Guest] = Field(..., min_length=1, max_length=20000)


# ------------------------------------------------------------------ serving model
class Serving:
    def __init__(self, reg, device):
        self.reg, self.device = reg, device
        d = ROOT / "output" / "saved_models"
        self.fp = joblib.load(d / reg["preprocessing"]["pipeline"])
        self.pca = joblib.load(d / reg["preprocessing"]["pca_viz"])
        self.sn = joblib.load(d / reg["preprocessing"]["sequence_normalizer"])
        self.hdb = HDBSCANModel.load(d / reg["vip_outlier_detector"]) if reg.get("vip_outlier_detector") else None
        t, f = reg["model_type"], reg["model_file"]
        self.type = t
        if t == "kmeans":
            self.m = KMeansGPU.load(d / f, device)
        elif t == "gmm":
            self.m = GMMGPU.load(d / f, device)
        elif t == "hdbscan":
            self.m = HDBSCANModel.load(d / f)
        elif t == "dec":
            self.m = load_dec(d / f, device)
        elif t == "beta_vae_kmeans":
            self.m = load_vae(d / f, device)
            self.km = KMeansGPU.load(d / reg["cluster_file"], device)
        elif t == "sequence_kmeans":
            self.m = load_seq(d / f, device)
            self.km = KMeansGPU.load(d / reg["cluster_file"], device)
        else:
            raise ValueError(t)
        self.names = {int(k): v for k, v in reg["segment_names"].items()}

    def _seq(self, guests: List[Guest]):
        L = 10
        seq = np.zeros((len(guests), L, 4), np.float32); mask = np.zeros((len(guests), L), np.float32)
        for i, g in enumerate(guests):
            hist = g.stay_history
            if not hist:  # synthesise a single-stay history from the profile summary
                n = g.total_stays or max(g.frequency_stays_12m, 1)
                prop = ["Urban Business", "Resort/Leisure", "Boutique", "Extended Stay"].index(g.property_type_affinity) \
                    if g.property_type_affinity in ["Urban Business", "Resort/Leisure", "Boutique", "Extended Stay"] else 0
                hist = [StayEvent(stay_gap_days=0, spend=g.monetary_total_spend / n, property_type_id=prop,
                                  lead_time=g.booking_lead_time_days)]
            k = len(hist)
            for j, e in enumerate(hist):
                seq[i, L - k + j] = [e.stay_gap_days, e.spend, e.property_type_id, e.lead_time]
            mask[i, L - k:] = 1
        return self.sn.transform(seq, mask)

    @torch.no_grad()
    def predict(self, guests: List[Guest]):
        df = pd.DataFrame([g.model_dump(exclude={"stay_history"}) for g in guests])
        df["total_stays"] = pd.to_numeric(df["total_stays"], errors="coerce").astype(float)
        df["total_stays"] = df["total_stays"].fillna(df["frequency_stays_12m"].clip(lower=1))
        X = self.fp.transform(df)
        pc = self.pca.transform(X)
        t = self.type
        if t == "kmeans":
            p = self.m.predict_proba(X); lat = pc
        elif t == "gmm":
            p = self.m.predict_proba(X); lat = pc
        elif t == "hdbscan":
            lab = self.m.predict(X); p = np.eye(len(self.names))[np.clip(lab, 0, None)]; lat = pc
            return lab, p.max(1), lat, self._vip(X)
        elif t == "dec":
            _, p, lat = dec_predict(self.m, X, self.device)
        elif t == "beta_vae_kmeans":
            lat = vae_latents(self.m, X, self.device); p = self.km.predict_proba(lat)
        else:  # sequence transformer
            c, pr, mk = self._seq(guests)
            lat = seq_embeddings(self.m, c, pr, mk, self.device); p = self.km.predict_proba(lat)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return p.argmax(1), p.max(1), lat[:, :3], self._vip(X)

    def _vip(self, X):
        return (self.hdb.predict(X) < 0) if self.hdb is not None else np.zeros(len(X), bool)


def _action(name):
    base = name.split(" (cluster")[0]
    return CFG["personas"].get(base, {}).get("next_best_offer", "General loyalty offer")


# ------------------------------------------------------------------ app
@asynccontextmanager
async def lifespan(app):
    reg_path = ROOT / "output" / "saved_models" / "model_registry.json"
    if not reg_path.exists():
        log.error("model_registry.json missing - run `python -m src.train_pipeline` first")
        STATE["serving"] = None
    else:
        reg = json.load(open(reg_path, encoding="utf-8"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        STATE["registry"], STATE["serving"] = reg, Serving(reg, device)
        STATE["personas"] = json.load(open(ROOT / "output" / "saved_models" / "segment_personas.json", encoding="utf-8"))
        log.info("Loaded model '%s' v%s on %s", reg["best_model"], reg["version"], device)
    yield


app = FastAPI(title="Hospitality Customer Segmentation API", version="1.0.0", lifespan=lifespan,
              description="Real-time guest segmentation with GPU inference and marketing playbooks.")


@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    request.state.inference_ms, request.state.segment = None, None
    try:
        resp = await call_next(request)
        status = resp.status_code
    except Exception as e:  # noqa
        log.exception("Unhandled error on %s: %s", request.url.path, e)
        raise
    tot = (time.perf_counter() - t0) * 1000
    log.info("%s %s status=%d total_ms=%.2f gpu_inference_ms=%s segment=%s client=%s", request.method, request.url.path,
             status, tot, "-" if request.state.inference_ms is None else f"{request.state.inference_ms:.2f}",
             request.state.segment or "-", request.client.host if request.client else "-")
    return resp


def _serving():
    s = STATE.get("serving")
    if s is None:
        raise HTTPException(503, "Model not loaded - run the training pipeline first")
    return s


@app.get("/health", summary="Service, GPU and model status")
def health():
    reg = STATE.get("registry")
    return {"status": "ok" if reg else "model_not_loaded",
            "gpu_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "model_name": reg["best_model"] if reg else None, "model_version": reg["version"] if reg else None,
            "uptime_seconds": round(time.time() - START, 1)}


def _result(s: Serving, lab, conf, lat, vip, i):
    sid = int(lab[i])
    name = s.names.get(sid, "VIP / Outlier (noise)" if sid < 0 else f"Segment {sid}")
    return {"segment_id": sid, "segment_name": name, "confidence_score": round(float(conf[i]), 4),
            "latent_coordinates": [round(float(v), 4) for v in lat[i]], "is_vip_outlier": bool(vip[i]),
            "recommended_hospitality_action": _action(name)}


@app.post("/predict/segment", summary="Score a single guest")
def predict_segment(guest: Guest, request: Request):
    s = _serving()
    t0 = time.perf_counter()
    lab, conf, lat, vip = s.predict([guest])
    request.state.inference_ms = (time.perf_counter() - t0) * 1000
    out = _result(s, lab, conf, lat, vip, 0)
    request.state.segment = out["segment_name"]
    out["inference_ms"] = round(request.state.inference_ms, 3)
    return out


@app.post("/predict/batch", summary="High-throughput batch scoring")
def predict_batch(req: BatchRequest, request: Request):
    s = _serving()
    t0 = time.perf_counter()
    lab, conf, lat, vip = s.predict(req.guests)
    ms = (time.perf_counter() - t0) * 1000
    request.state.inference_ms = ms
    results = [_result(s, lab, conf, lat, vip, i) for i in range(len(req.guests))]
    counts = pd.Series([r["segment_name"] for r in results]).value_counts().to_dict()
    request.state.segment = f"batch[{len(results)}]:{counts}"
    return {"count": len(results), "inference_ms": round(ms, 3),
            "throughput_guests_per_s": round(len(results) / (ms / 1000), 1) if ms > 0 else None,
            "segment_counts": counts, "predictions": results}


@app.get("/segments/personas", summary="Segment statistics and marketing playbooks")
def personas():
    _serving()
    return {"model": STATE["registry"]["best_model"], "segments": STATE["personas"]}
