"""FastAPI inference server for hospitality CLTV (single-guest, batch, health) with JSONL telemetry."""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .data_generator import CHANNELS, PROP_TIERS, REGIONS, ROOMS, TIERS
from .features import BOOKING_COLS, DAYS_PER_MONTH, build_features, build_sequences, encode_cats
from .logger import append_jsonl, get_device, get_logger, load_config
from .models.classical_ml import CatBoostTweedie, LGBHurdle, XGBTweedie
from .models.common import Inputs
from .models.deep_sequential import SeqZILN
from .models.deep_survival import WeibullSurvival
from .models.deep_tabular import FTZILN
from .models.probabilistic import BGNBDGammaGamma

VERSION = "1.0.0"
Tier = Literal["Member", "Silver", "Gold", "Platinum", "Titanium"]
Channel = Literal["Direct Web", "Mobile App", "OTA", "Corporate GDS"]
Region = Literal["North America", "Europe", "APAC", "Middle East", "LATAM"]
Prop = Literal["Luxury", "Premium", "Select Service"]
Room = Literal["Standard", "Deluxe", "Suite", "Club"]

STATE: dict[str, Any] = {}


class Stay(BaseModel):
    """One historical booking."""
    checkin_date: date
    length_of_stay: int = Field(ge=1, le=60)
    booking_lead_time: int = Field(0, ge=0, le=730)
    adr: float = Field(ge=0, description="Average daily rate ($)")
    fb_spend: float = Field(0, ge=0)
    spa_spend: float = Field(0, ge=0)
    golf_spend: float = Field(0, ge=0)
    parking_spend: float = Field(0, ge=0)
    is_cancelled: int = Field(0, ge=0, le=1)
    is_business: int = Field(0, ge=0, le=1)
    booking_channel: Channel = "Direct Web"
    property_tier: Prop = "Premium"
    room_type: Room = "Deluxe"


class Guest(BaseModel):
    """Guest profile + stay history observed before `as_of_date`."""
    guest_id: int | str = 0
    loyalty_tier: Tier
    acquisition_channel: Channel
    home_region: Region
    cobrand_card: int = Field(0, ge=0, le=1)
    is_business_guest: int = Field(0, ge=0, le=1)
    stays: list[Stay] = Field(min_length=1)
    model_config = {"json_schema_extra": {"example": {
        "guest_id": "G-100234", "loyalty_tier": "Gold", "acquisition_channel": "Mobile App", "home_region": "Europe",
        "cobrand_card": 1, "is_business_guest": 0,
        "stays": [
            {"checkin_date": "2023-03-10", "length_of_stay": 3, "booking_lead_time": 21, "adr": 289.0, "fb_spend": 120.0,
             "spa_spend": 90.0, "golf_spend": 0.0, "parking_spend": 54.0, "is_cancelled": 0, "is_business": 0,
             "booking_channel": "Mobile App", "property_tier": "Luxury", "room_type": "Suite"},
            {"checkin_date": "2023-11-22", "length_of_stay": 2, "booking_lead_time": 12, "adr": 255.0, "fb_spend": 60.0,
             "spa_spend": 0.0, "golf_spend": 0.0, "parking_spend": 36.0, "is_cancelled": 0, "is_business": 1,
             "booking_channel": "Direct Web", "property_tier": "Premium", "room_type": "Deluxe"},
            {"checkin_date": "2024-07-04", "length_of_stay": 5, "booking_lead_time": 45, "adr": 342.0, "fb_spend": 260.0,
             "spa_spend": 180.0, "golf_spend": 140.0, "parking_spend": 90.0, "is_cancelled": 0, "is_business": 0,
             "booking_channel": "Mobile App", "property_tier": "Luxury", "room_type": "Suite"}]}}}


class SingleRequest(BaseModel):
    guest: Guest
    as_of_date: date | None = Field(None, description="Scoring date (defaults to the model calibration cut-off)")
    model: str = Field("best", description="'best', 'ensemble', or a model name from /health")
    model_config = {"json_schema_extra": {"example": {"model": "best", "guest": Guest.model_config["json_schema_extra"]["example"]}}}


class BatchRequest(BaseModel):
    guests: list[Guest] = Field(min_length=1, max_length=20000)
    as_of_date: date | None = None
    model: str = "best"
    include_shap: bool = False

    @staticmethod
    def _example() -> dict:
        base = Guest.model_config["json_schema_extra"]["example"]
        g2 = {**base, "guest_id": "G-100235", "loyalty_tier": "Member", "acquisition_channel": "OTA", "home_region": "APAC",
              "cobrand_card": 0, "stays": [dict(base["stays"][0], adr=140.0, property_tier="Select Service", room_type="Standard",
                                               booking_channel="OTA", fb_spend=15.0, spa_spend=0.0, parking_spend=0.0)]}
        g3 = {**base, "guest_id": "G-100236", "loyalty_tier": "Titanium", "acquisition_channel": "Corporate GDS",
              "home_region": "North America", "is_business_guest": 1,
              "stays": [dict(base["stays"][2], checkin_date=d, is_business=1, booking_channel="Corporate GDS")
                        for d in ["2024-02-12", "2024-05-06", "2024-08-19", "2024-11-25"]]}
        return {"model": "best", "include_shap": False, "guests": [base, g2, g3]}


BatchRequest.model_config = {"json_schema_extra": {"example": BatchRequest._example()}}


def _frames(guests: list[Guest], start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    grows, brows = [], []
    for i, g in enumerate(guests):
        grows.append(dict(guest_id=i, loyalty_tier=g.loyalty_tier, acquisition_channel=g.acquisition_channel,
                          home_region=g.home_region, cobrand_card=g.cobrand_card, is_business_guest=g.is_business_guest))
        for s in g.stays:
            ok = 1 - s.is_cancelled
            room = s.adr * s.length_of_stay * ok
            anc = (s.fb_spend + s.spa_spend + s.golf_spend + s.parking_spend) * ok
            ci = pd.Timestamp(s.checkin_date)
            brows.append(dict(guest_id=i, checkin_day=float((ci - start).days), checkin_date=ci, total_revenue=room + anc,
                              room_revenue=room, ancillary_revenue=anc, adr=s.adr, length_of_stay=s.length_of_stay,
                              booking_lead_time=s.booking_lead_time, is_business=s.is_business, is_cancelled=s.is_cancelled,
                              booking_channel=s.booking_channel, property_tier=s.property_tier, room_type=s.room_type))
    return pd.DataFrame(grows), pd.DataFrame(brows)[BOOKING_COLS]


def _score(guests: list[Guest], as_of: date | None, model: str, shap_rows: bool) -> dict[str, Any]:
    cfg, S = STATE["cfg"], STATE
    start = pd.Timestamp(cfg["data"]["start_date"])
    cutoff = float((pd.Timestamp(as_of) - start).days) if as_of else S["cutoff_day"]
    g, b = _frames(guests, start)
    if (b["checkin_day"] >= cutoff).any():
        raise HTTPException(422, f"All stays must occur before as_of_date ({(start + pd.Timedelta(days=cutoff)).date()})")
    f = encode_cats(build_features(g, b, cutoff))
    seq, mask, _, sabs = build_sequences(b, f.index.values, cutoff, cfg["data"]["seq_len"])
    inp = Inputs(f, seq, mask, sabs)
    models = S["models"]
    if model == "best":
        names, label = [S["best"]], S["best"]
    elif model == "ensemble":
        names, label = S["ensemble"], "ensemble_top3(" + "+".join(S["ensemble"]) + ")"
    elif model in models:
        names, label = [model], model
    else:
        raise HTTPException(400, f"Unknown model '{model}'. Options: best, ensemble, {list(models)}")
    outs = [models[n].predict_full(inp) for n in names]
    cltv = np.mean([o["cltv"] for o in outs], 0)
    hurdle_p = models["lightgbm_hurdle"].predict_full(inp)["p_return"]
    p_ret = np.mean([o.get("p_return", hurdle_p) for o in outs], 0)
    n_st = [o["n_stays"] for o in outs if "n_stays" in o]
    per_stay = np.where(f["monetary_mean"].values > 0, f["monetary_mean"].values, S["avg_stay_value"])
    stays = np.mean(n_st, 0) if len(n_st) == len(outs) else cltv / np.maximum(per_stay, 1.0)
    cut = S["tier_cutoffs"]
    tier = np.where(cltv >= cut["VIP"], "VIP", np.where(cltv >= cut["High"], "High", np.where(cltv >= cut["Medium"], "Medium", "At-Risk")))
    res: dict[str, Any] = {"model_used": label, "cltv": cltv, "p": p_ret, "stays": stays, "tier": tier, "shap": None}
    if shap_rows:
        sv = models["xgboost_tweedie"].contribs(inp)[:, :-1]
        cols = models["xgboost_tweedie"].cols
        drivers = []
        for i in range(len(f)):
            top = np.argsort(-np.abs(sv[i]))[:5]
            drivers.append([{"feature": cols[j], "value": round(float(f.iloc[i][cols[j]]), 3), "shap_value": round(float(sv[i, j]), 4),
                             "direction": "increases CLTV" if sv[i, j] > 0 else "decreases CLTV"} for j in top])
        res["shap"] = drivers
    return res


class Prediction(BaseModel):
    guest_id: int | str
    predicted_12m_cltv: float
    return_probability: float
    expected_stays: float
    cltv_tier: Literal["VIP", "High", "Medium", "At-Risk"]
    top_shap_drivers: list[dict[str, Any]] | None = None


class SingleResponse(Prediction):
    model_used: str
    latency_ms: float


class BatchResponse(BaseModel):
    model_used: str
    n_guests: int
    latency_ms: float
    total_predicted_cltv: float
    tier_counts: dict[str, int]
    predictions: list[Prediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    out = cfg["out"]
    lg = get_logger(out, "api")
    dev = get_device(cfg, lg)
    mdir = out / "models"
    m = cfg["models"]
    H = cfg["data"]["horizon_months"] * DAYS_PER_MONTH
    models = {
        "bgnbd_gg": BGNBDGammaGamma(H).load(mdir), "xgboost_tweedie": XGBTweedie(m["xgboost"], dev).load(mdir),
        "catboost_tweedie": CatBoostTweedie(m["catboost"], dev).load(mdir), "lightgbm_hurdle": LGBHurdle(m["lightgbm_hurdle"]).load(mdir),
        "ft_transformer_ziln": FTZILN(m["ft_ziln"], dev).load(mdir), "gru_attention_ziln": SeqZILN(m["seq_ziln"], dev).load(mdir),
        "deep_weibull_survival": WeibullSurvival(m["survival"], H, dev).load(mdir),
    }
    bm = json.loads((mdir / "best_model.json").read_text())
    summary = json.loads((out / "logs" / "metrics_summary.json").read_text())
    bk = pd.read_parquet(out / "data" / "bookings.parquet", columns=["total_revenue", "is_cancelled"])
    STATE.update(cfg=cfg, device=dev, models=models, best=bm["best_model"], ensemble=bm["ensemble_members"], summary=summary,
                 tier_cutoffs=json.loads((mdir / "tier_cutoffs.json").read_text()), cutoff_day=cfg["data"]["calibration_months"] * DAYS_PER_MONTH,
                 avg_stay_value=float(bk.loc[bk.is_cancelled == 0, "total_revenue"].mean()), tele=out / "logs" / "api_telemetry.jsonl")
    lg.info(f"API ready | device={dev} | best={bm['best_model']} | ensemble={bm['ensemble_members']}")
    yield


app = FastAPI(title="Hospitality CLTV API", version=VERSION, lifespan=lifespan,
              description="12-month Customer Lifetime Value scoring for hotel loyalty guests: single guest, batch cohorts, "
                          "best-model or ensemble inference with SHAP drivers.")


@app.middleware("http")
async def telemetry(request: Request, call_next):
    t = time.perf_counter()
    request.state.n_guests = 0
    request.state.model = None
    resp = await call_next(request)
    if request.url.path not in ("/openapi.json", "/favicon.ico") and "tele" in STATE:
        append_jsonl(STATE["tele"], dict(path=request.url.path, method=request.method, status=resp.status_code,
                                         latency_ms=round((time.perf_counter() - t) * 1000, 2),
                                         n_guests=request.state.n_guests, model=request.state.model, device=STATE["device"]))
    return resp


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    """Model version, active device, and benchmark metrics of the served model."""
    best = STATE["best"]
    met = STATE["summary"]["metrics"]
    return {"status": "ok", "model_version": VERSION, "device": STATE["device"], "best_model": best,
            "ensemble_members": STATE["ensemble"], "available_models": list(STATE["models"]),
            "benchmark_metrics_test": {k: round(v, 4) for k, v in met[best].items()},
            "ensemble_metrics_test": {k: round(v, 4) for k, v in met["ensemble_top3"].items()},
            "tier_cutoffs_usd": STATE["tier_cutoffs"]}


@app.post("/predict/guest", response_model=SingleResponse, tags=["prediction"])
def predict_guest(req: SingleRequest, request: Request) -> SingleResponse:
    """Score one guest: 12-month CLTV, return probability, expected stays, tier and top SHAP drivers."""
    t = time.perf_counter()
    r = _score([req.guest], req.as_of_date, req.model, shap_rows=True)
    request.state.n_guests, request.state.model = 1, r["model_used"]
    return SingleResponse(guest_id=req.guest.guest_id, predicted_12m_cltv=round(float(r["cltv"][0]), 2),
                          return_probability=round(float(r["p"][0]), 4), expected_stays=round(float(r["stays"][0]), 3),
                          cltv_tier=str(r["tier"][0]), top_shap_drivers=r["shap"][0], model_used=r["model_used"],
                          latency_ms=round((time.perf_counter() - t) * 1000, 2))


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(req: BatchRequest, request: Request) -> BatchResponse:
    """Vectorised scoring of a guest cohort (features, sequences and models run once over the whole batch)."""
    t = time.perf_counter()
    r = _score(req.guests, req.as_of_date, req.model, shap_rows=req.include_shap)
    request.state.n_guests, request.state.model = len(req.guests), r["model_used"]
    preds = [Prediction(guest_id=g.guest_id, predicted_12m_cltv=round(float(r["cltv"][i]), 2),
                        return_probability=round(float(r["p"][i]), 4), expected_stays=round(float(r["stays"][i]), 3),
                        cltv_tier=str(r["tier"][i]), top_shap_drivers=r["shap"][i] if r["shap"] else None)
             for i, g in enumerate(req.guests)]
    return BatchResponse(model_used=r["model_used"], n_guests=len(preds), latency_ms=round((time.perf_counter() - t) * 1000, 2),
                         total_predicted_cltv=round(float(r["cltv"].sum()), 2),
                         tier_counts=dict(pd.Series(r["tier"]).value_counts()), predictions=preds)
