"""Collect every number/text the deck uses from the real artifacts -> presentation/deck_data.json.
Run from the project root:  python presentation/prepare_deck_data.py"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.common import load_config
from src.evaluator import decile_table
from src.predictor import PropensityEngine

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
cfg = load_config()
MET = ROOT / "outputs" / "metrics"
lb = pd.read_csv(MET / "master_leaderboard.csv")
best = json.loads((MET / "best_models.json").read_text())
ex = json.loads((ROOT / "outputs" / "examples" / "examples.json").read_text())
MODELS = ["xgboost", "lightgbm", "catboost", "pytorch_nn"]
SHORT = {"ota_to_direct": "OTA→Direct", "booking_abandonment": "Abandonment", "room_upgrade": "Room Upgrade",
         "ancillary_spend": "Ancillary Spend", "guest_churn": "Guest Churn", "property_repeat": "Property Repeat",
         "brand_repeat": "Brand Repeat", "cancel_noshow": "Cancel / No-Show"}
QA = {"ota_to_direct": ("Which OTA-reliant guests can we convert to book direct, and for whom does the offer change behaviour?", "Direct-booking offer, targeted by uplift"),
      "booking_abandonment": ("Which recent site/app visitors who did not book will still book direct within 60 days?", "Abandoned-search email with price hold"),
      "room_upgrade": ("Which upcoming reservations will buy a paid upgrade?", "Pre-arrival upgrade offer at T-72h"),
      "ancillary_spend": ("Which upcoming stays will spend $450+ on spa, F&B and experiences?", "Pre-sell spa / F&B / experience bundles"),
      "guest_churn": ("Which recently active guests will not book again in the next 12 months?", "Win-back offer plus bonus points"),
      "property_repeat": ("Which guests will return to the same property within 12 months?", "Anniversary-stay invitation"),
      "brand_repeat": ("Which guests will stay at a different brand property within 12 months?", "Cross-property destination packages"),
      "cancel_noshow": ("Which upcoming bookings will be cancelled or no-show?", "Deposit, reconfirmation, overbooking buffer")}

eng = PropensityEngine()
flows = []
for i, s in enumerate(cfg["scenarios"], 1):
    sc = cfg["scenarios"][s]
    rows = lb[lb.scenario == s].set_index("model")
    b = best[s]["model"]
    preds = pd.read_parquet(MET / s / f"preds_{b}.parquet")
    dec = decile_table(preds.y, preds.p_cal)
    shap = pd.read_csv(MET / s / "shap_importance.csv").head(5)
    k = 4 if s == "ota_to_direct" else 1
    L = ex[s]
    case = L[k - 1]
    r = eng.predict_scenario(s, pd.DataFrame([case["payload"]]), explain=True)[0]
    img = ROOT / "presentation" / "story_shots" / f"{s}.png"
    w, h = Image.open(img).size
    flows.append({
        "key": s, "n": i, "title": sc["title"], "short": SHORT[s], "question": QA[s][0], "action": QA[s][1],
        "window_days": sc["window_days"], "unit": sc["unit"], "base_rate": float(rows.base_rate.iloc[0]), "best_model": b,
        "models": {m: {c: float(rows.loc[m, c]) for c in ["roc_auc", "pr_auc", "logloss", "brier", "top_decile_lift", "ece", "ece_raw", "logloss_raw", "train_time_s", "tuning_time_s", "gpu_peak_mem_mb"]} | {"device": rows.loc[m, "device"], "calibration": rows.loc[m, "calibration"]} for m in MODELS},
        "decile_lift": [float(x) for x in dec.lift], "decile_gain": [float(x) for x in dec.cum_gain],
        "drivers": [{"f": a, "v": float(v)} for a, v in zip(shap.feature, shap.mean_abs_shap)],
        "ladder": [{"name": e["name"], "score": e["model_score_at_selection"], "actual": e["observed_y"], "outcome": e["observed_outcome"]} for e in L],
        "story": {"case_no": k, "case_name": case["name"], "text": case["story"], "score": r["propensity"], "decile": r["decile"],
                  "base": r["base_rate"], "nba": r["next_best_action"], "segment": r["uplift_segment"], "uplift": r["uplift"],
                  "outcome": case["observed_outcome"], "actual": case["observed_y"],
                  "drivers": [{"f": d["feature"], "v": d["value"], "dir": d["direction"], "shap": d["shap"]} for d in r["top_drivers"][:4]],
                  "image": f"story_shots/{s}.png", "img_w": w, "img_h": h}})

# cross-flow aggregates
agg = {}
for m in MODELS:
    d = lb[lb.model == m]
    agg[m] = {"roc": d.roc_auc.mean(), "pr": d.pr_auc.mean(), "ll": d.logloss.mean(), "brier": d.brier.mean(), "lift": d.top_decile_lift.mean(),
              "fit_s": d.train_time_s.mean(), "tune_s": d.tuning_time_s.mean(), "ece": d.ece.mean(), "ece_raw": d.ece_raw.mean(),
              "ll_raw": d.logloss_raw.mean(), "peak_mb": d.gpu_peak_mem_mb.mean(), "device": d.device.iloc[0],
              "roc_gap_pp": float(np.mean([(lb[lb.scenario == s].roc_auc.max() - lb[(lb.scenario == s) & (lb.model == m)].roc_auc.iloc[0]) * 100 for s in cfg["scenarios"]])),
              "pr_gap_pp": float(np.mean([(lb[lb.scenario == s].pr_auc.max() - lb[(lb.scenario == s) & (lb.model == m)].pr_auc.iloc[0]) * 100 for s in cfg["scenarios"]])),
              "wins_val": sum(1 for s in best if best[s]["model"] == m),
              "wins_test_roc": int(sum(lb[lb.scenario == s].sort_values("roc_auc").iloc[-1].model == m for s in cfg["scenarios"]))}
    agg[m] = {k: (float(v) if not isinstance(v, str) else v) for k, v in agg[m].items()}
cal = {"improved_or_equal": int((lb.logloss <= lb.logloss_raw + 1e-9).sum()), "n": int(len(lb)), "sigmoid": int((lb.calibration == "sigmoid").sum()),
       "isotonic": int((lb.calibration == "isotonic").sum()), "ece_raw_mean": float(lb.ece_raw.mean()), "ece_cal_mean": float(lb.ece.mean())}
up = json.loads((MET / "ota_to_direct" / "uplift.json").read_text())
tables = {t: len(pd.read_parquet(ROOT / "data" / "raw" / f"{t}.parquet")) for t in ["guests", "bookings", "folios", "web_sessions", "email_monthly", "surveys", "complaints"]}
out = {"flows": flows, "agg": agg, "cal": cal, "uplift": up, "tables": tables, "models": MODELS,
       "n_models": int(len(lb)), "n_trials": cfg["training"]["n_trials"],
       "cuda": eng.health()["gpu_device"], "tests": 28}
(ROOT / "presentation" / "deck_data.json").write_text(json.dumps(out, indent=1, default=float))
print("ok", len(flows), "flows;", {m: round(a["roc_gap_pp"], 2) for m, a in agg.items()}, cal)
