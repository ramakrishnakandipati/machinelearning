"""Unified training orchestrator: trains all CLTV paradigms, times them, saves artifacts and predictions."""
from __future__ import annotations

import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .features import Dataset
from .logger import append_jsonl, get_device
from .models.classical_ml import CatBoostTweedie, LGBHurdle, XGBTweedie
from .models.common import Inputs, subset
from .models.deep_sequential import SeqZILN
from .models.deep_survival import WeibullSurvival
from .models.deep_tabular import FTZILN
from .models.probabilistic import BGNBDGammaGamma

MODEL_ORDER = ["bgnbd_gg", "xgboost_tweedie", "catboost_tweedie", "lightgbm_hurdle", "ft_transformer_ziln",
               "gru_attention_ziln", "deep_weibull_survival"]


def _latency_ms_per_1k(model: Any, inp: Inputs, reps: int = 3) -> float:
    """Median wall-clock inference latency scaled to 1,000 guests (includes GPU sync)."""
    ts = []
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        model.predict(inp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) / len(inp.df) * 1000 * 1000)


def train_all(cfg: dict, ds: Dataset, lg) -> dict[str, Any]:
    """Train every model; returns {'predictions': DataFrame, 'timing': {...}, 'histories': {...}}."""
    out = cfg["out"]
    mdir = out / "models"
    dev = get_device(cfg, lg)
    torch.manual_seed(42)
    np.random.seed(42)
    tr, dtr = subset(ds, "train")
    va, dva = subset(ds, "val")
    te, dte = subset(ds, "test")
    m = cfg["models"]
    models = {
        "bgnbd_gg": BGNBDGammaGamma(ds.horizon_days),
        "xgboost_tweedie": XGBTweedie(m["xgboost"], dev),
        "catboost_tweedie": CatBoostTweedie(m["catboost"], dev),
        "lightgbm_hurdle": LGBHurdle(m["lightgbm_hurdle"]),
        "ft_transformer_ziln": FTZILN(m["ft_ziln"], dev),
        "gru_attention_ziln": SeqZILN(m["seq_ziln"], dev),
        "deep_weibull_survival": WeibullSurvival(m["survival"], ds.horizon_days, dev),
    }
    timing: dict[str, dict[str, float]] = {}
    hist: dict[str, dict] = {}
    preds = {"guest_id": np.concatenate([dva.index.values, dte.index.values]),
             "split": ["val"] * len(dva) + ["test"] * len(dte),
             "y": np.concatenate([dva["y"].values, dte["y"].values])}
    extras: dict[str, np.ndarray] = {}
    for name in MODEL_ORDER:
        mod = models[name]
        lg.info(f"=== Training {name} ===")
        t0 = time.perf_counter()
        if name == "bgnbd_gg":
            h = mod.fit(dtr, lg)
        elif name in ("xgboost_tweedie", "catboost_tweedie", "lightgbm_hurdle"):
            h = mod.fit(tr, dtr["y"].values, va, dva["y"].values, lg)
        else:
            h = mod.fit(tr, va, out, lg)
        if dev == "cuda":
            torch.cuda.synchronize()
        tt = time.perf_counter() - t0
        mod.save(mdir)
        lat = _latency_ms_per_1k(mod, te)
        pv, pt = mod.predict(va), mod.predict(te)
        preds[name] = np.concatenate([pv, pt])
        timing[name] = {"train_time_s": tt, "latency_ms_per_1k": lat}
        hist[name] = h
        append_jsonl(out / "logs" / "training_history.jsonl", {"model": name, "event": "done", "train_time_s": tt})
        lg.info(f"[{name}] trained in {tt:.1f}s | latency {lat:.2f} ms/1k guests | saved to {mdir}")
        if name == "gru_attention_ziln":
            extras["attention_test"] = mod.predict_full(te, with_attention=True)["attention"]
    pdf = pd.DataFrame(preds)
    pdf.to_parquet(out / "data" / "predictions.parquet", index=False)
    (out / "logs" / "loss_histories.json").write_text(json.dumps(hist, default=float))
    (out / "logs" / "timing.json").write_text(json.dumps(timing, indent=2))
    np.save(out / "data" / "attention_test.npy", extras["attention_test"])
    joblib.dump(models["xgboost_tweedie"].cols, mdir / "tree_feature_columns.joblib")
    return {"predictions": pdf, "timing": timing, "histories": hist, "models": models}
