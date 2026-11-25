"""GPU multi-model trainer: Optuna tuning (+pruning), stratified-group CV, calibration, artifact saving."""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from .calibration import Calibrator
from .common import cuda_info, get_logger, gpu_enabled, load_config, path_of
from .evaluator import compute_metrics
from .feature_engineering import build_preprocessor
from .models.deep_tabular import TorchTabularModel, suggest_nn
from .models.gbdt_models import CLASSES, SUGGEST, lightgbm_device
from .models.uplift_model import TLearnerUplift, auuc

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
MODEL_ORDER = ["xgboost", "lightgbm", "catboost", "pytorch_nn"]


class GPUMonitor:
    """Polls nvidia-smi in a thread to record peak GPU memory (MB) and utilisation during a block."""

    def __init__(self, interval=0.4):
        self.interval, self.peak_mem, self.peak_util, self._stop = interval, 0.0, 0.0, threading.Event()

    def _poll(self):
        while not self._stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                self.peak_mem = max(self.peak_mem, float(o[0])); self.peak_util = max(self.peak_util, float(o[1]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self.t = threading.Thread(target=self._poll, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self.t.join(timeout=3)


class Data:
    """Loaded + preprocessed scenario data."""

    def __init__(self, scen, cfg, fast):
        d = path_of(cfg, "processed") / scen
        self.schema = json.loads((d / "schema.json").read_text())
        self.tr, va, self.te = (pd.read_parquet(d / f"{s}.parquet") for s in ("train", "val", "test"))
        num, cat = self.schema["numeric"], self.schema["categorical"]
        self.num, self.cat, self.cols = num, cat, cat + num
        # val split in two disjoint, guest-hashed halves: early-stopping/tuning vs calibration/model-selection
        h = (va.guest_id.astype(np.int64) * 2654435761 % (2 ** 32)) % 2
        self.va_es, self.va_cal = va[h == 0].reset_index(drop=True), va[h == 1].reset_index(drop=True)
        if fast:
            self.tr = self.tr.sample(min(len(self.tr), 20000), random_state=cfg["seed"]).reset_index(drop=True)
        self.pre_g = build_preprocessor(num, cat, "gbdt").fit(self.tr[self.cols])
        self.pre_n = build_preprocessor(num, cat, "nn").fit(self.tr[self.cols])
        self.cat_cards = [len(c) for c in self.pre_n.named_transformers_["cat"].categories_]
        self.X = {}
        for mode, pre in (("gbdt", self.pre_g), ("nn", self.pre_n)):
            self.X[mode] = {k: pre.transform(df[self.cols]) for k, df in
                            dict(tr=self.tr, es=self.va_es, cal=self.va_cal, te=self.te).items()}
        self.y = {k: df["y"].to_numpy() for k, df in dict(tr=self.tr, es=self.va_es, cal=self.va_cal, te=self.te).items()}
        self.groups = self.tr["guest_id"].to_numpy()


def make_model(name, data: Data, cfg, use_gpu, fast, max_epochs=None):
    t = cfg["training"]
    if name == "pytorch_nn":
        return TorchTabularModel(data.cat, data.cat_cards, use_gpu, max_epochs or (8 if fast else t["nn_max_epochs"]), seed=cfg["seed"])
    return CLASSES[name](data.cat, use_gpu, es_rounds=t["early_stopping_rounds"], max_rounds=300 if fast else (500 if name == "catboost" else 1500), seed=cfg["seed"])


def train_one(scen, name, data: Data, cfg, use_gpu, fast, n_trials, log):
    t0 = time.time()
    mode = "nn" if name == "pytorch_nn" else "gbdt"
    X, y = data.X[mode], data.y
    rng = np.random.RandomState(cfg["seed"])
    n_tune = min(len(y["tr"]), 12000 if fast else (30000 if name == "catboost" else cfg["training"]["tune_max_rows"]))
    tidx = rng.choice(len(y["tr"]), n_tune, replace=False)
    Xt, yt = X["tr"].iloc[tidx].reset_index(drop=True), y["tr"][tidx]
    suggest = suggest_nn if name == "pytorch_nn" else SUGGEST[name]

    def objective(trial):
        params = suggest(trial)
        m = make_model(name, data, cfg, use_gpu, fast, max_epochs=12 if name == "pytorch_nn" and not fast else None).fit(
            Xt, yt, X["es"], y["es"], params, trial)
        p = np.clip(m.predict_proba(X["es"]), 1e-6, 1 - 1e-6)
        trial.set_user_attr("val_logloss", float(log_loss(y["es"], p)))
        return float(average_precision_score(y["es"], p))

    warm = 3 if name == "pytorch_nn" else 20
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=warm))
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    best = study.best_params
    tuning_s = time.time() - t0
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)

    # stratified (group-aware) K-fold CV on the tuning sample with the best params
    cv_scores, folds = [], 2 if fast else (3 if name == "catboost" else cfg["training"]["cv_folds"])
    for k, (a, b) in enumerate(StratifiedGroupKFold(folds, shuffle=True, random_state=cfg["seed"]).split(Xt, yt, data.groups[tidx])):
        m = make_model(name, data, cfg, use_gpu, fast).fit(Xt.iloc[a], yt[a], X["es"], y["es"], best)
        cv_scores.append(average_precision_score(yt[b], m.predict_proba(Xt.iloc[b])))

    # final fit on the full training set (timed, GPU memory monitored)
    mem_before = cuda_info().get("used_mem_gb")
    t1 = time.time()
    with GPUMonitor() as mon:
        model = make_model(name, data, cfg, use_gpu, fast).fit(X["tr"], y["tr"], X["es"], y["es"], best)
    fit_s = time.time() - t1

    # calibration on the held-out calibration half of validation
    p_cal_raw = model.predict_proba(X["cal"])
    cal = Calibrator(cfg["training"]["calibration"]).fit(p_cal_raw, y["cal"])
    p_te_raw = model.predict_proba(X["te"])
    p_te = cal.predict(p_te_raw)

    mdir = path_of(cfg, "models") / scen
    mdir.mkdir(parents=True, exist_ok=True)
    model.save(mdir / f"{name}.{model.ext}")
    joblib.dump(cal, mdir / f"{name}_calibrator.joblib")
    ometr = path_of(cfg, "metrics") / scen
    ometr.mkdir(parents=True, exist_ok=True)
    preds = pd.DataFrame({"guest_id": data.te.guest_id.values, "y": y["te"], "p_raw": p_te_raw, "p_cal": p_te})
    if "booking_id" in data.te:
        preds["booking_id"] = data.te["booking_id"].values
    preds.to_parquet(ometr / f"preds_{name}.parquet", index=False)
    res = {
        "scenario": scen, "model": name, "device": "cuda" if (use_gpu and (name != "lightgbm" or lightgbm_device() != "cpu")) else "cpu",
        "lightgbm_device": lightgbm_device() if name == "lightgbm" else None,
        "best_params": best, "n_trials": len(study.trials), "n_pruned": n_pruned, "best_trial_value": study.best_value,
        "tuning_seconds": round(tuning_s, 2), "final_fit_seconds": round(fit_s, 2), "best_iteration": getattr(model, "best_iteration", None),
        "cv": {"folds": folds, "pr_auc_mean": float(np.mean(cv_scores)), "pr_auc_std": float(np.std(cv_scores))},
        "calibration_method": cal.chosen, "calibration_cv_logloss": cal.scores,
        "val_pr_auc_raw": float(average_precision_score(y["cal"], p_cal_raw)),
        "test_raw": compute_metrics(y["te"], p_te_raw), "test_calibrated": compute_metrics(y["te"], p_te),
        "gpu_peak_mem_mb": mon.peak_mem, "gpu_peak_util_pct": mon.peak_util, "gpu_mem_gb_before": mem_before,
        "n_train_rows": int(len(y["tr"])), "n_features": len(data.cols),
    }
    (ometr / f"{name}.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame([{**t.params, "value": t.value, "state": t.state.name} for t in study.trials]).to_csv(ometr / f"optuna_{name}.csv", index=False)
    log.info("  %-11s PR-AUC=%.4f ROC=%.4f LL=%.4f(raw %.4f) lift@10=%.2f | tune %.0fs fit %.1fs | gpu peak %.0f MB | cal=%s",
             name, res["test_calibrated"]["pr_auc"], res["test_calibrated"]["roc_auc"], res["test_calibrated"]["logloss"],
             res["test_raw"]["logloss"], res["test_calibrated"]["top_decile_lift"], tuning_s, fit_s, mon.peak_mem, cal.chosen)
    return res


def train_uplift(data: Data, cfg, use_gpu, log, scen="ota_to_direct"):
    X, y = data.X["gbdt"], data.y
    t_tr, t_es, t_te = (data.tr.treatment.to_numpy(), data.va_es.treatment.to_numpy(), data.te.treatment.to_numpy())
    t0 = time.time()
    up = TLearnerUplift(use_gpu, cfg["seed"]).fit(X["tr"], y["tr"], t_tr, X["es"], y["es"], t_es)
    up.set_thresholds(up.predict(X["es"]))
    out = up.predict(X["te"])
    out["segment"] = up.segment(out)
    out["y"], out["treatment"], out["guest_id"] = y["te"], t_te, data.te.guest_id.values
    # oracle-free evaluation: AUUC for uplift ranking vs baseline-propensity ranking vs random
    metrics = {"auuc_uplift": auuc(out.y, out.treatment, out.uplift), "auuc_baseline_propensity": auuc(out.y, out.treatment, out.p_control + out.p_treated),
               "auuc_random": auuc(out.y, out.treatment, np.random.default_rng(0).random(len(out))),
               "ate_observed": float(out[out.treatment == 1].y.mean() - out[out.treatment == 0].y.mean()),
               "segments": out.groupby("segment").apply(lambda g: {"n": int(len(g)), "realised_uplift": float(g[g.treatment == 1].y.mean() - g[g.treatment == 0].y.mean())}).to_dict(),
               "fit_seconds": round(time.time() - t0, 2)}
    mdir = path_of(cfg, "models") / scen; ometr = path_of(cfg, "metrics") / scen
    up.save(mdir / "uplift_tlearner.joblib")
    joblib.dump(data.pre_g, mdir / "preprocessor_gbdt.joblib")
    out.to_parquet(ometr / "uplift_preds.parquet", index=False)
    (ometr / "uplift.json").write_text(json.dumps(metrics, indent=2))
    log.info("  uplift T-Learner AUUC=%.1f (baseline-propensity %.1f, random %.1f) segments=%s", metrics["auuc_uplift"],
             metrics["auuc_baseline_propensity"], metrics["auuc_random"], {k: v["n"] for k, v in metrics["segments"].items()})


def run_scenario(scen, cfg, models, fast, force, n_trials, log):
    use_gpu = gpu_enabled(cfg)
    log.info("=== %s | %s | window=%dd | GPU=%s %s", scen, cfg["scenarios"][scen]["title"], cfg["scenarios"][scen]["window_days"], use_gpu, cuda_info().get("device_name"))
    data = Data(scen, cfg, fast)
    log.info("  rows train=%d val_es=%d val_cal=%d test=%d | features=%d (cat=%d) | pos rate train=%.3f test=%.3f",
             len(data.tr), len(data.va_es), len(data.va_cal), len(data.te), len(data.cols), len(data.cat), data.y["tr"].mean(), data.y["te"].mean())
    mdir = path_of(cfg, "models") / scen; mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(data.pre_g, mdir / "preprocessor_gbdt.joblib"); joblib.dump(data.pre_n, mdir / "preprocessor_nn.joblib")
    (mdir / "schema.json").write_text(json.dumps({**data.schema, "cat_cards": data.cat_cards}, indent=2))
    for name in models:
        if (path_of(cfg, "metrics") / scen / f"{name}.json").exists() and not force:
            log.info("  %-11s skipped (checkpoint exists)", name); continue
        try:
            train_one(scen, name, data, cfg, use_gpu, fast, n_trials, log)
        except Exception as e:  # keep the pipeline going; surface the failure
            log.exception("  %s failed: %s", name, e)
    if cfg["scenarios"][scen].get("uplift") and "treatment" in data.tr:
        if force or not (path_of(cfg, "metrics") / scen / "uplift.json").exists():
            train_uplift(data, cfg, use_gpu, log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="smoke-test mode: tiny data, 3 trials")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    log = get_logger("trainer", cfg, "training.log")
    ci = cuda_info()
    log.info("CUDA verification: %s", ci)
    log.info("LightGBM device after probe: %s", lightgbm_device())
    n_trials = a.trials or (3 if a.fast else cfg["training"]["n_trials"])
    for scen in a.scenarios or list(cfg["scenarios"]):
        run_scenario(scen, cfg, a.models or cfg["training"]["models"], a.fast, a.force, n_trials, log)
    from .evaluator import write_leaderboard
    write_leaderboard(cfg)


if __name__ == "__main__":
    main()
