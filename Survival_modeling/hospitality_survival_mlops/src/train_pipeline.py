"""End-to-end orchestrator: data -> EDA -> split -> Optuna tuning -> training -> evaluation -> persistence."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

from . import data_generator, eda_visuals, evaluate  # noqa: E402
from .features import load_and_split
from .models_classical import (CoxPHModel, KaplanMeierModel, LogNormalAFTModel, NelsonAalenModel,
                               WeibullAFTModel)
from .models_deep import DeepHitModel, DeepSurvModel
from .models_tree import RSFModel, XGBSurvivalModel
from .registry import ARTIFACTS, BASELINES
from .utils import ROOT, get_device, get_logger, load_config, resolve, seed_everything

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
log = get_logger("pipeline")


# ----------------------------------------------------------------------------------------------
# Hyper-parameter tuning (validation C-index on an inner split of the TRAIN set only)
# ----------------------------------------------------------------------------------------------
def _val_c(model, Xv, tv, ev) -> float:
    return float(concordance_index_censored(ev.astype(bool), tv, model.risk(Xv))[0])


def tune(name: str, cfg: dict, dev, inner: dict) -> tuple[dict, float]:
    """Return (best_params, best_val_c). `inner` holds X_tr/t_tr/e_tr/X_va/t_va/e_va."""
    n_trials = cfg["optuna"]["n_trials"].get(name, 0)
    base = copy.deepcopy(cfg["models"][name])
    if not cfg["optuna"]["enabled"] or n_trials <= 0:
        return base, float("nan")
    Xt, tt, et, Xv, tv, ev = (inner[k] for k in ("X_tr", "t_tr", "e_tr", "X_va", "t_va", "e_va"))
    seed = cfg["seed"]
    best_iter: dict[int, int] = {}

    def objective(trial: optuna.Trial) -> float:
        if name == "coxph":
            p = dict(penalizer=trial.suggest_float("penalizer", 1e-3, 1.0, log=True),
                     l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0))
            m = CoxPHModel(**p)
        elif name == "rsf":
            p = dict(n_estimators=100, min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
                     max_depth=trial.suggest_int("max_depth", 6, 16),
                     max_features=trial.suggest_categorical("max_features", ["sqrt", 0.5]))
            m = RSFModel(seed=seed, **p)
        elif name in ("xgb_cox", "xgb_aft"):
            p = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                     max_depth=trial.suggest_int("max_depth", 2, 7),
                     subsample=trial.suggest_float("subsample", 0.5, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                     min_child_weight=trial.suggest_float("min_child_weight", 1, 20),
                     reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10, log=True))
            if name == "xgb_aft":
                p["aft_loss_distribution"] = trial.suggest_categorical("aft_loss_distribution",
                                                                       ["normal", "logistic", "extreme"])
                p["aft_loss_distribution_scale"] = trial.suggest_float("aft_loss_distribution_scale", 0.5, 2.0)
            m = XGBSurvivalModel(kind=name.split("_")[1], device=dev.xgb_device, seed=seed, n_estimators=1500, **p)
            m.fit(Xt, tt, et, X_val=Xv, t_val=tv, e_val=ev)
            trial.set_user_attr("best_iteration", int(m.booster.best_iteration) + 1)
            return _val_c(m, Xv, tv, ev)
        elif name == "deepsurv":
            p = dict(hidden=list(map(int, trial.suggest_categorical("hidden", ["64-32", "128-64-32", "256-128-64"])
                                     .split("-"))),
                     dropout=trial.suggest_float("dropout", 0.0, 0.5),
                     lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                     weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                     activation=trial.suggest_categorical("activation", ["selu", "relu"]),
                     batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048]))
            m = DeepSurvModel(epochs=min(cfg["models"]["deepsurv"]["epochs"], 80), patience=12,
                              device=dev.torch_device, seed=seed, **p)
        elif name == "deephit":
            p = dict(hidden=list(map(int, trial.suggest_categorical("hidden", ["64-32", "128-64", "256-128"])
                                     .split("-"))),
                     dropout=trial.suggest_float("dropout", 0.0, 0.5),
                     lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                     weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                     alpha=trial.suggest_float("alpha", 0.7, 1.0),
                     sigma=trial.suggest_float("sigma", 0.05, 0.5, log=True))
            m = DeepHitModel(epochs=min(cfg["models"]["deephit"]["epochs"], 80), patience=12,
                             batch_size=1024, device=dev.torch_device, seed=seed, **p)
        else:
            raise KeyError(name)
        m.fit(Xt, tt, et, X_val=Xv, t_val=tv, e_val=ev) if name in ("deepsurv", "deephit") else m.fit(Xt, tt, et)
        return _val_c(m, Xv, tv, ev)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    log.info("Optuna[%s] %d trials in %.1fs | best val C=%.4f | %s", name, n_trials, time.perf_counter() - t0,
             study.best_value, study.best_params)
    study.trials_dataframe().to_csv(resolve(cfg, "training_logs") / f"optuna_{name}.csv", index=False)

    params = dict(study.best_params)
    if name in ("deepsurv", "deephit"):
        params["hidden"] = list(map(int, params["hidden"].split("-")))
    if name.startswith("xgb"):
        params["n_estimators"] = int(study.best_trial.user_attrs["best_iteration"])
    return {**base, **params}, float(study.best_value)


# ----------------------------------------------------------------------------------------------
def build_final_models(cfg: dict, dev, tuned: dict) -> list:
    seed = cfg["seed"]
    mc = cfg["models"]

    def kw(name, drop=()):
        return {k: v for k, v in tuned.get(name, mc[name]).items() if k not in drop}

    return [
        KaplanMeierModel(), NelsonAalenModel(),
        CoxPHModel(**kw("coxph")),
        WeibullAFTModel(penalizer=0.01), LogNormalAFTModel(penalizer=0.01),
        RSFModel(seed=seed, **kw("rsf")),
        XGBSurvivalModel("cox", device=dev.xgb_device, seed=seed, **kw("xgb_cox")),
        XGBSurvivalModel("aft", device=dev.xgb_device, seed=seed, **kw("xgb_aft")),
        DeepSurvModel(device=dev.torch_device, seed=seed, **kw("deepsurv")),
        DeepHitModel(device=dev.torch_device, seed=seed, bins=cfg["time_horizons"]["deephit_bins"],
                     **kw("deephit")),
    ]


def run(config_path: str | None = None, quick: bool = False, n_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    if quick:                                         # smoke-test profile
        cfg["data"]["n_samples"] = n_override or 3000
        for k in cfg["optuna"]["n_trials"]:
            cfg["optuna"]["n_trials"][k] = 2
        for k in ("deepsurv", "deephit"):
            cfg["models"][k]["epochs"] = 30
        cfg["models"]["rsf"]["n_estimators"] = 60
    elif n_override:
        cfg["data"]["n_samples"] = n_override

    seed_everything(cfg["seed"])
    dev = get_device(cfg)
    log.info("=" * 90)
    log.info("Hospitality survival pipeline | python %s | device=%s (%s) | CUDA available=%s",
             platform.python_version(), dev.torch_device, dev.name, dev.cuda_available)
    t_start = time.perf_counter()

    # 1. data ---------------------------------------------------------------------------------
    df = data_generator.generate(cfg["data"]["n_samples"], cfg["data"]["target_censoring"],
                                 cfg["data"]["max_follow_up_days"], cfg["seed"])
    df.to_csv(resolve(cfg, "data"), index=False)

    # 2. EDA ----------------------------------------------------------------------------------
    eda_visuals.run(df, cfg)

    # 3. split + preprocessing ----------------------------------------------------------------
    split = load_and_split(resolve(cfg, "data"), cfg["split"]["test_size"], cfg["seed"])
    models_dir, logs_dir, graphs_dir = resolve(cfg, "models"), resolve(cfg, "training_logs"), resolve(cfg, "comparison_graphs")
    split["preprocessor"].save(models_dir / "preprocessor.pkl")
    log.info("train=%d test=%d | event rate train=%.3f test=%.3f", len(split["X_train"]), len(split["X_test"]),
             split["e_train"].mean(), split["e_test"].mean())

    # 4. tuning (inner validation split of the training set) -----------------------------------
    idx = np.arange(len(split["X_train"]))
    i_tr, i_va = train_test_split(idx, test_size=0.2, random_state=cfg["seed"], stratify=split["e_train"])
    inner = dict(X_tr=split["X_train"].iloc[i_tr], t_tr=split["t_train"][i_tr], e_tr=split["e_train"][i_tr],
                 X_va=split["X_train"].iloc[i_va], t_va=split["t_train"][i_va], e_va=split["e_train"][i_va])
    tuned, tuning_scores = {}, {}
    for name in ("coxph", "rsf", "xgb_cox", "xgb_aft", "deepsurv", "deephit"):
        tuned[name], tuning_scores[name] = tune(name, cfg, dev, inner)

    # 5. final training on the full training set -------------------------------------------------
    models = build_final_models(cfg, dev, tuned)
    fitted = {}
    for m in models:
        log.info("Training %s on %s ...", m.name, getattr(m, "device", "cpu"))
        m.fit(split["X_train"], split["t_train"], split["e_train"])
        if m.name == "coxph":
            ph = m.ph_violations()
            ph.to_csv(logs_dir / "coxph_proportional_hazards_test.csv")
            log.info("CoxPH PH-test violations (p<0.01): %s", list(ph.index[ph.violates_PH]))
        if hasattr(m, "history"):
            m.save_history(logs_dir)
        m.save(models_dir)
        fitted[m.name] = m
        log.info("  -> %s trained in %.2fs", m.name, m.train_seconds)

    # 6. evaluation -----------------------------------------------------------------------------
    results = [evaluate.evaluate_model(m, split, cfg) for m in fitted.values()]
    table = evaluate.metrics_table(results)
    table.to_csv(logs_dir / "model_comparison.csv", index=False)

    evaluate.plot_metric_bars(results, graphs_dir)
    evaluate.plot_time_auc(results, cfg, graphs_dir)
    evaluate.plot_personas(fitted, split["preprocessor"], graphs_dir)
    evaluate.plot_calibration(fitted, split, 180, graphs_dir)
    evaluate.plot_training_times(results, graphs_dir)
    evaluate.plot_shap_xgb(fitted["xgboost_cox"], split["X_test"], graphs_dir)

    # 7. pick the best deployable model ---------------------------------------------------------
    cand = table[~table["model"].isin(BASELINES)].copy()
    cand = cand[cand["integrated_brier_score"] < 0.25]
    best_row = cand.sort_values(["c_index_uno", "integrated_brier_score"], ascending=[False, True]).iloc[0]
    best = best_row["model"]
    (models_dir / "best_model.json").write_text(json.dumps({
        "name": best, "artifact": ARTIFACTS[best], "c_index_uno": float(best_row["c_index_uno"]),
        "integrated_brier_score": float(best_row["integrated_brier_score"]),
        "trained_on_device": getattr(fitted[best], "device", "cpu"),
    }, indent=2))
    log.info("Best model for serving: %s (Uno C=%.4f, IBS=%.4f)", best, best_row["c_index_uno"],
             best_row["integrated_brier_score"])

    # 8. metrics summary --------------------------------------------------------------------------
    summary = {
        "run": {"device": dev.torch_device, "gpu_name": dev.name, "cuda_available": dev.cuda_available,
                "n_samples": int(len(df)), "censoring_rate": float(1 - df.event_observed.mean()),
                "train_size": int(len(split["X_train"])), "test_size": int(len(split["X_test"])),
                "quick_mode": quick, "total_seconds": round(time.perf_counter() - t_start, 1)},
        "best_model": best,
        "tuning": {"best_params": tuned, "val_c_index": tuning_scores},
        "models": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
    }
    (logs_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info("Pipeline finished in %.1fs | metrics -> %s", summary["run"]["total_seconds"],
             logs_dir / "metrics_summary.json")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="small smoke-test run (N=3000, 2 Optuna trials)")
    ap.add_argument("--n", type=int, default=None, help="override number of synthetic guests")
    a = ap.parse_args()
    run(a.config, a.quick, a.n)
