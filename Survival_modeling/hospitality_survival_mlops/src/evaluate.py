"""Hold-out evaluation (Harrell C, Uno C, IBS, time-dependent AUC) and comparison graphs."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc,
                            integrated_brier_score)
from lifelines import KaplanMeierFitter

from .base import SurvivalModel
from .utils import get_logger

log = get_logger("evaluate")
DPI = 300

PERSONAS = {
    "High-Value Loyal Business": dict(loyalty_tier="Platinum", adr_avg=320.0, length_of_stay_avg=3.0,
                                      lead_time_days=7.0, days_since_last_stay=30.0, total_prior_stays=18,
                                      ancillary_spend_ratio=0.35, promo_email_open_rate=0.60,
                                      app_sessions_30d=25, channel_direct=1, trip_purpose_business=1,
                                      seasonality_index=1.05),
    "Occasional Leisure": dict(loyalty_tier="Silver", adr_avg=160.0, length_of_stay_avg=5.0,
                               lead_time_days=60.0, days_since_last_stay=150.0, total_prior_stays=3,
                               ancillary_spend_ratio=0.15, promo_email_open_rate=0.25, app_sessions_30d=4,
                               channel_direct=0, trip_purpose_business=0, seasonality_index=1.0),
    "At-Risk Churner": dict(loyalty_tier="Member", adr_avg=95.0, length_of_stay_avg=2.0,
                            lead_time_days=25.0, days_since_last_stay=400.0, total_prior_stays=1,
                            ancillary_spend_ratio=0.03, promo_email_open_rate=0.03, app_sessions_30d=0,
                            channel_direct=0, trip_purpose_business=0, seasonality_index=0.8),
}


def eval_grid(cfg: dict) -> np.ndarray:
    h = cfg["time_horizons"]
    return np.linspace(h["eval_grid_min"], h["eval_grid_max"], h["eval_grid_points"])


def evaluate_model(model: SurvivalModel, split: dict, cfg: dict) -> dict:
    """All metrics for one fitted model on the hold-out test set."""
    X, y_tr, y_te = split["X_test"], split["y_train"], split["y_test"]
    grid = eval_grid(cfg)
    auc_days = np.array(cfg["time_horizons"]["auc_days"], float)

    risk = model.risk(X)
    surv_grid = model.survival(X, grid)
    harrell = concordance_index_censored(y_te["event"], y_te["time"], risk)[0]
    tau = min(y_te["time"].max(), y_tr["time"].max()) - 1e-6
    uno = concordance_index_ipcw(y_tr, y_te, risk, tau=tau)[0]
    ibs = integrated_brier_score(y_tr, y_te, surv_grid, grid)

    # Time-dependent AUC uses the model's own 1 - S(t|x) as the time-specific risk score
    auc_traj, auc_mean = cumulative_dynamic_auc(y_tr, y_te, 1.0 - surv_grid, grid)
    auc_h, _ = cumulative_dynamic_auc(y_tr, y_te, 1.0 - model.survival(X, auc_days), auc_days)

    res = {
        "model": model.name,
        "device": getattr(model, "device", "cpu"),
        "train_seconds": round(float(model.train_seconds), 3),
        "c_index_harrell": float(harrell),
        "c_index_uno": float(uno),
        "integrated_brier_score": float(ibs),
        "one_minus_ibs": float(1 - ibs),
        "mean_time_auc": float(auc_mean),
        **{f"auc_{int(d)}d": float(a) for d, a in zip(auc_days, auc_h)},
        "_auc_trajectory": auc_traj.tolist(),
    }
    log.info("%-24s C=%.4f UnoC=%.4f IBS=%.4f AUC@90=%.4f  (%.1fs, %s)", model.name, harrell, uno, ibs,
             res["auc_90d"], model.train_seconds, res["device"])
    return res


def metrics_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def _palette(names):
    return dict(zip(names, sns.color_palette("tab10", len(names))))


def plot_metric_bars(results: list[dict], out: Path) -> Path:
    df = metrics_table(results)
    long = df.melt(id_vars="model", value_vars=["c_index_harrell", "c_index_uno", "one_minus_ibs"],
                   var_name="metric", value_name="value")
    long["metric"] = long["metric"].map({"c_index_harrell": "Harrell C-Index", "c_index_uno": "Uno IPCW C-Index",
                                         "one_minus_ibs": "1 - IBS"})
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=long, x="model", y="value", hue="metric", ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Model Comparison on Hold-out Test Set (higher is better)")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    path = out / "01_model_comparison_cindex_ibs.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return path


def plot_time_auc(results: list[dict], cfg: dict, out: Path) -> Path:
    grid = eval_grid(cfg)
    pal = _palette([r["model"] for r in results])
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        if r["model"] in ("kaplan_meier", "nelson_aalen"):
            continue
        ax.plot(grid, r["_auc_trajectory"], marker="o", ms=3, label=r["model"], color=pal[r["model"]])
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set(title="Time-dependent ROC-AUC (cumulative / dynamic)", xlabel="Days since last checkout",
           ylabel="AUC(t)")
    ax.legend(ncol=2, fontsize=8)
    path = out / "02_time_dependent_auc.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return path


def plot_personas(models: dict[str, SurvivalModel], preprocessor, out: Path, t_max: int = 365) -> Path:
    raw = pd.DataFrame(PERSONAS.values())
    X = preprocessor.transform(raw)
    t = np.linspace(1, t_max, 100)
    fig, axes = plt.subplots(1, len(PERSONAS), figsize=(17, 5), sharey=True)
    skip = ("kaplan_meier", "nelson_aalen")
    pal = _palette([n for n in models if n not in skip])
    for i, (ax, name) in enumerate(zip(axes, PERSONAS)):
        for mname, m in models.items():
            if mname in skip:
                continue
            ax.plot(t, m.survival(X.iloc[[i]], t)[0], label=mname, color=pal[mname], lw=1.6)
        ax.set_title(name)
        ax.set_xlabel("Days since last checkout")
        ax.axhline(0.5, ls=":", c="gray", lw=1)
    axes[0].set_ylabel("S(t | x)  = P(not yet rebooked)")
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Predicted Individual Survival Curves for Guest Personas", y=1.02)
    path = out / "03_persona_survival_curves.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return path


def plot_shap_xgb(model, X: pd.DataFrame, out: Path, max_rows: int = 2000) -> Path | None:
    import shap
    try:
        booster = model.booster
        booster.set_param({"device": "cpu"})                 # SHAP's tree explainer runs on CPU
        Xs = X[model.feature_names].sample(min(max_rows, len(X)), random_state=0)
        sv = shap.TreeExplainer(booster, model_output="raw").shap_values(Xs)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns).sort_values()
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(imp.index, imp.values, color=sns.color_palette("tab10")[0])
        ax.set(title=f"SHAP Feature Importance - {model.name} (mean |SHAP| of log-hazard)",
               xlabel="mean(|SHAP value|)")
        path = out / "04_shap_summary_xgboost.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
        # beeswarm companion
        plt.figure(figsize=(9, 7))
        shap.summary_plot(sv, Xs, show=False, plot_size=None)
        plt.title("SHAP Beeswarm - XGBoost Survival (Cox)")
        plt.savefig(out / "04b_shap_beeswarm_xgboost.png", dpi=DPI, bbox_inches="tight"); plt.close()
        return path
    except Exception as exc:                                   # SHAP must never break the pipeline
        log.warning("SHAP plot skipped: %s", exc)
        return None
    finally:
        try:
            booster.set_param({"device": model.device})
        except Exception:
            pass


def plot_calibration(models: dict[str, SurvivalModel], split: dict, horizon: int, out: Path, bins: int = 8) -> Path:
    """Predicted vs Kaplan-Meier-observed survival at `horizon`, by predicted-risk quantile."""
    X, t, e = split["X_test"], split["t_test"], split["e_test"]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    pal = _palette([n for n in models])
    for name, m in models.items():
        if name in ("kaplan_meier", "nelson_aalen"):
            continue
        pred = m.survival(X, np.array([horizon], float))[:, 0]
        q = pd.qcut(pd.Series(pred).rank(method="first"), bins, labels=False)
        xs, ys = [], []
        for b in range(bins):
            ix = (q == b).to_numpy()
            km = KaplanMeierFitter().fit(t[ix], e[ix])
            xs.append(pred[ix].mean()); ys.append(float(km.predict(horizon)))
        ax.plot(xs, ys, marker="o", ms=4, label=name, color=pal[name])
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title=f"Calibration at {horizon} days", xlabel="Predicted survival probability",
           ylabel="Observed (Kaplan-Meier) survival", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=7)
    path = out / "05_calibration.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return path


def plot_training_times(results: list[dict], out: Path) -> Path:
    df = metrics_table(results).sort_values("train_seconds")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2a9d8f" if d == "cuda" else "#8d99ae" for d in df["device"]]
    ax.barh(df["model"], df["train_seconds"], color=colors)
    ax.set(title="Training time (teal = CUDA, grey = CPU)", xlabel="seconds")
    ax.set_xscale("log")
    path = out / "06_training_time.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return path
