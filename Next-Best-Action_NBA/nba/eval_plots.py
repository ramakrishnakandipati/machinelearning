"""Module 5b - Validation & policy-comparison plots (outputs/validation_plots, outputs/comparison_plots)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve

from .artifact_manager import ArtifactManager
from .config import ACTION_COSTS, ACTION_IDS, ACTION_SHORT, SEGMENT_ORDER, TREATMENT_IDS, UNSUB_RATE_CAP
from .evaluation import PolicyReport, legacy_policy_probs
from .features import assign_segment
from .plot_utils import ACTION_COLORS, apply_style
from .synthetic_data import legacy_crm_distribution
from .uplift_model import make_outcome_regressor, make_propensity_model, with_action_onehot


# --------------------------------------------------------------------------- #
# Validation plots
# --------------------------------------------------------------------------- #
def ipw_gain(y: np.ndarray, a_logged: np.ndarray, p: np.ndarray, action: int) -> np.ndarray:
    """Per-row unbiased estimate of tau_action(x): 1[T=a]Y/p - 1[T=0]Y/p."""
    return (a_logged == action) * y / p - (a_logged == 0) * y / p


def auuc(gain: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Qini curve (cumulative IPW gain, descending score) and AUUC vs random."""
    order = np.argsort(-score)
    cum = np.cumsum(gain[order])
    frac = np.arange(1, len(gain) + 1) / len(gain)
    random = frac * cum[-1]
    return frac, cum, float(np.trapezoid(cum, frac)), float(np.trapezoid(random, frac))


def plot_qini(am: ArtifactManager, log: pd.DataFrame, tau_margin: np.ndarray) -> tuple[Path, dict[str, float]]:
    """Multi-treatment Qini curves + AUUC against random targeting."""
    apply_style()
    y = log["gross_margin_dollars"].to_numpy(); a = log["action_id"].to_numpy(); p = log["propensity_score"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    out: dict[str, float] = {}
    for j, (act, ax) in enumerate(zip(TREATMENT_IDS, axes.ravel())):
        frac, cum, area, rand = auuc(ipw_gain(y, a, p, act), tau_margin[:, j])
        ax.plot(frac, cum, color=ACTION_COLORS[act], lw=2, label=f"model (AUUC {area:,.0f})")
        ax.plot(frac, frac * cum[-1], color="#555555", ls="--", label=f"random (AUUC {rand:,.0f})")
        ax.fill_between(frac, frac * cum[-1], cum, color=ACTION_COLORS[act], alpha=0.12)
        ax.set_title(f"{ACTION_SHORT[act]}  | AUUC lift over random: {area - rand:+,.0f}")
        ax.set_xlabel("fraction of customers targeted (ranked by predicted uplift)")
        ax.set_ylabel("cumulative incremental margin ($, IPW)")
        out[f"auuc_lift_a{act}"] = area - rand
    return am.save_figure(fig, "validation_plots", "qini_and_uplift_curves",
                          "Qini / uplift curves vs. random targeting (holdout)"), out


def plot_cate_calibration(am: ArtifactManager, log: pd.DataFrame, tau_margin: np.ndarray, n_bins: int = 10) -> tuple[Path, dict[str, float]]:
    """Predicted CATE decile vs observed IPW-adjusted incremental margin."""
    apply_style()
    y = log["gross_margin_dollars"].to_numpy(); a = log["action_id"].to_numpy(); p = log["propensity_score"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    out: dict[str, float] = {}
    rng = np.random.default_rng(0)

    def hajek_diff(idx: np.ndarray, act: int) -> float:
        t, c = idx[a[idx] == act], idx[a[idx] == 0]
        if len(t) < 5 or len(c) < 5:
            return np.nan
        return float(np.average(y[t], weights=1 / p[t]) - np.average(y[c], weights=1 / p[c]))

    for j, (act, ax) in enumerate(zip(TREATMENT_IDS, axes.ravel())):
        dec = pd.qcut(tau_margin[:, j], n_bins, labels=False, duplicates="drop")
        pred, obs, lo, hi = [], [], [], []
        for d in np.unique(dec):
            idx = np.where(dec == d)[0]
            pred.append(tau_margin[idx, j].mean()); obs.append(hajek_diff(idx, act))
            boots = [hajek_diff(rng.choice(idx, len(idx)), act) for _ in range(150)]
            lo.append(np.nanpercentile(boots, 2.5)); hi.append(np.nanpercentile(boots, 97.5))
        pred, obs, lo, hi = map(np.array, (pred, obs, lo, hi))
        ax.errorbar(pred, obs, yerr=[np.clip(obs - lo, 0, None), np.clip(hi - obs, 0, None)], fmt="o-",
                    color=ACTION_COLORS[act], capsize=3, label="observed (IPW, 95% CI)")
        lim = [min(pred.min(), np.nanmin(lo)), max(pred.max(), np.nanmax(hi))]
        ax.plot(lim, lim, "k--", lw=1, label="perfect calibration")
        rho = spearmanr(pred, obs, nan_policy="omit")[0]
        ax.set_title(f"{ACTION_SHORT[act]}  (decile rank corr = {rho:.2f})")
        ax.set_xlabel("mean predicted CATE in decile ($)")
        ax.set_ylabel("observed incremental margin ($)")
        out[f"decile_spearman_a{act}"] = float(rho)
    return am.save_figure(fig, "validation_plots", "cate_calibration_by_decile",
                          "CATE calibration by predicted-uplift decile (holdout)"), out


def plot_nuisance_diagnostics(am: ArtifactManager, x_train: pd.DataFrame, log_train: pd.DataFrame,
                              x_test: pd.DataFrame, log_test: pd.DataFrame, seed: int = 42) -> Path:
    """Calibration + residual checks of the propensity and outcome nuisance models."""
    apply_style()
    t_tr, t_te = log_train["action_id"].to_numpy(), log_test["action_id"].to_numpy()
    prop = make_propensity_model(seed).fit(x_train, t_tr)
    pp = prop.predict_proba(x_test)
    om = make_outcome_regressor(seed).fit(with_action_onehot(x_train, t_tr), log_train["gross_margin_dollars"].to_numpy())
    yhat = om.predict(with_action_onehot(x_test, t_te))
    y = log_test["gross_margin_dollars"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    for act in ACTION_IDS:
        fx, fy = calibration_curve((t_te == act).astype(int), pp[:, act], n_bins=8, strategy="quantile")
        ax.plot(fx, fy, "o-", color=ACTION_COLORS[act], label=ACTION_SHORT[act])
    ax.plot([0, 0.7], [0, 0.7], "k--", lw=1)
    ax.set_title("Propensity model calibration (one-vs-rest)")
    ax.set_xlabel("predicted P(T=a | x)"); ax.set_ylabel("observed frequency")

    ax = axes[0, 1]
    pred_logged = pp[np.arange(len(t_te)), t_te]
    ax.scatter(log_test["propensity_score"], pred_logged, s=4, alpha=0.25, color="#0072B2")
    ax.plot([0, 0.7], [0, 0.7], "k--", lw=1, label="y = x")
    ax.set_title("Estimated vs. logged propensity (positivity / mis-specification)")
    ax.set_xlabel("logged propensity (ground truth of logging policy)"); ax.set_ylabel("model-estimated propensity")

    ax = axes[1, 0]
    bins = pd.qcut(pd.Series(yhat).rank(method="first"), 12, labels=False)
    g = pd.DataFrame({"pred": yhat, "obs": y, "bin": bins}).groupby("bin").mean()
    ax.plot(g["pred"], g["obs"], "o-", color="#009E73", label="binned mean")
    lim = [0, max(g["pred"].max(), g["obs"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="perfect calibration")
    ax.set_title("Outcome model calibration (30-day gross margin)")
    ax.set_xlabel("predicted margin ($)"); ax.set_ylabel("observed margin ($)")

    ax = axes[1, 1]
    resid = y - yhat
    ax.scatter(yhat, resid, s=4, alpha=0.2, color="#999999")
    rb = pd.DataFrame({"pred": yhat, "res": resid, "bin": bins}).groupby("bin").mean()
    ax.plot(rb["pred"], rb["res"], "o-", color="#D55E00", label="binned mean residual")
    ax.axhline(0, color="k", lw=1)
    ax.set_title("Outcome residuals vs. fitted")
    ax.set_xlabel("predicted margin ($)"); ax.set_ylabel("residual ($)")
    return am.save_figure(fig, "validation_plots", "nuisance_model_diagnostics",
                          "Nuisance estimator diagnostics (propensity & outcome), holdout")


# --------------------------------------------------------------------------- #
# Comparison plots
# --------------------------------------------------------------------------- #
def plot_policy_comparison(am: ArtifactManager, reports: list[PolicyReport]) -> Path:
    """SNIPS net margin with 95% bootstrap CIs for the four policies."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    vals = np.array([r.value for r in reports])
    err = np.array([[r.value - r.ci_low for r in reports], [r.ci_high - r.value for r in reports]])
    colors = ["#999999", "#E69F00", "#56B4E9", "#009E73"]
    bars = ax.bar([r.name for r in reports], vals, yerr=err, capsize=6, color=colors, edgecolor="black", lw=0.6)
    for b, r in zip(bars, reports):
        ax.text(b.get_x() + b.get_width() / 2, r.ci_high + 0.02 * abs(vals).max(), f"${r.value:.2f}\nESS {r.ess:,.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("SNIPS 30-day net margin per customer ($)")
    ax.set_xlabel("policy")
    ax.tick_params(axis="x", labelsize=8)
    return am.save_figure(fig, "comparison_plots", "policy_net_margin_comparison",
                          "Counterfactual policy value on holdout (SNIPS, 95% bootstrap CI)")


def plot_allocation_by_segment(am: ArtifactManager, x: pd.DataFrame, nba_actions: np.ndarray) -> Path:
    """Stacked action mix per segment: legacy CRM vs. new NBA policy."""
    apply_style()
    seg = assign_segment(x).to_numpy()
    legacy = legacy_crm_distribution(x)
    segs = [s for s in SEGMENT_ORDER if (seg == s).any()]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    width, gap = 0.38, 0.0
    for i, s in enumerate(segs):
        m = seg == s
        mixes = {"Legacy": legacy[m].mean(axis=0),
                 "NBA": np.bincount(nba_actions[m], minlength=5) / m.sum()}
        for k, (lab, mix) in enumerate(mixes.items()):
            xpos = i + (k - 0.5) * (width + gap)
            bottom = 0.0
            for act in ACTION_IDS:
                ax.bar(xpos, mix[act], width, bottom=bottom, color=ACTION_COLORS[act], edgecolor="white", lw=0.5,
                       label=ACTION_SHORT[act] if (i == 0 and k == 0) else None)
                if mix[act] > 0.06:
                    ax.text(xpos, bottom + mix[act] / 2, f"{mix[act]:.0%}", ha="center", va="center", fontsize=7, color="white")
                bottom += mix[act]
            ax.text(xpos, 1.01, lab, ha="center", fontsize=8)
    ax.set_xticks(range(len(segs)), [f"{s}\n(n={(seg == s).sum():,})" for s in segs])
    ax.set_ylabel("share of customers"); ax.set_xlabel("customer segment"); ax.set_ylim(0, 1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False)
    return am.save_figure(fig, "comparison_plots", "action_allocation_by_segment",
                          "Action allocation by segment: legacy CRM vs. NBA policy")


def plot_pareto(am: ArtifactManager, sweep: pd.DataFrame, lam_star: float) -> Path:
    """Margin-vs-unsubscribe frontier across risk-penalty lambda."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(sweep["unsub_rate"] * 100, sweep["net_margin"], "o-", color="#0072B2", lw=2, label="guardrailed NBA (lambda sweep)")
    ax.fill_between(sweep["unsub_rate"] * 100, sweep["ci_low"], sweep["ci_high"], color="#0072B2", alpha=0.12, label="95% bootstrap CI")
    for _, r in sweep.iterrows():
        ax.annotate(f"{r['lambda']:g}", (r["unsub_rate"] * 100, r["net_margin"]), textcoords="offset points", xytext=(5, 6), fontsize=7)
    star = sweep[sweep["lambda"] == lam_star].iloc[0]
    ax.scatter([star["unsub_rate"] * 100], [star["net_margin"]], s=140, color="#009E73", zorder=5, marker="*", label=f"selected lambda = {lam_star:g}")
    ax.axvline(UNSUB_RATE_CAP * 100, color="red", ls="--", label="0.5% unsubscribe cap")
    ax.set_xlabel("estimated unsubscribe rate (%)"); ax.set_ylabel("SNIPS 30-day net margin per customer ($)")
    return am.save_figure(fig, "comparison_plots", "margin_vs_unsubscribe_pareto",
                          "Pareto frontier: net margin vs. unsubscribe rate across lambda (labels = lambda)")
