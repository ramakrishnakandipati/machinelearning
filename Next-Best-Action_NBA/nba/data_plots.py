"""Module 2d - Exploratory & causal-assumption diagnostic plots (outputs/data_plots)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from .artifact_manager import ArtifactManager
from .config import ACTION_IDS, ACTION_SHORT, FEATURE_COLUMNS, SEED
from .plot_utils import ACTION_COLORS, apply_style
from .uplift_model import make_propensity_model


def estimate_propensities(x: pd.DataFrame, action: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Cross-fitted generalized propensity P(T=a | x), shape (n, 5)."""
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    return cross_val_predict(make_propensity_model(seed), x, action, cv=cv, method="predict_proba")


def weighted_smd(v: np.ndarray, w1: np.ndarray, w0: np.ndarray, m1: np.ndarray, m0: np.ndarray) -> float:
    """Standardized mean difference between two (weighted) groups."""
    a, b = np.average(v[m1], weights=w1[m1]), np.average(v[m0], weights=w0[m0])
    va = np.average((v[m1] - a) ** 2, weights=w1[m1])
    vb = np.average((v[m0] - b) ** 2, weights=w0[m0])
    return float((a - b) / np.sqrt((va + vb) / 2 + 1e-12))


def plot_propensity_overlap(am: ArtifactManager, x: pd.DataFrame, action: np.ndarray) -> Path:
    """Positivity check: P(T=a|x) for units that received a vs. everyone else."""
    apply_style()
    pr = estimate_propensities(x, action)
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8), sharey=False)
    for a, ax in zip(ACTION_IDS, axes):
        sns.kdeplot(pr[action == a, a], ax=ax, color=ACTION_COLORS[a], fill=True, alpha=0.4, label=f"received {a}", clip=(0, 1))
        sns.kdeplot(pr[action != a, a], ax=ax, color="#444444", fill=False, linestyle="--", label="received other", clip=(0, 1))
        ax.axvline(0.01, color="red", lw=0.8, ls=":")
        ax.set_title(f"{ACTION_SHORT[a]}  (min={pr[:, a].min():.3f}, max={pr[:, a].max():.2f})", fontsize=9)
        ax.set_xlabel(f"estimated P(T={a} | x)")
        ax.set_ylabel("density" if a == 0 else "")
        ax.set_xlim(0, 1)
    return am.save_figure(fig, "data_plots", "treatment_propensity_overlap",
                          "Common support: estimated propensity by treatment (red dotted = 0.01 positivity floor)")


def plot_correlations_and_drift(am: ArtifactManager, x: pd.DataFrame, action: np.ndarray, propensity: np.ndarray,
                                is_test: np.ndarray, top_k: int = 12) -> Path:
    """Feature correlation heatmap + SMD love plot (raw vs IPW, plus time drift)."""
    apply_style()
    # Top features by variance of association with treatment assignment (max |corr| to any arm).
    dummies = pd.get_dummies(pd.Series(action), prefix="T").astype(float)
    assoc = pd.concat([x.reset_index(drop=True), dummies], axis=1).corr().loc[FEATURE_COLUMNS, dummies.columns].abs().max(axis=1)
    top = assoc.sort_values(ascending=False).index[:top_k].tolist()

    w = 1.0 / np.clip(propensity, 1e-3, 1.0)  # IPW weight of the logged action
    uniform = np.ones(len(action))
    rows = []
    for f in top:
        v = x[f].to_numpy(dtype=float)
        before = max(abs(weighted_smd(v, uniform, uniform, action == a, action == 0)) for a in ACTION_IDS[1:])
        after = max(abs(weighted_smd(v, w, w, action == a, action == 0)) for a in ACTION_IDS[1:])
        drift = abs(weighted_smd(v, uniform, uniform, is_test, ~is_test))
        rows.append((f, before, after, drift))
    smd = pd.DataFrame(rows, columns=["feature", "before", "after", "drift"]).sort_values("before")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.1, 1]})
    sns.heatmap(x[top].corr(), ax=ax1, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".1f",
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    ax1.set_title("Correlation of top engineered features")
    yy = np.arange(len(smd))
    ax2.scatter(smd["before"], yy, color="#D55E00", label="raw (max |SMD| vs control)", zorder=3)
    ax2.scatter(smd["after"], yy, color="#0072B2", label="after IPW", zorder=3)
    ax2.scatter(smd["drift"], yy, color="#7f7f7f", marker="x", label="train->holdout drift", zorder=3)
    ax2.hlines(yy, 0, np.maximum(smd["before"], smd["after"]), color="#dddddd", zorder=1)
    ax2.axvline(0.1, color="red", ls=":", lw=1)
    ax2.set_yticks(yy, smd["feature"])
    ax2.set_xlabel("|standardized mean difference|")
    ax2.set_title("Treatment balance (love plot) & temporal drift")
    return am.save_figure(fig, "data_plots", "feature_correlations_and_drift",
                          "Feature correlations, treatment balance before/after IPW, and drift")


def plot_margin_and_returns(am: ArtifactManager, log: pd.DataFrame, discount_tier: np.ndarray) -> Path:
    """Net gross margin (converters) and return rate, by action x discount tier."""
    apply_style()
    df = log.copy()
    df["tier"] = pd.Series(discount_tier, index=df.index).map({0: "Full-price", 1: "Mixed", 2: "Markdown-driven"})
    df["Action"] = df["action_id"].map(ACTION_SHORT)
    order = [ACTION_SHORT[a] for a in ACTION_IDS]
    tiers = ["Full-price", "Mixed", "Markdown-driven"]
    conv = df[df["purchased_flag"] & (df["gross_margin_dollars"] > 0)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    sns.violinplot(data=conv, x="Action", y="net_margin_30d", hue="tier", order=order, hue_order=tiers, ax=ax1,
                   palette=["#56B4E9", "#E69F00", "#D55E00"], cut=0, inner="quartile", linewidth=0.8)
    ax1.set_title("Net gross margin per converting customer")
    ax1.set_ylabel("30-day net margin ($)")
    ax1.set_ylim(conv["net_margin_30d"].quantile(0.001), conv["net_margin_30d"].quantile(0.995))
    buyers = df[df["purchased_flag"]]
    sns.barplot(data=buyers, x="Action", y="returned_flag", hue="tier", order=order, hue_order=tiers, ax=ax2,
                palette=["#56B4E9", "#E69F00", "#D55E00"], errorbar=("ci", 95))
    ax2.set_title("Return rate among purchasers (30-day window)")
    ax2.set_ylabel("return rate")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=15)
        ax.set_xlabel("")
    return am.save_figure(fig, "data_plots", "margin_and_return_distributions",
                          "Margin and returns by action and discount-sensitivity tier")
