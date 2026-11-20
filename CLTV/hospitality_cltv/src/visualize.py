"""Publication-quality (300 DPI) plots: EDA, features/SHAP/attention, training curves, model comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_generator import TIERS
from .evaluate import decile_table
from .features import NUM_COLS, Dataset, tree_columns
from .models.common import Inputs

DPI = 300
# Okabe-Ito colour-blind-safe palette
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#444444"]
LABELS = {"bgnbd_gg": "BG/NBD+GG", "xgboost_tweedie": "XGBoost\nTweedie", "catboost_tweedie": "CatBoost\nTweedie",
          "lightgbm_hurdle": "LightGBM\nHurdle", "ft_transformer_ziln": "FT-Transf.\nZILN",
          "gru_attention_ziln": "GRU+Attn\nZILN", "deep_weibull_survival": "Weibull\nSurvival", "ensemble_top3": "Ensemble\nTop-3"}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": DPI, "axes.titleweight": "bold", "axes.titlesize": 12,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ EDA
def plot_eda(cfg: dict, ds: Dataset, guests: pd.DataFrame, bookings: pd.DataFrame, lg) -> None:
    _style()
    d = cfg["out"] / "plots" / "eda"
    df = ds.df
    y = df["y"].values
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].hist(y, bins=60, color=PAL[0])
    ax[0].set(title="12-month future CLTV (linear)", xlabel="Revenue ($)", ylabel="Guests")
    ax[0].annotate(f"{(y == 0).mean():.1%} of guests spend $0\n(zero-inflation)", xy=(0.45, 0.8), xycoords="axes fraction")
    ax[1].hist(np.log1p(y[y > 0]), bins=60, color=PAL[2])
    ax[1].set(title="Positive CLTV (log scale)", xlabel="log(1 + revenue)", ylabel="Guests")
    _save(fig, d / "cltv_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    g = df.groupby("loyalty_tier")["y"].agg(["mean", "median", "size"]).reindex(TIERS)
    x = np.arange(len(g))
    b1 = ax.bar(x - 0.2, g["mean"], 0.4, color=PAL[0], label="Mean")
    b2 = ax.bar(x + 0.2, g["median"], 0.4, color=PAL[1], label="Median")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="$%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, [f"{t}\n(n={int(n):,})" for t, n in zip(g.index, g["size"])])
    ax.set(title="Future 12-month spend by loyalty tier", ylabel="Revenue per guest ($)")
    ax.legend()
    _save(fig, d / "spend_by_loyalty_tier.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    s = bookings.sample(min(40000, len(bookings)), random_state=42)
    for c, (pt, grp) in zip(PAL, s.groupby("property_tier")):
        ax.scatter(grp["booking_lead_time"], grp["adr"], s=3, alpha=0.12, color=c)
        bins = pd.cut(grp["booking_lead_time"], np.arange(0, 181, 15))
        m = grp.groupby(bins, observed=True).agg(x=("booking_lead_time", "mean"), y=("adr", "mean"))
        ax.plot(m["x"], m["y"], color=c, lw=2.5, label=pt)
    ax.set(xlim=(0, 180), ylim=(0, 1200), xlabel="Booking lead time (days)", ylabel="ADR ($)",
           title="Booking lead time vs ADR (lines = binned mean)")
    ax.legend(title="Property tier", markerscale=1)
    _save(fig, d / "leadtime_vs_adr.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    b = bookings[bookings["is_cancelled"] == 0].merge(guests[["guest_id", "acq_day", "loyalty_tier"]], on="guest_id")
    b["m"] = np.floor((b["checkin_day"] - b["acq_day"]) / 30.4).astype(int)
    horizon_m = 36 - np.ceil(b.groupby("guest_id")["acq_day"].first() / 30.4)
    tier_of = guests.set_index("guest_id")["loyalty_tier"]
    for c, t in zip(PAL, ["All"] + TIERS):
        ids = guests["guest_id"] if t == "All" else guests.loc[guests["loyalty_tier"] == t, "guest_id"]
        curve = []
        for m in range(0, 25):
            elig = ids[(36 - np.ceil(guests.set_index("guest_id").loc[ids, "acq_day"] / 30.4) > m + 1).values]
            act = b[(b["m"] == m) & b["guest_id"].isin(elig)]["guest_id"].nunique()
            curve.append(100 * act / max(len(elig), 1))
        ax.plot(range(0, 25), curve, color=c, lw=2.8 if t == "All" else 1.8, label=t)
    ax.set(xlabel="Months since first stay", ylabel="% guests with a completed stay in month",
           title="Retention decay (only guests observable for that month)")
    ax.legend(ncol=2)
    _save(fig, d / "retention_decay.png")
    lg.info(f"EDA plots saved to {d}")


# ------------------------------------------------------------------------------------------ features
def plot_features(cfg: dict, ds: Dataset, models: dict, lg) -> None:
    import shap
    _style()
    d = cfg["out"] / "plots" / "features"
    cols = NUM_COLS + ["y"]
    corr = ds.df[cols].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Pearson r"},
                annot=True, fmt=".1f", annot_kws={"size": 6})
    ax.set_title("Feature correlation matrix (incl. 12m target `y`)")
    _save(fig, d / "correlation_matrix.png")

    xgbm = models["xgboost_tweedie"]
    te_idx = ds.idx("test")
    samp = te_idx[np.random.default_rng(42).permutation(len(te_idx))[:3000]]
    X = ds.df.iloc[samp]
    inp = Inputs(X)
    sv = xgbm.contribs(inp)[:, :-1]
    Xs = X[xgbm.cols]
    plt.figure(figsize=(9, 8))
    shap.summary_plot(sv, Xs, show=False, max_display=18)
    plt.title("SHAP summary (XGBoost Tweedie, log-link contributions)")
    _save(plt.gcf(), d / "shap_beeswarm.png")
    imp = pd.Series(np.abs(sv).mean(0), index=xgbm.cols).sort_values().tail(18)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, color=PAL[0])
    for i, v in enumerate(imp.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.set(title="Mean |SHAP| feature importance (XGBoost)", xlabel="mean |SHAP value|")
    _save(fig, d / "shap_importance_bar.png")

    # permutation importance (RMSE increase) on the test sample
    rng = np.random.default_rng(42)
    base = np.sqrt(np.mean((xgbm.predict(inp) - X["y"].values) ** 2))
    pi = {}
    for c in xgbm.cols:
        inc = []
        for _ in range(3):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            inc.append(np.sqrt(np.mean((xgbm.predict(Inputs(Xp)) - X["y"].values) ** 2)) - base)
        pi[c] = np.mean(inc)
    pi = pd.Series(pi).sort_values().tail(18)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(pi.index, pi.values, color=PAL[1])
    ax.set(title="Permutation importance (XGBoost)", xlabel="Increase in RMSE ($) when shuffled")
    _save(fig, d / "permutation_importance.png")

    att = np.load(cfg["out"] / "data" / "attention_test.npy")
    mask = ds.seq_mask[ds.idx("test")]
    tiers = ds.df.iloc[ds.idx("test")]["loyalty_tier"].values
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    rows = []
    for t in TIERS:
        sel = (tiers == t)
        rows.append(att[sel].sum(0) / np.maximum(mask[sel].sum(0), 1))
    L = att.shape[1]
    sns.heatmap(np.array(rows), cmap="mako_r", ax=ax[0], yticklabels=TIERS, annot=False,
                xticklabels=[f"-{L - 1 - i}" if i < L - 1 else "last" for i in range(L)], cbar_kws={"label": "mean attention"})
    ax[0].set(title="Mean attention per stay position, by loyalty tier", xlabel="Stay position (0=most recent)")
    full = np.argsort(-mask.sum(1), kind='stable')[:12]  # guests with the longest histories
    sns.heatmap(att[full], cmap="mako_r", ax=ax[1], cbar_kws={"label": "attention"}, yticklabels=[f"guest {i+1}" for i in range(len(full))])
    ax[1].set(title="Attention for the 12 guests with longest histories", xlabel="Stay position (oldest -> latest)")
    fig.tight_layout()
    _save(fig, d / "sequence_attention_heatmap.png")
    lg.info(f"Feature plots saved to {d}")


# ----------------------------------------------------------------------------------------- training
def plot_training(cfg: dict, hist: dict, lg) -> None:
    _style()
    d = cfg["out"] / "plots" / "training_loss"
    panels = [("ft_transformer_ziln", "FT-Transformer (ZILN loss)"), ("gru_attention_ziln", "GRU + Attention (ZILN loss)"),
              ("deep_weibull_survival", "Weibull survival (NLL)"), ("xgboost_tweedie", "XGBoost (Tweedie NLL / round)"),
              ("catboost_tweedie", "CatBoost (Tweedie / iteration)"), ("lightgbm_hurdle", "LightGBM hurdle stage 1 (logloss)")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (k, title) in zip(axes.ravel(), panels):
        h = hist[k]
        ax.plot(range(1, len(h["train"]) + 1), h["train"], color=PAL[0], lw=2, label="train")
        ax.plot(range(1, len(h["val"]) + 1), h["val"], color=PAL[3], lw=2, label="validation")
        bi = int(np.argmin(h["val"]))
        ax.axvline(bi + 1, color="grey", ls="--", lw=1)
        ax.annotate(f"best val {h['val'][bi]:.4f}\n@ {bi + 1}", xy=(bi + 1, h["val"][bi]), xytext=(0.55, 0.65),
                    textcoords="axes fraction", arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9)
        ax.set(title=title, xlabel="epoch" if k in ("ft_transformer_ziln", "gru_attention_ziln", "deep_weibull_survival") else "boosting round",
               ylabel="loss")
        ax.legend()
    fig.suptitle("Training vs validation loss", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, d / "training_validation_loss_all_models.png")
    for k, title in panels:  # also individual files
        h = hist[k]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h["train"], color=PAL[0], label="train")
        ax.plot(h["val"], color=PAL[3], label="validation")
        ax.set(title=title, xlabel="epoch / round", ylabel="loss")
        ax.legend()
        _save(fig, d / f"loss_{k}.png")
    lg.info(f"Training-loss plots saved to {d}")


# ----------------------------------------------------------------------------------------- compare
def plot_comparisons(cfg: dict, ev: dict, lg) -> None:
    _style()
    d = cfg["out"] / "plots" / "comparisons"
    rep = ev["report"].copy()
    rep["Top10_Lift"] = rep["Top10_Revenue_Capture_pct"] / 10.0
    models = list(rep["model"])
    colors = [PAL[i % len(PAL)] for i in range(len(models))]
    spec = [("RMSE", "RMSE ($) - lower is better", "{:,.0f}"), ("MAE", "MAE ($) - lower is better", "{:,.0f}"),
            ("Normalized_Gini", "Normalized Gini - higher is better", "{:.3f}"),
            ("Top10_Lift", "Top-decile lift (x) - higher is better", "{:.2f}")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (col, title, fmt) in zip(axes.ravel(), spec):
        bars = ax.bar(range(len(models)), rep[col], color=colors)
        for b, v in zip(bars, rep[col]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(models)), [LABELS[m] for m in models], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, rep[col].max() * 1.15)
    fig.suptitle("Model comparison on held-out test guests", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, d / "model_comparison_metrics.png")

    # secondary grouped chart: remaining business metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (col, title, fmt) in zip(axes, [("R2", "R²", "{:.3f}"), ("Spearman", "Spearman rank corr.", "{:.3f}"),
                                           ("Calibration_Ratio", "Calibration ratio (pred sum / actual sum)", "{:.3f}")]):
        bars = ax.bar(range(len(models)), rep[col], color=colors)
        for b, v in zip(bars, rep[col]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(models)), [LABELS[m] for m in models], fontsize=7)
        ax.set_title(title)
        if col == "Calibration_Ratio":
            ax.axhline(1.0, color="k", ls="--", lw=1)
    fig.tight_layout()
    _save(fig, d / "model_comparison_secondary.png")

    pdf = ev["predictions"]
    te = pdf[pdf.split == "test"]
    y = te["y"].values
    fig, ax = plt.subplots(figsize=(8, 6.5))
    n = len(y)
    for c, m in zip(colors, models):
        o = np.argsort(-te[m].values, kind="stable")
        cum = np.r_[0, np.cumsum(y[o]) / y.sum()]
        ax.plot(np.linspace(0, 100, n + 1), 100 * cum, color=c, lw=2, label=f"{m} (Gini {rep.set_index('model').loc[m, 'Normalized_Gini']:.3f})")
    o = np.argsort(-y, kind="stable")
    ax.plot(np.linspace(0, 100, n + 1), 100 * np.r_[0, np.cumsum(y[o]) / y.sum()], "k:", lw=1.5, label="Perfect")
    ax.plot([0, 100], [0, 100], color="grey", ls="--", label="Random")
    ax.set(xlabel="% of guests targeted (ranked by predicted CLTV)", ylabel="% of future revenue captured",
           title="Cumulative gain / Lorenz curve", xlim=(0, 100), ylim=(0, 101))
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, d / "cumulative_gain_lorenz.png")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for ax, c, m in zip(axes.ravel(), colors, models):
        t = decile_table(y, te[m].values)
        ax.plot([1, 3e4], [1, 3e4], "k--", lw=1)
        ax.loglog(t["predicted"].clip(lower=1), t["actual"].clip(lower=1), "o-", color=c, lw=2)
        ax.set_title(LABELS[m].replace("\n", " "), fontsize=10)
        ax.set_xlabel("mean predicted ($)")
        ax.set_ylabel("mean actual ($)")
        ax.set_xlim(left=1)
    fig.suptitle("Predicted vs actual CLTV by predicted-decile (dashed = perfect calibration)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, d / "calibration_by_decile.png")

    best = ev["best"]
    t = decile_table(y, te[best].values)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(10)
    b1 = ax.bar(x - 0.2, t["actual"], 0.4, color=PAL[1], label="Actual")
    b2 = ax.bar(x + 0.2, t["predicted"], 0.4, color=PAL[0], label="Predicted")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
    ax.set_xticks(x, [f"D{i}" for i in range(1, 11)])
    ax.set(title=f"Decile calibration - best model: {best}", xlabel="Predicted CLTV decile (D1 = highest)", ylabel="Mean CLTV ($)")
    ax.legend()
    _save(fig, d / "calibration_best_model.png")
    lg.info(f"Comparison plots saved to {d}")
