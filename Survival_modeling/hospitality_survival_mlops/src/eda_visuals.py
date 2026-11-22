"""Exploratory visuals saved at 300 DPI to output/data_visuals/."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test

from .utils import TIERS, get_logger, load_config, resolve

log = get_logger("eda")
DPI = 300
sns.set_theme(style="whitegrid", context="notebook")
NUM_FEATURES = ["adr_avg", "length_of_stay_avg", "lead_time_days", "days_since_last_stay",
                "total_prior_stays", "ancillary_spend_ratio", "promo_email_open_rate",
                "app_sessions_30d", "seasonality_index"]


def _save(fig, out: Path, name: str) -> Path:
    path = out / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path)
    return path


def plot_censoring(df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df.assign(status=df.event_observed.map({1: "Rebooked (event)", 0: "Censored"})),
                 x="duration_days", hue="status", bins=60, stat="density", common_norm=False,
                 kde=True, alpha=0.45, ax=ax)
    ax.set_title(f"Event vs Censoring Distribution of Duration (censored {100 * (1 - df.event_observed.mean()):.1f}%)")
    ax.set_xlabel("Days since last checkout")
    return _save(fig, out, "01_censoring_vs_event_distribution.png")


def plot_km_by(df: pd.DataFrame, col: str, order: list | None, title: str, fname: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    groups = order if order is not None else sorted(df[col].unique())
    for g in groups:
        sub = df[df[col] == g]
        kmf = KaplanMeierFitter().fit(sub.duration_days, sub.event_observed, label=f"{g} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15)
    res = multivariate_logrank_test(df.duration_days, df[col], df.event_observed)
    ptxt = "p < 0.001" if res.p_value < 1e-3 else f"p = {res.p_value:.4f}"
    ax.text(0.97, 0.95, f"Log-rank test: {ptxt}\nχ² = {res.test_statistic:.1f}", transform=ax.transAxes,
            ha="right", va="top", bbox=dict(boxstyle="round", fc="white", ec="gray"))
    ax.set_title(title)
    ax.set_xlabel("Days since last checkout")
    ax.set_ylabel("P(not yet rebooked)")
    ax.set_xlim(0, 365)
    return _save(fig, out, fname)


def plot_nelson_aalen(df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    for flag, label in [(1, "Direct channel"), (0, "Indirect (OTA / other)")]:
        sub = df[df.channel_direct == flag]
        NelsonAalenFitter().fit(sub.duration_days, sub.event_observed, label=f"{label} (n={len(sub)})") \
            .plot_cumulative_hazard(ax=ax, ci_alpha=0.15)
    p = logrank_test(df[df.channel_direct == 1].duration_days, df[df.channel_direct == 0].duration_days,
                     df[df.channel_direct == 1].event_observed, df[df.channel_direct == 0].event_observed).p_value
    ax.text(0.03, 0.95, f"Log-rank: {'p < 0.001' if p < 1e-3 else f'p = {p:.4f}'}", transform=ax.transAxes,
            va="top", bbox=dict(boxstyle="round", fc="white", ec="gray"))
    ax.set_title("Nelson-Aalen Cumulative Hazard by Booking Channel")
    ax.set_xlabel("Days since last checkout")
    ax.set_ylabel("Cumulative hazard")
    ax.set_xlim(0, 365)
    return _save(fig, out, "04_nelson_aalen_by_channel.png")


def plot_correlation(df: pd.DataFrame, out: Path) -> Path:
    cols = NUM_FEATURES + ["channel_direct", "trip_purpose_business", "duration_days", "event_observed"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, bool)), annot=True, fmt=".2f", cmap="vlag",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap")
    return _save(fig, out, "05_feature_correlation_heatmap.png")


def plot_violins(df: pd.DataFrame, out: Path) -> Path:
    d = df.assign(status=df.event_observed.map({1: "Rebooked", 0: "Censored"}))
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, col in zip(axes.ravel(), NUM_FEATURES):
        sns.violinplot(data=d, x="status", y=col, hue="status", palette="Set2", inner="quartile",
                       legend=False, ax=ax, cut=0)
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Numerical Features by Event Status", y=1.01, fontsize=14)
    fig.tight_layout()
    return _save(fig, out, "06_feature_violins_by_event.png")


def run(df: pd.DataFrame | None = None, cfg: dict | None = None) -> list[Path]:
    cfg = cfg or load_config()
    out = resolve(cfg, "data_visuals")
    if df is None:
        df = pd.read_csv(resolve(cfg, "data"))
    df = df.copy()
    df["trip_purpose"] = np.where(df.trip_purpose_business == 1, "Business", "Leisure")
    paths = [
        plot_censoring(df, out),
        plot_km_by(df, "loyalty_tier", TIERS, "Kaplan-Meier: Time to Next Booking by Loyalty Tier (95% CI)",
                   "02_km_by_loyalty_tier.png", out),
        plot_km_by(df, "trip_purpose", ["Business", "Leisure"],
                   "Kaplan-Meier: Time to Next Booking by Trip Purpose (95% CI)", "03_km_by_trip_purpose.png", out),
        plot_nelson_aalen(df, out),
        plot_correlation(df, out),
        plot_violins(df, out),
    ]
    return paths


if __name__ == "__main__":
    run()
