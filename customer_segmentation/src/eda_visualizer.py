"""EDA, feature and training-loss visualisations (all saved at 300 DPI)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif

from .data_generator import LOYALTY, PROPERTY_TYPES
from .feature_engineering import NUM_COLS
from .utils import get_logger

log = get_logger()
DPI = 300
PAL = ["#2f6db5", "#e07b39", "#3a9d6e", "#a55eb5", "#c9453f", "#7a7a7a", "#c8a600", "#2aa6b8"]
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_raw_distributions(g, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    specs = [("avg_daily_rate", "ADR ($)", False), ("booking_lead_time_days", "Booking lead time (days)", False),
             ("avg_length_of_stay", "Average length of stay (nights)", False),
             ("monetary_total_spend", "Total spend ($, log scale)", True)]
    for ax, (c, t, lg) in zip(axes.ravel(), specs):
        sns.boxplot(data=g, x="loyalty_tier", y=c, order=LOYALTY, palette=PAL[:5], ax=ax, fliersize=1.5)
        if lg:
            ax.set_yscale("log")
        ax.set_title(t); ax.set_xlabel("")
    fig.suptitle("Raw distributions by loyalty tier", fontsize=14)
    _save(fig, path)


def plot_channel_breakdown(g, path):
    d = g.copy()
    d["dominant_channel"] = d[["direct_app_booking_ratio", "ota_booking_ratio", "other_channel_ratio"]].idxmax(axis=1) \
        .map({"direct_app_booking_ratio": "Direct / App", "ota_booking_ratio": "OTA",
              "other_channel_ratio": "Other (phone/corporate)"})
    pivot = d.pivot_table(index="property_type_affinity", columns="dominant_channel",
                          values="monetary_total_spend", aggfunc="mean").reindex(PROPERTY_TYPES)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    pivot.plot(kind="bar", stacked=True, ax=axes[0], color=PAL[:3], width=.7)
    axes[0].set_title("Mean guest spend by property affinity (stacked by dominant booking channel)")
    axes[0].set_ylabel("Mean total spend ($)"); axes[0].set_xlabel(""); axes[0].tick_params(axis="x", rotation=15)
    sns.violinplot(data=d, x="property_type_affinity", y="monetary_total_spend", hue="dominant_channel",
                   order=PROPERTY_TYPES, palette=PAL[:3], ax=axes[1], cut=0, inner="quartile", density_norm="width")
    axes[1].set_yscale("log"); axes[1].set_title("Spend distribution by property type and channel")
    axes[1].set_xlabel(""); axes[1].tick_params(axis="x", rotation=15)
    _save(fig, path)


def plot_correlation_heatmap(g_feat, path):
    corr = g_feat[NUM_COLS].corr()
    cg = sns.clustermap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, figsize=(13, 12),
                        linewidths=.3, cbar_kws={"label": "Pearson r"})
    cg.fig.suptitle("Clustered correlation of engineered hospitality features", y=1.01, fontsize=14)
    cg.fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(cg.fig)
    log.info("Saved %s", path)


def plot_feature_importance(X, labels, names, path, top=15, seed=42):
    keep = labels >= 0
    Xk, lk = X[keep], labels[keep]
    F, _ = f_classif(Xk, lk)
    F = np.nan_to_num(F)
    rf = RandomForestClassifier(150, max_depth=8, n_jobs=-1, random_state=seed).fit(Xk[:15000], lk[:15000])
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, vals, t, c in [(axes[0], F, "ANOVA F-score", PAL[0]), (axes[1], rf.feature_importances_, "Random Forest importance", PAL[2])]:
        o = np.argsort(vals)[::-1][:top][::-1]
        ax.barh(np.array(names)[o], vals[o], color=c)
        ax.set_title(f"Top {top} discriminative features - {t}")
    _save(fig, path)


def _embed2d(Z, seed=42):
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.2).fit_transform(Z), "UMAP"
    except Exception as e:  # umap-learn missing
        log.warning("UMAP unavailable (%s) -> falling back to PCA", e)
        return PCA(2, random_state=seed).fit_transform(Z), "PCA"


def plot_latent_umap(spaces: dict, labels, path, n=6000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels), min(n, len(labels)), replace=False)
    fig, axes = plt.subplots(1, len(spaces), figsize=(6.2 * len(spaces), 6))
    for ax, (nm, Z) in zip(np.atleast_1d(axes), spaces.items()):
        E, meth = _embed2d(Z[idx], seed)
        lab = labels[nm][idx] if isinstance(labels, dict) else labels[idx]
        for i, c in enumerate(np.unique(lab)):
            m = lab == c
            ax.scatter(E[m, 0], E[m, 1], s=4, alpha=.6, color=PAL[i % len(PAL)] if c >= 0 else "#bbbbbb",
                       label=f"cluster {c}" if c >= 0 else "noise")
        ax.set_title(f"{nm} ({meth})"); ax.set_xticks([]); ax.set_yticks([])
    np.atleast_1d(axes)[-1].legend(markerscale=3, fontsize=8)
    fig.suptitle("2-D projection of raw feature space vs. learned latent spaces", fontsize=14)
    _save(fig, path)


def plot_dec_loss(h, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(h["pre_loss"], color=PAL[0], lw=2); axes[0].set_title("Pre-training: MSE reconstruction")
    axes[1].plot(h["dec_recon"], color=PAL[1], lw=2, label="reconstruction")
    axes[1].plot(h["dec_total"], color=PAL[3], lw=2, ls="--", label="total (recon + 0.1*KL)")
    axes[1].legend(); axes[1].set_title("DEC fine-tune: recon / total")
    axes[2].plot(h["dec_kl"], color=PAL[4], lw=2); axes[2].set_title("DEC fine-tune: KL(P || Q)")
    for a in axes:
        a.set_xlabel("epoch")
    _save(fig, path)


def plot_vae_loss(h, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(h["total"], color=PAL[0], lw=2); axes[0].set_title("Total ELBO loss (recon + beta_eff*KL)")
    axes[1].plot(h["recon"], color=PAL[1], lw=2, label="reconstruction")
    axes[1].plot(h["kl"], color=PAL[4], lw=2, label="KL divergence")
    axes[1].legend(); axes[1].set_title("Reconstruction vs KL")
    axes[2].plot(h["beta_eff"], color=PAL[2], lw=2); axes[2].set_title("KL annealing schedule (effective beta)")
    for a in axes:
        a.set_xlabel("epoch")
    _save(fig, path)


def plot_seq_loss(h, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h["train"], color=PAL[0], lw=2, label="train"); ax.plot(h["val"], color=PAL[1], lw=2, label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked MSE + 0.3*CE"); ax.legend()
    ax.set_title("Temporal Transformer autoencoder - sequence reconstruction loss")
    _save(fig, path)
