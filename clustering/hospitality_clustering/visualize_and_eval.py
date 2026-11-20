"""Metrics, persona labelling and all report plots."""
from __future__ import annotations

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

import config as C
import models_ml

log = logging.getLogger("hospitality")

# Okabe-Ito colour-blind-safe palette
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
NOISE = "#BBBBBB"
sns.set_theme(style="whitegrid", context="notebook")
SERVABLE = ("kmeans", "gmm", "agglomerative", "dec", "vae_gmm")  # eligible as served default
MODEL_ORDER = ["kmeans", "gmm", "agglomerative", "dbscan", "dec", "vae_gmm"]

# Archetype signatures: feature -> direction (+1 high / -1 low) used for persona naming
ARCHETYPES = {
    "Business Road Warrior": {"business_trip_ratio": 1, "annual_stays": 1, "weekend_stay_ratio": -1, "booking_lead_time_days": -1},
    "Luxury Leisure Spender": {"adr_usd": 1, "spa_golf_spend_usd": 1, "room_upgrade_rate": 1, "avg_length_of_stay": 1},
    "Weekend Family Getaway": {"weekend_stay_ratio": 1, "business_trip_ratio": -1, "booking_lead_time_days": 1, "annual_stays": -1},
    "Budget OTA Shopper": {"adr_usd": -1, "direct_booking_ratio": -1, "cancellation_rate": 1, "loyalty_tenure_years": -1},
    "Loyal Direct Regular": {"direct_booking_ratio": 1, "loyalty_tenure_years": 1, "cancellation_rate": -1, "recency_days": -1},
}


# ----------------------------------------------------------------------- metrics
def _metrics_for(X, labels, rng):
    m = labels >= 0
    out = dict(n_clusters=int(len(set(labels[m]))), noise_fraction=float(1 - m.mean()))
    if out["n_clusters"] < 2:
        out.update(silhouette=None, calinski_harabasz=None, davies_bouldin=None)
        return out
    Xm, lm = X[m], labels[m]
    sub = rng.choice(len(Xm), size=min(5000, len(Xm)), replace=False)
    out["silhouette"] = float(silhouette_score(Xm[sub], lm[sub])) if len(set(lm[sub])) > 1 else None
    out["calinski_harabasz"] = float(calinski_harabasz_score(Xm, lm))
    out["davies_bouldin"] = float(davies_bouldin_score(Xm, lm))
    return out


def evaluate(models: dict, X_train, X_test, truth, k_selection) -> dict:
    rng = np.random.default_rng(C.SEED)
    res = {}
    for name in MODEL_ORDER:
        if name not in models:
            continue
        mod, labels = models[name]["model"], models[name]["labels"]
        r = _metrics_for(X_train, labels, rng)
        r["latency_ms_per_1k"] = models_ml.timed_predict_ms_per_1k(mod.predict_proba, X_test)
        r["ari_vs_simulated_truth"] = float(adjusted_rand_score(truth[labels >= 0], labels[labels >= 0]))
        if hasattr(mod, "embed"):  # deep models: also score in latent space
            z = mod.embed(X_train)
            r["latent_space"] = _metrics_for(z, labels, rng)
        res[name] = r
        log.info("%-14s sil=%s CH=%s DB=%s k=%d noise=%.1f%% lat=%.2fms/1k", name,
                 *(f"{r[m]:.3f}" if r[m] is not None else "NA" for m in
                   ("silhouette", "calinski_harabasz", "davies_bouldin")),
                 r["n_clusters"], 100 * r["noise_fraction"], r["latency_ms_per_1k"])
    res["_best_model"] = pick_best(res)
    res["_k_selection"] = k_selection
    res["_device"] = C.DEVICE
    C.METRICS_FILE.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info("Best model (rank aggregate of sil/CH/DB): %s", res["_best_model"])
    return res


def pick_best(res: dict) -> str:
    cand = [n for n in SERVABLE if n in res and res[n]["silhouette"] is not None]
    df = pd.DataFrame({n: [res[n]["silhouette"], res[n]["calinski_harabasz"], res[n]["davies_bouldin"]]
                       for n in cand}, index=["sil", "ch", "db"]).T
    rank = df["sil"].rank(ascending=False) + df["ch"].rank(ascending=False) + df["db"].rank(ascending=True)
    return str(rank.idxmin())


# ----------------------------------------------------------------------- personas
def _profile(df_raw: pd.DataFrame, labels) -> pd.DataFrame:
    z = (df_raw[C.FEATURES] - df_raw[C.FEATURES].mean()) / df_raw[C.FEATURES].std()
    z["c"] = labels
    return z[z["c"] >= 0].groupby("c").mean()


def build_personas(models: dict, df_raw: pd.DataFrame) -> dict:
    """Name clusters for every model via Hungarian matching against archetype signatures."""
    names = list(ARCHETYPES)
    pmap = {}
    for mname, m in models.items():
        prof = _profile(df_raw, m["labels"])
        cl = list(prof.index)
        score = np.array([[sum(d * prof.loc[c, f] for f, d in ARCHETYPES[a].items()) / len(ARCHETYPES[a])
                           for a in names] for c in cl])
        r, c_ = linear_sum_assignment(-score)
        mp = {}
        for i, j in zip(r, c_):
            mp[str(cl[i])] = names[j] if score[i, j] > 0.15 else None
        for i, c in enumerate(cl):
            if mp.get(str(c)) is None:
                top = prof.loc[c].abs().sort_values(ascending=False).index[0]
                mp[str(c)] = f"Segment {c} ({'high' if prof.loc[c, top] > 0 else 'low'} {top})"
        pmap[mname] = mp
    C.PERSONA_FILE.write_text(json.dumps(pmap, indent=2), encoding="utf-8")
    return pmap


# ----------------------------------------------------------------------- plots
def _save(fig, name):
    fig.savefig(C.PLOTS_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("saved plot %s", name)


def plot_eda(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    for ax, f in zip(axes.ravel(), C.FEATURES):
        sns.histplot(df[f], bins=40, kde=True, color=PAL[0], ax=ax)
        ax.set_title(f, fontsize=10); ax.set_xlabel("")
    for ax in axes.ravel()[len(C.FEATURES):]:
        ax.axis("off")
    fig.suptitle(f"Guest feature distributions (n={len(df):,})", fontsize=15)
    fig.tight_layout()
    _save(fig, "01_eda_distributions.png")


def plot_corr_importance(df: pd.DataFrame, best_labels):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw=dict(width_ratios=[1.3, 1]))
    sns.heatmap(df[C.FEATURES].corr(), cmap="vlag", center=0, annot=True, fmt=".1f", annot_kws=dict(size=7),
                cbar_kws=dict(shrink=.8), ax=a1)
    a1.set_title("Feature correlation (Pearson)")
    m = best_labels >= 0
    rf = RandomForestClassifier(200, n_jobs=-1, random_state=C.SEED).fit(df.loc[m, C.FEATURES], best_labels[m])
    imp = pd.Series(rf.feature_importances_, index=C.FEATURES).sort_values()
    imp.plot.barh(color=PAL[0], ax=a2)
    a2.set_title("Feature importance for cluster separation\n(random forest on best model's clusters)")
    a2.set_xlabel("Gini importance")
    fig.tight_layout()
    _save(fig, "02_feature_correlation_importance.png")


def plot_losses():
    rows = [json.loads(l) for l in C.LOSS_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    d = df[(df.model == "dec") & (df.phase == "pretrain")]
    axes[0].plot(d.epoch, d.recon, color=PAL[0]); axes[0].set_title("DEC: AE pretrain (MSE)")
    d = df[(df.model == "dec") & (df.phase == "finetune")]
    axes[1].plot(d.epoch, d.kl, color=PAL[3], label="KL(P||Q)")
    axes[1].plot(d.epoch, d.recon, color=PAL[0], ls="--", label="recon (monitor)")
    axes[1].set_title("DEC: clustering fine-tune"); axes[1].legend()
    d = df[df.model == "vae"]
    axes[2].plot(d.epoch, d.recon, color=PAL[0], label="reconstruction")
    axes[2].plot(d.epoch, d.total, color=PAL[1], label="total (recon + beta*KL)")
    axes[2].set_title("beta-VAE: reconstruction & total"); axes[2].legend()
    axes[3].plot(d.epoch, d.kl, color=PAL[3]); axes[3].set_title("beta-VAE: KL divergence")
    for a in axes:
        a.set_xlabel("epoch"); a.set_ylabel("loss")
    fig.tight_layout()
    _save(fig, "03_dl_training_losses.png")


def plot_umap(X, models: dict):
    import umap
    rng = np.random.default_rng(C.SEED)
    sub = rng.choice(len(X), size=min(4000, len(X)), replace=False)
    emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=C.SEED).fit_transform(X[sub])
    names = [n for n in MODEL_ORDER if n in models]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, n in zip(axes.ravel(), names):
        lab = models[n]["labels"][sub]
        for c in sorted(set(lab)):
            mk = lab == c
            ax.scatter(emb[mk, 0], emb[mk, 1], s=5, alpha=.7,
                       color=NOISE if c < 0 else PAL[c % 7], label="noise" if c < 0 else f"C{c}")
        ax.set_title(n); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=3, fontsize=8, loc="best")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("UMAP projection of guests coloured by cluster", fontsize=15)
    fig.tight_layout()
    _save(fig, "04_umap_cluster_projections.png")


def plot_metrics(res: dict):
    names = [n for n in MODEL_ORDER if n in res]
    spec = [("silhouette", "Silhouette (higher better)"), ("calinski_harabasz", "Calinski-Harabasz (higher better)"),
            ("davies_bouldin", "Davies-Bouldin (lower better)"), ("latency_ms_per_1k", "Inference latency (ms / 1k guests)")]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    for ax, (key, title) in zip(axes, spec):
        vals = [res[n][key] if res[n][key] is not None else 0 for n in names]
        colors = [PAL[3] if n == res["_best_model"] else PAL[0] for n in names]
        bars = ax.bar(names, vals, color=colors)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_title(title, fontsize=11); ax.tick_params(axis="x", rotation=40)
        if key == "latency_ms_per_1k":
            ax.set_yscale("log")
    fig.suptitle(f"Model comparison (orange = selected: {res['_best_model']}; density-model scores exclude noise)")
    fig.tight_layout()
    _save(fig, "05_model_metrics_comparison.png")


def plot_persona_heatmap(df_raw, labels, pmap_for_model, model_name):
    prof = _profile(df_raw, labels)
    sizes = pd.Series(labels[labels >= 0]).value_counts().reindex(prof.index)
    prof.index = [f"C{c}: {pmap_for_model[str(c)]} (n={sizes[c]:,})" for c in prof.index]
    fig, ax = plt.subplots(figsize=(15, 0.9 * len(prof) + 3))
    sns.heatmap(prof, cmap="vlag", center=0, annot=True, fmt=".1f", vmin=-2, vmax=2, ax=ax,
                cbar_kws=dict(label="z-score vs. all guests"))
    ax.set_title(f"Cluster persona profiles - {model_name}")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    _save(fig, "06_cluster_persona_heatmap.png")


def make_all_plots(df_train, X_train, models, res, pmap):
    best = res["_best_model"]
    plot_eda(df_train)
    plot_corr_importance(df_train, models[best]["labels"])
    plot_losses()
    plot_umap(X_train, models)
    plot_metrics(res)
    plot_persona_heatmap(df_train, models[best]["labels"], pmap[best], best)
