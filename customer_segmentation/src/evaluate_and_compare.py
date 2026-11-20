"""Cluster validation, model comparison, persona naming and business KPIs."""
import json
import logging
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from .feature_engineering import NUM_COLS
from .utils import get_logger, out_dir, sync

log = get_logger()
PALETTE = ["#2f6db5", "#e07b39", "#3a9d6e", "#a55eb5", "#c9453f", "#7a7a7a"]


def audit_logger(cfg):
    lg = logging.getLogger("evaluation_audit")
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        fh = logging.FileHandler(out_dir(cfg, "logs") / "evaluation_audit.log", mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        lg.addHandler(fh)
        lg.propagate = False
    return lg


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def cluster_metrics(X, labels, sample=8000, seed=42):
    """Silhouette / DB / CH on non-noise points; also returns n_clusters & noise ratio."""
    labels = np.asarray(labels)
    keep = labels >= 0
    noise = 1 - keep.mean()
    Xk, lk = X[keep], labels[keep]
    k = len(np.unique(lk))
    if k < 2:
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan,
                    n_clusters=k, noise_ratio=noise)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xk), min(sample, len(Xk)), replace=False)
    sil = silhouette_score(Xk[idx], lk[idx]) if len(np.unique(lk[idx])) > 1 else np.nan
    return dict(silhouette=float(sil), davies_bouldin=float(davies_bouldin_score(Xk, lk)),
                calinski_harabasz=float(calinski_harabasz_score(Xk, lk)), n_clusters=int(k),
                noise_ratio=float(noise))


def stability(refit_fn, base_labels, n_boot=5, frac=0.8, seed=42):
    """Mean ARI between the base labelling and re-clusterings fitted on random 80% subsamples."""
    rng = np.random.default_rng(seed)
    n = len(base_labels)
    scores = []
    for b in range(n_boot):
        idx = rng.choice(n, int(n * frac), replace=False)
        lab = refit_fn(idx, seed + b)
        scores.append(adjusted_rand_score(base_labels, lab))
    return float(np.mean(scores))


def measure_latency(fn, repeats=5):
    fn(); sync()  # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts))


# --------------------------------------------------------------------------------------
# Personas & KPIs
# --------------------------------------------------------------------------------------
# +1 / -1 weights over (standardised) engineered features used to name clusters.
PERSONA_SIGNATURES = {
    "Corporate Road Warriors": {"business_center_usage": 1.5, "last_minute_booking_ratio": 1.0,
                                "weekend_vs_weekday_ratio": -1.5, "booking_lead_time_days": -1.0,
                                "frequency_stays_12m": 1.0, "avg_length_of_stay": -0.5},
    "Luxury Resort Families": {"avg_daily_rate": 1.0, "avg_length_of_stay": 1.5, "booking_lead_time_days": 1.0,
                               "golf_recreation_spend_ratio": 1.0, "dining_fb_spend_ratio": 1.0,
                               "monetary_total_spend": 1.0},
    "Weekend Spa Couples": {"spa_spend_ratio": 2.0, "weekend_vs_weekday_ratio": 1.5,
                            "business_center_usage": -0.5, "avg_length_of_stay": -0.5},
    "OTA Price-Sensitive Travelers": {"ota_booking_ratio": 2.0, "avg_daily_rate": -1.2,
                                      "direct_app_booking_ratio": -1.0, "cancellation_rate": 1.0,
                                      "email_open_rate": -0.5},
    "At-Risk Dormant VIPs": {"recency_days": 2.0, "tenure_months": 0.8, "avg_daily_rate": 0.7,
                             "email_open_rate": -1.0, "mobile_key_usage_rate": -0.7,
                             "frequency_stays_12m": -1.0},
}


def name_clusters(X, labels, cfg=None):
    """Assign persona names via Hungarian matching of cluster z-profiles to persona signatures."""
    names = list(PERSONA_SIGNATURES)
    ids = [c for c in np.unique(labels) if c >= 0]
    prof = np.stack([X[labels == c][:, :len(NUM_COLS)].mean(0) for c in ids])
    S = np.zeros((len(ids), len(names)))
    for j, nm in enumerate(names):
        for f, w in PERSONA_SIGNATURES[nm].items():
            S[:, j] += w * prof[:, NUM_COLS.index(f)]
    r, c = linear_sum_assignment(-S)
    mapping = {int(ids[i]): names[j] for i, j in zip(r, c)}
    ids = [int(c) for c in ids]
    for i, cid in enumerate(ids):  # extra clusters (K>5): best match with suffix
        if cid not in mapping:
            mapping[cid] = names[int(S[i].argmax())] + f" (cluster {cid})"
    if (labels < 0).any():
        mapping[-1] = "VIP / Outlier (noise)"
    return mapping, prof


def segment_kpis(guests, labels, mapping, cfg):
    df = guests.copy()
    df["segment_id"] = labels
    df["segment_name"] = df["segment_id"].map(mapping)
    df["churn_prob"] = 1 - np.exp(-df["recency_days"] / 365.0)
    df["ltv"] = df["spend_per_stay"] * df["frequency_stays_12m"].clip(lower=0.5) * 3 * (1 - df["churn_prob"])
    pers = cfg["personas"]
    rows = []
    for sid, g in df.groupby("segment_id"):
        nm = mapping[sid]
        base = nm.split(" (cluster")[0]
        rows.append(dict(
            segment_id=int(sid), segment_name=nm, size=int(len(g)), share_pct=round(100 * len(g) / len(df), 2),
            avg_ltv=round(float(g.ltv.mean()), 2), avg_total_spend=round(float(g.monetary_total_spend.mean()), 2),
            avg_adr=round(float(g.avg_daily_rate.mean()), 2), avg_los=round(float(g.avg_length_of_stay.mean()), 2),
            avg_lead_time=round(float(g.booking_lead_time_days.mean()), 1),
            avg_recency_days=round(float(g.recency_days.mean()), 1),
            churn_risk=round(float(g.churn_prob.mean()), 3),
            next_best_offer=pers.get(base, {}).get("next_best_offer", "General loyalty offer"),
            playbook=pers.get(base, {}).get("playbook", "Standard lifecycle marketing")))
    return pd.DataFrame(rows).sort_values("segment_id")


# --------------------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------------------
def compare_models(entries, X, guests, cfg):
    """entries: list of dicts (name, labels, space, refit_fn, infer_fn, train_time_s, can_serve)."""
    audit = audit_logger(cfg)
    ec = cfg["evaluation"]
    rows = []
    for e in entries:
        m = cluster_metrics(X, e["labels"], ec["silhouette_sample"], cfg["seed"])
        own = cluster_metrics(e["space"], e["labels"], ec["silhouette_sample"], cfg["seed"])
        stab = stability(e["refit_fn"], e["labels"], ec["stability_bootstraps"], seed=cfg["seed"])
        lat = measure_latency(e["infer_fn"])
        sizes = pd.Series(e["labels"]).value_counts().sort_index().to_dict()
        rows.append(dict(model=e["name"], paradigm=e["paradigm"], **m, latent_silhouette=own["silhouette"],
                         stability_ari=stab, train_time_ms=e["train_time_s"] * 1000, inference_latency_ms=lat,
                         ari_vs_planted_archetypes=float(adjusted_rand_score(guests["true_archetype"], e["labels"])),
                         can_serve=e["can_serve"]))
        audit.info("MODEL %-26s | K=%d noise=%.1f%% | sil(common)=%.4f DB=%.4f CH=%.1f | sil(latent)=%.4f | "
                   "stability ARI=%.3f | train=%.0fms infer=%.2fms", e["name"], m["n_clusters"],
                   100 * m["noise_ratio"], m["silhouette"], m["davies_bouldin"], m["calinski_harabasz"],
                   own["silhouette"], stab, e["train_time_s"] * 1000, lat)
        audit.info("   segment sizes: %s", sizes)
    df = pd.DataFrame(rows)
    # composite rank: lower is better. Silhouette, CH, stability higher-is-better; DB lower-is-better.
    rk = pd.DataFrame({
        "sil": df.silhouette.rank(ascending=False, na_option="bottom"),
        "ch": df.calinski_harabasz.rank(ascending=False, na_option="bottom"),
        "db": df.davies_bouldin.rank(ascending=True, na_option="bottom"),
        "stab": df.stability_ari.rank(ascending=False, na_option="bottom")})
    df["composite_rank_score"] = rk.mean(1)
    # penalise degenerate segmentations (noise-dominated / <3 clusters) - they can't drive marketing playbooks
    df.loc[(df.n_clusters < 3) | (df.noise_ratio > 0.3), "composite_rank_score"] += 10
    return df


def select_best(df, cfg):
    audit = audit_logger(cfg)
    cand = df[df.can_serve].sort_values(["composite_rank_score", "silhouette"], ascending=[True, False])
    best = cand.iloc[0]
    audit.info("=" * 100)
    audit.info("SELECTION: composite = mean rank of (silhouette, Calinski-Harabasz, Davies-Bouldin, stability) "
               "on the common standardised feature space; only models with an online `predict` are eligible.")
    for _, r in df.sort_values("composite_rank_score").iterrows():
        audit.info("   %-26s composite=%.2f eligible=%s", r.model, r.composite_rank_score, r.can_serve)
    audit.info("WINNER: %s (composite=%.2f, silhouette=%.4f, DB=%.4f, CH=%.1f, stability=%.3f)", best.model,
               best.composite_rank_score, best.silhouette, best.davies_bouldin, best.calinski_harabasz,
               best.stability_ari)
    return best.model


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_metric_comparison(df, path):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    specs = [("silhouette", "Silhouette (higher better)"), ("davies_bouldin", "Davies-Bouldin (LOWER better)"),
             ("calinski_harabasz", "Calinski-Harabasz (higher better)"),
             ("stability_ari", "Bootstrap stability ARI (higher better)"),
             ("train_time_ms", "Training latency (ms, log)"),
             ("inference_latency_ms", "Inference latency (ms / 5k guests)")]
    for ax, (col, title) in zip(axes.ravel(), specs):
        vals = df[col].fillna(0).values
        bars = ax.bar(range(len(df)), vals, color=PALETTE[:len(df)])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.model, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
        if "train" in col:
            ax.set_yscale("log")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Segmentation model comparison (common standardised feature space)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_radar(profile_z, names, sizes, path):
    feats = ["avg_daily_rate", "avg_length_of_stay", "booking_lead_time_days", "frequency_stays_12m",
             "recency_days", "spa_spend_ratio", "dining_fb_spend_ratio", "business_center_usage",
             "ota_booking_ratio", "direct_app_booking_ratio", "email_open_rate", "weekend_vs_weekday_ratio"]
    idx = [NUM_COLS.index(f) for f in feats]
    P = profile_z[:, idx]
    P = (P - P.min(0)) / (P.max(0) - P.min(0) + 1e-9)
    ang = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    ang += ang[:1]
    n = len(names)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.2 * rows), subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off"); continue
        v = P[i].tolist() + P[i][:1].tolist()
        ax.plot(ang, v, color=PALETTE[i % 6], lw=2); ax.fill(ang, v, color=PALETTE[i % 6], alpha=.25)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels([f.replace("_", "\n") for f in feats], fontsize=6.5)
        ax.set_yticklabels([]); ax.set_ylim(0, 1)
        ax.set_title(f"{names[i]}\n({sizes[i]:,} guests)", fontsize=10, pad=18)
    fig.suptitle("Winning model - hospitality persona profiles (min-max across segments)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_benchmark(df, kpis, folder, best_name):
    out = df.copy()
    out["selected"] = out.model == best_name
    out.to_csv(folder / "model_benchmark_summary.csv", index=False)
    payload = {"best_model": best_name,
               "models": json.loads(out.to_json(orient="records")),
               "winning_model_segment_kpis": json.loads(kpis.to_json(orient="records"))}
    with open(folder / "model_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    kpis.to_csv(folder / "segment_business_kpis.csv", index=False)
