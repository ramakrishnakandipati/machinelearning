"""Master orchestrator: data -> features -> EDA -> train 6 models -> evaluate -> save best.

Usage:  python -m src.train_pipeline [--quick] [--cpu]
"""
import argparse
import datetime as dt
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from . import eda_visualizer as viz
from .data_generator import build_and_save
from .evaluate_and_compare import (compare_models, name_clusters, plot_metric_comparison, plot_radar,
                                   save_benchmark, segment_kpis, select_best, audit_logger, cluster_metrics)
from .feature_engineering import FeaturePipeline, SequenceNormalizer, engineer_features
from .models.classical_ml import GMMGPU, HDBSCANModel, KMeansGPU, HAS_CUML
from .models.deep_autoencoder import dec_predict, save_dec, train_dec
from .models.deep_vae import save_vae, train_vae, vae_latents
from .models.sequence_transformer import save_seq, seq_embeddings, train_seq
from .utils import (JsonlMetricsLogger, ROOT, ensure_output_dirs, get_device, get_logger, load_config,
                    log_gpu_memory, out_dir, set_seed, sync, timed)

PERSONA_ORDER = None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke-test run")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.quick:
        q = cfg["training"]["quick"]
        cfg["training"].update({k: v for k, v in q.items() if k != "n_guests"})
        cfg["data"]["n_guests"] = q["n_guests"]
    ensure_output_dirs(cfg)
    # fresh run -> fresh structured metrics file
    mp = out_dir(cfg, "logs") / "training_metrics.jsonl"
    if mp.exists():
        mp.unlink()
    log = get_logger(cfg=cfg)
    audit = audit_logger(cfg)
    metrics = JsonlMetricsLogger(cfg)
    set_seed(cfg["seed"])
    device = get_device(cfg["use_gpu"] and not args.cpu, log)
    if not HAS_CUML:
        log.warning("cuML not installed -> using PyTorch GPU K-Means/GMM + sklearn HDBSCAN fallback")
    K = cfg["model"]["n_clusters"]
    saved = out_dir(cfg, "saved_models")
    T0 = time.perf_counter()
    log.info("Run config: quick=%s device=%s K=%d guests=%d", args.quick, device, K, cfg["data"]["n_guests"])

    # ---------------------------------------------------------------- data + features
    with timed("data generation", log):
        guests, events, seq, mask = build_and_save(cfg)
    with timed("feature engineering", log):
        gfeat = engineer_features(guests)
        fp = FeaturePipeline()
        X = fp.fit_transform(gfeat)
        fp.save(saved)
        sn = SequenceNormalizer().fit(seq, mask)
        joblib.dump(sn, saved / "sequence_normalizer.joblib")
        cont, prop, m = sn.transform(seq, mask)
    log.info("Shapes: X=%s | seq cont=%s prop=%s mask=%s", X.shape, cont.shape, prop.shape, m.shape)
    log_gpu_memory("after data", log)

    # ---------------------------------------------------------------- EDA
    with timed("EDA plots", log):
        viz.plot_raw_distributions(gfeat, out_dir(cfg, "data_visualizations") / "raw_distributions.png")
        viz.plot_channel_breakdown(gfeat, out_dir(cfg, "data_visualizations") / "booking_channel_breakdown.png")
        viz.plot_correlation_heatmap(gfeat, out_dir(cfg, "feature_visualizations") / "feature_correlation_heatmap.png")

    # ---------------------------------------------------------------- K sweep (audit)
    with timed("K sweep (K-Means / GMM)", log):
        for k in cfg["model"]["k_sweep"]:
            km_k = KMeansGPU(k, seed=cfg["seed"], device=device).fit(X)
            lab = km_k.predict(X)
            mk = cluster_metrics(X, lab, cfg["evaluation"]["silhouette_sample"])
            gm_k = GMMGPU(k, seed=cfg["seed"], device=device).fit(X)
            audit.info("K-SWEEP k=%d | KMeans inertia=%.1f sil=%.4f DB=%.4f CH=%.1f | GMM BIC=%.1f", k,
                       km_k.inertia_ if hasattr(km_k, "inertia_") else float("nan"), mk["silhouette"],
                       mk["davies_bouldin"], mk["calinski_harabasz"], gm_k.bic(len(X), X.shape[1]))

    entries, store = [], {}
    n_inf = min(5000, len(X))
    Xi = X[:n_inf]

    def kmeans_refit_factory(space, make):
        def refit(idx, seed):
            return make(seed).fit(space[idx]).predict(space)
        return refit

    # ---------------------------------------------------------------- 1. K-Means
    with timed("train K-Means", log):
        t = time.perf_counter()
        km = KMeansGPU(K, seed=cfg["seed"], device=device).fit(X); sync()
        tt = time.perf_counter() - t
    lab_km = km.predict(X)
    entries.append(dict(name="K-Means (GPU)", paradigm="Centroid", labels=lab_km, space=X, train_time_s=tt,
                        refit_fn=kmeans_refit_factory(X, lambda s: KMeansGPU(K, seed=s, device=device)),
                        infer_fn=lambda: km.predict(Xi)))
    km.save(saved / "kmeans.joblib")

    # ---------------------------------------------------------------- 2. GMM
    with timed("train GMM", log):
        t = time.perf_counter()
        gmm = GMMGPU(K, seed=cfg["seed"], device=device).fit(X); sync()
        tt = time.perf_counter() - t
    lab_gmm = gmm.predict(X)
    entries.append(dict(name="GMM (GPU EM)", paradigm="Probabilistic", labels=lab_gmm, space=X, train_time_s=tt,
                        refit_fn=kmeans_refit_factory(X, lambda s: GMMGPU(K, seed=s, device=device)),
                        infer_fn=lambda: gmm.predict_proba(Xi)))
    gmm.save(saved / "gmm.joblib")

    # ---------------------------------------------------------------- 3. HDBSCAN
    with timed("train HDBSCAN", log):
        t = time.perf_counter()
        hdb = HDBSCANModel(min_cluster_size=max(50, len(X) // 150), seed=cfg["seed"]).fit(X)
        tt = time.perf_counter() - t
    lab_h = hdb.labels_
    audit.info("HDBSCAN found %d clusters; noise/VIP-outlier ratio %.2f%%", len(hdb.cluster_ids_), 100 * (lab_h < 0).mean())
    entries.append(dict(name="HDBSCAN", paradigm="Density", labels=lab_h, space=X, train_time_s=tt,
                        refit_fn=lambda idx, s: HDBSCANModel(min_cluster_size=hdb.mcs, seed=s).fit(X[idx]).predict(X),
                        infer_fn=lambda: hdb.predict(Xi)))
    hdb.save(saved / "hdbscan.joblib")

    # ---------------------------------------------------------------- 4. DEC
    with timed("train DEC autoencoder", log):
        t = time.perf_counter()
        dec, h_dec = train_dec(X, K, device, cfg, metrics, cfg["seed"]); sync()
        tt = time.perf_counter() - t
    lab_dec, q_dec, Z_dec = dec_predict(dec, X, device)
    entries.append(dict(name="DEC Autoencoder", paradigm="Deep", labels=lab_dec, space=Z_dec, train_time_s=tt,
                        refit_fn=kmeans_refit_factory(Z_dec, lambda s: KMeansGPU(K, seed=s, device=device)),
                        infer_fn=lambda: dec_predict(dec, Xi, device)))
    save_dec(dec, saved / "dec_autoencoder.pt")
    viz.plot_dec_loss(h_dec, out_dir(cfg, "training_losses") / "dec_autoencoder_loss.png")
    log_gpu_memory("after DEC", log)

    # ---------------------------------------------------------------- 5. beta-VAE
    with timed("train beta-VAE", log):
        t = time.perf_counter()
        vae, h_vae = train_vae(X, device, cfg, metrics)
        mu = vae_latents(vae, X, device)
        km_v = KMeansGPU(K, seed=cfg["seed"], device=device).fit(mu); sync()
        tt = time.perf_counter() - t
    lab_vae = km_v.predict(mu)
    entries.append(dict(name="Beta-VAE + K-Means", paradigm="Deep generative", labels=lab_vae, space=mu, train_time_s=tt,
                        refit_fn=kmeans_refit_factory(mu, lambda s: KMeansGPU(K, seed=s, device=device)),
                        infer_fn=lambda: km_v.predict(vae_latents(vae, Xi, device))))
    km_v.save(saved / "beta_vae_kmeans.joblib")
    save_vae(vae, saved / "beta_vae.pt", "beta_vae_kmeans.joblib")
    viz.plot_vae_loss(h_vae, out_dir(cfg, "training_losses") / "beta_vae_loss.png")

    # ---------------------------------------------------------------- 6. Sequence Transformer
    with timed("train Sequence Transformer", log):
        t = time.perf_counter()
        stm, h_seq = train_seq(cont, prop, m, device, cfg, metrics, cfg["seed"])
        E = seq_embeddings(stm, cont, prop, m, device)
        km_s = KMeansGPU(K, seed=cfg["seed"], device=device).fit(E); sync()
        tt = time.perf_counter() - t
    lab_seq = km_s.predict(E)
    entries.append(dict(name="Sequence Transformer", paradigm="Sequential", labels=lab_seq, space=E, train_time_s=tt,
                        refit_fn=kmeans_refit_factory(E, lambda s: KMeansGPU(K, seed=s, device=device)),
                        infer_fn=lambda: km_s.predict(seq_embeddings(stm, cont[:n_inf], prop[:n_inf], m[:n_inf], device))))
    km_s.save(saved / "sequence_kmeans.joblib")
    save_seq(stm, saved / "sequence_transformer.pt")
    viz.plot_seq_loss(h_seq, out_dir(cfg, "training_losses") / "sequence_transformer_loss.png")

    for e in entries:  # eligible for serving: yields the target number of segments
        e["can_serve"] = bool(len(set(e["labels"]) - {-1}) >= K)

    # ---------------------------------------------------------------- evaluate + select
    with timed("evaluation", log):
        df = compare_models(entries, X, gfeat, cfg)
        best_name = select_best(df, cfg)
    best = next(e for e in entries if e["name"] == best_name)
    best_labels = best["labels"]
    log.info("Selected production model: %s", best_name)

    mapping, prof = name_clusters(X, best_labels, cfg)
    audit.info("PERSONA MAPPING (winner %s): %s", best_name, mapping)
    kpis = segment_kpis(gfeat, best_labels, mapping, cfg)
    for _, r in kpis.iterrows():
        audit.info("SEGMENT %d %-34s size=%6d (%.1f%%) LTV=%.0f churn=%.2f NBO=%s", r.segment_id, r.segment_name,
                   r["size"], r.share_pct, r.avg_ltv, r.churn_risk, r.next_best_offer)

    mc = out_dir(cfg, "model_comparisons")
    plot_metric_comparison(df, mc / "metrics_comparison_bar.png")
    ids = [c for c in np.unique(best_labels) if c >= 0]
    sizes = [int((best_labels == c).sum()) for c in ids]
    plot_radar(prof, [mapping[c] for c in ids], sizes, mc / "cluster_radar_personas.png")
    save_benchmark(df, kpis, mc, best_name)

    fv = out_dir(cfg, "feature_visualizations")
    viz.plot_feature_importance(X, best_labels, fp.feature_names_, fv / "feature_importance_anova.png", seed=cfg["seed"])
    viz.plot_latent_umap({"Raw feature space": X, "DEC Autoencoder latent": Z_dec, "Beta-VAE mu latent": mu},
                         {"Raw feature space": lab_km, "DEC Autoencoder latent": lab_dec, "Beta-VAE mu latent": lab_vae},
                         fv / "latent_space_umap.png", seed=cfg["seed"])

    # PCA for 3-D "latent coordinates" for non-neural winners
    pca = PCA(3, random_state=cfg["seed"]).fit(X)
    joblib.dump(pca, saved / "pca_viz.joblib")

    artefacts = {"K-Means (GPU)": ("kmeans", "kmeans.joblib"), "GMM (GPU EM)": ("gmm", "gmm.joblib"),
                 "HDBSCAN": ("hdbscan", "hdbscan.joblib"), "DEC Autoencoder": ("dec", "dec_autoencoder.pt"),
                 "Beta-VAE + K-Means": ("beta_vae_kmeans", "beta_vae.pt"),
                 "Sequence Transformer": ("sequence_kmeans", "sequence_transformer.pt")}
    mtype, mfile = artefacts[best_name]
    brow = df[df.model == best_name].iloc[0]
    registry = {
        "best_model": best_name, "model_type": mtype, "model_file": mfile,
        "cluster_file": {"beta_vae_kmeans": "beta_vae_kmeans.joblib", "sequence_kmeans": "sequence_kmeans.joblib"}.get(mtype),
        "version": dt.datetime.now().strftime("%Y%m%d-%H%M%S"), "trained_at": dt.datetime.now().isoformat(),
        "n_clusters": int(len(ids)), "n_features": int(X.shape[1]), "feature_names": fp.feature_names_,
        "segment_names": {str(k): v for k, v in mapping.items()},
        "metrics": {k: (None if pd.isna(brow[k]) else float(brow[k])) for k in
                    ["silhouette", "davies_bouldin", "calinski_harabasz", "stability_ari"]},
        "preprocessing": {"scaler": "scaler.joblib", "encoder": "encoder.joblib", "pipeline": "feature_pipeline.joblib",
                          "sequence_normalizer": "sequence_normalizer.joblib", "pca_viz": "pca_viz.joblib"},
        "vip_outlier_detector": "hdbscan.joblib",
        "all_models": {k: v[1] for k, v in artefacts.items()},
    }
    with open(saved / "model_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    kpis.to_json(saved / "segment_personas.json", orient="records", indent=2)

    seg = guests.copy(); seg["segment_id"] = best_labels
    seg["segment_name"] = seg.segment_id.map(mapping)
    seg.to_csv(ROOT / cfg["paths"]["data"] / "guests_segmented.csv", index=False)

    log.info("Pipeline finished in %.1fs. Winner: %s", time.perf_counter() - T0, best_name)
    log_gpu_memory("end", log)
    return registry


if __name__ == "__main__":
    main()
