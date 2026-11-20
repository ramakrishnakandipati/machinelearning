"""End-to-end orchestrator.  Usage: python run_pipeline.py [--quick] [--skip-api-shots]"""
from __future__ import annotations

import argparse
import sys
import time

import config as C

EXPECTED = [
    "logs/pipeline.log", "logs/training_losses.jsonl", "logs/metrics_summary.json",
    "models/preprocessor.joblib", "models/kmeans.joblib", "models/gmm.joblib", "models/dbscan.joblib",
    "models/agglomerative.joblib", "models/autoencoder_dec.pt", "models/vae_cluster.pt",
    "plots/01_eda_distributions.png", "plots/02_feature_correlation_importance.png",
    "plots/03_dl_training_losses.png", "plots/04_umap_cluster_projections.png",
    "plots/05_model_metrics_comparison.png", "plots/06_cluster_persona_heatmap.png",
    "api_screenshots/01_swagger_docs.png", "api_screenshots/02_health_check.png",
    "api_screenshots/03_predict_cluster_response.png",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few epochs / small data for smoke testing")
    ap.add_argument("--skip-api-shots", action="store_true")
    ap.add_argument("--n-guests", type=int, default=C.N_GUESTS)
    ap.add_argument("--csv", default=None, help="optional PMS/CRM csv with the feature columns")
    ap.add_argument("--k", type=int, default=None, help="force cluster count (default: best silhouette)")
    ap.add_argument("--epochs", type=int, default=None, help="override AE/VAE epochs")
    args = ap.parse_args()

    log = C.setup_logging()
    C.set_seed()
    C.log_hardware(log)
    hp = {}
    if args.quick:
        hp = dict(ae_epochs=8, dec_epochs=10, vae_epochs=10, kl_warmup=3)
    if args.epochs:
        hp.update(ae_epochs=args.epochs, vae_epochs=args.epochs)

    # heavy imports after logging is configured
    import data_pipeline as dp
    import models_dl
    import models_ml
    import visualize_and_eval as ve

    t0 = time.perf_counter()
    log.info("=== 1/5 data & preprocessing ===")
    df = dp.load_or_simulate(args.csv, 3000 if args.quick else args.n_guests)
    d = dp.prepare(df)
    Xtr, Xte, dtr = d["X_train"], d["X_test"], d["df_train"]
    truth = dtr["true_persona"].to_numpy() if "true_persona" in dtr else None
    log.info("train=%s test=%s", Xtr.shape, Xte.shape)

    log.info("=== 2/5 classical ML models ===")
    ml = models_ml.fit_all(Xtr, args.k)
    ks = ml.pop("_k_selection")

    log.info("=== 3/5 deep models (%s) ===", C.DEVICE)
    dl = models_dl.fit_all(Xtr, ks["best_k"], hp)
    models = {**ml, **dl}

    log.info("=== 4/5 evaluation & plots ===")
    if truth is None:
        import numpy as np
        truth = np.zeros(len(Xtr), dtype=int)
    res = ve.evaluate(models, Xtr, Xte, truth, ks)
    pmap = ve.build_personas(models, dtr)
    ve.make_all_plots(dtr, Xtr, models, res, pmap)

    if not args.skip_api_shots:
        log.info("=== 5/5 API screenshots ===")
        import capture_api_screenshots
        capture_api_screenshots.capture()

    missing = [p for p in EXPECTED if not (C.OUTPUT_DIR / p).exists() or (C.OUTPUT_DIR / p).stat().st_size == 0]
    if args.skip_api_shots:
        missing = [m for m in missing if not m.startswith("api_screenshots")]
    log.info("Done in %.1fs. Missing outputs: %s", time.perf_counter() - t0, missing or "none")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
