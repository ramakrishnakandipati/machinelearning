"""Single-command end-to-end pipeline: data -> features -> train -> evaluate -> plots -> API screenshots."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_generator import generate  # noqa: E402
from src.evaluate import evaluate_all  # noqa: E402
from src.features import build_dataset  # noqa: E402
from src.logger import ensure_dirs, get_device, get_logger, load_config  # noqa: E402
from src.train import train_all  # noqa: E402
from src.visualize import plot_comparisons, plot_eda, plot_features, plot_training  # noqa: E402

REQUIRED = [
    "models/best_model.json", "models/xgboost_tweedie.json", "models/catboost_tweedie.cbm", "models/ft_transformer_ziln.pt",
    "models/gru_attention_ziln.pt", "models/deep_weibull_survival.pt", "models/bgnbd_gg.joblib",
    "models/lightgbm_hurdle.joblib", "models/num_scaler.joblib",
    "plots/eda/cltv_distribution.png", "plots/eda/spend_by_loyalty_tier.png", "plots/eda/leadtime_vs_adr.png",
    "plots/eda/retention_decay.png", "plots/features/correlation_matrix.png", "plots/features/shap_beeswarm.png",
    "plots/features/sequence_attention_heatmap.png", "plots/training_loss/training_validation_loss_all_models.png",
    "plots/comparisons/model_comparison_metrics.png", "plots/comparisons/cumulative_gain_lorenz.png",
    "plots/comparisons/calibration_by_decile.png",
    "logs/execution.log", "logs/metrics_summary.json", "logs/evaluation_report.csv", "logs/training_history.jsonl",
]
SHOTS = ["swagger_ui_overview.png", "single_guest_prediction.png", "batch_cohort_prediction.png"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-guests", type=int, default=None, help="override guest count (smoke test)")
    ap.add_argument("--skip-screenshots", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.n_guests:
        cfg["data"]["n_guests"] = args.n_guests
        cfg["data"]["target_bookings"] = int(args.n_guests * 4.4)
    out = cfg["out"]
    ensure_dirs(out)
    for f in (out / "logs").glob("*.jsonl"):
        f.unlink()  # fresh training history per run
    lg = get_logger(out)
    t0 = time.time()
    lg.info("=" * 70 + "\nHospitality CLTV pipeline starting")
    get_device(cfg, lg)

    guests, bookings = generate(cfg, seed=cfg["seed"])
    guests.to_parquet(out / "data" / "guests.parquet", index=False)
    bookings.to_parquet(out / "data" / "bookings.parquet", index=False)
    lg.info(f"Data: {len(guests):,} guests | {len(bookings):,} bookings | cancel rate {bookings.is_cancelled.mean():.1%}")
    ds = build_dataset(guests, bookings, cfg, seed=cfg["seed"])
    lg.info(f"Features {ds.df.shape} | sequences {ds.seq.shape} | splits "
            f"{ {s: int((ds.split == s).sum()) for s in ('train', 'val', 'test')} } | "
            f"zero-spend share {(ds.df.y == 0).mean():.1%}")

    res = train_all(cfg, ds, lg)
    ev = evaluate_all(cfg, res["predictions"], res["timing"], lg)
    plot_eda(cfg, ds, guests, bookings, lg)
    plot_features(cfg, ds, res["models"], lg)
    plot_training(cfg, res["histories"], lg)
    plot_comparisons(cfg, ev, lg)

    if not args.skip_screenshots:
        lg.info("Capturing API screenshots ...")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "capture_api_screenshots.py")], cwd=ROOT)
        if r.returncode != 0:
            lg.error("Screenshot capture failed")

    missing = [p for p in REQUIRED if not (out / p).exists() or (out / p).stat().st_size == 0]
    if not args.skip_screenshots:
        missing += [f"api_screenshots/{s}" for s in SHOTS if not (out / "api_screenshots" / s).exists()]
        if not (out / "logs" / "api_telemetry.jsonl").exists():
            missing.append("logs/api_telemetry.jsonl")
    lg.info(f"Pipeline finished in {time.time() - t0:.0f}s | missing artifacts: {missing or 'none'}")
    return 1 if missing else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
