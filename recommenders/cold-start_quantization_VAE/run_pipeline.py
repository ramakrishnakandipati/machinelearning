"""Master orchestrator: Data -> Train -> Eval -> Viz -> API -> Screenshots.

    python run_pipeline.py                 # full run (populates every folder under ./outputs/)
    python run_pipeline.py --quick         # smoke test with tiny epoch counts
    python run_pipeline.py --skip-train    # reuse saved models, redo eval / viz / screenshots
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from src.config import LOGS_DIR, MODELS_DIR, OUT, SHOT_DIR, VIZ_DIR, DEVICE, get_config, seed_everything


def stage(name: str):
    print(f"\n{'=' * 8} {name} {'=' * 8}", flush=True)
    return time.time()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-screenshots", action="store_true")
    args = ap.parse_args()
    cfg = get_config(args.quick)
    seed_everything()
    times = {}
    print(f"device: {DEVICE}")

    from src.artifacts import load_bundle
    from src.data.generate_data import generate_dataset, save_dataset, summarize
    from src.training.evaluator import run_benchmark
    from src.training.train_all import train_all
    from src.visualization.plot_suite import make_all

    if not args.skip_train:
        t = stage("1/6 data generation")
        ds = generate_dataset(cfg.data)
        save_dataset(ds)
        summ = summarize(ds)
        print(json.dumps(summ, indent=1))
        (LOGS_DIR / "dataset_summary.json").write_text(json.dumps(summ, indent=2))
        times["data"] = time.time() - t

        t = stage("2/6 training VQ-VAE, RQ-VAE, generative model, codebook bridge")
        train_all(ds, cfg)
        times["train"] = time.time() - t

    t = stage("3/6 reload artifacts + evaluation")
    bundle = load_bundle()          # exactly what the API serves
    df = run_benchmark(bundle, cfg, LOGS_DIR / "evaluation_metrics.csv")
    times["eval"] = time.time() - t

    t = stage("4/6 visualizations")
    print(json.dumps(make_all(bundle, df), indent=1))
    times["viz"] = time.time() - t

    if not args.skip_screenshots:
        t = stage("5/6 API server + Playwright screenshots")
        from src.api.capture_screenshots import run_with_server
        shots = run_with_server()
        print("\n".join(shots))
        times["screenshots"] = time.time() - t

    stage("6/6 output manifest")
    manifest = {str(p.relative_to(OUT)): p.stat().st_size for d in (MODELS_DIR, LOGS_DIR, VIZ_DIR, SHOT_DIR)
                for p in sorted(d.glob("*")) if p.is_file()}
    for k, v in manifest.items():
        print(f"  {k:55s} {v / 1024:9.1f} KB")
    (LOGS_DIR / "pipeline_timing.json").write_text(json.dumps({k: round(v, 1) for k, v in times.items()}, indent=2))
    empty = [d.name for d in (MODELS_DIR, LOGS_DIR, VIZ_DIR, SHOT_DIR) if not any(d.iterdir())]
    print(f"\ntimings (s): {({k: round(v, 1) for k, v in times.items()})}")
    print("all output folders populated" if not empty else f"EMPTY folders: {empty}")


if __name__ == "__main__":
    main()
