#!/usr/bin/env bash
# End-to-end: install deps -> train/evaluate -> API tests -> Swagger screenshots.
# Works in Git Bash / WSL / Linux / macOS. Use run_all.ps1 for native PowerShell.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
LOG_DIR="output/training_logs"
mkdir -p "$LOG_DIR" output/{data_visuals,models,comparison_graphs,api_screenshots} data

echo "[run_all] Installing dependencies ..." | tee -a "$LOG_DIR/pipeline.log"
# For a CUDA build of PyTorch on a clean machine, install it first, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install -q -r requirements.txt 2>&1 | tee -a "$LOG_DIR/pipeline.log"
"$PY" -m playwright install chromium 2>&1 | tee -a "$LOG_DIR/pipeline.log"

echo "[run_all] Training + evaluation pipeline ..." | tee -a "$LOG_DIR/pipeline.log"
"$PY" -m src.train_pipeline "$@" 2>&1 | tee -a "$LOG_DIR/pipeline.log"
test -f "$LOG_DIR/metrics_summary.json"

echo "[run_all] API tests ..." | tee -a "$LOG_DIR/pipeline.log"
"$PY" -m pytest -q tests/test_api.py 2>&1 | tee -a "$LOG_DIR/pipeline.log"

echo "[run_all] Capturing API screenshots ..." | tee -a "$LOG_DIR/pipeline.log"
"$PY" tests/capture_api_screenshots.py 2>&1 | tee -a "$LOG_DIR/pipeline.log"

echo "[run_all] Done. Metrics: $LOG_DIR/metrics_summary.json | Plots: output/*/" | tee -a "$LOG_DIR/pipeline.log"
