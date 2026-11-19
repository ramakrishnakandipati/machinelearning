#!/usr/bin/env bash
# Hardware-aware setup: detects an NVIDIA GPU, installs CUDA torch / faiss-gpu when possible, else CPU fallbacks.
set -u
PY=${PYTHON:-python}
OS=$(uname -s 2>/dev/null || echo unknown)
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then HAS_GPU=1; fi
echo "[setup] NVIDIA GPU detected: $HAS_GPU (OS: $OS)"

$PY -m pip install --upgrade pip >/dev/null
if $PY -c "import torch,sys; sys.exit(0 if (torch.cuda.is_available() or $HAS_GPU == 0) else 1)" 2>/dev/null; then
  echo "[setup] torch already usable"
elif [ "$HAS_GPU" = 1 ]; then
  echo "[setup] installing CUDA torch"
  $PY -m pip install torch --index-url https://download.pytorch.org/whl/cu124
else
  echo "[setup] installing CPU torch"
  $PY -m pip install torch
fi

# faiss-gpu only ships wheels for Linux; native Windows / no GPU -> faiss-cpu
case "$OS" in
  Linux*)
    if [ "$HAS_GPU" = 1 ]; then $PY -m pip install faiss-gpu-cu12 || $PY -m pip install faiss-cpu; else $PY -m pip install faiss-cpu; fi ;;
  *)
    echo "[setup] no faiss-gpu wheel for $OS -> faiss-cpu (use WSL/conda for GPU ANN)"
    $PY -m pip install faiss-cpu ;;
esac

# optional GPU array stack (Linux + CUDA only); code falls back to pandas/numpy
if [ "$HAS_GPU" = 1 ] && [ "$OS" = "Linux" ]; then $PY -m pip install cupy-cuda12x || true; fi

grep -vE '^(torch|faiss|#)' requirements.txt > req_rest.tmp
$PY -m pip install -r req_rest.tmp
rm -f req_rest.tmp
$PY -m playwright install chromium || echo "[setup] chromium unavailable -> Pillow screenshot fallback"
$PY -m src.check_gpu
