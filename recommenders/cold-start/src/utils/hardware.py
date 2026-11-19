"""Hardware diagnostics, deterministic seeding, AMP helpers and memory tracking."""
from __future__ import annotations

import contextlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(use_gpu: bool = True) -> torch.device:
    """Return CUDA device if requested and available, else CPU."""
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")


def faiss_gpu_available() -> bool:
    """True when faiss was built with GPU support and a GPU is visible."""
    try:
        import faiss  # noqa: F401

        return bool(hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0)
    except Exception:
        return False


def amp_enabled(device: torch.device, use_amp: bool = True) -> bool:
    """AMP is only meaningful on CUDA."""
    return bool(use_amp and device.type == "cuda")


@contextlib.contextmanager
def autocast_ctx(device: torch.device, enabled: bool) -> Iterator[None]:
    """Autocast context that is a no-op on CPU / when disabled."""
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def make_grad_scaler(device: torch.device, enabled: bool):
    """Create a GradScaler across torch versions (disabled when not on CUDA)."""
    on = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=on)
    except Exception:  # older torch
        return torch.cuda.amp.GradScaler(enabled=on)


def gpu_memory_mb() -> Dict[str, float]:
    """Current/peak allocated CUDA memory in MB (zeros on CPU)."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "peak_mb": 0.0}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 2),
        "peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }


def collect_diagnostics() -> Dict[str, Any]:
    """Collect environment, PyTorch/CUDA and FAISS status."""
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [],
        "numpy_version": np.__version__,
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        info["gpus"].append(
            {
                "index": i,
                "name": p.name,
                "total_vram_gb": round(p.total_memory / 2**30, 2),
                "compute_capability": f"{p.major}.{p.minor}",
            }
        )
    try:
        import faiss

        info["faiss_version"] = getattr(faiss, "__version__", "unknown")
        info["faiss_gpu"] = faiss_gpu_available()
        info["faiss_mode"] = "GPU" if info["faiss_gpu"] else "CPU"
    except Exception as exc:  # pragma: no cover
        info["faiss_version"] = None
        info["faiss_gpu"] = False
        info["faiss_mode"] = f"unavailable ({exc})"
    try:
        import sentence_transformers

        info["sentence_transformers"] = sentence_transformers.__version__
    except Exception:
        info["sentence_transformers"] = None
    info["memory"] = gpu_memory_mb()
    return info


def write_diagnostics(path: str | Path) -> Dict[str, Any]:
    """Write diagnostics JSON and return the dict."""
    info = collect_diagnostics()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
