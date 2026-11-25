"""Shared helpers: config loading, paths, logging, GPU/CUDA introspection."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_of(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str, cfg: dict | None = None, logfile: str = "pipeline.log") -> logging.Logger:
    cfg = cfg or load_config()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(path_of(cfg, "logs") / logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def cuda_info() -> dict:
    """CUDA availability + device name + memory (torch is the source of truth)."""
    info = {"cuda_available": False, "device_name": None, "total_mem_gb": None, "used_mem_gb": None}
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info.update(
                cuda_available=True,
                device_name=torch.cuda.get_device_name(0),
                total_mem_gb=round(total / 1e9, 2),
                used_mem_gb=round((total - free) / 1e9, 2),
                torch_cuda=torch.version.cuda,
            )
    except Exception:  # pragma: no cover
        pass
    return info


def nvidia_smi_used_mb() -> float | None:
    """Total GPU memory in use (all processes) in MB, via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return None


def gpu_enabled(cfg: dict) -> bool:
    return bool(cfg["gpu"]["enabled"]) and cuda_info()["cuda_available"]
