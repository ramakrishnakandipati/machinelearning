"""Centralized logging: console + execution.log, JSONL metric/epoch tracking, GPU diagnostics."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUBDIRS = ["models", "data", "plots/eda", "plots/features", "plots/training_loss",
           "plots/comparisons", "logs", "api_screenshots"]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load YAML config and resolve the output directory to an absolute path."""
    cfg = yaml.safe_load((path or ROOT / "config" / "config.yaml").read_text())
    cfg["out"] = (ROOT / cfg["paths"]["output"]).resolve()
    return cfg


def ensure_dirs(out: Path) -> None:
    for s in SUBDIRS:
        (out / s).mkdir(parents=True, exist_ok=True)


def get_logger(out: Path, name: str = "cltv") -> logging.Logger:
    """Return a logger writing to console and output/logs/execution.log."""
    ensure_dirs(out)
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out / "logs" / "execution.log", encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(sh)
    lg.addHandler(fh)
    return lg


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), **record}, default=float) + "\n")


def gpu_memory_mb() -> float:
    try:
        import torch
        return float(torch.cuda.memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0


def log_epoch(out: Path, model: str, epoch: int, train_loss: float, val_loss: float, lr: float) -> None:
    """Append one epoch record to training_history.jsonl."""
    append_jsonl(out / "logs" / "training_history.jsonl", dict(
        model=model, epoch=epoch, train_loss=float(train_loss), val_loss=float(val_loss),
        lr=float(lr), gpu_mem_mb=gpu_memory_mb()))


def get_device(cfg: dict[str, Any], lg: logging.Logger | None = None) -> str:
    """Return 'cuda' if available and enabled, else 'cpu'; log hardware details."""
    import torch
    use = cfg["gpu"]["enabled"] and torch.cuda.is_available()
    if lg:
        if use:
            p = torch.cuda.get_device_properties(0)
            lg.info(f"GPU: {p.name} | VRAM {p.total_memory/2**30:.1f} GB | torch {torch.__version__} | CUDA {torch.version.cuda}")
        else:
            lg.info("CUDA unavailable/disabled -> CPU")
    return "cuda" if use else "cpu"
