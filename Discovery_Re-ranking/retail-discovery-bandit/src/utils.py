"""Shared utilities: config, device handling, seeding, structured logging, stage timers."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def output_dir(cfg: Dict[str, Any], *sub: str) -> Path:
    p = ROOT / cfg["paths"]["output_dir"]
    p = p.joinpath(*sub)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> torch.device:
    """CUDA when available, CPU only as a clean fallback."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def xgb_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device": "cpu"}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "vram_total_mb": round(total / 2**20, 1),
        "vram_used_mb": round((total - free) / 2**20, 1),
        "torch_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
        "torch_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }


def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


class JsonlLogger:
    """Append-only structured JSONL log."""

    def __init__(self, path: Path, truncate: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if truncate and self.path.exists():
            self.path.unlink()
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=_jsonable) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class StageTimer:
    """Accumulates per-stage latencies (ms), synchronising CUDA for honest GPU timings."""

    def __init__(self):
        self.current: Dict[str, float] = {}
        self.history: Dict[str, list] = {}

    @contextmanager
    def stage(self, name: str):
        sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            sync()
            self.current[name] = self.current.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def commit(self) -> Dict[str, float]:
        cur = dict(self.current)
        cur["total"] = float(sum(cur.values()))
        for k, v in cur.items():
            self.history.setdefault(k, []).append(v)
        self.current = {}
        return cur

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for k, v in self.history.items():
            a = np.asarray(v)
            out[k] = {
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)),
                "mean": float(a.mean()),
                "n": int(a.size),
            }
        return out


def save_json(obj: Any, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def ensure_src_on_path() -> None:
    r = str(ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
