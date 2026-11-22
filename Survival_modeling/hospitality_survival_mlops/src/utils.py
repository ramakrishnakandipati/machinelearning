"""Shared helpers: config loading, logging, device detection, seeding, timing."""
from __future__ import annotations

import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = [
    "loyalty_tier", "adr_avg", "length_of_stay_avg", "lead_time_days",
    "days_since_last_stay", "total_prior_stays", "ancillary_spend_ratio",
    "promo_email_open_rate", "app_sessions_30d", "channel_direct",
    "trip_purpose_business", "seasonality_index",
]
TIERS = ["Member", "Silver", "Gold", "Platinum"]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a configured relative path against the project root and create dirs."""
    p = ROOT / cfg["paths"][key]
    target = p.parent if p.suffix else p
    target.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str = "survival", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_file = log_file or (ROOT / "output" / "training_logs" / "pipeline.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


@dataclass
class DeviceInfo:
    torch_device: str      # "cuda" | "cpu"
    xgb_device: str        # "cuda" | "cpu"
    cuda_available: bool
    name: str


def get_device(cfg: dict | None = None) -> DeviceInfo:
    import torch
    want = True if cfg is None else cfg.get("gpu", {}).get("use_cuda", True)
    ok = bool(want and torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if ok else "CPU"
    return DeviceInfo("cuda" if ok else "cpu", "cuda" if ok else "cpu",
                      torch.cuda.is_available(), name)


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def timer():
    """Usage: with timer() as t: ...; t['seconds']"""
    box = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - start


def make_surv_array(time_, event):
    """scikit-survival structured array."""
    arr = np.empty(len(time_), dtype=[("event", "?"), ("time", "<f8")])
    arr["event"] = np.asarray(event).astype(bool)
    arr["time"] = np.asarray(time_, dtype=float)
    return arr
