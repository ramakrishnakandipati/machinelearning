"""Structured logging: stdout + rotating file; stage timer with RAM / VRAM usage."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import LOGS

FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str, logfile: str | Path | None = None) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter(FMT)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if logfile:
        p = Path(logfile)
        p = p if p.is_absolute() else LOGS / p
        fh = RotatingFileHandler(p, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def mem_snapshot() -> str:
    parts = []
    try:
        import psutil
        parts.append(f"rss={psutil.Process().memory_info().rss / 2**20:.0f}MB")
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            parts.append(f"vram_alloc={torch.cuda.memory_allocated() / 2**20:.0f}MB "
                         f"vram_peak={torch.cuda.max_memory_allocated() / 2**20:.0f}MB")
    except Exception:
        pass
    return " ".join(parts)


@contextmanager
def stage_timer(log: logging.Logger, stage: str):
    log.info("START %s | %s", stage, mem_snapshot())
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log.info("END   %s | %.2fs | %s", stage, time.perf_counter() - t0, mem_snapshot())
