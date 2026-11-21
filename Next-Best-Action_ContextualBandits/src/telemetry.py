"""Structured JSONL telemetry."""
import json
import threading
import time
from pathlib import Path

from .config import LOGS_DIR

_lock = threading.Lock()


def _default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(o)


def log_jsonl(path, record: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=_default) + "\n")


def training_logger(path=None):
    path = Path(path) if path else LOGS_DIR / "training_metrics.jsonl"

    def _log(record: dict):
        log_jsonl(path, record)

    return _log
