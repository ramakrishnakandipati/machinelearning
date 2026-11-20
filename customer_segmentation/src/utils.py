"""Device management, logging, seeding, config helpers."""
import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
_LOGGER_NAME = "segmentation"


def load_config(path=None):
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def out_dir(cfg=None, *parts):
    base = ROOT / ((cfg or {}).get("paths", {}).get("output", "output"))
    p = base.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_output_dirs(cfg=None):
    for sub in ["logs", "data_visualizations", "feature_visualizations", "training_losses",
                "model_comparisons", "saved_models", "api_screenshots"]:
        out_dir(cfg, sub)


def get_logger(name=_LOGGER_NAME, logfile="pipeline_execution.log", cfg=None):
    """Logger streaming to console and output/logs/<logfile>."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(out_dir(cfg, "logs") / logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(use_gpu=True, logger=None):
    import torch
    log = logger or get_logger()
    if use_gpu and torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        log.info("CUDA device: %s | VRAM total %.1f GB | torch %s | CUDA %s",
                 props.name, props.total_memory / 1e9, torch.__version__, torch.version.cuda)
    else:
        dev = torch.device("cpu")
        log.warning("Running on CPU (%s)", "GPU disabled by flag" if not use_gpu else "CUDA not available")
    return dev


def log_gpu_memory(tag="", logger=None):
    import torch
    if torch.cuda.is_available():
        (logger or get_logger()).info("GPU mem [%s]: allocated %.1f MB | reserved %.1f MB", tag,
                                      torch.cuda.memory_allocated() / 1e6, torch.cuda.memory_reserved() / 1e6)


@contextmanager
def timed(label, logger=None):
    log = logger or get_logger()
    t0 = time.perf_counter()
    log.info("START %s", label)
    try:
        yield
    finally:
        log.info("END   %s (%.2fs)", label, time.perf_counter() - t0)


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class JsonlMetricsLogger:
    """Append structured per-epoch records to output/logs/training_metrics.jsonl."""

    def __init__(self, cfg=None, filename="training_metrics.jsonl"):
        self.path = out_dir(cfg, "logs") / filename

    def log(self, **record):
        record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")

    def read(self, model=None, phase=None):
        rows = []
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if (model is None or r.get("model") == model) and (phase is None or r.get("phase") == phase):
                        rows.append(r)
        return rows


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5
