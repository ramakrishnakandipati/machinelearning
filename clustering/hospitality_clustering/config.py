"""Central configuration, paths, hardware detection and logging."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
API_SHOTS_DIR = OUTPUT_DIR / "api_screenshots"
DATA_DIR = OUTPUT_DIR / "data"
for _d in (LOGS_DIR, MODELS_DIR, PLOTS_DIR, API_SHOTS_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / "pipeline.log"
LOSS_LOG = LOGS_DIR / "training_losses.jsonl"
METRICS_FILE = LOGS_DIR / "metrics_summary.json"
PERSONA_FILE = MODELS_DIR / "persona_map.json"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"

SEED = 42
N_GUESTS = 10_000
K_RANGE = list(range(2, 11))
DEFAULT_K = 5

FEATURES = [
    "recency_days", "annual_stays", "avg_length_of_stay", "booking_lead_time_days",
    "adr_usd", "fb_spend_ratio", "spa_golf_spend_usd", "room_upgrade_rate",
    "loyalty_tenure_years", "direct_booking_ratio", "cancellation_rate",
    "weekend_stay_ratio", "business_trip_ratio",
]
# skewed features -> log1p before scaling
LOG_FEATURES = ["recency_days", "booking_lead_time_days", "adr_usd", "spa_golf_spend_usd",
                "avg_length_of_stay", "annual_stays"]
# valid ranges used for simulation clipping and API validation
RANGES = {
    "recency_days": (0, 1500), "annual_stays": (0, 60), "avg_length_of_stay": (1, 30),
    "booking_lead_time_days": (0, 365), "adr_usd": (30, 2500), "fb_spend_ratio": (0, 1),
    "spa_golf_spend_usd": (0, 5000), "room_upgrade_rate": (0, 1),
    "loyalty_tenure_years": (0, 40), "direct_booking_ratio": (0, 1),
    "cancellation_rate": (0, 1), "weekend_stay_ratio": (0, 1), "business_trip_ratio": (0, 1),
}

# Deep learning hyperparameters (overridable via run_pipeline CLI)
DL = dict(
    latent_dim=8, hidden=(64, 32), noise_std=0.1, batch_size=256, lr=1e-3,
    ae_epochs=60, dec_epochs=40, dec_update_interval=5, dec_tol=0.001,
    vae_epochs=80, beta=1.0, kl_warmup=10,
)
# density clustering
HDBSCAN = dict(min_cluster_size=150, min_samples=15)

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    torch = None
    HAS_CUDA = False
DEVICE = "cuda" if HAS_CUDA else "cpu"

try:  # optional RAPIDS
    import cuml  # noqa: F401
    import cudf  # noqa: F401
    HAS_CUML = HAS_CUDA
except Exception:
    HAS_CUML = False


def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if HAS_CUDA:
            torch.cuda.manual_seed_all(seed)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("hospitality")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def log_hardware(logger: logging.Logger) -> None:
    logger.info("Device: %s", DEVICE)
    if torch is not None:
        logger.info("torch %s | CUDA available: %s", torch.__version__, HAS_CUDA)
        if HAS_CUDA:
            logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("RAPIDS cuml/cudf: %s", "enabled" if HAS_CUML
                else "not installed -> scikit-learn CPU fallback")
