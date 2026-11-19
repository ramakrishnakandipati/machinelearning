"""Shared paths, constants and hyper-parameters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
LOGS, METRICS, CHARTS, MODELS, SHOTS = (OUT / d for d in ("logs", "metrics", "charts", "models", "screenshots"))
DATA = OUT / "data"
for _d in (LOGS, METRICS, CHARTS, MODELS, SHOTS, DATA):
    _d.mkdir(parents=True, exist_ok=True)

EMB_DIM = 128
HIST_LEN = 20
SEED = 42


@dataclass
class DataCfg:
    n_users: int = 20000
    n_items: int = 5000
    n_days: int = 180
    n_interactions: int = 1_200_000
    train_end: int = 126
    val_end: int = 153
    cold_user_frac: float = 0.05
    cold_item_frac: float = 0.02
    latent_dim: int = 16


@dataclass
class TrainCfg:
    epochs: int = 30
    batch_size: int = 4096
    lr: float = 2e-3
    weight_decay: float = 1e-5
    patience: int = 4
    init_temperature: float = 0.07
    hard_neg: bool = True
    logq: bool = True
    dropout: float = 0.1
