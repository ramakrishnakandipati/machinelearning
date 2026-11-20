"""Central configuration: seeds, paths, device, hyper-parameters."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODELS_DIR = OUT / "models"
LOGS_DIR = OUT / "logs"
VIZ_DIR = OUT / "visualizations"
SHOT_DIR = OUT / "api_screenshots"

for _d in (MODELS_DIR, LOGS_DIR, VIZ_DIR, SHOT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"  # torch.cuda.amp with automatic CPU fallback


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DataConfig:
    n_users: int = 5000
    n_items: int = 2000
    n_interactions: int = 50000
    item_cold_frac: float = 0.10
    user_cold_frac: float = 0.10
    text_dim: int = 32
    img_dim: int = 32
    latent_dim: int = 16


@dataclass
class QuantConfig:
    latent_dim: int = 64          # D
    hidden_dim: int = 256
    n_res_blocks: int = 2
    codebook_size: int = 256      # K
    n_levels: int = 4             # M (RQ-VAE)
    beta: float = 0.25            # commitment weight
    ema_decay: float = 0.97
    dead_code_threshold: float = 0.3
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    patience: int = 30
    dropout: float = 0.05


@dataclass
class GenConfig:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    max_hist: int = 8
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 25
    beam_width: int = 64
    dropout: float = 0.1
    ctx_dropout: float = 0.2


@dataclass
class BridgeConfig:
    emb_dim: int = 32
    hidden: int = 256
    batch_size: int = 512
    lr: float = 2e-3
    epochs: int = 40
    temperature: float = 0.07
    hist_dropout: float = 0.5
    click_dropout: float = 0.3
    aux_weight: float = 0.3


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    eval_k: tuple = (10, 20)
    eval_max_users: int = 800

    def quick(self) -> "Config":
        """Small settings for smoke tests."""
        self.quant.epochs = 12
        self.quant.patience = 6
        self.gen.epochs = 2
        self.bridge.epochs = 3
        self.eval_max_users = 150
        return self


def get_config(quick: bool = False) -> Config:
    cfg = Config()
    if quick or os.environ.get("PIPELINE_QUICK") == "1":
        cfg.quick()
    return cfg
