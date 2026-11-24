"""Multi-objective utility:  U(u,i) = w1 * p_click + w2 * p_buy * margin + w3 * novelty(u,i)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from src.data.state import TIER_NAMES


def novelty_from_tier(tier: np.ndarray, cfg: Dict) -> np.ndarray:
    """tier: 3 unseen subcategory -> 1.0, 2 unseen brand -> 0.5, 1 new SKU -> 0.25, 0 previously bought -> 0.0."""
    u = cfg["utility"]
    table = np.array([u["novelty_bought"], u["novelty_new_sku"], u["novelty_unseen_brand"], u["novelty_unseen_subcat"]],
                     dtype=np.float32)
    return table[tier]


def tier_names(tier: np.ndarray):
    return [TIER_NAMES[t] for t in tier]


def utility(p_click: torch.Tensor, p_buy: torch.Tensor, margin: torch.Tensor, novelty: torch.Tensor,
            w1: float, w2: float, w3: float) -> torch.Tensor:
    """All inputs are (n,) tensors on the compute device."""
    return w1 * p_click + w2 * p_buy * margin + w3 * novelty


def utility_weights(cfg: Dict, use_novelty: bool = True):
    u = cfg["utility"]
    return u["w1"], u["w2"], (u["w3"] if use_novelty else 0.0)
