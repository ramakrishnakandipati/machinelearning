"""Margin-aware re-ranking: blend normalised relevance with expected profit, never promoting items
below a relevance floor."""
from __future__ import annotations

import numpy as np

from src.common.utils import minmax


def rank_norm(scores: np.ndarray) -> np.ndarray:
    """Rank-based percentile in [0,1] (robust to differing score scales across LTR models)."""
    if len(scores) <= 1:
        return np.ones(len(scores))
    r = np.empty(len(scores))
    r[np.argsort(scores)] = np.arange(len(scores))
    return r / (len(scores) - 1)


def margin_rerank(rel_scores: np.ndarray, expected_profit: np.ndarray, margin_weight: float, floor: float,
                  penalties: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    r = rank_norm(rel_scores)
    p = minmax(np.sqrt(np.maximum(expected_profit, 0)))
    eligible = r >= floor
    final = (1 - margin_weight) * r + margin_weight * p * eligible
    if penalties is not None:
        final = final * penalties
    return final, {"relevance_norm": r, "profit_norm": p, "eligible": eligible}
