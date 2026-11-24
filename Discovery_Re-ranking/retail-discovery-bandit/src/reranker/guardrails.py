"""Hard business guardrails with per-rule drop accounting.

1. impression fatigue : exclude items with > N unclicked impressions in the last 7 days
2. price band         : *explored* items (cold-start or unseen subcategory/brand) must be within +-30% of the
                        user's average item price
3. subcategory cap    : at most 2 items per subcategory in a slate (enforced inside MMR, validated here)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class Guardrails:
    def __init__(self, cfg: Dict):
        g = cfg["guardrails"]
        self.max_per_sub = g["max_per_subcategory"]
        self.max_unclicked = g["max_unclicked_impressions_7d"]
        self.window = g["impression_window_days"]
        self.band = g["price_band"]
        self.min_candidates = g["min_candidates"]
        self.totals: Dict[str, int] = {"fatigue": 0, "price_band": 0, "fallback_relaxed_price": 0,
                                       "fallback_relaxed_fatigue": 0, "subcat_cap_violations": 0, "requests": 0}

    def filter(self, state, u: int, items: np.ndarray, tier: np.ndarray, is_cold: np.ndarray, price: np.ndarray,
               user_avg_price: float, day: int, k: int) -> Tuple[np.ndarray, Dict[str, int]]:
        n = len(items)
        unclicked = state.unclicked_counts(u, items, day, self.window)
        fatigue = unclicked > self.max_unclicked
        explored = is_cold | (tier >= 2)
        outside = (price < (1 - self.band) * user_avg_price) | (price > (1 + self.band) * user_avg_price)
        band_drop = explored & outside
        keep = ~fatigue & ~band_drop
        info = {"n_in": n, "dropped_fatigue": int(fatigue.sum()), "dropped_price_band": int((band_drop & ~fatigue).sum()),
                "relaxed_price_band": 0, "relaxed_fatigue": 0}
        need = max(k, self.min_candidates)
        if keep.sum() < need:                       # fallback 1: relax price band
            keep = ~fatigue
            info["relaxed_price_band"] = 1
            info["dropped_price_band"] = 0
        if keep.sum() < need:                       # fallback 2: relax fatigue
            keep = np.ones(n, dtype=bool)
            info["relaxed_fatigue"] = 1
            info["dropped_fatigue"] = 0
        info["n_out"] = int(keep.sum())
        self.totals["fatigue"] += info["dropped_fatigue"]
        self.totals["price_band"] += info["dropped_price_band"]
        self.totals["fallback_relaxed_price"] += info["relaxed_price_band"]
        self.totals["fallback_relaxed_fatigue"] += info["relaxed_fatigue"]
        self.totals["requests"] += 1
        return keep, info

    def validate_slate(self, subcats: List[int]) -> int:
        """Return number of subcategory-cap violations in a slate."""
        _, counts = np.unique(subcats, return_counts=True)
        v = int(np.maximum(counts - self.max_per_sub, 0).sum())
        self.totals["subcat_cap_violations"] += v
        return v
