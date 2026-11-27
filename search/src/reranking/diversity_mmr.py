"""Maximal Marginal Relevance with brand / subcategory caps."""
from __future__ import annotations

import numpy as np


def mmr_select(scores: np.ndarray, emb: np.ndarray, brands, subcats, k: int, lam: float = 0.75,
               max_per_brand: int = 4, max_per_sub: int = 12) -> list[int]:
    """Greedy MMR. `scores` normalised to [0,1]; returns positions in selection order."""
    n = len(scores)
    k = min(k, n)
    chosen: list[int] = []
    avail = np.ones(n, dtype=bool)
    bcount: dict = {}
    scount: dict = {}
    max_sim = np.zeros(n)
    while len(chosen) < k and avail.any():
        val = lam * scores - (1 - lam) * max_sim
        val = np.where(avail, val, -np.inf)
        j = int(np.argmax(val))
        if bcount.get(brands[j], 0) >= max_per_brand or scount.get(subcats[j], 0) >= max_per_sub:
            avail[j] = False
            continue
        chosen.append(j)
        avail[j] = False
        bcount[brands[j]] = bcount.get(brands[j], 0) + 1
        scount[subcats[j]] = scount.get(subcats[j], 0) + 1
        max_sim = np.maximum(max_sim, emb @ emb[j])
    # if caps starved the list, backfill with best remaining
    if len(chosen) < k:
        rest = [i for i in np.argsort(-scores) if i not in set(chosen)]
        chosen += rest[: k - len(chosen)]
    return chosen
