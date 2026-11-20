"""Reference baselines for the benchmark table."""
from __future__ import annotations

import numpy as np

from .base import Catalog, ColdStartRecommender, UserContext


class RandomRecommender(ColdStartRecommender):
    name = "random"

    def __init__(self, catalog: Catalog, seed: int = 0):
        super().__init__(catalog)
        self.rng = np.random.default_rng(seed)

    def score_all(self, ctx: UserContext) -> np.ndarray:
        return self.rng.random(self.cat.N)


class PopularityRecommender(ColdStartRecommender):
    """Interaction-log popularity (not available on Day-0; zero for brand-new items)."""
    name = "popularity"

    def __init__(self, catalog: Catalog, seed: int = 0):
        super().__init__(catalog)
        self.rng = np.random.default_rng(seed)

    def score_all(self, ctx: UserContext) -> np.ndarray:
        return self.cat.pop + 1e-3 * self.rng.random(self.cat.N)


class RawContentKNNRecommender(ColdStartRecommender):
    """Cosine similarity on the raw (un-quantized) multimodal features -- shows what quantization adds/costs."""
    name = "raw_content_knn"

    def __init__(self, catalog: Catalog, item_feats: np.ndarray):
        super().__init__(catalog)
        self.X = item_feats / (np.linalg.norm(item_feats, axis=1, keepdims=True) + 1e-9)

    def score_all(self, ctx: UserContext) -> np.ndarray:
        seeds, w = self.cat.seed_items(ctx)
        if not seeds:
            return np.zeros(self.cat.N)
        v = (self.X[seeds] * np.asarray(w)[:, None]).sum(0)
        return self.X @ (v / (np.linalg.norm(v) + 1e-9))
