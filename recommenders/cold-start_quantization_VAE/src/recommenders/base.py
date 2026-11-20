"""Shared building blocks: user context, tokenized item catalog and recommender interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import DEVICE

META_FIELDS = ["channel", "device", "region", "age_band", "intent_cat", "intent_sub"]


@dataclass
class UserContext:
    """Everything known about the requester. Any field may be missing (cold-start)."""
    history: List[int] = field(default_factory=list)       # interacted item ids (oldest -> newest)
    click_item: Optional[int] = None                         # first item clicked in the session
    channel: Optional[str] = None
    device: Optional[str] = None
    region: Optional[str] = None
    age_band: Optional[str] = None
    intent_cat: Optional[str] = None                         # landing / intent category name
    intent_sub: Optional[str] = None                         # intent sub-category name
    regime: Optional[str] = None                             # 'warm' | 'user_cold' | 'system_cold' (auto if None)

    def has_meta(self) -> bool:
        return any(getattr(self, f) for f in ("channel", "device", "region", "age_band"))

    def auto_regime(self) -> str:
        if self.regime:
            return self.regime
        if len(self.history) >= 3:
            return "warm"
        if self.history or self.click_item is not None or self.has_meta():
            return "user_cold"
        return "system_cold"


class Catalog:
    """Items + their discrete (Semantic ID) and dense (z_q) representations from one tokenizer."""

    def __init__(self, name: str, dataset: dict, model, codes: np.ndarray, z_q: np.ndarray, z_e: np.ndarray,
                 pop: np.ndarray, num_codes: int):
        self.name, self.model, self.dataset = name, model, dataset
        self.items: pd.DataFrame = dataset["items"]
        self.N = len(self.items)
        self.codes = codes.astype(np.int64)
        self.M = codes.shape[1]
        self.K = num_codes
        self.zq = z_q.astype(np.float32)
        self.ze = z_e.astype(np.float32)
        self.zq_n = self.zq / (np.linalg.norm(self.zq, axis=1, keepdims=True) + 1e-9)
        self.pop = pop.astype(np.float32)
        self.cat_idx = self.items["cat_idx"].values
        self.sub_idx = self.items["sub_idx"].values
        self.margin = self.items["margin"].values.astype(np.float32)
        self.is_cold = self.items["is_cold"].values
        self.cold_ids = np.where(self.is_cold)[0]
        self.warm_ids = np.where(~self.is_cold)[0]
        # prefix ids per depth (hash of the first d codes) used by the trie / overlap logic
        self.prefix_ids: List[np.ndarray] = []
        for d in range(1, self.M + 1):
            _, inv = np.unique(self.codes[:, :d], axis=0, return_inverse=True)
            self.prefix_ids.append(inv.reshape(-1))
        # disambiguation index for items colliding on the full ID
        self.dedup = np.zeros(self.N, dtype=np.int64)
        seen: Dict[tuple, int] = {}
        for i in range(self.N):
            key = tuple(self.codes[i])
            self.dedup[i] = seen.get(key, 0)
            seen[key] = seen.get(key, 0) + 1
        self.n_dup = {k: v for k, v in seen.items() if v > 1}
        # node popularity at depth min(2, M): used as cold-friendly tie-break
        d2 = self.prefix_ids[min(2, self.M) - 1]
        node_pop = np.bincount(d2, weights=self.pop, minlength=d2.max() + 1)
        self.node_pop = node_pop[d2]
        self._proto_cache: Dict[tuple, np.ndarray] = {}

    # ---- identifiers
    def semantic_id(self, i: int) -> List[int]:
        return [int(c) for c in self.codes[i]]

    def semantic_id_str(self, i: int) -> str:
        s = "-".join(str(int(c)) for c in self.codes[i])
        return s if tuple(self.codes[i]) not in self.n_dup else f"{s}#{int(self.dedup[i])}"

    def id_map(self) -> dict:
        return {int(i): {"codes": self.semantic_id(i), "dedup_index": int(self.dedup[i]), "id": self.semantic_id_str(i),
                         "is_cold": bool(self.is_cold[i])} for i in range(self.N)}

    # ---- inductive encoding for unseen items (API)
    @torch.no_grad()
    def encode_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        codes = self.model.encode_to_codes(x)
        zq = self.model.codes_to_embedding(codes)
        ze = self.model.encode(x)
        return codes.cpu().numpy(), zq.cpu().numpy(), ze.cpu().numpy()

    # ---- context handling
    def prototypes(self, cat: Optional[str] = None, sub: Optional[str] = None, n: int = 5) -> np.ndarray:
        """Most central items of a category / sub-category (content only -> usable on Day-0)."""
        key = (cat, sub, n)
        if key in self._proto_cache:
            return self._proto_cache[key]
        mask = np.ones(self.N, bool)
        if sub:
            mask &= (self.items["subcategory"].values == sub)
        elif cat:
            mask &= (self.items["category"].values == cat)
        ids = np.where(mask)[0]
        if len(ids) == 0:
            ids = np.arange(self.N)
        centroid = self.zq_n[ids].mean(0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        order = np.argsort(-(self.zq_n[ids] @ centroid))[:n]
        out = ids[order]
        self._proto_cache[key] = out
        return out

    def seed_items(self, ctx: UserContext, n_proto: int = 5) -> Tuple[List[int], List[float]]:
        """Items that represent the user's taste: history (recency weighted) + click + intent prototypes."""
        seeds, weights = [], []
        hist = list(ctx.history)[-8:]
        for age, it in enumerate(reversed(hist)):
            seeds.append(int(it)); weights.append(0.9 ** age)
        if ctx.click_item is not None:
            seeds.append(int(ctx.click_item)); weights.append(1.3)
        if not seeds:
            if ctx.intent_sub:
                protos = self.prototypes(sub=ctx.intent_sub, n=n_proto)
            elif ctx.intent_cat:
                protos = self.prototypes(cat=ctx.intent_cat, n=n_proto)
            else:
                protos = []
            for p in protos:
                seeds.append(int(p)); weights.append(1.0)
        return seeds, weights


class ColdStartRecommender:
    """Common interface. Subclasses implement `score_all` (N,) or override `recommend`."""
    name = "base"

    def __init__(self, catalog: Catalog):
        self.cat = catalog

    def score_all(self, ctx: UserContext) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def recommend(self, ctx: UserContext, k: int = 10, candidates: Optional[Sequence[int]] = None,
                  exclude: Optional[set] = None) -> List[Tuple[int, float]]:
        scores = self.score_all(ctx).astype(np.float64)
        return self._top_k(scores, k, candidates, exclude if exclude is not None else set(ctx.history))

    def _top_k(self, scores: np.ndarray, k: int, candidates, exclude) -> List[Tuple[int, float]]:
        scores = scores.copy()
        if candidates is not None:
            mask = np.full(self.cat.N, True)
            mask[np.asarray(candidates, dtype=np.int64)] = False
            scores[mask] = -np.inf
        if exclude:
            scores[np.fromiter(exclude, dtype=np.int64)] = -np.inf
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]
