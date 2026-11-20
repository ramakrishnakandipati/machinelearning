"""Approach 1: Semantic-ID trie / prefix-tree matching.

Items are stored in a prefix tree keyed by their hierarchical Semantic ID (c1, c2, c3, c4). A user is represented by
the Semantic IDs of the seed items (history / click / intent prototypes). Candidates are retrieved from the deepest
shared prefix first and fall back to coarser prefixes (c1,c2,c3) -> (c1,c2) -> (c1).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import Catalog, ColdStartRecommender, UserContext


class SemanticTrie:
    """Prefix tree: node (tuple of codes) -> item ids below it, plus children lookup."""

    def __init__(self, codes: np.ndarray, item_ids: Optional[Sequence[int]] = None):
        self.M = codes.shape[1]
        ids = np.arange(len(codes)) if item_ids is None else np.asarray(item_ids, dtype=np.int64)
        self.items: Dict[tuple, List[int]] = {}
        self.children: Dict[tuple, set] = {(): set()}
        for i in ids:
            path = tuple(int(c) for c in codes[i])
            for d in range(1, self.M + 1):
                node = path[:d]
                self.items.setdefault(node, []).append(int(i))
                self.children.setdefault(node[:-1], set()).add(node[-1])
                self.children.setdefault(node, set())

    def items_under(self, prefix: tuple) -> List[int]:
        return self.items.get(tuple(prefix), [])

    def child_codes(self, prefix: tuple) -> List[int]:
        return sorted(self.children.get(tuple(prefix), ()))

    def longest_prefix(self, path: tuple) -> int:
        """Depth of the longest prefix of `path` present in the trie."""
        d = 0
        for d in range(1, len(path) + 1):
            if tuple(path[:d]) not in self.items:
                return d - 1
        return len(path)


class SemanticTrieRecommender(ColdStartRecommender):
    name = "semantic_trie"

    def __init__(self, catalog: Catalog, tiebreak_weight: float = 5e-4):
        super().__init__(catalog)
        self.trie = SemanticTrie(catalog.codes)
        self.depth_w = np.array([2.0 ** d for d in range(catalog.M)], dtype=np.float64)   # deeper match dominates
        self.depth_w /= self.depth_w.sum()
        npop = np.log1p(catalog.node_pop)
        self.node_pop_n = npop / (npop.max() + 1e-9)
        self.tiebreak_weight = tiebreak_weight

    def profile_scores(self, seeds: List[int], weights: List[float]) -> np.ndarray:
        """score_i = sum_d w_d * (weighted fraction of seeds sharing item i's depth-d prefix)."""
        c = self.cat
        score = np.zeros(c.N)
        if not seeds:
            return score
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        for d in range(c.M):
            pid = c.prefix_ids[d]
            cnt = np.bincount(pid[np.asarray(seeds)], weights=w, minlength=pid.max() + 1)
            score += self.depth_w[d] * cnt[pid]
        return score

    def score_all(self, ctx: UserContext) -> np.ndarray:
        seeds, weights = self.cat.seed_items(ctx)
        return self.profile_scores(seeds, weights) + self.tiebreak_weight * self.node_pop_n

    # ---- explicit prefix-fallback retrieval (used by the API / explanations)
    def retrieve(self, ctx: UserContext, k: int = 10, candidates: Optional[Sequence[int]] = None,
                 exclude: Optional[set] = None) -> List[Tuple[int, int]]:
        """Return (item, matched_prefix_depth) pairs gathered from the deepest prefix down to the coarsest."""
        exclude = exclude if exclude is not None else set(ctx.history)
        allowed = None if candidates is None else set(int(x) for x in candidates)
        seeds, weights = self.cat.seed_items(ctx)
        out, seen = [], set()
        order = np.argsort(-np.asarray(weights)) if seeds else []
        for depth in range(self.cat.M, 0, -1):
            for si in order:
                path = tuple(int(c) for c in self.cat.codes[seeds[si]][:depth])
                for it in self.trie.items_under(path):
                    if it in seen or it in exclude or (allowed is not None and it not in allowed):
                        continue
                    seen.add(it)
                    out.append((it, depth))
                    if len(out) >= k:
                        return out
        return out

    def match_depth(self, item: int, ctx: UserContext) -> Tuple[Optional[int], int]:
        """(best matching seed item, shared prefix depth) for explanation strings."""
        seeds, _ = self.cat.seed_items(ctx)
        best, best_d = None, 0
        for s in seeds:
            d = 0
            while d < self.cat.M and self.cat.codes[s, d] == self.cat.codes[item, d]:
                d += 1
            if d > best_d:
                best, best_d = s, d
        return best, best_d
