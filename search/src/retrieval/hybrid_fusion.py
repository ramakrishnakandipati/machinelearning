"""Hybrid fusion: Reciprocal Rank Fusion and alpha-weighted score normalisation."""
from __future__ import annotations

import numpy as np


def rrf(lists: list[np.ndarray], k: int = 60, weights: list[float] | None = None):
    """lists: ranked arrays of doc ids (best first). Returns (ids, scores) sorted desc."""
    weights = weights or [1.0] * len(lists)
    score: dict[int, float] = {}
    for w, ids in zip(weights, lists):
        for r, d in enumerate(ids):
            score[int(d)] = score.get(int(d), 0.0) + w / (k + r + 1)
    ids = np.fromiter(score.keys(), dtype=np.int64, count=len(score))
    sc = np.fromiter(score.values(), dtype=np.float64, count=len(score))
    o = np.argsort(-sc)
    return ids[o], sc[o]


def _norm(s: np.ndarray) -> np.ndarray:
    if len(s) == 0:
        return s
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi - lo > 1e-12 else np.ones_like(s)


def alpha_fuse(dense_ids, dense_scores, sparse_ids, sparse_scores, alpha: float = 0.6):
    """alpha * S_dense + (1-alpha) * S_sparse with per-list min-max normalisation."""
    dn, sn = _norm(np.asarray(dense_scores, float)), _norm(np.asarray(sparse_scores, float))
    score: dict[int, float] = {}
    for d, s in zip(dense_ids, dn):
        score[int(d)] = alpha * s
    for d, s in zip(sparse_ids, sn):
        score[int(d)] = score.get(int(d), 0.0) + (1 - alpha) * s
    ids = np.fromiter(score.keys(), dtype=np.int64, count=len(score))
    sc = np.fromiter(score.values(), dtype=np.float64, count=len(score))
    o = np.argsort(-sc)
    return ids[o], sc[o]


def fuse(mode: str, dense_ids, dense_scores, sparse_ids, sparse_scores, alpha=0.6, rrf_k=60, n=200):
    if mode == "rrf":
        ids, sc = rrf([dense_ids, sparse_ids], k=rrf_k)
    elif mode == "dense":
        ids, sc = np.asarray(dense_ids), np.asarray(dense_scores)
    elif mode == "sparse":
        ids, sc = np.asarray(sparse_ids), np.asarray(sparse_scores)
    else:
        ids, sc = alpha_fuse(dense_ids, dense_scores, sparse_ids, sparse_scores, alpha)
    return ids[:n], sc[:n]
