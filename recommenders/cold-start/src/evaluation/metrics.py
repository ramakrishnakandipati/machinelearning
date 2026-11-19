"""Ranking metrics and the cold-start evaluation harness."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.data.preprocessor import EvalCase


def recall_at_k(recs: Sequence[int], targets: set, k: int) -> float:
    """Fraction of relevant items retrieved in the top-k."""
    return len(set(recs[:k]) & targets) / max(len(targets), 1)


def ndcg_at_k(recs: Sequence[int], targets: set, k: int) -> float:
    """Binary-relevance NDCG@k."""
    dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(recs[:k]) if i in targets)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(targets), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recs: Sequence[int], targets: set, k: int) -> float:
    return float(len(set(recs[:k]) & targets) > 0)


def catalog_coverage(all_recs: List[Sequence[int]], n_catalog: int) -> float:
    """Share of the (candidate) catalog that appears in at least one recommendation list."""
    return len({i for r in all_recs for i in r}) / max(n_catalog, 1)


def novelty(all_recs: List[Sequence[int]], pop: np.ndarray) -> float:
    """Mean self-information -log2(p(item)) of recommended items (higher = less mainstream)."""
    p = (pop + 1.0) / (pop.sum() + len(pop))
    vals = [-np.log2(p[i]) for r in all_recs for i in r]
    return float(np.mean(vals)) if vals else 0.0


def serendipity(all_recs: List[Sequence[int]], all_targets: List[set], pop_top: set, k: int) -> float:
    """Share of recommended slots that are relevant AND not in the globally-popular top set."""
    hits = [sum(1 for i in r[:k] if i in t and i not in pop_top) / k for r, t in zip(all_recs, all_targets)]
    return float(np.mean(hits)) if hits else 0.0


def top_k_from_scores(scores: np.ndarray, cands: np.ndarray | None, exclude: set, k: int) -> List[int]:
    """Top-k positions of ``scores`` restricted to candidates and excluding seen items."""
    s = scores.astype(np.float64).copy()
    if cands is not None:
        m = np.full(len(s), -np.inf)
        c = cands[cands < len(s)]
        m[c] = 0.0
        s = s + m
    if exclude:
        s[[e for e in exclude if e < len(s)]] = -np.inf
    k = min(k, len(s))
    idx = np.argpartition(-s, k - 1)[:k]
    idx = idx[np.argsort(-s[idx])]
    return [int(i) for i in idx if np.isfinite(s[i])]


def evaluate_strategy(name: str, recommend: Callable[[EvalCase], List[int]], cases: Dict[str, List[EvalCase]],
                      k: int, pop_counts: np.ndarray, cold_item_mask: np.ndarray, n_candidates: Dict[str, int],
                      pop_top: set) -> List[Dict[str, Any]]:
    """Run one strategy over every tier; returns one metrics row per tier."""
    rows = []
    for tier, cs in cases.items():
        recs, tgts, r, n, hr = [], [], [], [], []
        for c in cs:
            rl = recommend(c)
            t = set(c.targets)
            recs.append(rl); tgts.append(t)
            r.append(recall_at_k(rl, t, k)); n.append(ndcg_at_k(rl, t, k)); hr.append(hit_rate_at_k(rl, t, k))
        rows.append(dict(
            strategy=name, tier=tier, n_cases=len(cs), **{f"recall@{k}": float(np.mean(r)) if r else 0.0,
                                                         f"ndcg@{k}": float(np.mean(n)) if n else 0.0,
                                                         f"hit_rate@{k}": float(np.mean(hr)) if hr else 0.0},
            coverage=catalog_coverage(recs, n_candidates[tier]),
            cold_item_coverage=catalog_coverage([[i for i in rl if cold_item_mask[i]] for rl in recs],
                                                int(cold_item_mask.sum())),
            novelty=novelty(recs, pop_counts), serendipity=serendipity(recs, tgts, pop_top, k)))
    return rows


def add_cold_start_lift(df: pd.DataFrame, metric: str = "recall@10", baseline: str = "Popularity") -> pd.DataFrame:
    """Add ``lift_vs_popularity`` = (metric / popularity metric) - 1 for each tier."""
    base = df[df.strategy == baseline].set_index("tier")[metric]
    df = df.copy()
    df["lift_vs_popularity"] = [(m / base[t] - 1.0) if base.get(t, 0) > 0 else np.nan
                                for t, m in zip(df.tier, df[metric])]
    return df
