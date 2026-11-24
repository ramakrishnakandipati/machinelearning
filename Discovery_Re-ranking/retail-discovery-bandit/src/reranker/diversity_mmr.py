"""Maximal Marginal Relevance sequential slate assembly on GPU.

    next = argmax_{i in C \\ S} [ lambda * Score(u,i) - (1 - lambda) * max_{j in S} Sim(i,j) ]

Sim blends item-embedding cosine similarity with a same-subcategory penalty. The subcategory cap
(hard guardrail) is enforced inside the greedy loop.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def similarity_matrix(emb: torch.Tensor, subcat: torch.Tensor, w_embed: float, w_sub: float) -> torch.Tensor:
    """emb: (n,d) L2-normalised. Returns (n,n) similarity in [0, w_embed + w_sub]."""
    cos = (emb @ emb.T).clamp(min=0.0)
    same = (subcat[:, None] == subcat[None, :]).float()
    return w_embed * cos + w_sub * same


def minmax(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


@torch.no_grad()
def mmr_select(scores: torch.Tensor, emb: torch.Tensor, subcat: torch.Tensor, k: int, lam: float,
               w_embed: float = 0.6, w_sub: float = 0.4, max_per_subcat: int = 2
               ) -> Tuple[List[int], List[float], List[float]]:
    """Greedy MMR. Returns (selected indices, mmr penalties (1-lam)*max-sim at pick time, mmr objective values)."""
    n = scores.shape[0]
    k = min(k, n)
    s = minmax(scores.float())
    sim = similarity_matrix(emb, subcat, w_embed, w_sub)
    max_sim = torch.zeros(n, device=scores.device)
    selected = torch.zeros(n, dtype=torch.bool, device=scores.device)
    n_sub = int(subcat.max().item()) + 1
    sub_count = torch.zeros(n_sub, device=scores.device)
    picks, pens, vals = [], [], []
    for _ in range(k):
        val = lam * s - (1.0 - lam) * max_sim
        val = torch.where(sub_count[subcat] >= max_per_subcat, val - 1e6, val)   # cap (relaxed only if unavoidable)
        val = val.masked_fill(selected, -1e9)
        j = torch.argmax(val)
        picks.append(j)
        pens.append((1.0 - lam) * max_sim[j])
        vals.append(val[j])
        selected[j] = True
        sub_count[subcat[j]] += 1
        max_sim = torch.maximum(max_sim, sim[j])
    picks_t = torch.stack(picks).cpu().tolist()
    return picks_t, torch.stack(pens).cpu().tolist(), torch.stack(vals).cpu().tolist()
