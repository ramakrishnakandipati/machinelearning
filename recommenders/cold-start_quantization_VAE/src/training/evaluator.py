"""Metrics: codebook health (perplexity / utilization / collisions) and ranking metrics
(Recall@K, NDCG@K, MRR, HitRate, Coverage) plus the benchmark harness over cold-start regimes."""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- codebook health
def perplexity_from_counts(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def codebook_stats(codes: np.ndarray, num_codes: int) -> dict:
    """codes: (N, M) int array. Per-level perplexity & utilization, plus full-ID collision rate."""
    n, m = codes.shape
    level_ppl, level_util = [], []
    for l in range(m):
        counts = np.bincount(codes[:, l], minlength=num_codes)
        level_ppl.append(perplexity_from_counts(counts))
        level_util.append(float((counts > 0).mean() * 100))
    tuples, inv, cnt = np.unique(codes, axis=0, return_inverse=True, return_counts=True)
    collision_rate = float((cnt[inv.reshape(-1)] > 1).mean())          # fraction of items sharing their full ID
    return {
        "level_perplexity": level_ppl, "level_utilization_pct": level_util,
        "mean_perplexity": float(np.mean(level_ppl)), "mean_utilization_pct": float(np.mean(level_util)),
        "unique_ids": int(len(tuples)), "n_items": int(n), "collision_rate": collision_rate,
    }


# --------------------------------------------------------------------------- ranking metrics
def recall_at_k(ranked: list, truth: set, k: int) -> float:
    if not truth:
        return 0.0
    return len(set(ranked[:k]) & truth) / len(truth)


def hit_at_k(ranked: list, truth: set, k: int) -> float:
    return float(len(set(ranked[:k]) & truth) > 0)


def ndcg_at_k(ranked: list, truth: set, k: int) -> float:
    if not truth:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(ranked[:k]) if it in truth)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / idcg


def mrr(ranked: list, truth: set, k: int = 100) -> float:
    for i, it in enumerate(ranked[:k]):
        if it in truth:
            return 1.0 / (i + 1)
    return 0.0


# --------------------------------------------------------------------------- benchmark harness
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.recommenders.base import UserContext

REGIMES = ["warm", "item_cold", "user_cold", "system_cold"]


def build_eval_requests(ds: dict, max_users: int, seed: int = 0) -> Dict[str, List[dict]]:
    """Each request: {'ctx', 'truth', 'candidates', 'exclude'} for the four regimes."""
    rng = np.random.default_rng(seed)
    inter, users, items = ds["interactions"], ds["users"].set_index("user_id"), ds["items"]
    cold_ids = np.where(items["is_cold"].values)[0]
    warm_ids = np.where(~items["is_cold"].values)[0]
    by_split = {s: g.groupby("user_id")["item_id"].apply(list).to_dict() for s, g in inter.groupby("split")}
    tr = inter[inter["split"].isin(["train", "val"])].sort_values(["user_id", "ts"]).groupby("user_id")["item_id"].apply(list).to_dict()

    def base_ctx(u, hist):
        r = users.loc[u]
        return UserContext(history=list(hist), click_item=int(r.click_item), channel=r.channel, device=r.device,
                           region=r.region, age_band=r.age_band, intent_cat=r.intent_cat, intent_sub=r.intent_sub)

    def sample(keys):
        keys = list(keys)
        rng.shuffle(keys)
        return keys[:max_users]

    req = {r: [] for r in REGIMES}
    for u in sample(by_split.get("test", {}).keys()):                              # warm
        ctx = base_ctx(u, tr.get(u, []))
        req["warm"].append(dict(ctx=ctx, truth=set(by_split["test"][u]), candidates=warm_ids,
                                exclude=set(ctx.history) | {ctx.click_item}))
    for u in sample(by_split.get("item_cold_test", {}).keys()):                    # item cold
        ctx = base_ctx(u, tr.get(u, []))
        req["item_cold"].append(dict(ctx=ctx, truth=set(by_split["item_cold_test"][u]), candidates=cold_ids,
                                     exclude=set()))
    cold_users = [u for u in by_split.get("user_cold_test", {}).keys()]
    for u in sample(cold_users):                                                   # user cold
        ctx = base_ctx(u, by_split.get("user_cold_history", {}).get(u, []))
        req["user_cold"].append(dict(ctx=ctx, truth=set(by_split["user_cold_test"][u]), candidates=warm_ids,
                                     exclude=set(ctx.history) | {ctx.click_item}))
    for u in sample(cold_users):                                                   # system cold (Day-0: landing category only)
        r = users.loc[u]
        truth = set(by_split["user_cold_test"][u]) | set(by_split.get("user_cold_history", {}).get(u, []))
        req["system_cold"].append(dict(ctx=UserContext(intent_cat=r.intent_cat, regime="system_cold"), truth=truth,
                                       candidates=warm_ids, exclude=set()))
    return req


def evaluate_recommender(rec, requests: List[dict], ks=(10, 20)) -> dict:
    kmax = max(ks)
    agg = defaultdict(list)
    recommended, lat = set(), []
    cand_n = len(requests[0]["candidates"]) if requests else 1
    for rq in requests:
        t0 = time.perf_counter()
        out = rec.recommend(rq["ctx"], kmax, rq["candidates"], rq["exclude"])
        lat.append((time.perf_counter() - t0) * 1000)
        ranked = [i for i, _ in out]
        for k in ks:
            agg[f"recall@{k}"].append(recall_at_k(ranked, rq["truth"], k))
            agg[f"ndcg@{k}"].append(ndcg_at_k(ranked, rq["truth"], k))
            agg[f"hit@{k}"].append(hit_at_k(ranked, rq["truth"], k))
        agg["mrr"].append(mrr(ranked, rq["truth"], kmax))
        recommended.update(ranked[:10])
    row = {m: float(np.mean(v)) for m, v in agg.items()}
    row.update({"coverage@10": len(recommended) / cand_n, "latency_ms": float(np.mean(lat)),
                "latency_p95_ms": float(np.percentile(lat, 95)), "n_requests": len(requests)})
    return row


def run_benchmark(bundle, cfg, out_csv: Optional[Path] = None, verbose: bool = True) -> pd.DataFrame:
    reqs = build_eval_requests(bundle.dataset, cfg.eval_max_users)
    rows = []
    for name, rec in bundle.recs.items():
        for regime in REGIMES:
            if not reqs[regime]:
                continue
            r = evaluate_recommender(rec, reqs[regime], cfg.eval_k)
            rows.append({"approach": name, "regime": regime, **r})
            if verbose:
                print(f"  {name:16s} {regime:12s} R@10 {r['recall@10']:.3f} NDCG@10 {r['ndcg@10']:.3f} "
                      f"MRR {r['mrr']:.3f} cov {r['coverage@10']:.2f} {r['latency_ms']:.1f}ms")
    df = pd.DataFrame(rows)
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
    return df
