"""Offline metrics: retrieval recall (incl. cold-start), ranking NDCG/MRR, business value (expected
revenue & margin per result page, position-discounted exactly like the click simulator) and latency."""
from __future__ import annotations

import numpy as np

from src.common.utils import percentiles


def _gain(x):
    return np.power(2.0, x.astype(float)) - 1.0


def ndcg_at_k(order: np.ndarray, labels: np.ndarray, k: int) -> float:
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    g = _gain(labels[order[:k]])
    dcg = float((g * disc[: len(g)]).sum())
    ig = np.sort(_gain(labels))[::-1][:k]
    idcg = float((ig * disc[: len(ig)]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def eval_ranked(data: dict, world, ranked: list[np.ndarray], k: int = 10) -> dict:
    """ranked[i] = positions (within list i) in final order. Missing items are simply not shown."""
    off = data["offsets"]
    res = {f"ndcg_rel@{k}": [], f"ndcg_eng@{k}": [], "mrr": [], f"precision@{k}": [],
           f"expected_revenue@{k}": [], f"expected_margin@{k}": [], f"oos_rate@{k}": [], f"avg_return_prob@{k}": []}
    for i, order in enumerate(ranked):
        s, e = off[i], off[i + 1]
        rel, eng = data["rel"][s:e], data["y_eng"][s:e]
        order = np.asarray(order, dtype=int)
        res[f"ndcg_rel@{k}"].append(ndcg_at_k(order, rel, k))
        res[f"ndcg_eng@{k}"].append(ndcg_at_k(order, eng, k))
        top = order[:k]
        hit = np.where(rel[order] >= 3)[0]
        res["mrr"].append(1.0 / (hit[0] + 1) if len(hit) else 0.0)
        res[f"precision@{k}"].append(float((rel[top] >= 3).mean()) if len(top) else 0.0)
        exam = 1.0 / (1.0 + 0.35 * np.arange(len(top)))
        pbuy = (data["p_click"][s:e] * data["p_atc"][s:e] * data["p_cvr"][s:e])[top]
        sku = data["idx"][s:e][top]
        res[f"expected_revenue@{k}"].append(float((exam * pbuy * world.price[sku]).sum()))
        res[f"expected_margin@{k}"].append(float((exam * pbuy * world.margin_usd[sku] * (1 - data["p_ret"][s:e][top])).sum()))
        u = data["uidx"][i]
        store = world.user_store[u]
        oos = (world.stock["WH"][sku] <= 0) & (world.stock[store][sku] <= 0)
        res[f"oos_rate@{k}"].append(float(oos.mean()) if len(top) else 0.0)
        res[f"avg_return_prob@{k}"].append(float(data["p_ret"][s:e][top].mean()) if len(top) else 0.0)
    return {m: float(np.mean(v)) for m, v in res.items()}


def eval_scores(data: dict, world, scores: np.ndarray, k: int = 10) -> dict:
    off = data["offsets"]
    ranked = [np.argsort(-scores[off[i]:off[i + 1]], kind="stable") for i in range(len(off) - 1)]
    return eval_ranked(data, world, ranked, k)


def quick_ndcg(data: dict, scores: np.ndarray, k: int = 10, key: str = "rel") -> float:
    off = data["offsets"]
    return float(np.mean([ndcg_at_k(np.argsort(-scores[off[i]:off[i + 1]], kind="stable"),
                                     data[key][off[i]:off[i + 1]], k) for i in range(len(off) - 1)]))


# ---------------------------------------------------------------------------------------------------
def retrieval_recall(engine, queries, mode: str, alpha: float | None = None, n: int = 200, use_typo: bool = True) -> dict:
    """Recall@N against graded-relevant (rel==3, purchasable online) SKUs; normalised by min(|rel|, N).
    Cold-start recall is computed on the cold-start subset of the relevant set."""
    w = engine.world
    rec, cold_rec, hit, lat = [], [], [], []
    import time

    for q in queries.itertuples():
        raw = q.typo_text if use_typo else q.text
        t0 = time.perf_counter()
        ctx = engine.understand(raw)
        cand = engine.retrieve(ctx, mode=mode, alpha=alpha, n=n)
        lat.append((time.perf_counter() - t0) * 1e3)
        rel = (w.rel_true(q.query_id) >= 3) & (w.stock["WH"] > 0)
        rel_ids = np.where(rel)[0]
        if len(rel_ids) == 0:
            continue
        got = np.isin(rel_ids, cand.idx)
        rec.append(got.sum() / min(len(rel_ids), n))
        hit.append(float(got.any()))
        cold_ids = rel_ids[w.cold[rel_ids]]
        if len(cold_ids):
            cold_rec.append(np.isin(cold_ids, cand.idx).sum() / min(len(cold_ids), n))
    return {f"recall@{n}": float(np.mean(rec)), "hit_rate": float(np.mean(hit)),
            "cold_start_recall": float(np.mean(cold_rec)) if cold_rec else float("nan"),
            "latency_ms": percentiles(lat)}
