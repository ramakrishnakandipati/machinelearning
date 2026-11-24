"""Ranking, discovery, diversity, calibration, exposure-fairness and latency metrics."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


# ----------------------------------------------------------------------------------------
# Stage 3 ranker metrics
# ----------------------------------------------------------------------------------------
def ndcg_at_k(users: np.ndarray, scores: np.ndarray, rel: np.ndarray, k: int) -> float:
    """Mean NDCG@k over users that have at least one positive in their list."""
    df = pd.DataFrame({"u": users, "s": scores, "r": rel})
    vals = []
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for _, g in df.groupby("u", sort=False):
        if g["r"].max() <= 0:
            continue
        r = g["r"].values
        order = np.argsort(-g["s"].values, kind="stable")[:k]
        dcg = float(((2.0 ** r[order] - 1.0) * disc[: len(order)]).sum())
        ideal = np.sort(r)[::-1][:k]
        idcg = float(((2.0 ** ideal - 1.0) * disc[: len(ideal)]).sum())
        vals.append(dcg / idcg)
    return float(np.mean(vals)) if vals else float("nan")


def binary_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    out = {"logloss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)),
           "base_rate": float(np.mean(y)), "mean_pred": float(np.mean(p)), "n": int(len(y))}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return out


def ranker_metrics(users, y_click, y_buy, p_click, p_buy) -> Dict[str, Dict[str, float]]:
    rel = np.asarray(y_click, dtype=float) + np.asarray(y_buy, dtype=float)  # graded: 0 / 1 click / 2 buy
    util = np.asarray(p_click) + np.asarray(p_buy)
    res = {"click": binary_metrics(y_click, p_click), "buy": binary_metrics(y_buy, p_buy)}
    res["ranking"] = {"ndcg@5": ndcg_at_k(users, util, rel, 5), "ndcg@10": ndcg_at_k(users, util, rel, 10)}
    return res


def calibration_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    qs = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, qs[1:-1]), 0, len(qs) - 2)
    rows = []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b, "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean()), "n": int(m.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Diversity / exposure helpers
# ----------------------------------------------------------------------------------------
def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def shannon_entropy(counts: np.ndarray, base: float = 2.0) -> float:
    p = np.asarray(counts, dtype=np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(base))


def intra_list_diversity(emb: np.ndarray) -> float:
    """Average pairwise cosine distance of a slate (emb: (k, d), rows L2-normalised)."""
    k = emb.shape[0]
    if k < 2:
        return 0.0
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sim = e @ e.T
    iu = np.triu_indices(k, 1)
    return float((1.0 - sim[iu]).mean())


# ----------------------------------------------------------------------------------------
# Stage 4 slate metrics (impression-level records)
# ----------------------------------------------------------------------------------------
def slate_metrics(rec: pd.DataFrame, n_items: int, in_stock_items: int) -> Dict[str, float]:
    """rec: one row per displayed (request, slot). Required columns documented in pipeline.record."""
    if len(rec) == 0:
        return {}
    n_imp = len(rec)
    n_req = rec["request"].nunique()
    buys = rec["buy"].sum()
    exposure = np.bincount(rec["item"].values, minlength=n_items)
    disc = rec["tier"].values >= 2                      # unseen subcategory or unseen brand
    cold = rec["is_cold"].values.astype(bool)
    serendip = (disc & ~rec["ranker_topk"].values.astype(bool))
    slates = rec.groupby("request")
    out = {
        "n_requests": int(n_req), "n_impressions": int(n_imp),
        "ctr": float(rec["click"].mean()),
        "add_to_cart_rate": float(rec["buy"].mean()),   # purchase events stand in for add-to-cart in the simulator
        "buy_rate_per_impression": float(rec["buy"].mean()),
        "avg_reward_per_impression": float(rec["reward"].mean()),
        "avg_margin_per_slate": float(slates["margin"].mean().mean()),
        "realized_profit_per_slate": float((rec["buy"] * rec["price"] * rec["margin"]).groupby(rec["request"]).sum().mean()),
        "repeat_purchase_rate_7d": float(rec["repeat"].sum() / buys) if buys > 0 else 0.0,
        "discovery_impression_share": float(disc.mean()),
        "discovery_conversion_rate": float(rec["buy"].values[disc].mean()) if disc.any() else 0.0,
        "discovery_click_rate": float(rec["click"].values[disc].mean()) if disc.any() else 0.0,
        "discovery_purchase_share": float(rec["buy"].values[disc].sum() / buys) if buys > 0 else 0.0,
        "unseen_subcat_conversion_rate": float(rec["buy"].values[rec["tier"].values == 3].mean()) if (rec["tier"].values == 3).any() else 0.0,
        "cold_start_impression_share": float(cold.mean()),
        "cold_start_conversion_rate": float(rec["buy"].values[cold].mean()) if cold.any() else 0.0,
        "cold_start_click_rate": float(rec["click"].values[cold].mean()) if cold.any() else 0.0,
        "warm_conversion_rate": float(rec["buy"].values[~cold].mean()) if (~cold).any() else 0.0,
        "catalog_coverage_pct": float(100.0 * (exposure > 0).sum() / n_items),
        "instock_coverage_pct": float(100.0 * (exposure > 0).sum() / max(in_stock_items, 1)),
        "gini_exposure": gini(exposure),
        "shannon_entropy_bits": shannon_entropy(exposure),
        "shannon_entropy_normalised": float(shannon_entropy(exposure) / np.log2(n_items)),
        "ild": float(slates["ild"].first().mean()),
        "serendipity_score": float((serendip & (rec["buy"].values == 1)).sum() / n_imp),
        "serendipity_click_score": float((serendip & (rec["click"].values == 1)).sum() / n_imp),
        "avg_p_buy_true": float(rec["p_buy_true"].mean()) if "p_buy_true" in rec else float("nan"),
    }
    return out


def slice_metrics(rec: pd.DataFrame, by: str, n_items: int, in_stock_items: int) -> pd.DataFrame:
    rows = []
    for key, g in rec.groupby(by):
        m = slate_metrics(g, n_items, in_stock_items)
        m[by] = key
        rows.append(m)
    return pd.DataFrame(rows)


def exposure_by_group(rec: pd.DataFrame, group_col: str, catalog_items: pd.DataFrame, catalog_col: str) -> pd.DataFrame:
    """Exposure share vs catalog share (supply-side fairness)."""
    exp = rec.groupby(group_col).size() / len(rec)
    cat = catalog_items.groupby(catalog_col).size() / len(catalog_items)
    out = pd.DataFrame({"exposure_share": exp, "catalog_share": cat}).fillna(0.0)
    out["exposure_ratio"] = out["exposure_share"] / out["catalog_share"].replace(0, np.nan)
    return out.reset_index().rename(columns={"index": group_col})


def latency_report(timer_summary: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = [{"stage": k, **v} for k, v in timer_summary.items()]
    return pd.DataFrame(rows)
