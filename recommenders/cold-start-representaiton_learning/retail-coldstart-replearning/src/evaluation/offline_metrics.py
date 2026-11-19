"""Stratified offline evaluation: Warm / User cold-start / Item cold-start / System cold-start.

Protocol: for every evaluation user in a slice, the query is built from train-visible information
(profile, train history, context of the user's first evaluation event), all 2,500 catalogue items
(warm AND cold) are ranked, items already in the user's train history are excluded, and the
ground truth is the set of items the user interacts with inside that slice/split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.common import SLICES, alignment, l2n, uniformity


@dataclass
class SliceData:
    name: str
    u: torch.Tensor
    ctx_cat: torch.Tensor
    tod: torch.Tensor
    hist: torch.Tensor
    nhist: torch.Tensor
    target: torch.Tensor      # (n,nI) bool
    seen: torch.Tensor        # (n,nI) bool

    def __len__(self):
        return len(self.u)

    def subset(self, mask: torch.Tensor) -> "SliceData":
        return SliceData(self.name, self.u[mask], self.ctx_cat[mask], self.tod[mask], self.hist[mask], self.nhist[mask],
                         self.target[mask], self.seen[mask])


def build_eval_sets(bundle, df: pd.DataFrame, max_users: int = 1500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    dev = bundle.device
    sets = {}
    for sl in SLICES:
        d = df[df["slice"] == sl]
        if d.empty:
            continue
        users = d.user_id.unique()
        if len(users) > max_users:
            users = rng.choice(users, max_users, replace=False)
        first = d[d.user_id.isin(users)].groupby("user_id").head(1).set_index("user_id")
        u_idx = np.array([bundle.user_index[x] for x in first.index])
        tgt = torch.zeros(len(first), bundle.n_items, dtype=torch.bool)
        grp = d[d.user_id.isin(users)].groupby("user_id").item_id.apply(lambda s: [bundle.item_index[i] for i in s])
        for r, uid in enumerate(first.index):
            tgt[r, grp[uid]] = True
        u_t = torch.as_tensor(u_idx, device=dev)
        hist = torch.as_tensor(bundle.hist_np[u_idx], device=dev)
        seen = torch.zeros(len(first), bundle.n_items, dtype=torch.bool, device=dev)
        hv = hist.clamp(min=1) - 1
        seen.scatter_(1, hv, hist > 0)
        ctx = bundle.enc.transform_context(first.hour.values, first.device.values, first.store_id.values, first.channel.values)
        sets[sl] = SliceData(sl, u_t, torch.as_tensor(ctx["cat"], dtype=torch.long, device=dev),
                             torch.as_tensor(ctx["tod"], device=dev), hist, (hist > 0).sum(1), tgt.to(dev), seen)
    return sets


@torch.no_grad()
def rank_metrics(scores: torch.Tensor, sd: SliceData, cold_mask: torch.Tensor, ks=(10, 20)) -> tuple:
    """scores (n,nI). Returns (per-user metric tensors, top-10 item ids)."""
    scores = scores.float().masked_fill(sd.seen & ~sd.target, -1e9)
    order = torch.argsort(scores, 1, descending=True)
    hits = sd.target.gather(1, order)
    n_t = sd.target.sum(1).clamp(min=1).float()
    out = {}
    for k in ks:
        out[f"recall@{k}"] = (hits[:, :k].sum(1) / n_t)
    disc = 1.0 / torch.log2(torch.arange(2, 12, device=scores.device).float())
    dcg = (hits[:, :10].float() * disc).sum(1)
    ideal = torch.stack([disc[:int(min(t, 10))].sum() for t in n_t.tolist()])
    out["ndcg@10"] = dcg / ideal
    first = hits.float().argmax(1)
    out["mrr"] = torch.where(hits.any(1), 1.0 / (first.float() + 1), torch.zeros_like(first, dtype=torch.float))
    top10 = order[:, :10]
    return out, top10


def aggregate(per_user: dict, top10: torch.Tensor, sd: SliceData, cold_mask: torch.Tensor, items_mat, queries) -> dict:
    nI = len(cold_mask)
    res = {k: float(v.mean()) for k, v in per_user.items()}
    flat = top10.reshape(-1)
    res["catalog_coverage_pct"] = 100.0 * len(torch.unique(flat)) / nI
    res["cold_items_recommended_pct"] = 100.0 * float(cold_mask[flat].float().mean())          # share of slots
    res["cold_item_coverage_pct"] = 100.0 * len(torch.unique(flat[cold_mask[flat]])) / max(int(cold_mask.sum()), 1)
    if items_mat is not None and queries is not None:
        pr, pc = sd.target.nonzero(as_tuple=True)
        res["alignment"] = alignment(queries[pr], items_mat[pc])
        res["uniformity_users"] = uniformity(queries)
    res["n_users"] = len(sd)
    return res


@torch.no_grad()
def evaluate_method(name: str, query_fn, items_mat, sets: dict, cold_mask: torch.Tensor, pop_scores=None, ks=(10, 20)) -> list:
    rows = []
    unif_items = uniformity(items_mat) if items_mat is not None else float("nan")
    for sl, sd in sets.items():
        if pop_scores is not None:
            scores = pop_scores[None, :].expand(len(sd), -1).clone()
            q = None
        else:
            q = l2n(query_fn(sd))
            scores = q @ items_mat.T
        pu, top10 = rank_metrics(scores, sd, cold_mask, ks)
        r = aggregate(pu, top10, sd, cold_mask, items_mat, q)
        r.update({"method": name, "slice": sl, "uniformity_items": unif_items})
        rows.append(r)
    return rows


def save_metrics(rows: list, out_dir: Path, stem: str = "offline_evaluation"):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cols = ["method", "slice", "n_users", "recall@10", "recall@20", "ndcg@10", "mrr", "catalog_coverage_pct",
            "cold_items_recommended_pct", "cold_item_coverage_pct", "alignment", "uniformity_users", "uniformity_items"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    json.dump(json.loads(df.to_json(orient="records")), open(out_dir / f"{stem}.json", "w"), indent=2)
    return df
