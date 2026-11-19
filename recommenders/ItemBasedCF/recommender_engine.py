"""Online inference engine for IBCF: artifact loader, scoring, guardrails, cold-start fallback."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_BOOST = 2.0
MAX_PER_SUBCATEGORY = 3
MAX_CANDIDATES = 500


class RecommenderEngine:
    def __init__(self) -> None:
        self.loaded = False

    # ------------------------------------------------------------------ loading
    def load(self, artifact_dir: str | Path) -> "RecommenderEngine":
        d = Path(artifact_dir)
        meta = json.loads((d / "model_meta.json").read_text())
        catalog = pd.read_parquet(d / "catalog.parquet")
        sim = pd.read_parquet(d / "item_sim_map.parquet")
        hist = pd.read_parquet(d / "user_history.parquet")
        fallback = json.loads((d / "fallback_popular.json").read_text())

        self.meta = meta
        self.version: str = meta["model_version"]
        self.items: list[str] = catalog["item_id"].tolist()
        self.idx = {k: i for i, k in enumerate(self.items)}
        self.category = catalog["category"].to_numpy()
        self.sub_category = catalog["sub_category"].to_numpy()
        self.price = catalog["price"].to_numpy()
        self.consumable = catalog["is_consumable"].to_numpy(dtype=bool)
        self.in_stock = catalog["in_stock"].to_numpy(dtype=bool)
        n = len(self.items)

        self.neighbors: list[tuple[np.ndarray, np.ndarray]] = [(np.empty(0, int), np.empty(0))] * n
        sim = sim.assign(i=sim["item_id"].map(self.idx), j=sim["neighbor_id"].map(self.idx))
        for i, g in sim.groupby("i"):
            g = g.sort_values("rank")
            self.neighbors[int(i)] = (g["j"].to_numpy(int), g["score"].to_numpy(float))

        self.ref_ts = pd.Timestamp(meta["reference_time"])
        self.repurchase_s = meta.get("consumable_repurchase_days", 14) * 86400
        self.history: dict[str, dict] = {}
        hist = hist.assign(j=hist["item_id"].map(self.idx))
        mx = hist.groupby("user_id")["score"].transform("max").replace(0, 1)
        hist["w"] = hist["score"] / mx
        for uid, g in hist.groupby("user_id"):
            seeds = g[g["w"] > 0]
            purchased = g[g["n_purchases"] > 0]
            last = {int(j): (self.ref_ts - ts).total_seconds() for j, ts in zip(purchased["j"], purchased["last_purchase_ts"])}
            self.history[uid] = {"seed_idx": seeds["j"].to_numpy(int), "seed_w": seeds["w"].to_numpy(float),
                                 "purchased_age_s": last}
        self.fallback = [(self.idx[f["item_id"]], f["score"]) for f in fallback if f["item_id"] in self.idx]
        covered = set()
        for nb, _ in self.neighbors:
            covered.update(nb.tolist())
        self.catalog_coverage = len(covered) / n
        self.loaded = True
        return self

    def status(self) -> dict:
        return {"loaded": self.loaded, "model_version": getattr(self, "version", None),
                "n_items": len(self.items) if self.loaded else 0,
                "n_users": len(self.history) if self.loaded else 0,
                "catalog_coverage": round(self.catalog_coverage, 4) if self.loaded else 0.0,
                "params": {"beta": self.meta.get("beta"), "K": self.meta.get("K")} if self.loaded else {}}

    # --------------------------------------------------------------- guardrails
    def _suppressed(self, user_hist: dict | None) -> set[int]:
        """Durable items already purchased are never re-recommended; consumables are allowed
        again after the replenishment window."""
        out: set[int] = set()
        if not user_hist:
            return out
        for j, age in user_hist["purchased_age_s"].items():
            if (not self.consumable[j]) or age < self.repurchase_s:
                out.add(j)
        return out

    def _select(self, ranked, limit, blocked, in_stock_only, reasons, strategy_reason):
        per_sub: dict[str, int] = {}
        out = []
        for j, s in ranked:
            if j in blocked or (in_stock_only and not self.in_stock[j]):
                continue
            sub = self.sub_category[j]
            if per_sub.get(sub, 0) >= MAX_PER_SUBCATEGORY:
                continue
            per_sub[sub] = per_sub.get(sub, 0) + 1
            out.append(self._fmt(j, s, reasons.get(j, strategy_reason)))
            blocked.add(j)
            if len(out) >= limit:
                break
        return out

    def _fmt(self, j: int, score: float, reason: str) -> dict:
        return {"item_id": self.items[j], "score": round(float(score), 5), "category": str(self.category[j]),
                "sub_category": str(self.sub_category[j]), "price": float(self.price[j]),
                "in_stock": bool(self.in_stock[j]), "reason": reason}

    def _fallback(self, limit, blocked, in_stock_only=True):
        return self._select(self.fallback, limit, set(blocked), in_stock_only, {}, "popular_margin_weighted")

    # ---------------------------------------------------------------- scoring
    def _score(self, seed_idx, seed_w):
        s = np.zeros(len(self.items))
        for j, w in zip(seed_idx, seed_w):
            nb, sc = self.neighbors[j]
            s[nb] += sc * w
        return s

    def _rank(self, s):
        cand = np.flatnonzero(s > 0)
        if len(cand) > MAX_CANDIDATES:
            cand = cand[np.argpartition(-s[cand], MAX_CANDIDATES)[:MAX_CANDIDATES]]
        cand = cand[np.argsort(-s[cand])]
        return [(int(j), float(s[j])) for j in cand]

    # --------------------------------------------------------------- public API
    def recommend_item(self, item_id: str, limit: int = 10, in_stock_only: bool = True) -> dict:
        j = self.idx.get(item_id)
        if j is None:
            return {"strategy": "fallback", "found": False, "items": self._fallback(limit, set(), in_stock_only)}
        nb, sc = self.neighbors[j]
        items = self._select(zip(nb.tolist(), sc.tolist()), limit, {j}, in_stock_only, {}, "similar_item")
        if not items:
            return {"strategy": "fallback", "found": True, "items": self._fallback(limit, {j}, in_stock_only)}
        return {"strategy": "item_similarity", "found": True, "items": items}

    def recommend_user(self, user_id: str, recent_item_ids=(), exclude_item_ids=(), limit: int = 10) -> dict:
        hist = self.history.get(user_id)
        sess = [self.idx[i] for i in recent_item_ids if i in self.idx]
        blocked = {self.idx[i] for i in exclude_item_ids if i in self.idx} | set(sess) | self._suppressed(hist)
        if hist is None and not sess:
            return {"strategy": "fallback", "items": self._fallback(limit, blocked)}
        h_scores = self._score(hist["seed_idx"], hist["seed_w"]) if hist else np.zeros(len(self.items))
        s_scores = self._score(sess, [SESSION_BOOST] * len(sess)) if sess else np.zeros(len(self.items))
        total = h_scores + s_scores
        reasons = {int(j): ("session_affinity" if s_scores[j] > h_scores[j] else "purchase_history")
                   for j in np.flatnonzero(total > 0)}
        items = self._select(self._rank(total), limit, blocked, True, reasons, "purchase_history")
        if not items:
            return {"strategy": "fallback", "items": self._fallback(limit, blocked)}
        if len(items) < limit:  # top up from margin-weighted bestsellers, still guardrailed
            used = blocked | {self.idx[i["item_id"]] for i in items}
            subs = {}
            for i in items:
                subs[i["sub_category"]] = subs.get(i["sub_category"], 0) + 1
            for f in self._fallback(limit * 3, used):
                if len(items) >= limit:
                    break
                if subs.get(f["sub_category"], 0) < MAX_PER_SUBCATEGORY:
                    subs[f["sub_category"]] = subs.get(f["sub_category"], 0) + 1
                    items.append(f)
        return {"strategy": "personalized" if hist else "session", "items": items}
