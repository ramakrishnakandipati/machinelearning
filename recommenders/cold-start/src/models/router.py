"""Meta-router: picks the cold-start strategy per request and fuses the individual models.

Routing (by number of known interactions):
    0          -> ``cold_zero_history``     contextual popularity + onboarding two-tower (+ content) + LinUCB
    1 .. 4     -> ``semi_cold_session``     session content-based + two-tower (+ light popularity)
    5+         -> ``warm_personalized``     two-tower candidates re-ranked with the bandit
    new SKUs   -> semantic injection slots  (content-based) whenever ``include_new_items_boost`` is set
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.data.preprocessor import Preprocessor
from src.models.content_based import ContentBasedRecommender, ItemIndex
from src.models.contextual_bandit import ContextualBandit
from src.models.hybrid_two_tower import TwoTowerRecommender
from src.models.non_personalized import ContextualPopularity

STRATEGY_ZERO, STRATEGY_SEMI, STRATEGY_WARM = "cold_zero_history", "semi_cold_session", "warm_personalized"
WEIGHTS = {
    STRATEGY_ZERO: dict(pop=0.15, tt=0.40, content=0.30, bandit=0.15),
    STRATEGY_SEMI: dict(pop=0.10, tt=0.50, content=0.30, bandit=0.10),
    STRATEGY_WARM: dict(pop=0.00, tt=0.85, content=0.00, bandit=0.15),
}
NEW_ITEM_TAG = "Matched via New-Item Semantic Similarity"


def _norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score over the candidate mask, clipped to [-3, 3] and rescaled to [-1, 1] (non-candidates -> 0)."""
    out = np.zeros_like(x, dtype=np.float64)
    v = x[mask].astype(np.float64)
    if v.size == 0:
        return out
    sd = v.std()
    out[mask] = np.clip((v - v.mean()) / (sd if sd > 1e-9 else 1.0), -3, 3) / 3.0
    return out


@dataclass
class RankedItem:
    position: int
    item_id: str
    score: float
    tags: List[str]
    is_new_item: bool = False


@dataclass
class RouteResult:
    strategy: str
    items: List[RankedItem]
    latency_ms: float
    n_interactions: int
    components: Dict[str, float] = field(default_factory=dict)


class MetaRouter:
    """Orchestrates popularity / content / two-tower / bandit models and the live item catalog."""

    def __init__(self, cfg: Dict[str, Any], pre: Preprocessor, items_df, pop: ContextualPopularity,
                 index: ItemIndex, tower: TwoTowerRecommender, bandit: ContextualBandit, sales: np.ndarray):
        self.cfg, self.pre, self.pop, self.index, self.tower, self.bandit = cfg, pre, pop, index, tower, bandit
        self.content = ContentBasedRecommender(index)
        rc = cfg["router"]
        self.warm_th, self.cold_th = int(rc["warm_threshold"]), int(rc["cold_threshold"])
        self.pool, self.new_slots = int(rc["candidate_pool"]), int(rc["new_item_slots"])
        self.new_max_sales = int(rc["new_item_max_sales"])
        self.meta: List[Dict[str, Any]] = [
            {k: r[k] for k in ("item_id", "title", "brand", "dept", "subcat", "price")} for r in items_df.to_dict("records")]
        self.sales = np.asarray(sales, dtype=np.float64).copy()
        self.subcat_idx = np.array([bandit.arm_idx.get(s, 0) for s in pre.item_subcat], dtype=np.int64)
        self._lock = threading.RLock()
        self._refresh_priors()

    # ----------------------------------------------------------------- catalog
    def _refresh_priors(self) -> None:
        """Category-mean popularity prior used for zero-sales SKUs (Bayesian smoothing toward the category)."""
        g = self.pop.global_score
        n = len(self.subcat_idx)
        g = np.pad(g, (0, max(0, n - len(g))))[:n]
        warm = self.sales > 0
        cat_mean = np.zeros(len(self.bandit.arms))
        for a in range(len(cat_mean)):
            m = warm & (self.subcat_idx == a)
            cat_mean[a] = g[m].mean() if m.any() else g[warm].mean()
        self.cat_prior = cat_mean

    @property
    def n_items(self) -> int:
        return len(self.pre.item_ids)

    def new_item_mask(self) -> np.ndarray:
        return self.sales[: self.n_items] <= self.new_max_sales

    def onboard_item(self, item: Dict[str, Any]) -> int:
        """Embed a brand-new SKU and inject it into FAISS / two-tower / registries with no downtime."""
        p = self.pre
        with self._lock:
            if item["item_id"] in p.item_pos:
                raise ValueError(f"item_id {item['item_id']} already exists")
            enc = p.encode_new_item(item)
            pos = len(p.item_ids)
            if self.index.size != pos:
                raise RuntimeError("index/catalog out of sync")
            it = enc["item"]
            p.item_ids = np.append(p.item_ids, it["item_id"])
            p.item_pos[it["item_id"]] = pos
            p.item_cats = np.vstack([p.item_cats, enc["cats"][None]])
            p.item_text = np.vstack([p.item_text, enc["text"][None]])
            p.item_price = np.append(p.item_price, enc["price"])
            p.item_id_idx = np.append(p.item_id_idx, 0)
            p.item_content = np.vstack([p.item_content, enc["content"][None]])
            p.is_cold_item = np.append(p.is_cold_item, True)
            p.item_subcat = np.append(p.item_subcat, it["subcat"])
            p.item_dept = np.append(p.item_dept, it["dept"])
            self.index.add(enc["content"])
            self.tower.add_item(enc)
            self.pop.grow(pos + 1)
            self.sales = np.append(self.sales, 0.0)
            self.subcat_idx = np.append(self.subcat_idx, self.bandit.arm_idx.get(it["subcat"], 0))
            self.bandit.record_impressions([])  # make sure arrays exist
            self.meta.append({k: it.get(k) for k in ("item_id", "title", "brand", "dept", "subcat", "price")})
            self._refresh_priors()
            return pos

    # ------------------------------------------------------------------ routing
    def route(self, n_interactions: int) -> str:
        if n_interactions < self.cold_th:
            return STRATEGY_ZERO
        if n_interactions < self.warm_th:
            return STRATEGY_SEMI
        return STRATEGY_WARM

    def components(self, user_row: Dict[str, Any], ctx: Dict[str, Any], history: Sequence[int],
                   strategy: str, explore: bool = True) -> Dict[str, np.ndarray]:
        """Raw per-model score arrays over all current items."""
        n = self.n_items
        history = list(history)
        has_user_feats = any(user_row.get(k) for k in ("region", "device", "acq_channel", "age_band", "onboarding_categories"))
        pop_ctx = dict(region=user_row.get("region"), device=user_row.get("device"), acq_channel=user_row.get("acq_channel"),
                       season=ctx.get("season"), hour=ctx.get("hour"), channel=ctx.get("channel"))
        pop = self.pop.score_all(pop_ctx)[:n].astype(np.float64)
        pop = np.pad(pop, (0, n - len(pop)))
        new = self.new_item_mask()
        pop = np.where(new, self.cat_prior[self.subcat_idx], pop)  # zero-sales SKUs -> category prior
        out = {"pop": pop}
        profile = self.pre.profile_vec(history, user_row.get("onboarding_categories"))
        out["content"] = self.content.score_all(profile, n)[:n].astype(np.float64) if profile is not None else np.zeros(n)
        if has_user_feats or history:
            out["tt"] = self.tower.score_all(user_row, ctx, history)[:n].astype(np.float64)
        else:
            out["tt"] = np.zeros(n)
        x = self.bandit.context_vector(user_row, ctx)
        out["bandit"] = self.bandit.score_items(x, self.subcat_idx[:n], explore)
        return out

    def recommend(self, user_row: Dict[str, Any], ctx: Dict[str, Any], history: Sequence[int], k: int = 10,
                  include_new_items_boost: bool = False, exclude: Optional[set] = None,
                  candidates: Optional[np.ndarray] = None, explore: bool = True,
                  n_interactions: Optional[int] = None, record_impressions: bool = False) -> RouteResult:
        t0 = time.perf_counter()
        history = list(history)
        n_int = len(history) if n_interactions is None else int(n_interactions)
        strategy = self.route(n_int)
        n = self.n_items
        raw = self.components(user_row, ctx, history, strategy, explore)
        mask = np.zeros(n, bool)
        if candidates is None:
            mask[:] = True
        else:
            mask[np.asarray(candidates)[np.asarray(candidates) < n]] = True
        if exclude:
            mask[[e for e in exclude if e < n]] = False
        w = dict(WEIGHTS[strategy])
        has_signal = {c: bool(np.any(raw[c] != 0)) for c in raw}
        if not has_signal["tt"]:  # anonymous: no side-features at all -> popularity + exploration
            w = dict(pop=0.7, tt=0.0, content=0.0, bandit=0.3)
        elif not has_signal["content"]:
            w["pop"] += w["content"]; w["content"] = 0.0
        normed = {c: _norm(raw[c], mask) for c in raw}
        if strategy == STRATEGY_WARM:   # bandit only re-ranks the two-tower shortlist
            tt_m = np.where(mask, raw["tt"], -np.inf)
            shortlist = [i for i in np.argsort(-tt_m)[: self.pool] if np.isfinite(tt_m[i])]
            m2 = np.zeros(n, bool); m2[shortlist] = True
            normed = {c: _norm(raw[c], m2) for c in raw}
            mask = m2
        total = sum(w[c] * normed[c] for c in w)
        total = np.where(mask, total, -np.inf)
        order = np.argsort(-total)[: k + (self.new_slots if include_new_items_boost else 0)]
        order = [int(i) for i in order if np.isfinite(total[i])]
        ranked = order[:k]
        new_mask = self.new_item_mask()
        injected: List[int] = []
        if include_new_items_boost and self.new_slots > 0:
            injected = self._new_item_slots(raw, mask if candidates is not None or exclude else None, exclude, ranked, user_row,
                                            ctx, history)
            if injected:
                ranked = self._interleave(ranked, injected, k)
        items = []
        for i in ranked:
            contrib = {c: w[c] * normed[c][i] for c in w}
            items.append(RankedItem(i, str(self.pre.item_ids[i]), float(total[i]) if np.isfinite(total[i]) else 0.0,
                                    self._tags(i, contrib, strategy, user_row, ctx, i in injected, bool(new_mask[i])),
                                    bool(new_mask[i])))
        if record_impressions:
            self.bandit.record_impressions(ranked)
        return RouteResult(strategy, items, (time.perf_counter() - t0) * 1000, n_int, {c: float(w[c]) for c in w})

    # ----------------------------------------------------------- new-item slots
    def _new_item_slots(self, raw, mask, exclude, ranked, user_row, ctx, history) -> List[int]:
        n = self.n_items
        new = self.new_item_mask().copy()
        if exclude:
            new[[e for e in exclude if e < n]] = False
        if not new.any():
            return []
        sim = raw["content"] if np.any(raw["content"] != 0) else raw["tt"]
        s = _norm(sim, new) + 0.5 * _norm(raw["bandit"], new)
        s = np.where(new, s, -np.inf)
        return [int(i) for i in np.argsort(-s)[: self.new_slots] if np.isfinite(s[i])]

    @staticmethod
    def _interleave(ranked: List[int], injected: List[int], k: int) -> List[int]:
        """Place injected new SKUs at slots 2, 5, 8... (0-based 1, 4, 7) keeping total length k."""
        injected = injected[: max(1, k // 2)]
        base = [i for i in ranked if i not in injected][: max(0, k - len(injected))]
        out = list(base)
        for j, it in enumerate(injected):
            out.insert(min(1 + 3 * j, len(out)), it)
        return out[:k]

    # ----------------------------------------------------------------- explain
    def _tags(self, i: int, contrib: Dict[str, float], strategy: str, user_row: Dict[str, Any], ctx: Dict[str, Any],
              injected: bool, is_new: bool) -> List[str]:
        tags: List[str] = []
        if injected:
            tags.append(NEW_ITEM_TAG)
        pos = {c: v for c, v in contrib.items() if v > 0}
        for c, _ in sorted(pos.items(), key=lambda kv: -kv[1])[:2]:
            if c == "pop":
                seg = "/".join(str(user_row.get(k)) for k in ("region", "device") if user_row.get(k)) or "global"
                tags.append(f"Contextual Popularity ({seg}{', ' + str(ctx['season']) if ctx.get('season') else ''})")
            elif c == "content":
                tags.append(NEW_ITEM_TAG if is_new else
                            ("Session Content Similarity" if strategy == STRATEGY_SEMI else "Onboarding Category Similarity"))
            elif c == "tt":
                tags.append("Onboarding Category + Two-Tower" if strategy == STRATEGY_ZERO else
                            ("Session-Aware Two-Tower" if strategy == STRATEGY_SEMI else "Personalized Two-Tower"))
            elif c == "bandit":
                tags.append("Onboarding Category + LinUCB Exploration" if strategy == STRATEGY_ZERO else "LinUCB Exploration")
        return list(dict.fromkeys(tags))

    # ------------------------------------------------------------ eval helper
    def score_for_eval(self, user_row, ctx, history) -> np.ndarray:
        """Total fused score vector (no exploration noise) - used by the offline evaluator."""
        res = self.recommend(user_row, ctx, history, k=self.n_items, explore=False)
        out = np.full(self.n_items, -np.inf)
        for r, it in enumerate(res.items):
            out[it.position] = -r
        return out
