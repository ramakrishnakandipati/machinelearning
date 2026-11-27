"""End-to-end funnel: understanding -> hybrid retrieval -> LTR + multi-task scoring -> business re-ranking."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.common.utils import load_config, resolve, setup_logging
from src.common.world import World
from src.engine import SearchEngine
from src.ranking.feature_engineering import FeatureBuilder
from src.ranking.ltr_models import load_models
from src.ranking.multi_task_scorer import MultiTaskScorer
import joblib

from src.reranking.business_rules import Constraints, apply_hard_filters, soft_penalties
from src.reranking.diversity_mmr import mmr_select
from src.reranking.margin_optimizer import margin_rerank


@dataclass
class RerankResult:
    order: np.ndarray            # positions (into candidate arrays) in final order
    final: np.ndarray            # final score for every candidate (-inf if filtered)
    keep: np.ndarray
    reasons: dict
    route: np.ndarray
    flags: list
    parts: dict


def rerank(world: World, cfg: dict, doc_emb: np.ndarray, idx: np.ndarray, rel_scores: np.ndarray,
           expected_profit: np.ndarray, user_idx, constraints: Constraints, parsed_max_price: float,
           top_k: int, margin_weight: float | None = None, apply_rules: bool = True,
           margin_opt: bool = True, diversify: bool = True, brand_intent: bool = False) -> RerankResult:
    rc = cfg["rerank"]
    n = len(idx)
    keep, reasons, route = (np.ones(n, bool), {}, np.array([""] * n, dtype=object))
    flags = [[] for _ in range(n)]
    pen = np.ones(n, dtype=np.float32)
    if apply_rules:
        keep, reasons, route = apply_hard_filters(world, idx, constraints, parsed_max_price, cfg)
        pen, flags = soft_penalties(world, idx, user_idx, constraints.store_id, cfg)
    kpos = np.where(keep)[0]
    final = np.full(n, -np.inf)
    parts = {"relevance_norm": np.zeros(n), "profit_norm": np.zeros(n), "penalty": pen}
    if len(kpos) == 0:
        return RerankResult(np.array([], dtype=int), final, keep, reasons, route, flags, parts)
    mw = rc["margin_weight"] if margin_weight is None else margin_weight
    if not margin_opt:
        mw = 0.0
    f, p = margin_rerank(rel_scores[kpos], expected_profit[kpos], mw, cfg["business_rules"]["min_relevance_floor"], pen[kpos])
    final[kpos] = f
    parts["relevance_norm"][kpos], parts["profit_norm"][kpos] = p["relevance_norm"], p["profit_norm"]
    o = kpos[np.argsort(-f)]
    if diversify:
        pool = o  # all hard-filter survivors (<=200): MMR is O(k*n) so this is cheap
        sc = final[pool]
        sc = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
        sel = mmr_select(sc, doc_emb[idx[pool]], world.brand[idx[pool]], world.subcat[idx[pool]], top_k,
                         rc["mmr_lambda"], top_k if brand_intent else rc["max_per_brand"], rc["max_per_subcategory"])
        o = pool[sel]
    else:
        o = o[:top_k]
    return RerankResult(o, final, keep, reasons, route, flags, parts)


class SearchService:
    def __init__(self, cfg, world, engine, models, mtl):
        self.cfg, self.world, self.engine, self.models, self.mtl = cfg, world, engine, models, mtl
        self.fb = FeatureBuilder(world)

    @classmethod
    def load(cls, cfg: dict | None = None) -> "SearchService":
        cfg = cfg or load_config()
        tuned = resolve(cfg, "models") / "tuned.json"
        if tuned.exists():
            import json

            cfg["retrieval"]["alpha"] = json.loads(tuned.read_text())["alpha"]
        world = World(cfg)
        engine = SearchEngine.load(cfg, world)
        mdir = resolve(cfg, "models")
        models = load_models(mdir / "ltr_models.joblib", cfg)
        mtl = joblib.load(mdir / "multitask.joblib")
        logger.info("SearchService ready: {} rankers", list(models))
        return cls(cfg, world, engine, models, mtl)

    def search(self, query: str, user_id: str | None = None, store_id: str | None = None, zip_code: str | None = None,
               fulfillment: str = "any", brand_allow=None, brand_deny=None, min_price: float = 0.0,
               max_price: float = 0.0, ranker: str | None = None, fusion: str | None = None, alpha: float | None = None,
               top_k: int | None = None, margin_weight: float | None = None, apply_rules: bool = True,
               margin_opt: bool = True, diversify: bool = True, n_candidates: int | None = None) -> dict:
        cfg, w = self.cfg, self.world
        T: dict = {}
        t_all = time.perf_counter()
        top_k = top_k or cfg["rerank"]["top_k"]
        uidx = w.user_index.get(user_id) if user_id else None
        store = store_id or w.store_for_zip(zip_code) or (w.user_store[uidx] if uidx is not None else None)
        ctx = self.engine.understand(query)
        T["query_understanding_ms"] = ctx.timings["understand_ms"]
        cand = self.engine.retrieve(ctx, mode=fusion, alpha=alpha, n=n_candidates)
        T["retrieval_ms"] = cand.timings["retrieval_ms"]
        t = time.perf_counter()
        X = self.fb.build(ctx, cand, uidx, store)
        T["feature_ms"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        rname = ranker or cfg["ranking"]["default_ranker"]
        model = self.models.get(rname) or self.models[cfg["ranking"]["default_ranker"]]
        rel_scores = model.predict(X)
        heads = self.mtl.predict(X)
        util = self.mtl.utility(heads, w.price[cand.idx], w.margin_usd[cand.idx])
        T["ranking_ms"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        cons = Constraints(store_id=store, fulfillment=fulfillment, brand_allow=brand_allow or [],
                           brand_deny=brand_deny or [], min_price=min_price, max_price=max_price)
        rr = rerank(w, cfg, self.engine.doc_emb, cand.idx, rel_scores, util["expected_profit"], uidx, cons,
                    ctx.parsed.max_price, top_k, margin_weight, apply_rules, margin_opt, diversify, bool(ctx.parsed.brand))
        T["rerank_ms"] = (time.perf_counter() - t) * 1e3
        T["total_ms"] = (time.perf_counter() - t_all) * 1e3
        items = []
        for rank, j in enumerate(rr.order, 1):
            i = int(cand.idx[j])
            items.append({
                "rank": rank, "sku_id": w.catalog.sku_id.values[i], "title": w.catalog.title.values[i],
                "brand": w.brand[i], "category": w.category[i], "subcategory": w.subcat[i], "color": w.color[i],
                "price": float(w.price[i]), "gross_margin_pct": float(w.margin_pct[i]),
                "return_rate": float(w.return_rate[i]), "is_cold_start": bool(w.cold[i]),
                "stock_online": int(w.stock["WH"][i]), "stock_store": int(w.stock[store][i]) if store in w.stock else 0,
                "fulfillment": str(rr.route[j]),
                "scores": {"final": float(rr.final[j]), "ranker": float(rel_scores[j]),
                           "relevance_norm": float(rr.parts["relevance_norm"][j]),
                           "dense_cos": float(cand.dense_cos[j]), "bm25": float(cand.bm25[j]),
                           "p_click": float(heads["pctr"][j]), "p_add_to_cart": float(heads["patc"][j]),
                           "p_purchase_given_cart": float(heads["pcvr"][j]), "p_return": float(heads["pret"][j]),
                           "expected_revenue": float(util["expected_revenue"][j]),
                           "expected_profit": float(util["expected_profit"][j])},
                "flags": rr.flags[j],
            })
        p = ctx.parsed
        return {
            "query": query, "corrected_query": ctx.corrected, "spell_edits": ctx.edits,
            "understanding": {**p.facets(), "intent_confidence": p.intent_conf, "feature_terms": p.feature_terms,
                              "synonyms": ctx.expansion["synonyms"], "embedding_neighbors": ctx.expansion["neighbors"],
                              "learned_sparse_terms": ctx.learned},
            "ranker": rname, "fusion": fusion or cfg["retrieval"]["fusion"], "store_id": store,
            "funnel": {"retrieved": int(len(cand.idx)), "after_hard_filters": int(rr.keep.sum()),
                       "returned": len(items), "dropped_reasons": rr.reasons,
                       "cold_start_in_results": int(sum(i["is_cold_start"] for i in items))},
            "latency_ms": {k: round(v, 2) for k, v in T.items()},
            "results": items,
        }
