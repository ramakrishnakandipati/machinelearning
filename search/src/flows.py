"""End-to-end scenario flows. Each example is pushed through the real funnel and scored against the
simulator's ground truth *at every stage*, so the UI/README can show what each stage contributes.

Stage variants compared per example (all use the same 200 hybrid candidates):
  A retrieval order -> B + LTR ensemble -> C + hard rules/penalties -> D + margin re-rank -> E + MMR (final)
"""
from __future__ import annotations

import json
import time

import numpy as np

from src.data_gen import generate_retail_data as G
from src.evaluation.offline_metrics import ndcg_at_k
from src.reranking.business_rules import Constraints, availability
from src.search_service import rerank

K = 10

FLOWS = [
    {"id": "vocab_gap", "title": "1 · Vocabulary gap",
     "blurb": "Colloquial shopper language that never appears in the catalog. Stage 1 expands it; hybrid retrieval bridges the gap.",
     "examples": [
         {"label": "waterproof trail kicks", "query": "waterproof trail kicks", "user_id": "U00012", "store_id": "S1",
          "truth": {"subs": ["Trail Running Shoes"], "feat": "Gore-Tex waterproof"}},
         {"label": "breathable gym tee", "query": "breathable gym tee", "user_id": "U00020", "store_id": "S2",
          "truth": {"subs": ["Athletic T-Shirts"], "feat": "moisture-wicking"}},
         {"label": "office chic heels", "query": "office chic heels", "user_id": "U00033", "store_id": "S3",
          "truth": {"subs": ["Heels"], "feat": "block heel"}},
         {"label": "warm winter puffer", "query": "warm winter puffer", "user_id": "U00045", "store_id": "S4",
          "truth": {"subs": ["Puffer Jackets"], "feat": "insulated down"}}]},
    {"id": "typo_price", "title": "2 · Typos + price constraint",
     "blurb": "Misspellings are fixed against catalog vocabulary and 'under $N' becomes a hard price filter.",
     "examples": [
         {"label": "gortex snekers under $100", "query": "gortex snekers under $100", "user_id": "U00012", "store_id": "S1",
          "truth": {"subs": ["Trail Running Shoes"], "feat": "Gore-Tex waterproof", "max_price": 100}},
         {"label": "breathble gym tee under $30", "query": "breathble gym tee under $30", "user_id": "U00021", "store_id": "S5",
          "truth": {"subs": ["Athletic T-Shirts"], "feat": "moisture-wicking", "max_price": 30}},
         {"label": "warm pufer jaket below $200", "query": "warm pufer jaket below $200", "user_id": "U00046", "store_id": "S3",
          "truth": {"subs": ["Puffer Jackets"], "feat": "insulated down", "max_price": 200}},
         {"label": "legings with pockets under $60", "query": "legings with pockets under $60", "user_id": "U00058", "store_id": "S2",
          "truth": {"subs": ["Leggings"], "feat": "pockets", "max_price": 60}}]},
    {"id": "navigational", "title": "3 · Navigational (brand) intent",
     "blurb": "Brand-led queries: the intent classifier flags navigational intent, brand features dominate, and the brand diversity cap is lifted.",
     "examples": [
         {"label": "aeroflux running shoes", "query": "aeroflux running shoes", "user_id": "U00005", "store_id": "S1",
          "truth": {"subs": ["Trail Running Shoes", "Road Running Shoes"], "brand": "Aeroflux"}},
         {"label": "terrapeak jackets", "query": "terrapeak jackets", "user_id": "U00014", "store_id": "S4",
          "truth": {"subs": ["Puffer Jackets", "Rain Jackets", "Fleece Jackets"], "brand": "TerraPeak"}},
         {"label": "hearthly bedding", "query": "hearthly bedding", "user_id": "U00027", "store_id": "S5",
          "truth": {"subs": ["Bedding Sets"], "brand": "Hearthly"}},
         {"label": "voltra headphones", "query": "voltra headphones", "user_id": "U00039", "store_id": "S2",
          "truth": {"subs": ["Headphones"], "brand": "Voltra"}}]},
    {"id": "regional", "title": "4 · Regional stock & BOPIS",
     "blurb": "Buy-online-pick-up-in-store: only items stocked at (or near) the shopper's store may be shown. Compare unfulfillable results before vs after rules.",
     "examples": [
         {"label": "running shoes · BOPIS Chicago (S3)", "query": "running shoes", "user_id": "U00008", "store_id": "S3", "fulfillment": "bopis",
          "truth": {"subs": ["Trail Running Shoes", "Road Running Shoes"]}},
         {"label": "winter boots · BOPIS Denver (S5)", "query": "warm winter boots", "user_id": "U00016", "store_id": "S5", "fulfillment": "bopis",
          "truth": {"subs": ["Winter Boots"], "feat": "insulated"}},
         {"label": "office chair · BOPIS Austin (S2)", "query": "back friendly desk chair", "user_id": "U00024", "store_id": "S2", "fulfillment": "bopis",
          "truth": {"subs": ["Office Chairs"], "feat": "lumbar support"}}]},
    {"id": "cold_start", "title": "5 · Cold-start SKUs",
     "blurb": "Brand-new drops with zero interaction history are retrieved from text+image+tabular content embeddings and still reach the page.",
     "examples": "COLD"},
    {"id": "margin", "title": "6 · Margin-aware re-ranking",
     "blurb": "Same relevance, higher expected profit: the margin optimiser (weight 0.6 here) re-orders relevant items by E[profit] without crossing the relevance floor.",
     "examples": [
         {"label": "running shoes · margin weight 0.6", "query": "running shoes", "user_id": "U00010", "store_id": "S1", "margin_weight": 0.6,
          "truth": {"subs": ["Trail Running Shoes", "Road Running Shoes"]}},
         {"label": "headphones · margin weight 0.6", "query": "headphones", "user_id": "U00031", "store_id": "S2", "margin_weight": 0.6,
          "truth": {"subs": ["Headphones"]}},
         {"label": "backpacks · margin weight 0.6", "query": "backpacks", "user_id": "U00052", "store_id": "S3", "margin_weight": 0.6,
          "truth": {"subs": ["Backpacks"]}}]},
    {"id": "personalization", "title": "7 · Personalisation",
     "blurb": "The same query for three different shoppers. User brand/price affinity features change the order; 'preferred-brand share' shows it.",
     "examples": [
         {"label": "running shoes · user U00001", "query": "running shoes", "user_id": "U00001", "store_id": "S1",
          "truth": {"subs": ["Trail Running Shoes", "Road Running Shoes"]}},
         {"label": "running shoes · user U00100", "query": "running shoes", "user_id": "U00100", "store_id": "S2",
          "truth": {"subs": ["Trail Running Shoes", "Road Running Shoes"]}},
         {"label": "running shoes · user U00200", "query": "running shoes", "user_id": "U00200", "store_id": "S3",
          "truth": {"subs": ["Trail Running Shoes", "Road Running Shoes"]}}]},
]


def cold_examples(world, n: int = 3):
    rng = np.random.default_rng(7)
    ids = rng.choice(np.where(world.cold & (world.stock["WH"] > 5))[0], size=n * 6, replace=False)
    out, seen = [], set()
    for i in ids:
        row = world.catalog.iloc[i]
        if row.subcategory in seen:
            continue
        seen.add(row.subcategory)
        out.append({"label": f"new drop: {row.title.split(' - ')[0][:48]}", "query": f"{row.color} {row.subcategory.lower()} {row.brand.lower()}",
                    "user_id": f"U{int(rng.integers(0, 300)):05d}", "store_id": "S1", "target_idx": int(i),
                    "truth": {"subs": [row.subcategory], "color": row.color, "brand": row.brand}})
        if len(out) == n:
            break
    return out


def get_flows(world):
    fl = []
    for f in FLOWS:
        g = dict(f)
        if g["examples"] == "COLD":
            g["examples"] = cold_examples(world)
        fl.append(g)
    return fl


# ----------------------------------------------------------------------------------------------------
def _truth_rel(world, truth: dict, claimed: bool) -> np.ndarray:
    q = {"subs": truth["subs"], "feat": truth.get("feat", ""), "color": truth.get("color", ""),
         "brand": truth.get("brand", ""), "max_price": truth.get("max_price", 0.0)}
    arr = world._claimed_arr if claimed else world._true_arr
    return G.relevance_for_query(q, world.sub_idx, arr, world.brand, world.color, world.price, world._subs_index)


def _probs(world, truth, uidx, idx):
    rt = _truth_rel(world, truth, False)[idx].astype(float)
    rc = _truth_rel(world, truth, True)[idx].astype(float)
    if uidx is None:
        bm, med = np.zeros(len(idx)), float(np.median(world.user_median))
    else:
        bm = np.array([world.brand[i] in world.users_pref[uidx] for i in idx], float)
        med = float(world.user_median[uidx])
    return G.latent_probs(rt, rc, world.l_quality[idx], bm, G.price_fit(world.price[idx], med), world.l_ret[idx], exam=1.0)


def _recall(rel_ids, cand_idx, n):
    if len(rel_ids) == 0:
        return float("nan")
    return float(np.isin(rel_ids, cand_idx).sum() / min(len(rel_ids), n))


def run_example(svc, ex: dict) -> dict:
    """Run one example through every stage and score it against simulator ground truth."""
    w, eng, cfg = svc.world, svc.engine, svc.cfg
    truth = ex["truth"]
    uidx = w.user_index.get(ex["user_id"])
    store, fulfil = ex.get("store_id"), ex.get("fulfillment", "any")
    mw = ex.get("margin_weight")
    q = ex["query"]
    # ---- stage 1: query understanding ----------------------------------------------------------------
    ctx = eng.understand(q)
    p = ctx.parsed
    checks = {"subcategory": bool(set(p.subcategories) & set(truth["subs"]))}
    if truth.get("max_price"):
        checks["price_cap"] = abs(p.max_price - truth["max_price"]) < 1e-6
    if truth.get("brand"):
        checks["brand"] = p.brand == truth["brand"]
    if truth.get("color"):
        checks["color"] = p.color == truth["color"]
    stage1 = {"corrected": ctx.corrected, "edits": [list(e) for e in ctx.edits], "intent": p.intent_type,
              "intent_conf": p.intent_conf, "facets": p.facets(), "synonyms": ctx.expansion["synonyms"],
              "neighbors": ctx.expansion["neighbors"], "learned": ctx.learned, "n_expansion_terms": len(ctx.weighted_terms),
              "facet_checks": checks, "facet_accuracy": float(np.mean(list(checks.values()))),
              "ms": ctx.timings["understand_ms"]}
    # ---- stage 2: retrieval ---------------------------------------------------------------------------
    rel_t = _truth_rel(w, truth, False)
    rel_ids = np.where((rel_t >= 3) & (w.stock["WH"] > 0))[0]
    cold_rel = rel_ids[w.cold[rel_ids]]
    cand = eng.retrieve(ctx)
    ret = {}
    for name, mode in [("sparse", "sparse"), ("dense", "dense"), ("hybrid", "alpha")]:
        c50, c200 = eng.retrieve(ctx, mode=mode, n=50), eng.retrieve(ctx, mode=mode, n=200)
        ret[name] = {"recall@50": _recall(rel_ids, c50.idx, 50), "recall@200": _recall(rel_ids, c200.idx, 200)}
    stage2 = {"n_relevant_in_catalog": int(len(rel_ids)), "n_candidates": int(len(cand.idx)),
              "n_relevant_in_candidates": int(np.isin(rel_ids, cand.idx).sum()),
              "cold_relevant": int(len(cold_rel)), "cold_in_candidates": int(np.isin(cold_rel, cand.idx).sum()),
              "retrievers": ret, "ms": cand.timings["retrieval_ms"]}
    # ---- stage 3/4: ranking + re-ranking --------------------------------------------------------------
    X = svc.fb.build(ctx, cand, uidx, store)
    t = time.perf_counter()
    scores = svc.models[cfg["ranking"]["default_ranker"]].predict(X)
    heads = svc.mtl.predict(X)
    util = svc.mtl.utility(heads, w.price[cand.idx], w.margin_usd[cand.idx])
    rank_ms = (time.perf_counter() - t) * 1e3
    cons = Constraints(store_id=store, fulfillment=fulfil)
    idx = cand.idx
    rel_c = rel_t[idx]
    pc, pa, pv, pr = _probs(w, truth, uidx, idx)
    pbuy = pc * pa * pv
    pref = w.users_pref[uidx] if uidx is not None else set()
    avail, _ = availability(w, idx, store, fulfil, cfg["business_rules"]["bopis_max_km"])

    def metrics(order):
        order = np.asarray(order, dtype=int)
        top = order[:K]
        exam = 1.0 / (1.0 + 0.35 * np.arange(len(top)))
        sku = idx[top]
        return {"ndcg@10": ndcg_at_k(order, rel_c, K), "precision@10": float((rel_c[top] >= 3).mean()) if len(top) else 0.0,
                "exp_revenue@10": float((exam * pbuy[top] * w.price[sku]).sum()),
                "exp_margin@10": float((exam * pbuy[top] * w.margin_usd[sku] * (1 - pr[top])).sum()),
                "unfulfillable@10": int((~avail[top]).sum()), "avg_return_prob@10": float(pr[top].mean()) if len(top) else 0.0,
                "avg_price@10": float(w.price[sku].mean()) if len(top) else 0.0,
                "pref_brand_share@10": float(np.mean([w.brand[i] in pref for i in sku])) if len(top) else 0.0,
                "cold_in_top10": int(w.cold[sku].sum()), "shown": int(len(top))}

    def rr(**kw):
        r = rerank(w, cfg, eng.doc_emb, idx, scores, util["expected_profit"], uidx, cons, p.max_price, 50,
                   mw, brand_intent=bool(p.brand), **kw)
        return r

    orders = {"A retrieval order": np.argsort(-cand.fused), "B + LTR ensemble": np.argsort(-scores)}
    r_c = rr(apply_rules=True, margin_opt=False, diversify=False)
    r_d = rr(apply_rules=True, margin_opt=True, diversify=False)
    r_e = rerank(w, cfg, eng.doc_emb, idx, scores, util["expected_profit"], uidx, cons, p.max_price, 20, mw,
                 True, True, True, bool(p.brand))
    orders["C + hard rules & penalties"] = r_c.order
    orders["D + margin re-rank"] = r_d.order
    orders["E + MMR (final page)"] = r_e.order
    stages = {k: metrics(v) for k, v in orders.items()}
    ranking_stage = {"ranker": cfg["ranking"]["default_ranker"], "ms": rank_ms,
                     "mean_pCTR@10": float(heads["pctr"][orders["B + LTR ensemble"][:K]].mean()),
                     "mean_pCVR@10": float(heads["pcvr"][orders["B + LTR ensemble"][:K]].mean()),
                     "mean_pReturn@10": float(heads["pret"][orders["B + LTR ensemble"][:K]].mean())}
    rules_stage = {"retrieved": int(len(idx)), "after_hard_filters": int(r_c.keep.sum()), "dropped": r_c.reasons,
                   "high_return_flagged": int(sum("high_return_rate" in f for f in r_c.flags)),
                   "low_stock_flagged": int(sum("low_stock" in f for f in r_c.flags))}
    # ---- the real served response (latency + final page) ------------------------------------------------
    out = svc.search(q, user_id=ex["user_id"], store_id=store, fulfillment=fulfil, margin_weight=mw, top_k=20)
    final = [{"rank": r["rank"], "title": r["title"], "price": r["price"], "margin_pct": round(r["gross_margin_pct"] * 100, 1),
              "cold": r["is_cold_start"], "route": r["fulfillment"], "E_profit": round(r["scores"]["expected_profit"], 3),
              "flags": r["flags"]} for r in out["results"]]
    res = {"example": {k: v for k, v in ex.items() if k != "truth"}, "truth": truth, "stage1": stage1, "stage2": stage2,
           "ranking": ranking_stage, "rules": rules_stage, "variants": stages, "final": final, "latency_ms": out["latency_ms"]}
    if "target_idx" in ex:
        ti = ex["target_idx"]
        pos = lambda arr: int(np.where(np.asarray(arr) == ti)[0][0] + 1) if (np.asarray(arr) == ti).any() else None
        res["cold_target"] = {"title": w.catalog.title.values[ti], "impressions": int(w.catalog.impressions.values[ti]),
                              "days_since_launch": int(w.days_since_launch[ti]),
                              "rank_sparse": pos(eng.retrieve(ctx, mode="sparse").idx), "rank_dense": pos(eng.retrieve(ctx, mode="dense").idx),
                              "rank_hybrid": pos(cand.idx), "rank_final": pos([w.sku_index[r["sku_id"]] for r in out["results"]])}
    return res


def run_flow(svc, flow: dict) -> list[dict]:
    return [run_example(svc, ex) for ex in flow["examples"]]
