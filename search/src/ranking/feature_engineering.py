"""Query / document / user / cross features for LTR. Document features come from a precomputed
in-memory feature store (numpy arrays aligned to the catalog) so online feature assembly is O(candidates)."""
from __future__ import annotations

import numpy as np

from src.common.text import tokens
from src.common.world import World

INTENT_CODE = {"navigational": 0, "category_browsing": 1, "attribute_specific": 2}

QUERY_FEATURES = ["q_len", "q_ntokens", "q_max_price", "q_has_price", "q_has_brand", "q_has_color",
                  "q_intent", "q_entropy", "q_n_expansions", "q_n_edits", "q_top_dense", "q_top_bm25"]
DOC_FEATURES = ["d_price", "d_log_price", "d_margin_pct", "d_margin_usd", "d_bayes_ctr", "d_bayes_cvr",
                "d_return_rate", "d_days_since_launch", "d_is_cold", "d_stock_online", "d_stock_store",
                "d_brand_tier", "d_log_impr"]
USER_FEATURES = ["u_brand_aff", "u_cat_aff", "u_price_dist", "u_price_tier", "u_recent_same_sub", "u_price_fit"]
CROSS_FEATURES = ["x_dense_cos", "x_bm25", "x_bm25_norm", "x_char", "x_fused", "x_dense_rank", "x_sparse_rank",
                  "x_brand_match", "x_color_match", "x_sub_match", "x_cat_match", "x_attr_overlap",
                  "x_feat_overlap", "x_price_ratio", "x_over_budget", "x_dense_cos_rel"]
FEATURE_NAMES = QUERY_FEATURES + DOC_FEATURES + USER_FEATURES + CROSS_FEATURES


class FeatureBuilder:
    def __init__(self, world: World):
        self.w = world

    def build(self, ctx, cand, user_idx: int, store_id: str | None = None) -> np.ndarray:
        w = self.w
        idx = cand.idx
        n = len(idx)
        pq = ctx.parsed
        anon = user_idx is None
        store_id = store_id or (w.user_store[user_idx] if not anon else "S1")
        qt = set(tokens(ctx.corrected))
        # --- query-level
        sub_counts = np.bincount(w.sub_code[idx[:50]], minlength=int(w.sub_code.max()) + 1).astype(float)
        p = sub_counts[sub_counts > 0] / max(sub_counts.sum(), 1)
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        qf = np.tile(np.array([
            len(ctx.corrected), len(qt), pq.max_price, float(pq.max_price > 0), float(bool(pq.brand)),
            float(bool(pq.color)), INTENT_CODE.get(pq.intent_type, 1), entropy,
            len(ctx.expansion["synonyms"]) + len(ctx.expansion["neighbors"]) + len(ctx.learned),
            len(ctx.edits), float(cand.dense_cos.max() if n else 0), float(cand.bm25.max() if n else 0)],
            dtype=np.float32), (n, 1))
        # --- doc-level
        price = w.price[idx]
        df_ = np.stack([price, np.log1p(price), w.margin_pct[idx], w.margin_usd[idx], w.bayes_ctr[idx],
                        w.bayes_cvr[idx], w.return_rate[idx], w.days_since_launch[idx], w.cold[idx].astype(np.float32),
                        w.stock["WH"][idx], w.stock[store_id][idx] if store_id in w.stock else np.zeros(n),
                        w.tier[idx], w.log_impr[idx]], 1).astype(np.float32)
        # --- user-level
        pref = set() if anon else w.users_pref[user_idx]
        aff = {} if anon else w.users_aff[user_idx]
        med = float(np.median(w.user_median)) if anon else float(w.user_median[user_idx])
        pdist = np.abs(np.log1p(price) - np.log1p(med))
        brand_aff = np.array([float(w.brand[i] in pref) for i in idx], dtype=np.float32)
        cat_aff = np.array([aff.get(w.category[i], 1.0 / 7) for i in idx], dtype=np.float32)
        recent = np.zeros(n, dtype=np.float32) if anon else np.array([float(w.subcat[i] in w.recent_sub[user_idx]) for i in idx], dtype=np.float32)
        pfit = np.exp(-0.5 * (pdist / 0.7) ** 2)
        uf = np.stack([brand_aff, cat_aff, pdist, np.full(n, 1 if anon else w.user_tier[user_idx], dtype=np.float32), recent, pfit], 1)
        # --- cross
        feat_terms = set(pq.feature_terms)
        sub_set = set(pq.subcategories)
        brand_match = np.array([float(bool(pq.brand) and w.brand[i] == pq.brand) for i in idx], dtype=np.float32)
        color_match = np.array([float(bool(pq.color) and w.color[i] == pq.color) for i in idx], dtype=np.float32)
        sub_match = np.array([float(w.subcat[i] in sub_set) for i in idx], dtype=np.float32)
        cat_match = np.array([float(bool(pq.category) and w.category[i] == pq.category) for i in idx], dtype=np.float32)
        overlap = np.array([len(qt & w.doc_tokens[i]) / max(len(qt), 1) for i in idx], dtype=np.float32)
        foverlap = np.array([len(feat_terms & w.doc_tokens[i]) / max(len(feat_terms), 1) if feat_terms else 0.0
                             for i in idx], dtype=np.float32)
        if pq.max_price > 0:
            ratio = price / pq.max_price
            over = (price > pq.max_price).astype(np.float32)
        else:
            ratio = np.zeros(n, dtype=np.float32)
            over = np.zeros(n, dtype=np.float32)
        bm = cand.bm25
        cf = np.stack([cand.dense_cos, bm, bm / (bm.max() + 1e-9), cand.char, cand.fused.astype(np.float32),
                       cand.dense_rank, cand.sparse_rank, brand_match, color_match, sub_match, cat_match,
                       overlap, foverlap, ratio, over, cand.dense_cos - cand.dense_cos.max()], 1)
        X = np.hstack([qf, df_, uf, cf]).astype(np.float32)
        return X
