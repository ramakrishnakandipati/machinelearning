"""Approach 4: hybrid semantic-content graph with collaborative smoothing and business-rule re-ranking.

* Dense reconstructed embeddings z_q (RQ-VAE) -> FAISS inner-product index -> content k-NN graph.
* Collaborative smoothing: item CF vectors (SVD of train interactions) and popularity of cold / rarely seen items are
  borrowed from their semantic neighbours, shrunk by evidence n/(n+m).
* One ranker for three regimes (warm / user cold / item cold via candidates / system cold) that re-weights content,
  collaborative, popularity, demographic-segment prior and margin signals; system-cold adds MMR diversification.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from .base import Catalog, ColdStartRecommender, UserContext

REGIME_WEIGHTS = {
    #               content  cf    pop   segment margin central
    "warm":        dict(content=0.40, cf=0.35, pop=0.15, seg=0.00, margin=0.05, central=0.05),
    "user_cold":   dict(content=0.35, cf=0.15, pop=0.15, seg=0.25, margin=0.05, central=0.05),
    "system_cold": dict(content=0.30, cf=0.00, pop=0.00, seg=0.00, margin=0.40, central=0.30),
}


def _mm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo + 1e-9)


class HybridColdStartRecommender(ColdStartRecommender):
    name = "hybrid"

    def __init__(self, catalog: Catalog, dataset: dict, n_neighbors: int = 15, cf_dim: int = 32, shrink_m: float = 5.0,
                 mmr_lambda: float = 0.6, max_per_cluster: int = 2):
        super().__init__(catalog)
        c = catalog
        self.n_neighbors, self.mmr_lambda, self.max_per_cluster = n_neighbors, mmr_lambda, max_per_cluster
        self.index = faiss.IndexFlatIP(c.zq_n.shape[1])
        self.index.add(np.ascontiguousarray(c.zq_n))
        self._sub_idx_cache: Dict[bytes, Tuple[faiss.Index, np.ndarray]] = {}

        # ---- collaborative signals from training interactions only
        inter = dataset["interactions"]
        tr = inter[inter["split"] == "train"]
        U = dataset["users"]["user_id"].max() + 1
        R = sp.csr_matrix((np.ones(len(tr), np.float32), (tr["user_id"].values, tr["item_id"].values)), shape=(U, c.N))
        svd = TruncatedSVD(cf_dim, random_state=0).fit(R)
        cf = svd.components_.T * np.sqrt(svd.singular_values_)[None]
        cf = cf / (np.linalg.norm(cf, axis=1, keepdims=True) + 1e-9)
        cf[c.is_cold] = 0.0                                             # cold items have no collaborative evidence
        counts = np.asarray(R.sum(0)).ravel()
        self.counts = counts

        # ---- semantic neighbour graph over *warm* items (collaborative smoothing)
        warm_idx = faiss.IndexFlatIP(c.zq_n.shape[1])
        warm_idx.add(np.ascontiguousarray(c.zq_n[c.warm_ids]))
        sims, nb = warm_idx.search(np.ascontiguousarray(c.zq_n), n_neighbors + 1)
        nb = c.warm_ids[nb]
        w = np.exp((sims - sims.max(1, keepdims=True)) / 0.1)
        w[nb == np.arange(c.N)[:, None]] = 0.0                          # do not self-smooth
        w /= w.sum(1, keepdims=True) + 1e-9
        cf_nb = np.einsum("nk,nkd->nd", w, cf[nb])
        pop_nb = np.einsum("nk,nk->n", w, np.log1p(counts)[nb])
        alpha = counts / (counts + shrink_m)                            # evidence weight
        self.cf = alpha[:, None] * cf + (1 - alpha[:, None]) * cf_nb
        self.cf /= (np.linalg.norm(self.cf, axis=1, keepdims=True) + 1e-9)
        self.pop = _mm(alpha * np.log1p(counts) + (1 - alpha) * pop_nb)
        self.neighbors, self.neighbor_w = nb, w

        # ---- business / structure priors
        self.margin_n = _mm(c.margin)
        cat_cent = np.stack([c.zq_n[c.cat_idx == k].mean(0) for k in range(c.cat_idx.max() + 1)])
        cat_cent /= (np.linalg.norm(cat_cent, axis=1, keepdims=True) + 1e-9)
        self.central = _mm(np.einsum("nd,nd->n", c.zq_n, cat_cent[c.cat_idx]))
        self._build_segment_prior(dataset, tr)

    # ------------------------------------------------------------------ demographic segment prior
    def _build_segment_prior(self, dataset: dict, tr) -> None:
        n_sub = len(dataset["sub_names"])
        users = dataset["users"].set_index("user_id")
        sub_of_item = self.cat.sub_idx
        df = tr[["user_id", "item_id"]].copy()
        df["sub"] = sub_of_item[df["item_id"].values]
        for f in ("region", "age_band", "channel"):
            df[f] = users.loc[df["user_id"].values, f].values
        base = np.bincount(df["sub"], minlength=n_sub) + 1.0
        base = base / base.sum()
        self.seg_tables: Dict[str, Dict[str, np.ndarray]] = {}
        for f in ("region", "age_band", "channel"):
            tab = {}
            for v, g in df.groupby(f):
                p = np.bincount(g["sub"], minlength=n_sub) + 1.0
                tab[v] = (p / p.sum()) / base                               # lift of sub-category given segment value
            self.seg_tables[f] = tab

    def segment_prior(self, ctx: UserContext) -> Optional[np.ndarray]:
        lift = np.ones(len(self.seg_tables["region"][next(iter(self.seg_tables["region"]))]))
        used = False
        for f in ("region", "age_band", "channel"):
            v = getattr(ctx, f)
            if v in self.seg_tables[f]:
                lift = lift * self.seg_tables[f][v]
                used = True
        if not used:
            return None
        return _mm(lift)[self.cat.sub_idx]

    # ------------------------------------------------------------------ similarity helpers
    def _sub_index(self, candidates: np.ndarray):
        key = np.unique(candidates).tobytes()
        if key not in self._sub_idx_cache:
            ids = np.unique(candidates)
            idx = faiss.IndexFlatIP(self.cat.zq_n.shape[1])
            idx.add(np.ascontiguousarray(self.cat.zq_n[ids]))
            if len(self._sub_idx_cache) > 16:
                self._sub_idx_cache.clear()
            self._sub_idx_cache[key] = (idx, ids)
        return self._sub_idx_cache[key]

    def content_scores(self, vec: np.ndarray, candidates: Optional[np.ndarray], top: int = 400) -> np.ndarray:
        """k-NN content similarity via FAISS (scores outside the retrieved neighbourhood are 0)."""
        out = np.zeros(self.cat.N, np.float32)
        q = np.ascontiguousarray(vec[None].astype(np.float32))
        if candidates is None:
            sims, ids = self.index.search(q, min(top, self.cat.N))
            out[ids[0]] = sims[0]
        else:
            idx, cand_ids = self._sub_index(candidates)
            sims, ids = idx.search(q, min(top, len(cand_ids)))
            out[cand_ids[ids[0]]] = sims[0]
        return out

    def similar_items(self, item_vec: np.ndarray, k: int = 10, exclude: Optional[set] = None,
                      candidates: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
        v = item_vec / (np.linalg.norm(item_vec) + 1e-9)
        s = self.content_scores(v, candidates, top=k + 1 + len(exclude or ()))
        s[s == 0] = -np.inf
        for e in (exclude or ()):
            s[e] = -np.inf
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top if np.isfinite(s[i])]

    # ------------------------------------------------------------------ ranking
    def user_vectors(self, ctx: UserContext) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        seeds, weights = self.cat.seed_items(ctx)
        if not seeds:
            return None, None
        w = np.asarray(weights)[:, None]
        zv = (self.cat.zq_n[seeds] * w).sum(0)
        cv = (self.cf[seeds] * w).sum(0)
        return zv / (np.linalg.norm(zv) + 1e-9), cv / (np.linalg.norm(cv) + 1e-9)

    def component_scores(self, ctx: UserContext, candidates: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        zv, cv = self.user_vectors(ctx)
        N = self.cat.N
        comp = {k: np.zeros(N) for k in ("content", "cf", "pop", "seg", "margin", "central")}
        if zv is not None:
            comp["content"] = self.content_scores(zv, candidates)
            comp["cf"] = np.clip(self.cf @ cv, 0, None)
        comp["pop"] = self.pop
        seg = self.segment_prior(ctx)
        comp["seg"] = seg if seg is not None else np.zeros(N)
        comp["margin"] = self.margin_n
        comp["central"] = self.central
        return comp

    def weights_for(self, ctx: UserContext) -> Tuple[str, Dict[str, float]]:
        regime = ctx.auto_regime()
        w = dict(REGIME_WEIGHTS[regime])
        if regime == "user_cold" and not ctx.has_meta():
            w["seg"] = 0.0
        if self.cat.dataset is not None and not ctx.history and ctx.click_item is None and not ctx.intent_cat \
                and not ctx.intent_sub:
            w["content"] = 0.0                                         # nothing to match: popularity/margin/centrality only
            if regime == "user_cold":
                w["seg"] += 0.1
        return regime, w

    def score_all(self, ctx: UserContext, candidates: Optional[np.ndarray] = None) -> np.ndarray:
        _, w = self.weights_for(ctx)
        comp = self.component_scores(ctx, candidates)
        if candidates is not None:
            base = np.zeros(self.cat.N, bool); base[candidates] = True
        else:
            base = np.ones(self.cat.N, bool)
        s = np.zeros(self.cat.N)
        for k, wt in w.items():
            if wt > 0:
                x = comp[k]
                s += wt * ((x - x[base].min()) / (x[base].max() - x[base].min() + 1e-9))
        return s

    def recommend(self, ctx: UserContext, k: int = 10, candidates: Optional[Sequence[int]] = None,
                  exclude: Optional[set] = None) -> List[Tuple[int, float]]:
        exclude = exclude if exclude is not None else set(ctx.history)
        cand = None if candidates is None else np.unique(np.asarray(candidates, dtype=np.int64))
        scores = self.score_all(ctx, cand)
        regime = ctx.auto_regime()
        if regime == "system_cold":
            return self._mmr(scores, k, cand, exclude)
        return self._top_k(scores, k, cand, exclude)

    def _mmr(self, scores: np.ndarray, k: int, cand, exclude, pool: int = 100) -> List[Tuple[int, float]]:
        """Greedy maximal-marginal-relevance with a cap per level-1 semantic cluster."""
        top = self._top_k(scores, pool, cand, exclude)
        if not top:
            return []
        ids = np.array([i for i, _ in top])
        rel = _mm(np.array([s for _, s in top]))
        sim = self.cat.zq_n[ids] @ self.cat.zq_n[ids].T
        c1 = self.cat.codes[ids, 0]
        chosen, per_cluster = [], {}
        while len(chosen) < min(k, len(ids)):
            best, best_val = None, -np.inf
            for j in range(len(ids)):
                if j in chosen or per_cluster.get(int(c1[j]), 0) >= self.max_per_cluster:
                    continue
                red = max((sim[j, q] for q in chosen), default=0.0)
                val = self.mmr_lambda * rel[j] - (1 - self.mmr_lambda) * red
                if val > best_val:
                    best, best_val = j, val
            if best is None:                                            # cluster caps exhausted -> relax
                per_cluster = {}
                continue
            chosen.append(best)
            per_cluster[int(c1[best])] = per_cluster.get(int(c1[best]), 0) + 1
        return [(int(ids[j]), float(scores[ids[j]])) for j in chosen]
