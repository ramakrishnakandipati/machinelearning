"""Sparse retrieval: field-weighted BM25 + char n-gram TF-IDF + learned (SPLADE-style) term expansion.

BM25F-style scoring is implemented on scipy.sparse matrices (a vocab x docs impact matrix) so a query
touches only the rows of its terms; this is ~100x faster than `rank_bm25` on pure-python lists.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.common.text import tokens


class SparseRetriever:
    def __init__(self, field_weights: dict, k1: float = 1.4, b: float = 0.75):
        self.fw, self.k1, self.b = field_weights, k1, b
        self.vocab: dict[str, int] = {}
        self.Wt: sp.csr_matrix | None = None  # (vocab, docs) impact matrix
        self.idf: np.ndarray | None = None
        self.char_vec: TfidfVectorizer | None = None
        self.char_mat: sp.csr_matrix | None = None
        self.n_docs = 0
        self.learned: dict[str, list[tuple[str, float]]] = {}

    # ---- indexing -------------------------------------------------------------------------
    def fit(self, catalog: pd.DataFrame) -> "SparseRetriever":
        self.n_docs = len(catalog)
        fields = {"title": catalog.title, "brand": catalog.brand, "attributes": catalog.attributes,
                  "description": catalog.description}
        cv = CountVectorizer(analyzer=lambda s: tokens(s), lowercase=False)
        mats, lens = {}, {}
        all_docs = (catalog.title + " " + catalog.brand + " " + catalog.attributes + " " + catalog.description)
        cv.fit(all_docs.values)
        self.vocab = cv.vocabulary_
        for name, col in fields.items():
            m = cv.transform(col.astype(str).values).astype(np.float32).tocsr()
            mats[name] = m
            lens[name] = np.asarray(m.sum(1)).ravel()
        tf = sum(self.fw[n] * mats[n] for n in fields)
        dl = sum(self.fw[n] * lens[n] for n in fields)
        avgdl = dl.mean()
        df_t = np.asarray((tf > 0).sum(0)).ravel()
        self.idf = np.log(1 + (self.n_docs - df_t + 0.5) / (df_t + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        norm = self.k1 * (1 - self.b + self.b * dl[tf.row] / avgdl)
        w = tf.data * (self.k1 + 1) / (tf.data + norm) * self.idf[tf.col]
        W = sp.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.Wt = W.T.tocsr()
        # char n-grams over title+brand (model numbers, partial morphological matches)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, dtype=np.float32)
        self.char_mat = self.char_vec.fit_transform((catalog.title + " " + catalog.brand).values).tocsr()
        return self

    def fit_learned_expansion(self, catalog: pd.DataFrame, sessions: pd.DataFrame, queries: pd.DataFrame,
                              top_k: int = 4, min_count: int = 15) -> None:
        """SPLADE-style *learned* sparse expansion: map each query term to the catalog terms that
        co-occur in the titles of purchased/clicked products (click-supervised, train queries only)."""
        sku_pos = {s: i for i, s in enumerate(catalog.sku_id.values)}
        q_text = dict(zip(queries.query_id, queries.text))
        title_toks = [set(tokens(t)) for t in (catalog.title + " " + catalog.attributes).values]
        dfc: Counter = Counter(t for s in title_toks for t in s)
        banned = {t for b in catalog.brand.unique() for t in tokens(b)} | {t for c in catalog.color.unique() for t in tokens(c)}
        ok = {t for t, c in dfc.items() if c >= 300 and t.isalpha() and t not in banned}
        title_toks = [s & ok for s in title_toks]
        pos = sessions[(sessions.add_to_cart + sessions.purchase) > 0]
        pair_cnt: dict[str, Counter] = defaultdict(Counter)
        q_cnt: Counter = Counter()
        d_cnt: Counter = Counter()
        n = 0
        for qid, sku, w in zip(pos.query_id.values, pos.sku_id.values, 1 + pos.purchase.values):
            qt = set(tokens(q_text.get(qid, ""))) & ok
            dt = title_toks[sku_pos[sku]]
            n += w
            for a in qt:
                q_cnt[a] += w
                for d in dt:
                    pair_cnt[a][d] += w
            for d in dt:
                d_cnt[d] += w
        for a, ctr in pair_cnt.items():
            cands = []
            for d, c in ctr.items():
                if c < min_count or d == a:
                    continue
                pmi = np.log((c * n) / (q_cnt[a] * d_cnt[d]))
                p_d_given_a = c / q_cnt[a]
                if pmi > 0.4 and p_d_given_a > 0.25:
                    cands.append((d, float(p_d_given_a * min(pmi, 2.0))))
            cands.sort(key=lambda x: -x[1])
            self.learned[a] = cands[:top_k]

    def learned_terms(self, query_terms: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in query_terms:
            for d, w in self.learned.get(t, []):
                out[d] = max(out.get(d, 0.0), min(0.5, w * 0.5))
        return out

    # ---- scoring -----------------------------------------------------------------------------
    def bm25_all(self, weighted_terms: dict[str, float]) -> np.ndarray:
        idx, w = [], []
        for t, wt in weighted_terms.items():
            j = self.vocab.get(t)
            if j is not None:
                idx.append(j)
                w.append(wt)
        if not idx:
            return np.zeros(self.n_docs, dtype=np.float32)
        sub = self.Wt[idx]
        return np.asarray(sub.T @ np.asarray(w, dtype=np.float32)).ravel()

    def char_all(self, text: str) -> np.ndarray:
        q = self.char_vec.transform([text])
        return np.asarray((self.char_mat @ q.T).todense()).ravel()

    def search(self, weighted_terms: dict[str, float], raw_text: str, n: int = 200, char_weight: float = 0.35):
        bm = self.bm25_all(weighted_terms)
        ch = self.char_all(raw_text)
        bmn = bm / (bm.max() + 1e-9)
        score = bmn + char_weight * ch / (ch.max() + 1e-9)
        score[(bm <= 0) & (ch < 0.15)] = 0.0
        if n < len(score):
            top = np.argpartition(-score, n)[:n]
        else:
            top = np.arange(len(score))
        top = top[np.argsort(-score[top])]
        top = top[score[top] > 0]
        return top, score[top], bm, ch
