"""Query expansion for the vocabulary gap: curated synonym graph + embedding nearest neighbours."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.common.text import raw_tokens, tokens


class QueryExpander:
    def __init__(self, synonyms: dict, n_neighbors: int = 3, min_cos: float = 0.45):
        self.syn = {k.lower(): v for k, v in synonyms.items()}
        self.n_neighbors = n_neighbors
        self.min_cos = min_cos
        self.vocab_terms: list[str] = []
        self.vocab_emb: np.ndarray | None = None

    def build_vocab(self, catalog: pd.DataFrame, encode_fn, max_terms: int = 600) -> None:
        """Candidate expansion terms: frequent title/attribute uni+bi-grams, embedded once offline."""
        cnt: Counter = Counter()
        for text in (catalog.title.astype(str) + " " + catalog.attributes.astype(str)).values:
            tk = [t for t in raw_tokens(text) if len(t) > 2 and not t.replace(".", "").isdigit()]
            cnt.update(tk)
            cnt.update(" ".join(p) for p in zip(tk, tk[1:]))
        brands = {b.lower() for b in catalog.brand.unique()}
        colors = {c for c in catalog.color.unique()}
        terms = [t for t, c in cnt.most_common(4000)
                 if c >= 25 and not any(w in brands for w in t.split()) and not any(c_ in t for c_ in colors)]
        self.vocab_terms = terms[:max_terms]
        self.vocab_emb = encode_fn(self.vocab_terms).astype(np.float32)

    def expand(self, query: str, query_vec: np.ndarray | None = None) -> dict:
        """Return {'synonyms': [...], 'neighbors': [...], 'weighted_terms': {term: weight}}."""
        base = tokens(query)
        weights: dict[str, float] = {t: 1.0 for t in base}
        syns: list[str] = []
        for raw in raw_tokens(query):
            for s in self.syn.get(raw, []):
                syns.append(s)
                for t in tokens(s):
                    weights[t] = max(weights.get(t, 0.0), 0.6)
        nbrs: list[str] = []
        if query_vec is not None and self.vocab_emb is not None:
            sims = self.vocab_emb @ query_vec
            order = np.argsort(-sims)[: self.n_neighbors * 3]
            for j in order:
                term = self.vocab_terms[j]
                if sims[j] < self.min_cos or len(nbrs) >= self.n_neighbors:
                    continue
                if all(t in weights for t in tokens(term)):
                    continue
                nbrs.append(term)
                for t in tokens(term):
                    weights[t] = max(weights.get(t, 0.0), 0.35)
        return {"synonyms": syns, "neighbors": nbrs, "weighted_terms": weights}
