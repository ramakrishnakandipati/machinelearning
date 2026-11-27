"""Rule + gazetteer attribute extraction and a TF-IDF/logreg intent classifier."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.common.text import stem, tokens
from src.data_gen import taxonomy as T

PRICE_RE = re.compile(r"(?:under|below|less than|max|up to|<=?|cheaper than)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
PRICE_RE2 = re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(?:or less|max)", re.I)
GENDER = {"men": "men", "mens": "men", "women": "women", "womens": "women", "ladies": "women",
          "kids": "kids", "boys": "kids", "girls": "kids"}


@dataclass
class ParsedQuery:
    raw: str
    corrected: str
    brand: str = ""
    color: str = ""
    subcategories: list = field(default_factory=list)
    category: str = ""
    feature_terms: list = field(default_factory=list)
    max_price: float = 0.0
    gender: str = ""
    intent_type: str = "category_browsing"
    intent_conf: float = 0.0

    def facets(self) -> dict:
        return {"brand": self.brand, "color": self.color, "category": self.category,
                "subcategories": self.subcategories, "max_price": self.max_price,
                "gender": self.gender, "intent_type": self.intent_type}


class QueryNER:
    def __init__(self, synonyms: dict | None = None):
        self.brands: dict[str, str] = {}
        self.colors: set[str] = set()
        self.sub_terms: dict[str, set] = {}
        self.sub_cat: dict[str, str] = {}
        self.feat_terms: set[str] = set()
        self.synonyms = synonyms or {}
        self.clf = None

    def fit(self, catalog: pd.DataFrame, train_queries: pd.DataFrame | None = None) -> "QueryNER":
        for b in catalog.brand.unique():
            self.brands[b.lower()] = b
        self.colors = set(catalog.color.unique())
        for sub, cat in catalog[["subcategory", "category"]].drop_duplicates().values:
            terms = set(tokens(sub))
            ty = T.TYPE_BY_SUB.get(sub)
            if ty:
                for n in ty["nouns"]:
                    terms |= set(tokens(n))
                for f in ty["feats"]:
                    self.feat_terms |= set(tokens(f))
            for k, subs in T.CATEGORY_TERMS.items():
                if sub in subs:
                    terms |= set(tokens(k))
            self.sub_terms[sub] = terms
            self.sub_cat[sub] = cat
        if train_queries is not None and len(train_queries):
            self.clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                                     LogisticRegression(max_iter=400, C=4.0))
            self.clf.fit(train_queries.text.values, train_queries.intent_type.values)
        return self

    def parse(self, raw: str, corrected: str, expanded_tokens: list[str] | None = None) -> ParsedQuery:
        pq = ParsedQuery(raw=raw, corrected=corrected)
        text = corrected.lower()
        m = PRICE_RE.search(text) or PRICE_RE2.search(text)
        if m:
            pq.max_price = float(m.group(1))
        padded = f" {text} "
        for bl, b in self.brands.items():
            if f" {bl} " in padded:
                pq.brand = b
                break
        for c in self.colors:
            if f" {c} " in padded:
                pq.color = c
                break
        for w, g in GENDER.items():
            if f" {w} " in padded:
                pq.gender = g
                break
        qtoks = set(tokens(text))
        if expanded_tokens:
            qtoks_ext = qtoks | set(expanded_tokens)
        else:
            qtoks_ext = qtoks
        scores = {}
        for sub, terms in self.sub_terms.items():
            direct = len(qtoks & terms)
            ext = len(qtoks_ext & terms)
            s = direct + 0.5 * ext
            if s > 0:
                scores[sub] = s / (1.0 + 0.15 * len(terms))
        if scores:
            top = max(scores.values())
            pq.subcategories = [s for s, v in scores.items() if v >= 0.85 * top][:3]
            pq.category = self.sub_cat[pq.subcategories[0]]
        pq.feature_terms = sorted(qtoks & self.feat_terms)
        if self.clf is not None:
            proba = self.clf.predict_proba([corrected])[0]
            k = int(proba.argmax())
            pq.intent_type = str(self.clf.classes_[k])
            pq.intent_conf = float(proba[k])
        return pq
