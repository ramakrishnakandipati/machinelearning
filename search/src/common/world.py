"""Loads the generated world and exposes aligned numpy views used by every stage."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from src.common.text import tokens
from src.common.utils import resolve
from src.data_gen import generate_retail_data as G

REF_DATE = pd.Timestamp("2026-11-01")


class World:
    def __init__(self, cfg: dict, load_latent: bool = False):
        d = resolve(cfg, "data")
        self.cfg = cfg
        self.catalog = pd.read_parquet(d / "catalog.parquet")
        self.inventory = pd.read_parquet(d / "inventory.parquet")
        self.users = pd.read_parquet(d / "users.parquet")
        self.queries = pd.read_parquet(d / "queries.parquet")
        self.stores = pd.read_parquet(d / "stores.parquet")
        self.img = np.load(d / "image_embeddings.npy")
        with open(d / "synonyms.json") as f:
            self.synonyms = json.load(f)
        self.sku_index = {s: i for i, s in enumerate(self.catalog.sku_id.values)}
        self.user_index = {u: i for i, u in enumerate(self.users.user_id.values)}
        self.query_index = {q: i for i, q in enumerate(self.queries.query_id.values)}
        self.store_ids = [s for s in self.stores.store_id.values]
        self._prep()
        self.latent = None
        self._true_arr = self._claimed_arr = None
        if load_latent:
            self.load_latent(d)

    # ---- aligned document arrays ---------------------------------------------------------------
    def _prep(self):
        c = self.catalog
        self.n = len(c)
        self.price = c.price.values.astype(np.float32)
        self.margin_pct = c.gross_margin_pct.values.astype(np.float32)
        self.margin_usd = c.gross_margin.values.astype(np.float32)
        self.return_rate = c.return_rate.values.astype(np.float32)
        self.cold = c.is_cold_start.values.astype(bool)
        self.brand = c.brand.values
        self.color = c.color.values
        self.category = c.category.values
        self.subcat = c.subcategory.values
        self.sub_idx = c.sub_idx.values
        self.tier = c.brand_tier.values
        self.days_since_launch = (REF_DATE - pd.to_datetime(c.created_at)).dt.days.values.astype(np.float32)
        # Bayesian-smoothed popularity priors (category prior, strength m)
        cat_ctr = c.groupby("category").apply(lambda g: g.clicks.sum() / max(g.impressions.sum(), 1), include_groups=False)
        cat_cvr = c.groupby("category").apply(lambda g: g.purchases.sum() / max(g.clicks.sum(), 1), include_groups=False)
        pc, pv = c.category.map(cat_ctr).values, c.category.map(cat_cvr).values
        self.bayes_ctr = ((c.clicks.values + 60 * pc) / (c.impressions.values + 60)).astype(np.float32)
        self.bayes_cvr = ((c.purchases.values + 25 * pv) / (c.clicks.values + 25)).astype(np.float32)
        self.log_impr = np.log1p(c.impressions.values).astype(np.float32)
        inv = self.inventory
        self.stock = {s: inv[f"stock_{s}"].values.astype(np.int32) for s in self.store_ids}
        self.doc_tokens = [set(tokens(t)) for t in (c.title + " " + c.attributes).values]
        sizes = {s: i for i, s in enumerate(sorted(set(self.subcat)))}
        self.sub_code = np.array([sizes[s] for s in self.subcat])
        self.users_pref = [set(json.loads(p)) for p in self.users.preferred_brands.values]
        self.users_aff = [json.loads(p) for p in self.users.category_affinity.values]
        self.users_recent = [set(json.loads(p)) for p in self.users.recent_purchases.values]
        self.user_median = self.users.median_spend.values.astype(np.float32)
        self.user_tier = self.users.price_tier.values
        self.user_store = self.users.home_store.values
        self.recent_sub = [{self.subcat[self.sku_index[s]] for s in r if s in self.sku_index} for r in self.users_recent]
        self.store_ll = {r.store_id: (r.lat, r.lon) for r in self.stores.itertuples()}

    # ---- evaluation-only ground truth -------------------------------------------------------------
    def load_latent(self, d=None):
        d = d or resolve(self.cfg, "data")
        self.latent = pd.read_parquet(d / "catalog_latent.parquet")
        with open(d / "true_features.json") as f:
            self._true_arr = G.feature_arrays([set(x) for x in json.load(f)])
        with open(d / "claimed_features.json") as f:
            self._claimed_arr = G.feature_arrays([set(x) for x in json.load(f)])
        self.l_quality = self.latent.latent_quality.values
        self.l_ret = self.latent.latent_return_base.values
        self.l_mis = self.latent.is_misleading.values
        self._subs_index = {s: i for i, s in enumerate(G.SUBS)}

    @lru_cache(maxsize=4096)
    def rel_true(self, query_id: str) -> np.ndarray:
        q = self.queries.iloc[self.query_index[query_id]]
        return G.relevance_for_query(q, self.sub_idx, self._true_arr, self.brand, self.color, self.price, self._subs_index)

    @lru_cache(maxsize=4096)
    def rel_claimed(self, query_id: str) -> np.ndarray:
        q = self.queries.iloc[self.query_index[query_id]]
        return G.relevance_for_query(q, self.sub_idx, self._claimed_arr, self.brand, self.color, self.price, self._subs_index)

    def true_probs(self, query_id: str, user_idx: int, sku_idx: np.ndarray):
        """Latent per-impression funnel probabilities at position 0 (exam=1)."""
        rt = self.rel_true(query_id)[sku_idx].astype(float)
        rc = self.rel_claimed(query_id)[sku_idx].astype(float)
        bm = np.array([self.brand[i] in self.users_pref[user_idx] for i in sku_idx], dtype=float)
        pf = G.price_fit(self.price[sku_idx], float(self.user_median[user_idx]))
        return G.latent_probs(rt, rc, self.l_quality[sku_idx], bm, pf, self.l_ret[sku_idx], exam=1.0)

    # ---- store logistics ----------------------------------------------------------------------------
    def distance_km(self, a: str, b: str) -> float:
        (la1, lo1), (la2, lo2) = self.store_ll[a], self.store_ll[b]
        r = 6371.0
        p1, p2 = np.radians(la1), np.radians(la2)
        dphi, dl = p2 - p1, np.radians(lo2 - lo1)
        h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return float(2 * r * np.arcsin(np.sqrt(h)))

    def store_for_zip(self, zip_code: str | None) -> str | None:
        if not zip_code:
            return None
        m = self.stores[self.stores.zip_code == str(zip_code)]
        return m.store_id.iloc[0] if len(m) else None
