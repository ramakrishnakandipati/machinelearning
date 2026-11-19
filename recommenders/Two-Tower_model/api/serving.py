"""Model bundle (user tower + FAISS + metadata) and the retrieval / filtering logic used by the endpoints."""
from __future__ import annotations

import datetime as dt
import time

import faiss
import joblib
import numpy as np
import pandas as pd
import torch

from src.config import MODELS
from src.data_generator import CHANNEL_DEVICE
from src.train import U_KEYS

from .feature_store import FeatureStore


class ModelBundle:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> "ModelBundle":
        pp = joblib.load(MODELS / "preprocessors.joblib")
        self.pre = pp["pre"]
        self.tower = torch.jit.load(str(MODELS / "user_tower_serving.pt"), map_location=self.device).eval()
        self.index = faiss.read_index(str(MODELS / "retail_items.faiss"))
        self.item_emb = np.load(MODELS / "item_embeddings.npy")
        seg = np.load(MODELS / "segment_embeddings.npz")
        self.segments = {k: seg[k] for k in seg.files}
        meta = pd.read_parquet(MODELS / "item_metadata.parquet")
        self.meta = meta
        self.cat1 = meta.category_l1.to_numpy()
        self.cat1_lower = np.char.lower(self.cat1.astype(str))
        self.cat2, self.brand = meta.category_l2.to_numpy(), meta.brand.to_numpy()
        self.price, self.in_stock = meta.price.to_numpy(), meta.in_stock.to_numpy()
        self.pop = np.log1p(meta.popularity_7d.to_numpy())
        self.pop = self.pop / max(self.pop.max(), 1e-9)
        self.n_items = len(meta)
        mt = (MODELS / "two_tower_best.pt").stat().st_mtime
        self.version = f"two-tower-{int(mt)}"
        warm = {"channel": "Web", "tier": "Bronze", "region": "North", "recency_days": 5.0, "frequency_90d": 3.0,
                "monetary_value_90d": 50.0, "avg_basket_size": 1.0, "history": [1, 2, 3]}
        for _ in range(5):   # CUDA / cuDNN warm-up so the first real request is not slow
            self.embed_user(warm, None, 0)
        return self

    # ------------------------------------------------------------------ helpers
    def embed_user(self, feats: dict, channel: str | None, user_id: int) -> np.ndarray:
        now = dt.datetime.now()
        ch = channel or feats["channel"]
        d = np.array([[feats["recency_days"], feats["frequency_90d"], feats["monetary_value_90d"], feats["avg_basket_size"]]], float)
        hist = np.full((1, self.pre.hist_len), -1)
        h = feats["history"][-self.pre.hist_len:]
        if len(h):
            hist[0, -len(h):] = h
        enc = self.pre.encode_users([user_id], [feats["tier"]], [feats["channel"]], [feats["region"]], d, hist,
                                    [now.weekday()], [now.hour], [CHANNEL_DEVICE[ch][0]])
        with torch.no_grad():
            u = self.tower(*[torch.as_tensor(enc[k]).to(self.device) for k in U_KEYS])
        return u.cpu().numpy().astype(np.float32)

    def _search(self, q: np.ndarray, k: int):
        t = time.perf_counter()
        D, I = self.index.search(q, k)
        return D[0], I[0], (time.perf_counter() - t) * 1000

    def _filter(self, ids, scores, top_k, exclude: set[int], category: str | None, in_stock_only: bool):
        out = []
        cat = category.lower() if category else None
        for i, s in zip(ids, scores):
            if i < 0 or i in exclude:
                continue
            if in_stock_only and not self.in_stock[i]:
                continue
            if cat and self.cat1_lower[i] != cat:
                continue
            out.append({"item_id": int(i), "score": float(s), "category_l1": str(self.cat1[i]), "category_l2": str(self.cat2[i]),
                        "brand": str(self.brand[i]), "price": float(self.price[i]), "in_stock": bool(self.in_stock[i])})
            if len(out) >= top_k:
                break
        return out

    def retrieve(self, q: np.ndarray, top_k: int, exclude: set[int], category, in_stock_only, rerank_pop: float = 0.0):
        """Over-fetch 3x top_k for post-retrieval filtering; widen (x2) if filters leave too few."""
        k, ann_ms = min(3 * top_k, self.n_items), 0.0
        while True:
            D, I, ms = self._search(q, k)
            ann_ms += ms
            if rerank_pop:
                D = D + rerank_pop * self.pop[np.clip(I, 0, None)]
                o = np.argsort(-D)
                D, I = D[o], I[o]
            items = self._filter(I, D, top_k, exclude, category, in_stock_only)
            if len(items) >= top_k or k >= self.n_items:
                return items, ann_ms
            k = min(k * 2, self.n_items)

    # ----------------------------------------------------------------- public
    def recommend_user(self, store: FeatureStore, user_id: str, top_k, channel, exclude_purchased, category, in_stock_only):
        t0 = time.perf_counter()
        uid = int(user_id) if user_id.isdigit() else -1
        feats = store.get_user(uid) if uid >= 0 else None
        exclude: set[int] = store.purchased(uid) if (feats and exclude_purchased) else set()
        t1 = time.perf_counter()
        if feats is not None:
            q, strategy, rerank = self.embed_user(feats, channel, uid), "personalized", 0.0
        else:   # cold start: channel segment embedding (mean of warm users) + popularity-weighted re-rank
            q = self.segments.get(channel or "global", self.segments["global"])[None, :].astype(np.float32)
            strategy, rerank = "cold_start_segment", 0.05
        inf_ms = (time.perf_counter() - t1) * 1000 + (t1 - t0) * 1000
        items, ann_ms = self.retrieve(q, top_k, exclude, category, in_stock_only, rerank)
        return {"strategy": strategy, "cold_start": feats is None, "model_version": self.version, "items": items,
                "telemetry": {"inference_ms": round(inf_ms, 3), "ann_search_ms": round(ann_ms, 3),
                              "total_ms": round((time.perf_counter() - t0) * 1000, 3)}}

    def similar_items(self, item_id: int, top_k, category, in_stock_only):
        t0 = time.perf_counter()
        q = self.item_emb[item_id:item_id + 1]
        items, ann_ms = self.retrieve(q, top_k, {item_id}, category, in_stock_only)
        total = (time.perf_counter() - t0) * 1000
        return {"strategy": "item_similarity", "cold_start": False, "model_version": self.version, "items": items,
                "telemetry": {"inference_ms": round(total - ann_ms, 3), "ann_search_ms": round(ann_ms, 3), "total_ms": round(total, 3)}}
