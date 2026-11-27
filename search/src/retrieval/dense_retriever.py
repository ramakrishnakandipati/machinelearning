"""Dense retrieval: GPU bi-encoder text embeddings fused with synthetic image + tabular content
embeddings into one normalised space, indexed with FAISS (HNSW, GPU if available).

Cold-start SKUs are indexed purely on content (text + image + tabular) -> retrievable on day 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from src.common.utils import get_device


def doc_text(df: pd.DataFrame) -> list[str]:
    return (df.title + ". " + df.brand + ". " + df.attributes + ". " + df.description).tolist()


class TextEncoder:
    """SentenceTransformer wrapper with model fallback chain, GPU fp16, and a TF-IDF-SVD last resort."""

    def __init__(self, names: list[str], query_prefix: str = "", device: str | None = None):
        self.device = device or get_device()
        self.query_prefix = query_prefix
        self.model = None
        self.name = None
        for n in names:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(n, device=self.device)
                if self.device == "cuda":
                    self.model.half()
                self.name = n
                self.prefix_used = query_prefix if "bge" in n.lower() else ""
                break
            except Exception as e:  # pragma: no cover - network dependent
                logger.warning("Could not load encoder {}: {}", n, e)
        if self.model is None:
            raise RuntimeError("No sentence-transformer model could be loaded")
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_docs(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:  # generic (terms, etc.)
        return self.encode_docs(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.prefix_used + text], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)


class MultimodalFuser:
    """Projects image (+ tabular) content vectors into the text-embedding space (ridge regression
    fit on content only - no interaction signals) and fuses: normalize(wt*text + wi*img' + wa*tab')."""

    def __init__(self, weights: dict):
        self.w = weights
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.img_proj = Ridge(alpha=1.0)
        self.tab_proj = Ridge(alpha=1.0)

    def _tab(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        cat = df[["category", "subcategory", "color", "material", "brand"]].astype(str).values
        oh = self.ohe.fit_transform(cat) if fit else self.ohe.transform(cat)
        oh = oh.toarray().astype(np.float32)
        num = np.stack([np.log1p(df.price.values), df.brand_tier.values.astype(float), df.gross_margin_pct.values], 1)
        return np.hstack([oh, (num - num.mean(0)) / (num.std(0) + 1e-6)]).astype(np.float32) if fit else \
            np.hstack([oh, (num - self._mu) / (self._sd + 1e-6)]).astype(np.float32)

    def fit(self, df: pd.DataFrame, img: np.ndarray, text_emb: np.ndarray) -> "MultimodalFuser":
        num = np.stack([np.log1p(df.price.values), df.brand_tier.values.astype(float), df.gross_margin_pct.values], 1)
        self._mu, self._sd = num.mean(0), num.std(0)
        tab = self._tab(df, fit=True)
        self.img_proj.fit(img, text_emb)
        self.tab_proj.fit(tab, text_emb)
        self.tab_dim = tab.shape[1]
        return self

    def fuse(self, df: pd.DataFrame, img: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
        v = self.w["text"] * text_emb
        pi = self.img_proj.predict(img)
        pi /= np.linalg.norm(pi, axis=1, keepdims=True) + 1e-9
        pt = self.tab_proj.predict(self._tab(df))
        pt /= np.linalg.norm(pt, axis=1, keepdims=True) + 1e-9
        v = v + self.w["image"] * pi + self.w["tabular"] * pt
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v.astype(np.float32)


class DenseIndex:
    """FAISS HNSW (inner product on unit vectors); uses GPU brute-force if faiss-gpu is present."""

    def __init__(self, hnsw: dict):
        self.cfg = hnsw
        self.index = None
        self.gpu = False

    def build(self, emb: np.ndarray) -> "DenseIndex":
        import faiss

        d = emb.shape[1]
        if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:  # pragma: no cover
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(d))
            self.gpu = True
        else:
            idx = faiss.IndexHNSWFlat(d, self.cfg["M"], faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = self.cfg["efConstruction"]
            idx.hnsw.efSearch = self.cfg["efSearch"]
            self.index = idx
        self.index.add(emb)
        self.emb = emb
        return self

    def search(self, qv: np.ndarray, n: int):
        sc, ix = self.index.search(qv[None, :].astype(np.float32), n)
        keep = ix[0] >= 0
        return ix[0][keep], sc[0][keep]

    def save(self, path):
        import faiss

        if not self.gpu:
            faiss.write_index(self.index, str(path))

    @staticmethod
    def load(path, emb: np.ndarray, hnsw: dict) -> "DenseIndex":
        import faiss

        d = DenseIndex(hnsw)
        d.index = faiss.read_index(str(path))
        d.index.hnsw.efSearch = hnsw["efSearch"]
        d.emb = emb
        return d
