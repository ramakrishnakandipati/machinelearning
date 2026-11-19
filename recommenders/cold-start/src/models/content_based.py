"""Strategy 2 - content-based semantic retrieval over a FAISS index (new-item cold-start)."""
from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


class ItemIndex:
    """Thread-safe inner-product item index with live insertion.

    Uses ``faiss-gpu`` when available (``use_gpu``), otherwise ``faiss-cpu``; if FAISS is missing
    altogether a NumPy exact-search fallback keeps the system functional.
    """

    def __init__(self, dim: int, use_gpu: bool = True):
        self.dim = dim
        self._lock = threading.RLock()
        self._vecs = np.zeros((0, dim), np.float32)
        self.gpu = False
        self._res = None
        self.index = None
        if faiss is not None:
            self.index = faiss.IndexFlatIP(dim)
            if use_gpu and hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
                try:
                    self._res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(self._res, 0, self.index)
                    self.gpu = True
                except Exception:  # pragma: no cover
                    self.index, self.gpu = faiss.IndexFlatIP(dim), False
        self.backend = ("faiss-gpu" if self.gpu else "faiss-cpu") if self.index is not None else "numpy"

    @property
    def size(self) -> int:
        return int(self._vecs.shape[0])

    def build(self, vectors: np.ndarray) -> None:
        with self._lock:
            self._vecs = np.ascontiguousarray(vectors, dtype=np.float32)
            if self.index is not None:
                self.index.reset()
                self.index.add(self._vecs)

    def add(self, vec: np.ndarray) -> int:
        """Append one vector; returns its position. Searches in flight are unaffected (lock-protected)."""
        v = np.ascontiguousarray(vec, dtype=np.float32).reshape(1, -1)
        with self._lock:
            pos = self.size
            self._vecs = np.vstack([self._vecs, v])
            if self.index is not None:
                self.index.add(v)
            return pos

    def search(self, q: np.ndarray, k: int, allowed: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k (positions, scores) for one query vector; optional boolean ``allowed`` mask over positions."""
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(1, -1)
        with self._lock:
            n = self.size
            if n == 0:
                return np.zeros(0, int), np.zeros(0, np.float32)
            ks = n if allowed is not None else min(k, n)
            if self.index is not None:
                sc, ix = self.index.search(q, ks)
                sc, ix = sc[0], ix[0]
            else:
                s = (self._vecs @ q[0])
                ix = np.argsort(-s)[:ks]
                sc = s[ix]
        keep = ix >= 0
        if allowed is not None:
            a = np.zeros(max(n, len(allowed)), bool)
            a[: len(allowed)] = allowed
            keep &= a[np.clip(ix, 0, len(a) - 1)]
        ix, sc = ix[keep][:k], sc[keep][:k]
        return ix.astype(int), sc.astype(np.float32)

    def score_all(self, q: np.ndarray) -> np.ndarray:
        """Similarity of the query to every indexed item (position-aligned)."""
        ix, sc = self.search(q, self.size)
        out = np.full(self.size, -1.0, np.float32)
        out[ix] = sc
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if faiss is not None:
                cpu = faiss.index_gpu_to_cpu(self.index) if self.gpu else self.index
                faiss.write_index(cpu, str(path))
            else:  # pragma: no cover
                with open(path, "wb") as f:
                    pickle.dump(self._vecs, f)

    @classmethod
    def load(cls, path: str | Path, use_gpu: bool = False) -> "ItemIndex":
        if faiss is not None:
            cpu = faiss.read_index(str(path))
            obj = cls(cpu.d, use_gpu)
            vecs = cpu.reconstruct_n(0, cpu.ntotal)
            obj.build(vecs)
            return obj
        with open(path, "rb") as f:  # pragma: no cover
            vecs = pickle.load(f)
        obj = cls(vecs.shape[1], False)
        obj.build(vecs)
        return obj


class ContentBasedRecommender:
    """Retrieves items semantically similar to a session/onboarding profile vector."""

    def __init__(self, index: ItemIndex):
        self.index = index

    def recommend(self, profile: Optional[np.ndarray], k: int, allowed: Optional[np.ndarray] = None,
                  exclude: Optional[set] = None) -> Tuple[List[int], List[float], float]:
        """Return (positions, scores, latency_ms)."""
        if profile is None:
            return [], [], 0.0
        t = time.perf_counter()
        extra = len(exclude) if exclude else 0
        ix, sc = self.index.search(profile, k + extra, allowed)
        pairs = [(int(i), float(s)) for i, s in zip(ix, sc) if not exclude or int(i) not in exclude][:k]
        ms = (time.perf_counter() - t) * 1000
        return [p[0] for p in pairs], [p[1] for p in pairs], ms

    def score_all(self, profile: Optional[np.ndarray], n: int) -> np.ndarray:
        if profile is None:
            return np.zeros(n, np.float32)
        s = self.index.score_all(profile)
        return s if len(s) >= n else np.pad(s, (0, n - len(s)), constant_values=-1.0)
