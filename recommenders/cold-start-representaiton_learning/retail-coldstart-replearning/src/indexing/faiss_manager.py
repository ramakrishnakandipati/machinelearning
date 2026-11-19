"""FAISS ANN index builder, metadata-filtered search and latency / recall benchmarker.

Indexes: IndexFlatIP (exact baseline), IndexHNSWFlat (inner product), IndexIVFPQ (compressed).
All vectors are L2-normalised so inner product == cosine similarity.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np


class FaissManager:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.flat = self.hnsw = self.ivfpq = None
        self.meta = None

    # ------------------------------------------------------------------ build
    def build(self, vectors: np.ndarray, meta: dict | None = None, hnsw_m=32, ef_construction=200, ef_search=128,
              ivfpq=(32, 16, 8)):
        x = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(x)
        self.x = x
        self.flat = faiss.IndexFlatIP(self.dim)
        self.flat.add(x)
        self.hnsw = faiss.IndexHNSWFlat(self.dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.hnsw.hnsw.efConstruction = ef_construction
        self.hnsw.add(x)
        self.hnsw.hnsw.efSearch = ef_search
        nlist, m, nbits = ivfpq
        quant = faiss.IndexFlatIP(self.dim)
        self.ivfpq = faiss.IndexIVFPQ(quant, self.dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        self.ivfpq.train(x)
        self.ivfpq.add(x)
        self.ivfpq.nprobe = 8
        self.meta = meta or {}
        return self

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.flat, str(out_dir / "items_flat.index"))
        faiss.write_index(self.hnsw, str(out_dir / "items_hnsw.index"))
        faiss.write_index(self.ivfpq, str(out_dir / "items_ivfpq.index"))

    @staticmethod
    def load(out_dir: Path, meta: dict | None = None) -> "FaissManager":
        fm = FaissManager()
        fm.flat = faiss.read_index(str(out_dir / "items_flat.index"))
        fm.hnsw = faiss.read_index(str(out_dir / "items_hnsw.index"))
        fm.ivfpq = faiss.read_index(str(out_dir / "items_ivfpq.index"))
        fm.dim = fm.flat.d
        fm.meta = meta or {}
        return fm

    # ------------------------------------------------------------------ search
    def search(self, q: np.ndarray, k: int, kind: str = "hnsw", allowed: np.ndarray | None = None, exclude: set | None = None):
        """Top-k with optional metadata filter (boolean mask over item ids) and exclusion set.
        Filtering is done by over-fetching and doubling the fetch size until k survivors are found."""
        idx = {"flat": self.flat, "hnsw": self.hnsw, "ivfpq": self.ivfpq}[kind]
        q = np.ascontiguousarray(np.atleast_2d(q), dtype=np.float32)
        n_total = idx.ntotal
        fetch = k if (allowed is None and not exclude) else min(n_total, max(4 * k, 50))
        while True:
            fetch = min(fetch, n_total)
            if kind == "hnsw":
                self.hnsw.hnsw.efSearch = max(self.hnsw.hnsw.efSearch, fetch)
            D, I = idx.search(q, fetch)
            outs_i, outs_d, ok = [], [], True
            for d_row, i_row in zip(D, I):
                keep = [(d, i) for d, i in zip(d_row, i_row) if i >= 0 and (allowed is None or allowed[i]) and not (exclude and i in exclude)]
                if len(keep) < k and fetch < n_total:
                    ok = False
                    break
                keep = keep[:k]
                outs_i.append([i for _, i in keep]); outs_d.append([d for d, _ in keep])
            if ok:
                return outs_d, outs_i
            fetch *= 2

    # ------------------------------------------------------------------ benchmark
    def benchmark(self, queries: np.ndarray, ks=(10,), ef_sweep=(8, 16, 32, 64, 128, 256), nprobe_sweep=(1, 2, 4, 8, 16, 32),
                  out_path: Path | None = None, repeats: int = 1) -> dict:
        faiss.omp_set_num_threads(1)
        q = np.ascontiguousarray(queries, dtype=np.float32)
        faiss.normalize_L2(q)
        k = max(ks)
        _, truth = self.flat.search(q, k)

        def run(idx, label, params):
            lats = []
            ids = np.zeros((len(q), k), dtype=np.int64)
            for r in range(repeats):
                for n in range(len(q)):
                    t = time.perf_counter()
                    _, I = idx.search(q[n:n + 1], k)
                    lats.append((time.perf_counter() - t) * 1000)
                    ids[n] = I[0]
            rec = float(np.mean([len(set(ids[n][:10]) & set(truth[n][:10])) / 10 for n in range(len(q))]))
            return {"index": label, **params, "recall@10_vs_exact": rec, "p50_ms": float(np.percentile(lats, 50)),
                    "p95_ms": float(np.percentile(lats, 95)), "p99_ms": float(np.percentile(lats, 99)), "mean_ms": float(np.mean(lats))}

        res = {"n_items": int(self.flat.ntotal), "dim": self.dim, "n_queries": int(len(q)), "results": []}
        res["results"].append(run(self.flat, "IndexFlatIP", {}))
        for ef in ef_sweep:
            self.hnsw.hnsw.efSearch = ef
            res["results"].append(run(self.hnsw, "IndexHNSWFlat", {"efSearch": ef}))
        for npb in nprobe_sweep:
            self.ivfpq.nprobe = npb
            res["results"].append(run(self.ivfpq, "IndexIVFPQ", {"nprobe": npb}))
        res["index_bytes"] = {"flat": int(self.flat.ntotal * self.dim * 4), "ivfpq_code_bytes": int(self.ivfpq.ntotal * self.ivfpq.pq.code_size)}
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(out_path, "w"), indent=2)
        return res
