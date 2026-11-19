"""FAISS ANN indexing + benchmark: exact IndexFlatIP vs IndexHNSWFlat vs IndexIVFPQ (inner product, L2-normed)."""
from __future__ import annotations

import time

import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402

from .config import CHARTS, METRICS, MODELS, SEED  # noqa: E402
from .logger import get_logger, stage_timer  # noqa: E402

log = get_logger("ann", "04_ann_indexing.log")
GPU = hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0 and hasattr(faiss, "StandardGpuResources")
_RES = faiss.StandardGpuResources() if GPU else None


def maybe_gpu(index, name: str):
    """Move to GPU when faiss-gpu is present. IndexHNSW has no GPU implementation in FAISS -> stays on CPU."""
    if not GPU:
        return index
    if "HNSW" in name:
        log.info("%s: HNSW is CPU-only in FAISS; keeping on CPU", name)
        return index
    log.info("%s: transferring to GPU via faiss.index_cpu_to_gpu", name)
    return faiss.index_cpu_to_gpu(_RES, 0, index)


def size_mb(index) -> float:
    cpu = faiss.index_gpu_to_cpu(index) if GPU and not isinstance(index, (faiss.IndexHNSWFlat, faiss.IndexFlat)) and hasattr(faiss, "index_gpu_to_cpu") and "Gpu" in type(index).__name__ else index
    return len(faiss.serialize_index(cpu)) / 2**20


def build_hnsw(emb, M=32, efc=200):
    idx = faiss.IndexHNSWFlat(emb.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = efc
    idx.add(emb)
    return idx


def build_ivfpq(emb, nlist=32, m=16, nbits=6):
    q = faiss.IndexFlatIP(emb.shape[1])
    idx = faiss.IndexIVFPQ(q, emb.shape[1], nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    idx.train(emb)
    idx.add(emb)
    return idx


def bench(index, queries, gt, k=10, n_single=500) -> dict:
    index.search(queries[:16], k)  # warm-up
    t0 = time.perf_counter()
    _, I = index.search(queries, k)
    qps_batch = len(queries) / (time.perf_counter() - t0)
    lat = []
    for q in queries[:n_single]:
        t = time.perf_counter()
        index.search(q[None, :], k)
        lat.append((time.perf_counter() - t) * 1000)
    rec = np.mean([len(set(I[i]) & set(gt[i])) / k for i in range(len(queries))])
    return {"recall_at_10": float(rec), "qps": float(1000 / np.mean(lat)), "qps_batch": float(qps_batch),
            "p50_latency_ms": float(np.percentile(lat, 50)), "p99_latency_ms": float(np.percentile(lat, 99))}


def run(emb: np.ndarray, queries: np.ndarray) -> pd.DataFrame:
    emb, queries = np.ascontiguousarray(emb, dtype=np.float32), np.ascontiguousarray(queries, dtype=np.float32)
    faiss.omp_set_num_threads(max(1, faiss.omp_get_max_threads()))
    rows = []
    with stage_timer(log, "build IndexFlatIP"):
        flat = faiss.IndexFlatIP(emb.shape[1])
        flat.add(emb)
    gt = flat.search(queries, 10)[1]
    flat_g = maybe_gpu(flat, "IndexFlatIP")
    rows.append({"index": "IndexFlatIP", "param": "exact", **bench(flat_g, queries, gt), "index_size_mb": size_mb(flat),
                 "backend": "gpu" if flat_g is not flat else "cpu"})
    with stage_timer(log, "build IndexHNSWFlat(M=32, efC=200)"):
        hnsw = build_hnsw(emb)
    for ef in (8, 16, 32, 64, 128, 256):
        hnsw.hnsw.efSearch = ef
        rows.append({"index": "IndexHNSWFlat", "param": f"efSearch={ef}", **bench(hnsw, queries, gt),
                     "index_size_mb": size_mb(hnsw), "backend": "cpu"})
    with stage_timer(log, "build IndexIVFPQ(nlist=32, m=16, nbits=6)"):
        ivf = build_ivfpq(emb)
    ivf_g = maybe_gpu(ivf, "IndexIVFPQ")
    for nprobe in (1, 2, 4, 8, 16, 32):
        if ivf_g is ivf:
            ivf.nprobe = nprobe
        else:
            faiss.GpuParameterSpace().set_index_parameter(ivf_g, "nprobe", nprobe)
        rows.append({"index": "IndexIVFPQ", "param": f"nprobe={nprobe}", **bench(ivf_g, queries, gt),
                     "index_size_mb": size_mb(ivf), "backend": "gpu" if ivf_g is not ivf else "cpu"})
    df = pd.DataFrame(rows)
    for r in rows:
        log.info("%-14s %-14s recall@10=%.3f qps=%8.0f p50=%.3fms p99=%.3fms size=%.2fMB [%s]", r["index"], r["param"],
                 r["recall_at_10"], r["qps"], r["p50_latency_ms"], r["p99_latency_ms"], r["index_size_mb"], r["backend"])
    return df


def chart(df: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for name, g in df.groupby("index"):
        if name == "IndexFlatIP":
            for a, col in zip(ax, ("p50_latency_ms", "p99_latency_ms")):
                a.scatter(g.recall_at_10, g[col], marker="*", s=250, label="IndexFlatIP (exact)", color="k", zorder=5)
            continue
        for a, col in zip(ax, ("p50_latency_ms", "p99_latency_ms")):
            a.plot(g.recall_at_10, g[col], "o-", label=name)
            for _, r in g.iterrows():
                a.annotate(r.param.split("=")[1], (r.recall_at_10, r[col]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for a, t in zip(ax, ("p50 latency (ms)", "p99 latency (ms)")):
        a.set(xlabel="recall@10 vs exact search", ylabel=t, title=f"ANN {t} vs recall (labels: efSearch / nprobe)")
        a.set_yscale("log")
        a.legend()
    fig.tight_layout(); fig.savefig(CHARTS / "ann_latency_vs_recall.png", dpi=150); plt.close(fig)


def main(prep=None, model=None):
    from . import dataset
    from .evaluate import load_model
    from .train import U_KEYS, device_of, to_dev
    dev = device_of()
    log.info("faiss=%s gpu_available=%s -> backend=%s", faiss.__version__, GPU, "faiss-gpu" if GPU else "faiss-cpu")
    if prep is None:
        users, items, ev = (pd.read_parquet(f"outputs/data/{n}.parquet") for n in ("users", "items", "events"))
        prep = dataset.prepare(users, items, ev)
    model = model or load_model(prep, dev)
    emb = np.load(MODELS / "item_embeddings.npy")
    te = prep.splits["test"]
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(te["item"]), 2000, replace=False)
    with torch.no_grad():
        q = model.user_tower(*[torch.as_tensor(te[k][sel]).to(dev) for k in U_KEYS]).cpu().numpy()
    df = run(emb, q)
    df.to_csv(METRICS / "ann_benchmark_metrics.csv", index=False)
    chart(df)

    # persist serving index: smallest efSearch with recall>=0.95, else the largest
    h = df[df["index"] == "IndexHNSWFlat"]
    ok = h[h.recall_at_10 >= 0.95]
    ef = int(((ok if len(ok) else h).iloc[0 if len(ok) else -1]).param.split("=")[1])
    idx = build_hnsw(emb)
    idx.hnsw.efSearch = ef
    faiss.write_index(idx, str(MODELS / "retail_items.faiss"))
    meta = prep.items.copy()
    meta["in_stock"] = meta.in_stock_ratio > 0.1
    meta["popularity_7d"] = prep.item_pop7d
    meta.to_parquet(MODELS / "item_metadata.parquet", index=False)
    (METRICS / "serving_index.json").write_text(f'{{"index": "IndexHNSWFlat", "M": 32, "efSearch": {ef}, "n_items": {len(emb)}}}')
    log.info("persisted retail_items.faiss (HNSW efSearch=%d) + item_metadata.parquet", ef)
    return df


if __name__ == "__main__":
    main()
