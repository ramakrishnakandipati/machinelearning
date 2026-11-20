"""Classical, probabilistic and density-based clustering (RAPIDS cuML when available)."""
from __future__ import annotations

import logging
import time

import joblib
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.cluster import KMeans as SkKMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

import config as C

log = logging.getLogger("hospitality")

if C.HAS_CUML:  # pragma: no cover - exercised only on RAPIDS machines
    from cuml.cluster import KMeans as CuKMeans
else:
    CuKMeans = None


class CentroidModel:
    """Nearest-centroid predictor with softmax confidences.

    Used to serve models without a native `predict` (Agglomerative, DBSCAN/HDBSCAN)
    and to give K-Means probabilistic-style confidences.
    """

    def __init__(self, centroids: np.ndarray, tau: float, source: str):
        self.centroids_ = np.asarray(centroids, dtype=np.float64)
        self.tau_ = float(max(tau, 1e-6))
        self.source = source

    @classmethod
    def fit(cls, X: np.ndarray, labels: np.ndarray, source: str):
        ids = np.array(sorted(i for i in np.unique(labels) if i >= 0))
        cents = np.stack([X[labels == i].mean(0) for i in ids])
        m = labels >= 0
        own = np.linalg.norm(X[m] - cents[np.searchsorted(ids, labels[m])], axis=1)
        return cls(cents, own.mean(), source)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(X[:, None, :] - self.centroids_[None], axis=2)
        z = -(d / self.tau_) ** 2
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(X[:, None, :] - self.centroids_[None], axis=2)
        return d.argmin(1)


def _kmeans(k: int):
    if CuKMeans is not None:  # pragma: no cover
        return CuKMeans(n_clusters=k, random_state=C.SEED, n_init=10)
    return SkKMeans(n_clusters=k, random_state=C.SEED, n_init=10)


def _np(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def select_k(X: np.ndarray) -> dict:
    """Elbow (inertia) + silhouette sweep for K-Means. Picks k by best silhouette."""
    rng = np.random.default_rng(C.SEED)
    sub = rng.choice(len(X), size=min(4000, len(X)), replace=False)
    inertias, sils = [], []
    for k in C.K_RANGE:
        km = _kmeans(k).fit(X)
        inertias.append(float(km.inertia_))
        sils.append(float(silhouette_score(X[sub], _np(km.predict(X[sub])))))
    best_k = C.K_RANGE[int(np.argmax(sils))]
    log.info("K selection: silhouettes=%s -> best_k=%d", [round(s, 3) for s in sils], best_k)
    return dict(ks=C.K_RANGE, inertia=inertias, silhouette=sils, best_k=best_k)


def _save(model, name: str):
    joblib.dump(model, C.MODELS_DIR / f"{name}.joblib")


def fit_kmeans(X, k):
    km = _kmeans(k).fit(X)
    labels = _np(km.predict(X)).astype(int)
    model = CentroidModel.fit(X, labels, "kmeans")  # centroids + confidence scale from data
    _save(model, "kmeans")
    return model, labels


def fit_gmm(X, k):
    gmm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=C.SEED, reg_covar=1e-4).fit(X)
    labels = gmm.predict(X)
    _save(gmm, "gmm")
    return gmm, labels


def fit_agglomerative(X, k, max_n=6000):
    """Ward linkage (O(n^2) memory) on a subsample, then label all points by nearest centroid."""
    rng = np.random.default_rng(C.SEED)
    sub = rng.choice(len(X), size=min(max_n, len(X)), replace=False)
    sub_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X[sub])
    model = CentroidModel.fit(X[sub], sub_labels, "agglomerative")
    labels = model.predict(X)
    _save(model, "agglomerative")
    return model, labels


def fit_density(X):
    """HDBSCAN (falls back to DBSCAN with k-distance eps). Noise label = -1."""
    labels = HDBSCAN(**C.HDBSCAN).fit_predict(X)
    algo = "hdbscan"
    n_cl = len(set(labels) - {-1})
    if n_cl < 2:
        nn = NearestNeighbors(n_neighbors=C.HDBSCAN["min_samples"]).fit(X)
        kd = np.sort(nn.kneighbors(X)[0][:, -1])
        eps = float(np.percentile(kd, 90))
        labels = DBSCAN(eps=eps, min_samples=C.HDBSCAN["min_samples"]).fit_predict(X)
        algo = f"dbscan(eps={eps:.2f})"
        n_cl = len(set(labels) - {-1})
    log.info("Density clustering via %s: %d clusters, noise=%.1f%%", algo, n_cl, 100 * (labels == -1).mean())
    if n_cl < 1:
        return None, labels
    model = CentroidModel.fit(X, labels, algo)
    _save(model, "dbscan")
    return model, labels


def timed_predict_ms_per_1k(fn, X, repeats: int = 5) -> float:
    fn(X[:64])  # warmup
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn(X)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000 / (len(X) / 1000))


def fit_all(X: np.ndarray, k_override: int | None = None) -> dict:
    """Fit every classical model. Returns {name: dict(model, labels)} plus k-selection info."""
    ks = select_k(X)
    if k_override:
        ks["best_k"] = k_override
    k = ks["best_k"]
    out = {"_k_selection": ks}
    for name, fn in (("kmeans", lambda: fit_kmeans(X, k)), ("gmm", lambda: fit_gmm(X, k)),
                     ("agglomerative", lambda: fit_agglomerative(X, k)), ("dbscan", lambda: fit_density(X))):
        t = time.perf_counter()
        model, labels = fn()
        log.info("%-14s fit in %.1fs", name, time.perf_counter() - t)
        if model is not None:
            out[name] = dict(model=model, labels=labels)
    return out
