"""Classical clustering: GPU K-Means, GMM (torch EM), HDBSCAN. cuML used when installed."""
import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA

from ..utils import get_logger

log = get_logger()

try:  # RAPIDS (Linux / WSL2 only)
    from cuml.cluster import KMeans as CuKMeans  # type: ignore
    HAS_CUML = True
except Exception:
    HAS_CUML = False

try:
    import hdbscan as _hdbscan_pkg  # type: ignore
except Exception:
    _hdbscan_pkg = None


def _t(X, device):
    return torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)


def _sqdist(X, C, chunk=16384):
    out = []
    for i in range(0, X.shape[0], chunk):
        out.append(torch.cdist(X[i:i + chunk], C) ** 2)
    return torch.cat(out)


class KMeansGPU:
    """K-Means (k-means++ init, Lloyd iterations) on the GPU via PyTorch; cuML if available."""

    def __init__(self, n_clusters=5, n_init=3, max_iter=100, tol=1e-4, seed=42, device=None):
        self.k, self.n_init, self.max_iter, self.tol, self.seed = n_clusters, n_init, max_iter, tol, seed
        self.device = device or torch.device("cpu")
        self.centers_ = None

    def _init_pp(self, X, gen):
        n = X.shape[0]
        idx = torch.randint(0, n, (1,), generator=gen, device="cpu").item()
        C = X[idx:idx + 1]
        for _ in range(1, self.k):
            d = _sqdist(X, C).min(1).values
            p = (d / d.sum()).clamp_min(1e-12).cpu()
            idx = torch.multinomial(p, 1, generator=gen).item()
            C = torch.cat([C, X[idx:idx + 1]])
        return C

    def fit(self, X):
        if HAS_CUML and self.device.type == "cuda":
            log.info("KMeans: using cuML backend")
            km = CuKMeans(n_clusters=self.k, random_state=self.seed, n_init=self.n_init).fit(np.asarray(X))
            self.centers_ = _t(km.cluster_centers_, self.device)
            return self
        X = _t(X, self.device)
        gen = torch.Generator().manual_seed(self.seed)
        best = (float("inf"), None)
        for _ in range(self.n_init):
            C = self._init_pp(X, gen)
            prev = float("inf")
            for _ in range(self.max_iter):
                d = _sqdist(X, C)
                lab = d.argmin(1)
                inertia = d.gather(1, lab[:, None]).sum().item()
                newC = torch.zeros_like(C).index_add_(0, lab, X)
                cnt = torch.bincount(lab, minlength=self.k).clamp_min(1).float()[:, None]
                newC = newC / cnt
                empty = torch.bincount(lab, minlength=self.k) == 0
                newC[empty] = C[empty]
                C = newC
                if abs(prev - inertia) < self.tol * max(inertia, 1):
                    break
                prev = inertia
            if inertia < best[0]:
                best = (inertia, C.clone())
        self.inertia_, self.centers_ = best
        return self

    @torch.no_grad()
    def predict(self, X):
        return _sqdist(_t(X, self.device), self.centers_).argmin(1).cpu().numpy()

    def predict_proba(self, X, temp=1.0):
        d = _sqdist(_t(X, self.device), self.centers_)
        return torch.softmax(-d / (2 * temp * d.mean()), 1).cpu().numpy()

    def save(self, path):
        joblib.dump({"k": self.k, "centers": self.centers_.cpu().numpy()}, path)

    @classmethod
    def load(cls, path, device=None):
        d = joblib.load(path)
        m = cls(d["k"], device=device)
        m.centers_ = _t(d["centers"], m.device)
        return m


class GMMGPU:
    """Gaussian mixture (diagonal covariance) fitted with EM in PyTorch -> soft assignments."""

    def __init__(self, n_components=5, max_iter=100, tol=1e-4, reg=1e-4, seed=42, device=None):
        self.k, self.max_iter, self.tol, self.reg, self.seed = n_components, max_iter, tol, reg, seed
        self.device = device or torch.device("cpu")

    def _log_resp(self, X):
        # log N(x|mu,diag var) + log pi
        var = self.var_
        ll = -0.5 * (((X[:, None, :] - self.mu_[None]) ** 2 / var[None]).sum(-1)
                     + torch.log(var).sum(-1)[None] + X.shape[1] * np.log(2 * np.pi))
        return ll + torch.log(self.pi_)[None]

    def fit(self, X):
        X = _t(X, self.device)
        km = KMeansGPU(self.k, n_init=2, seed=self.seed, device=self.device).fit(X.cpu().numpy())
        lab = km.predict(X.cpu().numpy())
        lab = torch.as_tensor(lab, device=self.device)
        self.mu_ = km.centers_.clone()
        self.var_ = torch.stack([X[lab == j].var(0) if (lab == j).sum() > 1 else X.var(0)
                                 for j in range(self.k)]) + self.reg
        self.pi_ = torch.bincount(lab, minlength=self.k).float() / len(lab)
        prev = -float("inf")
        for it in range(self.max_iter):
            lr = self._log_resp(X)
            lse = torch.logsumexp(lr, 1, keepdim=True)
            resp = torch.exp(lr - lse)
            ll = lse.mean().item()
            nk = resp.sum(0) + 1e-8
            self.mu_ = (resp.T @ X) / nk[:, None]
            self.var_ = (resp.T @ (X ** 2)) / nk[:, None] - self.mu_ ** 2 + self.reg
            self.var_ = self.var_.clamp_min(self.reg)
            self.pi_ = nk / nk.sum()
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        self.loglik_, self.n_iter_ = ll, it + 1
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        lr = self._log_resp(_t(X, self.device))
        return torch.softmax(lr, 1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def bic(self, n, d):
        p = self.k * d * 2 + self.k - 1
        return -2 * self.loglik_ * n + p * np.log(n)

    def save(self, path):
        joblib.dump({"k": self.k, "mu": self.mu_.cpu().numpy(), "var": self.var_.cpu().numpy(),
                     "pi": self.pi_.cpu().numpy()}, path)

    @classmethod
    def load(cls, path, device=None):
        d = joblib.load(path)
        m = cls(d["k"], device=device)
        m.mu_, m.var_, m.pi_ = (_t(d[x], m.device) for x in ("mu", "var", "pi"))
        return m


class HDBSCANModel:
    """Density-based clustering with noise (-1) for outlier / VIP-whale detection.

    Fitted on a PCA projection for speed. `predict` assigns to the nearest cluster centroid, or
    -1 if farther than the cluster's 95th-percentile radius (approximate_predict analogue).
    """

    def __init__(self, min_cluster_size=150, min_samples=15, pca_dim=10, seed=42):
        self.mcs, self.ms, self.pca_dim, self.seed = min_cluster_size, min_samples, pca_dim, seed

    def fit(self, X):
        X = np.asarray(X)
        self.pca_ = PCA(self.pca_dim, random_state=self.seed).fit(X)
        Z = self.pca_.transform(X)
        if _hdbscan_pkg is not None:
            log.info("HDBSCAN: using `hdbscan` package")
            self.labels_ = _hdbscan_pkg.HDBSCAN(min_cluster_size=self.mcs, min_samples=self.ms).fit_predict(Z)
        else:
            from sklearn.cluster import HDBSCAN
            log.info("HDBSCAN: using sklearn.cluster.HDBSCAN")
            self.labels_ = HDBSCAN(min_cluster_size=self.mcs, min_samples=self.ms).fit_predict(Z)
        ids = [c for c in np.unique(self.labels_) if c >= 0]
        self.cluster_ids_ = ids
        self.centers_ = np.stack([Z[self.labels_ == c].mean(0) for c in ids]) if ids else np.zeros((0, Z.shape[1]))
        self.radius_ = np.array([np.percentile(np.linalg.norm(Z[self.labels_ == c] - self.centers_[i], axis=1), 95)
                                 for i, c in enumerate(ids)]) if ids else np.zeros(0)
        return self

    def predict(self, X):
        if len(self.cluster_ids_) == 0:
            return -np.ones(len(X), dtype=int)
        Z = self.pca_.transform(np.asarray(X))
        d = np.linalg.norm(Z[:, None] - self.centers_[None], axis=-1)
        j = d.argmin(1)
        lab = np.array(self.cluster_ids_)[j]
        lab[d[np.arange(len(Z)), j] > self.radius_[j] * 1.5] = -1
        return lab

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path, device=None):
        return joblib.load(path)
