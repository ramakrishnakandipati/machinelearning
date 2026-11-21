"""Linear contextual bandits: LinUCB (disjoint + hybrid) and Linear Thompson Sampling (GPU, float64)."""
from pathlib import Path

import numpy as np
import torch

from ..config import ACTION_ATTR, DEVICE
from .base import BaseBandit


def aug(X):
    return np.hstack([X, np.ones((len(X), 1), np.float32)])


def arm_stats(Phi, a, r, K, lam=1.0, device=DEVICE):
    """Per-arm ridge posterior. Returns Ainv (K,D,D), mu (K,D), sigma (K,) [float64, on device]."""
    P = torch.as_tensor(Phi, dtype=torch.float64, device=device)
    at = torch.as_tensor(a, device=device)
    rt = torch.as_tensor(r, dtype=torch.float64, device=device)
    D = P.shape[1]
    I = torch.eye(D, dtype=torch.float64, device=device)
    Ainv = torch.empty(K, D, D, dtype=torch.float64, device=device)
    mu = torch.zeros(K, D, dtype=torch.float64, device=device)
    sig = torch.ones(K, dtype=torch.float64, device=device)
    for k in range(K):
        m = at == k
        Xk, rk = P[m], rt[m]
        Ai = torch.linalg.inv(lam * I + Xk.T @ Xk)
        Ainv[k] = Ai
        if len(rk):
            mu[k] = Ai @ (Xk.T @ rk)
            res = rk - Xk @ mu[k]
            sig[k] = torch.sqrt(torch.clamp((res ** 2).mean(), min=1e-3))
    return Ainv, mu, sig


class ArmPosterior:
    """Bayesian linear posterior per arm with batched Thompson sampling."""

    def __init__(self, Ainv, mu, sig):
        self.Ainv, self.mu, self.sig = Ainv, mu, sig
        sym = 0.5 * (Ainv + Ainv.transpose(1, 2)) + 1e-9 * torch.eye(Ainv.shape[1], dtype=Ainv.dtype, device=Ainv.device)
        self.L = torch.linalg.cholesky(sym).float()
        self.mu32, self.sig32 = mu.float(), sig.float()

    def mean(self, Phi_t):
        return Phi_t @ self.mu32.T

    def sample_scores(self, Phi_t, v=1.0, chunk=4096):
        K, D = self.mu32.shape
        out = []
        for i in range(0, len(Phi_t), chunk):
            P = Phi_t[i:i + chunk]
            eps = torch.randn(len(P), K, D, device=P.device)
            theta = self.mu32[None] + v * self.sig32[None, :, None] * torch.einsum("ked,nkd->nke", self.L, eps)
            out.append(torch.einsum("nd,nkd->nk", P, theta))
        return torch.cat(out)

    def state(self):
        return {"Ainv": self.Ainv.cpu().numpy(), "mu": self.mu.cpu().numpy(), "sig": self.sig.cpu().numpy()}

    @classmethod
    def from_state(cls, s):
        t = lambda k: torch.as_tensor(s[k], dtype=torch.float64, device=DEVICE)
        return cls(t("Ainv"), t("mu"), t("sig"))


class LinUCB(BaseBandit):
    """Disjoint / hybrid LinUCB (Li et al., 2010) with exact batch ridge solutions on GPU."""
    DEFAULTS = dict(alpha=0.3, lam=1.0, hybrid=False)

    def __init__(self, n_features, n_actions=6, name=None, **hp):
        hybrid = hp.get("hybrid", False)
        super().__init__(n_features, n_actions, name or ("LinUCB-Hybrid" if hybrid else "LinUCB-Disjoint"), **hp)
        self.state = None

    def _fit(self, X, a, r, p, warm=False):
        lam, K = self.hp["lam"], self.K
        Phi = aug(X)
        if not self.hp["hybrid"]:
            Ainv, mu, _ = arm_stats(Phi, a, r, K, lam)
            self.state = {"Ainv": Ainv, "mu": mu}
            return
        dev = DEVICE
        P = torch.as_tensor(Phi, dtype=torch.float64, device=dev)
        at = torch.as_tensor(a, device=dev)
        rt = torch.as_tensor(r, dtype=torch.float64, device=dev)
        attr = torch.as_tensor(ACTION_ATTR, dtype=torch.float64, device=dev)
        D, m = P.shape[1], attr.shape[1]
        kd = D * m
        I, Ik = torch.eye(D, dtype=torch.float64, device=dev), torch.eye(kd, dtype=torch.float64, device=dev)
        A0, b0 = lam * Ik.clone(), torch.zeros(kd, dtype=torch.float64, device=dev)
        per = []
        for k in range(K):
            msk = at == k
            Xk, rk = P[msk], rt[msk]
            Zk = (Xk[:, :, None] * attr[k][None, None, :]).reshape(len(Xk), kd)
            Aik = torch.linalg.inv(lam * I + Xk.T @ Xk)
            Bk, bk = Xk.T @ Zk, Xk.T @ rk
            A0 = A0 + Zk.T @ Zk - Bk.T @ Aik @ Bk
            b0 = b0 + Zk.T @ rk - Bk.T @ Aik @ bk
            per.append((Aik, Bk, bk))
        A0inv = torch.linalg.inv(A0)
        beta = A0inv @ b0
        self.state = {
            "beta": beta, "A0inv": A0inv,
            "Ainv": torch.stack([p_[0] for p_ in per]),
            "B": torch.stack([p_[1] for p_ in per]),
            "mu": torch.stack([p_[0] @ (p_[2] - p_[1] @ beta) for p_ in per]),
        }

    @torch.no_grad()
    def _predict(self, X):
        P = torch.as_tensor(aug(X), dtype=torch.float64, device=DEVICE)
        s = self.state
        mean = P @ s["mu"].T
        var = torch.einsum("nd,kde,ne->nk", P, s["Ainv"], P)
        if self.hp["hybrid"]:
            attr = torch.as_tensor(ACTION_ATTR, dtype=torch.float64, device=DEVICE)
            n, D = P.shape
            for k in range(self.K):
                z = (P[:, :, None] * attr[k][None, None, :]).reshape(n, -1)
                u = P @ s["Ainv"][k]                      # (n,D) = xᵀAk⁻¹
                w = u @ s["B"][k]                         # (n,kd) = xᵀAk⁻¹Bk
                zA = z @ s["A0inv"]
                mean[:, k] += z @ s["beta"]
                var[:, k] = ((zA * z).sum(1) - 2 * (zA * w).sum(1) + (P * u).sum(1) + ((w @ s["A0inv"]) * w).sum(1))
        return mean, var.clamp_min(0)

    def _mean(self, X):
        return self._predict(X)[0].float().cpu().numpy()

    def _acting(self, X):
        mean, var = self._predict(X)
        return (mean + self.hp["alpha"] * var.sqrt()).float().cpu().numpy()

    def _save_weights(self, stem):
        np.savez_compressed(f"{stem}.npz", **{k: v.cpu().numpy() for k, v in self.state.items()})

    def _load_weights(self, stem):
        z = np.load(f"{stem}.npz")
        self.state = {k: torch.as_tensor(z[k], dtype=torch.float64, device=DEVICE) for k in z.files}


class LinearThompsonSampling(BaseBandit):
    """Bayesian linear regression per arm; acts on a joint posterior sample theta_a ~ N(mu_a, v^2 s_a^2 A_a^-1)."""
    DEFAULTS = dict(v=1.0, lam=1.0)
    stochastic = True
    default_name = "LinearTS"

    def _fit(self, X, a, r, p, warm=False):
        self.post = ArmPosterior(*arm_stats(aug(X), a, r, self.K, self.hp["lam"]))

    def _phi(self, X):
        return torch.as_tensor(aug(X), device=DEVICE)

    @torch.no_grad()
    def _mean(self, X):
        return self.post.mean(self._phi(X)).cpu().numpy()

    @torch.no_grad()
    def _acting(self, X):
        return self.post.sample_scores(self._phi(X), self.hp["v"]).cpu().numpy()

    def _save_weights(self, stem):
        np.savez_compressed(f"{stem}.npz", **self.post.state())

    def _load_weights(self, stem):
        self.post = ArmPosterior.from_state(np.load(f"{stem}.npz"))
