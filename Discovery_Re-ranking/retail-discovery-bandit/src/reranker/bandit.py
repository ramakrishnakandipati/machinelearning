"""GPU contextual bandits: Disjoint / Hybrid LinUCB (Sherman-Morrison) and Bayesian linear Thompson Sampling.

Arms are *grouped* (default: one disjoint ridge model per product category) so that brand-new SKUs
with zero history inherit statistics from their group and from shared hybrid parameters.

    Score_bandit(u,i) = U(u,i) [+ theta^T x] + alpha * sqrt(x^T A^-1 x)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


class LinUCBBandit:
    def __init__(self, dim: int, n_groups: int, device: torch.device, alpha: float = 0.35, lam: float = 1.0,
                 hybrid: bool = True, shared_idx: Optional[Sequence[int]] = None, refresh_every: int = 200):
        self.d, self.G, self.dev = dim, n_groups, device
        self.dt = torch.float64            # float64: hybrid Schur-complement updates are ill-conditioned in float32
        self.alpha, self.lam, self.hybrid = alpha, lam, hybrid
        self.refresh_every = refresh_every
        eye = torch.eye(dim, device=device, dtype=self.dt)
        self.A = lam * eye.expand(n_groups, dim, dim).clone()
        self.A_inv = (1.0 / lam) * eye.expand(n_groups, dim, dim).clone()
        self.b = torch.zeros(n_groups, dim, device=device, dtype=self.dt)
        self.shared_idx = torch.as_tensor(list(shared_idx) if shared_idx is not None else list(range(min(dim, 6))),
                                          device=device, dtype=torch.long)
        self.k = int(self.shared_idx.numel()) if hybrid else 0
        if hybrid:
            ek = torch.eye(self.k, device=device, dtype=self.dt)
            self.A0 = lam * ek.clone()
            self.A0_inv = (1.0 / lam) * ek.clone()
            self.b0 = torch.zeros(self.k, device=device, dtype=self.dt)
            self.B = torch.zeros(n_groups, dim, self.k, device=device, dtype=self.dt)
        self.n_updates = 0
        self.group_counts = torch.zeros(n_groups, device=device, dtype=self.dt)

    # ----- scoring --------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, X: torch.Tensor, groups: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, sqrt-variance) for contexts X (n,d) in arm-groups `groups` (n,)."""
        X = X.to(self.dt)
        Ainv = self.A_inv[groups]                                  # (n,d,d)
        v = torch.einsum("nde,ne->nd", Ainv, X)                   # A_g^-1 x
        if not self.hybrid:
            theta = torch.einsum("gde,ge->gd", self.A_inv, self.b)  # (G,d)
            mean = (X * theta[groups]).sum(-1)
            var = (X * v).sum(-1)
            return mean.float(), var.clamp(min=0).sqrt().float()
        Z = X[:, self.shared_idx]                                  # (n,k)
        beta = self.A0_inv @ self.b0                               # (k,)
        Bg = self.B[groups]                                        # (n,d,k)
        theta_g = torch.einsum("gde,ge->gd", self.A_inv, self.b - torch.einsum("gdk,k->gd", self.B, beta))
        mean = Z @ beta + (X * theta_g[groups]).sum(-1)
        w = torch.einsum("ndk,nd->nk", Bg, v)                      # B_g^T A_g^-1 x   (n,k)
        zA = Z @ self.A0_inv                                       # (n,k)
        var = (zA * Z).sum(-1) - 2.0 * (zA * w).sum(-1) + (X * v).sum(-1) + ((w @ self.A0_inv) * w).sum(-1)
        return mean.float(), var.clamp(min=0).sqrt().float()

    @torch.no_grad()
    def scores(self, X: torch.Tensor, groups: torch.Tensor, alpha: Optional[float] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (learned_mean_residual, exploration_bonus = alpha * sqrt(x^T A^-1 x))."""
        a = self.alpha if alpha is None else alpha
        mean, sd = self.predict(X, groups)
        return mean, a * sd

    # ----- online update --------------------------------------------------------------------
    @torch.no_grad()
    def update(self, x: torch.Tensor, group: int, reward: float) -> None:
        """Rank-1 (Sherman-Morrison) online update with a single observation."""
        g = int(group)
        x = x.to(self.dev, self.dt).reshape(-1)
        Ainv = self.A_inv[g]
        if self.hybrid:
            z = x[self.shared_idx]
            Bg = self.B[g]
            BtAinv = Bg.T @ Ainv                                   # (k,d)
            self.A0 += BtAinv @ Bg
            self.b0 += BtAinv @ self.b[g]
        Ax = Ainv @ x
        denom = 1.0 + x @ Ax
        self.A_inv[g] = Ainv - torch.outer(Ax, Ax) / denom
        self.A[g] += torch.outer(x, x)
        self.b[g] += reward * x
        if self.hybrid:
            self.B[g] += torch.outer(x, z)
            BtAinv = self.B[g].T @ self.A_inv[g]
            self.A0 += torch.outer(z, z) - BtAinv @ self.B[g]
            self.b0 += reward * z - BtAinv @ self.b[g]
            self.A0_inv = torch.linalg.inv(self.A0)
        self.n_updates += 1
        self.group_counts[g] += 1
        if self.refresh_every and self.n_updates % self.refresh_every == 0:
            self.refresh()

    @torch.no_grad()
    def refresh(self) -> float:
        """Recompute A^-1 exactly with torch.linalg.solve to bound Sherman-Morrison drift. Returns max drift."""
        eye = torch.eye(self.d, device=self.dev, dtype=self.dt).expand(self.G, self.d, self.d)
        exact = torch.linalg.solve(self.A, eye)
        drift = float((exact - self.A_inv).abs().max())
        self.A_inv = exact.clone()
        if self.hybrid:
            self.A0_inv = torch.linalg.solve(self.A0, torch.eye(self.k, device=self.dev, dtype=self.dt))
        return drift

    @torch.no_grad()
    def max_inverse_error(self) -> float:
        eye = torch.eye(self.d, device=self.dev, dtype=self.dt).expand(self.G, self.d, self.d)
        return float((self.A @ self.A_inv - eye).abs().max())

    # ----- persistence ----------------------------------------------------------------------
    def state_dict(self) -> Dict:
        s = {"kind": type(self).__name__, "A": self.A.cpu(), "A_inv": self.A_inv.cpu(), "b": self.b.cpu(),
             "alpha": self.alpha, "lam": self.lam, "hybrid": self.hybrid, "dim": self.d, "n_groups": self.G,
             "shared_idx": self.shared_idx.cpu(), "n_updates": self.n_updates, "group_counts": self.group_counts.cpu()}
        if self.hybrid:
            s.update({"A0": self.A0.cpu(), "A0_inv": self.A0_inv.cpu(), "b0": self.b0.cpu(), "B": self.B.cpu()})
        return s

    def load_state_dict(self, s: Dict) -> None:
        for k in ("A", "A_inv", "b", "group_counts"):
            setattr(self, k, s[k].to(self.dev))
        self.n_updates = s["n_updates"]
        if self.hybrid and "A0" in s:
            for k in ("A0", "A0_inv", "b0", "B"):
                setattr(self, k, s[k].to(self.dev))


class ThompsonBandit(LinUCBBandit):
    """Bayesian linear Thompson sampling (disjoint groups): theta~ ~ N(theta_hat, v^2 A^-1)."""

    def __init__(self, dim, n_groups, device, v: float = 0.05, lam: float = 1.0, refresh_every: int = 200):
        super().__init__(dim, n_groups, device, alpha=v, lam=lam, hybrid=False, refresh_every=refresh_every)

    @torch.no_grad()
    def scores(self, X, groups, alpha=None):
        v = self.alpha if alpha is None else alpha
        X = X.to(self.dt)
        theta = torch.einsum("gde,ge->gd", self.A_inv, self.b)
        mean = (X * theta[groups]).sum(-1)
        uniq = torch.unique(groups)
        L = torch.linalg.cholesky(0.5 * (self.A_inv[uniq] + self.A_inv[uniq].transpose(1, 2))
                                  + 1e-9 * torch.eye(self.d, device=self.dev, dtype=self.dt))
        eps = torch.randn(uniq.numel(), self.d, device=self.dev, dtype=self.dt)
        noise = torch.zeros(self.G, self.d, device=self.dev, dtype=self.dt)
        noise[uniq] = v * torch.einsum("gde,ge->gd", L, eps)
        return mean.float(), (X * noise[groups]).sum(-1).float()


def save_bandits(bandits: Dict[str, LinUCBBandit], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: b.state_dict() for k, b in bandits.items()}, path)


def load_bandit_states(path: Path) -> Dict[str, Dict]:
    return torch.load(path, map_location="cpu", weights_only=False)
