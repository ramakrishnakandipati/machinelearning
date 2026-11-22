"""GPU deep survival models in PyTorch: DeepSurv (Cox partial likelihood) and DeepHit (discrete time)."""
from __future__ import annotations

import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from .base import SurvivalModel, breslow_baseline, breslow_survival, monotone_survival
from .utils import get_logger, timer

log = get_logger("deep")


def _act(name: str) -> nn.Module:
    return {"selu": nn.SELU, "relu": nn.ReLU}[name]()


def _mlp(in_dim: int, hidden: list[int], out_dim: int, dropout: float, activation: str) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), _act(activation), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")   # LeCun-style, suits SELU
            nn.init.zeros_(m.bias)
    return net


# ----------------------------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------------------------
def neg_log_partial_likelihood(h: torch.Tensor, t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Cox negative log partial likelihood (Breslow ties) averaged over observed events."""
    order = torch.argsort(t, descending=True)
    h, e = h[order], e[order]
    log_risk = torch.logcumsumexp(h, dim=0)          # risk set = all with time >= t_i
    n_ev = e.sum().clamp(min=1.0)
    return -((h - log_risk) * e).sum() / n_ev


def deephit_loss(logits: torch.Tensor, k: torch.Tensor, e: torch.Tensor, t: torch.Tensor,
                 alpha: float, sigma: float) -> torch.Tensor:
    pmf = torch.softmax(logits, dim=1)
    cif = torch.cumsum(pmf, dim=1)                                   # F(bin_k) = P(T <= upper edge of bin k)
    idx = torch.arange(len(k), device=k.device)
    ll_event = torch.log(pmf[idx, k].clamp(min=1e-8))
    cif_prev = torch.where(k > 0, cif[idx, (k - 1).clamp(min=0)], torch.zeros_like(pmf[:, 0]))
    ll_cens = torch.log((1.0 - cif_prev).clamp(min=1e-8))            # alive at start of censoring bin
    nll = -(e * ll_event + (1 - e) * ll_cens).mean()

    # ranking loss over comparable pairs (i event, j with t_j > t_i)
    F_ii = cif[idx, k]                                               # (B,)
    F_ji = cif[:, k]                                                 # (B_j, B_i) -> F_j(k_i)
    diff = F_ii.unsqueeze(0) - F_ji                                  # [j,i] = F_i(k_i) - F_j(k_i)
    mask = (e.unsqueeze(0) > 0) & (t.unsqueeze(1) > t.unsqueeze(0))  # [j,i]: i event, t_j > t_i
    rank = (torch.exp(-diff / sigma) * mask).sum() / mask.sum().clamp(min=1)
    return alpha * nll + (1 - alpha) * rank


# ----------------------------------------------------------------------------------------------
# Shared training loop
# ----------------------------------------------------------------------------------------------
class _DeepBase(SurvivalModel):
    kind: str = ""

    def __init__(self, hidden=(128, 64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4, epochs=150,
                 batch_size=1024, patience=20, activation="selu", device="cuda", seed=42, **extra):
        self.hidden, self.dropout, self.lr, self.weight_decay = list(hidden), dropout, lr, weight_decay
        self.epochs, self.batch_size, self.patience = epochs, batch_size, patience
        self.activation, self.device, self.seed = activation, device, seed
        self.extra = extra
        self.history: list[dict] = []

    # subclass hooks
    def _build(self, in_dim): raise NotImplementedError
    def _prepare_targets(self, t, e): raise NotImplementedError
    def _batch_loss(self, out, y): raise NotImplementedError
    def _post_fit(self, X, t, e): pass

    def _fwd(self, x):
        return self.net(x)

    def fit(self, X, t, e, X_val=None, t_val=None, e_val=None):
        self.feature_names = list(X.columns)
        torch.manual_seed(self.seed)
        if X_val is None:   # internal 15% validation split for early stopping
            X, X_val, t, t_val, e, e_val = train_test_split(X, t, e, test_size=0.15, random_state=self.seed,
                                                            stratify=e)
        dev = torch.device(self.device)
        self.net = self._build(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(3, self.patience // 4))

        xtr = torch.tensor(np.asarray(X, np.float32), device=dev)
        xva = torch.tensor(np.asarray(X_val, np.float32), device=dev)
        ytr, yva = self._prepare_targets(t, e, dev), self._prepare_targets(t_val, e_val, dev)
        n = len(xtr)
        best, best_state, bad = np.inf, None, 0
        self.history = []

        with timer() as tm:
            for ep in range(1, self.epochs + 1):
                self.net.train()
                perm = torch.randperm(n, device=dev)
                tot, nb = 0.0, 0
                for s in range(0, n, self.batch_size):
                    ix = perm[s:s + self.batch_size]
                    if len(ix) < 32:                      # skip tiny tail batch (BatchNorm / ranking stability)
                        continue
                    opt.zero_grad(set_to_none=True)
                    loss = self._batch_loss(self._fwd(xtr[ix]), tuple(y[ix] for y in ytr))
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                    opt.step()
                    tot += loss.item(); nb += 1
                self.net.eval()
                with torch.no_grad():
                    val = self._batch_loss(self._fwd(xva), yva).item()
                sched.step(val)
                self.history.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": val})
                if val < best - 1e-5:
                    best, best_state, bad = val, copy.deepcopy(self.net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        log.info("%s early stop at epoch %d (best val %.4f)", self.name, ep, best)
                        break
            self.net.load_state_dict(best_state)
            self.net.eval()
            self._post_fit(X, t, e)
        self.train_seconds = tm["seconds"]
        self.best_val_loss = float(best)
        return self

    @torch.no_grad()
    def _raw(self, X) -> np.ndarray:
        self.net.eval()
        dev = next(self.net.parameters()).device
        x = torch.tensor(np.asarray(X[self.feature_names], np.float32), device=dev)
        outs = [self._fwd(x[i:i + 8192]) for i in range(0, len(x), 8192)]
        return torch.cat(outs).float().cpu().numpy()

    def save_history(self, directory: str | Path) -> Path:
        d = Path(directory)
        pd.DataFrame(self.history).to_csv(d / f"{self.name}_loss_history.csv", index=False)
        h = pd.DataFrame(self.history)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(h.epoch, h.train_loss, label="train")
        ax.plot(h.epoch, h.val_loss, label="validation")
        ax.axvline(h.val_loss.idxmin() + 1, ls="--", c="gray", lw=1, label="best epoch")
        ax.set(title=f"{self.name} loss curves ({self.device})", xlabel="epoch", ylabel="loss")
        ax.legend()
        out = d / f"{self.name}_loss_curve.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    def _config(self) -> dict:
        return dict(hidden=self.hidden, dropout=self.dropout, lr=self.lr, weight_decay=self.weight_decay,
                    epochs=self.epochs, batch_size=self.batch_size, patience=self.patience,
                    activation=self.activation, seed=self.seed, **self.extra)


class DeepSurvModel(_DeepBase):
    name = "deepsurv"

    def _build(self, in_dim):
        return _mlp(in_dim, self.hidden, 1, self.dropout, self.activation)

    def _prepare_targets(self, t, e, dev):
        return (torch.tensor(np.asarray(t, np.float32), device=dev),
                torch.tensor(np.asarray(e, np.float32), device=dev))

    def _batch_loss(self, out, y):
        return neg_log_partial_likelihood(out.squeeze(-1), y[0], y[1])

    def _post_fit(self, X, t, e):
        m = self._raw(X).ravel()
        self.base_times, self.base_H0 = breslow_baseline(m, t, e)

    def risk(self, X):
        return self._raw(X).ravel()

    def survival(self, X, times):
        return breslow_survival(self.base_times, self.base_H0, self._raw(X).ravel(), times)

    def save(self, directory):
        path = Path(directory) / f"{self.name}.pt"
        torch.save({"state": self.net.state_dict(), "config": self._config(), "in_dim": len(self.feature_names),
                    "feature_names": self.feature_names, "base_times": self.base_times, "base_H0": self.base_H0,
                    "train_seconds": self.train_seconds}, path)
        return path

    @staticmethod
    def load(path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        m = DeepSurvModel(device=device, **ck["config"])
        m.net = m._build(ck["in_dim"]).to(device)
        m.net.load_state_dict(ck["state"]); m.net.eval()
        m.feature_names, m.base_times, m.base_H0 = ck["feature_names"], ck["base_times"], ck["base_H0"]
        m.train_seconds = ck.get("train_seconds", 0.0)
        return m


class DeepHitModel(_DeepBase):
    """Discrete-time DeepHit (single risk). Bin edges always contain the required cut points
    (30, 60, 90, 180, 365 days), refined with midpoints, plus one tail bin beyond the last edge."""
    name = "deephit"

    def __init__(self, bins=(30, 60, 90, 180, 365), alpha=0.5, sigma=0.1, **kw):
        super().__init__(alpha=alpha, sigma=sigma, bins=list(bins), **kw)
        self.alpha, self.sigma = alpha, sigma
        base = np.array(sorted(bins), float)
        mids = (np.concatenate([[0.0], base[:-1]]) + base) / 2.0
        self.edges = np.unique(np.concatenate([mids, base]))   # upper edges of the first K-1 bins
        self.K = len(self.edges) + 1                           # + tail bin

    def _build(self, in_dim):
        return _mlp(in_dim, self.hidden, self.K, self.dropout, self.activation)

    def _bin(self, t):
        return np.searchsorted(self.edges, np.asarray(t, float), side="left").clip(0, self.K - 1)

    def _prepare_targets(self, t, e, dev):
        return (torch.tensor(self._bin(t), dtype=torch.long, device=dev),
                torch.tensor(np.asarray(e, np.float32), device=dev),
                torch.tensor(np.asarray(t, np.float32), device=dev))

    def _batch_loss(self, out, y):
        return deephit_loss(out, y[0], y[1], y[2], self.alpha, self.sigma)

    def _cif(self, X) -> np.ndarray:
        logits = torch.tensor(self._raw(X))
        return torch.cumsum(torch.softmax(logits, dim=1), dim=1).numpy()

    def risk(self, X):
        cif = self._cif(X)[:, :len(self.edges)]
        return cif.mean(axis=1)                                 # area under the CIF -> earlier events score higher

    def survival(self, X, times):
        cif = self._cif(X)[:, :len(self.edges)]                  # F at each upper edge
        grid = np.concatenate([[0.0], self.edges])
        cif = np.concatenate([np.zeros((len(cif), 1)), cif], axis=1)
        s = 1.0 - np.vstack([np.interp(times, grid, row) for row in cif])
        return monotone_survival(s)

    def save(self, directory):
        path = Path(directory) / f"{self.name}.pt"
        cfg = self._config()
        torch.save({"state": self.net.state_dict(), "config": cfg, "in_dim": len(self.feature_names),
                    "feature_names": self.feature_names, "train_seconds": self.train_seconds}, path)
        return path

    @staticmethod
    def load(path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = dict(ck["config"])
        m = DeepHitModel(device=device, **cfg)
        m.net = m._build(ck["in_dim"]).to(device)
        m.net.load_state_dict(ck["state"]); m.net.eval()
        m.feature_names = ck["feature_names"]
        m.train_seconds = ck.get("train_seconds", 0.0)
        return m
