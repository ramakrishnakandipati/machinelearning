"""Shared model utilities: inputs container, numeric preprocessing, ZILN loss, generic torch trainer."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..features import CAT_COLS, NUM_COLS, Dataset
from ..logger import log_epoch


@dataclass
class Inputs:
    """Model inputs for a set of guests (tabular frame + optional sequence tensors)."""
    df: pd.DataFrame
    seq: np.ndarray | None = None
    mask: np.ndarray | None = None
    sabs: np.ndarray | None = None


def subset(ds: Dataset, name: str) -> tuple[Inputs, pd.DataFrame]:
    idx = ds.idx(name)
    return Inputs(ds.df.iloc[idx], ds.seq[idx], ds.seq_mask[idx], ds.seq_abs[idx]), ds.df.iloc[idx]


class NumPrep:
    """Signed-log1p + standardisation for numeric features (fit on train only)."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    @staticmethod
    def _t(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, df: pd.DataFrame) -> "NumPrep":
        x = self._t(df[NUM_COLS].values.astype(np.float64))
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = self._t(df[NUM_COLS].values.astype(np.float64))
        return np.clip((x - self.mean) / self.std, -6, 6).astype(np.float32)

    def codes(self, df: pd.DataFrame) -> np.ndarray:
        return df[[c + "_code" for c in CAT_COLS]].values.astype(np.int64)


def ziln_loss(logit: torch.Tensor, mu: torch.Tensor, raw_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Zero-Inflated Log-Normal loss: BCE(return) + positive-only log-normal NLL (constants dropped)."""
    pos = (y > 0).float()
    sigma = F.softplus(raw_sigma) + 1e-3
    ly = torch.log(y.clamp_min(1e-6))
    nll = torch.log(sigma) + 0.5 * ((ly - mu) / sigma) ** 2
    bce = F.binary_cross_entropy_with_logits(logit, pos, reduction="none")
    return (bce + pos * nll).mean()


def ziln_mean(logit: torch.Tensor, mu: torch.Tensor, raw_sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (expected value, return probability)."""
    sigma = F.softplus(raw_sigma) + 1e-3
    p = torch.sigmoid(logit)
    return p * torch.exp((mu + 0.5 * sigma ** 2).clamp(max=14.0)), p


def fit_torch(model: torch.nn.Module, train_t: tuple[torch.Tensor, ...], val_t: tuple[torch.Tensor, ...],
              loss_fn: Callable[[torch.nn.Module, tuple[torch.Tensor, ...]], torch.Tensor], hp: dict, name: str,
              out: Path, lg: logging.Logger) -> dict[str, list[float]]:
    """Generic mini-batch trainer: AdamW, ReduceLROnPlateau, early stopping, best-state restore, JSONL epoch logs."""
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    n, bs = train_t[0].shape[0], hp["batch_size"]
    hist: dict[str, list[float]] = {"train": [], "val": []}
    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, hp["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, device=train_t[0].device)
        tot = 0.0
        for i in range(0, n, bs):
            ix = perm[i:i + bs]
            loss = loss_fn(model, tuple(t[ix] for t in train_t))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(ix)
        model.eval()
        with torch.no_grad():
            m = val_t[0].shape[0]
            vl = sum(loss_fn(model, tuple(t[j:j + 4096] for t in val_t)).item() * len(val_t[0][j:j + 4096])
                     for j in range(0, m, 4096)) / m
        tl = tot / n
        sched.step(vl)
        hist["train"].append(tl)
        hist["val"].append(vl)
        log_epoch(out, name, ep, tl, vl, opt.param_groups[0]["lr"])
        if vl < best - 1e-5:
            best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= hp["patience"]:
                lg.info(f"[{name}] early stop @ epoch {ep} (best val {best:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    lg.info(f"[{name}] epochs={len(hist['train'])} best_val={best:.4f}")
    return hist
