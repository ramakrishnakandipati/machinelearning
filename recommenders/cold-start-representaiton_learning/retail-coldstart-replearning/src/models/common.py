"""Shared building blocks: tensors on device, interaction tensors, history matrix, GRN, losses."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2n(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, dim=dim, eps=1e-8)


class GRN(nn.Module):
    """Gated Residual Network (Lim et al. 2021): LN(skip(x) + GLU(W2 ELU(W1 x)))."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_hidden = d_hidden or d_out
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out * 2)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        h = self.fc2(self.drop(F.elu(self.fc1(x))))
        a, b = h.chunk(2, dim=-1)
        return self.norm(self.skip(x) + a * torch.sigmoid(b))


@dataclass
class FeatureTensors:
    item_cat: torch.Tensor
    item_cont: torch.Tensor
    item_text: torch.Tensor
    item_vis: torch.Tensor
    user_cat: torch.Tensor
    user_quiz: torch.Tensor
    item_cold: torch.Tensor

    @staticmethod
    def from_features(f, device) -> "FeatureTensors":
        t = lambda a, dt=None: torch.as_tensor(a, dtype=dt).to(device)
        return FeatureTensors(t(f.item_cat, torch.long), t(f.item_cont), t(f.item_text), t(f.item_vis),
                              t(f.user_cat, torch.long), t(f.user_quiz), t(f.item_cold, torch.bool))


@dataclass
class InteractionTensors:
    u: torch.Tensor
    i: torch.Tensor
    tod: torch.Tensor        # (N,2)
    ctx_cat: torch.Tensor    # (N,3) device, store, channel
    ts: np.ndarray
    slice: np.ndarray

    def __len__(self):
        return len(self.u)


def build_interactions(df: pd.DataFrame, enc, user_index: dict, item_index: dict, device) -> InteractionTensors:
    ctx = enc.transform_context(df.hour.values, df.device.values, df.store_id.values, df.channel.values)
    return InteractionTensors(
        u=torch.as_tensor(df.user_id.map(user_index).values, dtype=torch.long, device=device),
        i=torch.as_tensor(df.item_id.map(item_index).values, dtype=torch.long, device=device),
        tod=torch.as_tensor(ctx["tod"], device=device),
        ctx_cat=torch.as_tensor(ctx["cat"], dtype=torch.long, device=device),
        ts=df.ts.values, slice=df["slice"].values,
    )


def build_history(train_df: pd.DataFrame, user_index: dict, item_index: dict, n_users: int, max_hist: int) -> np.ndarray:
    """(n_users, H) of item_idx+1, most-recent-last, 0 = pad. Train interactions only (no leakage)."""
    H = np.zeros((n_users, max_hist), dtype=np.int64)
    d = train_df.sort_values("ts")
    for uid, g in d.groupby("user_id"):
        it = g.item_id.map(item_index).values[-max_hist:] + 1
        H[user_index[uid], -len(it):] = it
    return H


def info_nce(q: torch.Tensor, k: torch.Tensor, tau, pos_ids=None, log_q=None) -> torch.Tensor:
    """In-batch InfoNCE; q,k L2-normalised (B,d). Duplicate positives in batch are masked out;
    optional logQ correction removes popularity bias of in-batch negatives."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        logits = (q.float() @ k.float().T) / tau
        if log_q is not None:
            logits = logits - log_q[None, :].float()
        if pos_ids is not None:
            dup = (pos_ids[:, None] == pos_ids[None, :]) & ~torch.eye(len(pos_ids), dtype=torch.bool, device=q.device)
            logits = logits.masked_fill(dup, -1e9)
        return F.cross_entropy(logits, torch.arange(len(q), device=q.device))


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    """SimCLR NT-Xent over 2B views."""
    B = z1.shape[0]
    with torch.autocast(device_type=z1.device.type, enabled=False):
        z = l2n(torch.cat([z1, z2], 0).float())
        sim = z @ z.T / tau
        sim = sim.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=z.device), -1e9)
        tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
        return F.cross_entropy(sim, tgt)


def uniformity(x: torch.Tensor, t: float = 2.0, max_n: int = 2000) -> float:
    x = l2n(x.float())
    if len(x) > max_n:
        x = x[torch.randperm(len(x), device=x.device)[:max_n]]
    sq = torch.pdist(x, p=2).pow(2)
    return float(torch.log(torch.exp(-t * sq).mean() + 1e-12))


def alignment(a: torch.Tensor, b: torch.Tensor, alpha: int = 2) -> float:
    return float((l2n(a.float()) - l2n(b.float())).norm(dim=1).pow(alpha).mean())


def scaled_epochs(n: int, quick: bool, scale: float) -> int:
    return max(2, int(math.ceil(n * scale))) if quick else n


SLICES = ["warm", "user_cold", "item_cold", "system_cold"]


@dataclass
class Bundle:
    cfg: dict
    device: torch.device
    items: pd.DataFrame
    users: pd.DataFrame
    stores: pd.DataFrame
    inter: pd.DataFrame
    enc: object
    feats: object
    ft: FeatureTensors
    user_index: dict
    item_index: dict
    tr_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    tr: InteractionTensors
    va: InteractionTensors
    hist: torch.Tensor
    hist_np: np.ndarray
    item_logq: torch.Tensor
    cat_sizes: dict
    text_dim: int
    vis_dim: int
    n_quiz: int

    @property
    def n_items(self):
        return len(self.items)

    @property
    def n_users(self):
        return len(self.users)


def build_bundle(cfg: dict, root, device) -> Bundle:
    from pathlib import Path
    from src.features.representation_encoders import RepresentationEncoders, load_data
    root = Path(root)
    items, users, stores, inter = load_data(root / cfg["paths"]["data_dir"])
    enc = RepresentationEncoders(cfg["features"]["text_svd_dim"], cfg["features"]["tfidf_max_features"], cfg["data"]["visual_dim"], cfg["seed"])
    enc.fit(items, users, stores, inter)
    feats = enc.build_features(items, users)
    ft = FeatureTensors.from_features(feats, device)
    ui = {u: k for k, u in enumerate(users.user_id)}
    ii = {i: k for k, i in enumerate(items.item_id)}
    tr_df = inter[inter.split == "train"].reset_index(drop=True)
    official = inter["slice"].isin(SLICES)
    val_df = inter[(inter.split == "val") & official].reset_index(drop=True)
    test_df = inter[(inter.split == "test") & official].reset_index(drop=True)
    hist_np = build_history(tr_df, ui, ii, len(users), cfg["train"]["two_tower"]["max_history"])
    cnt = np.bincount(tr_df.item_id.map(ii).values, minlength=len(items)).astype(np.float64) + 1.0
    logq = torch.as_tensor(np.log(cnt / cnt.sum()), dtype=torch.float32, device=device)
    return Bundle(cfg, device, items, users, stores, inter, enc, feats, ft, ui, ii, tr_df, val_df, test_df,
                  build_interactions(tr_df, enc, ui, ii, device), build_interactions(val_df, enc, ui, ii, device),
                  torch.as_tensor(hist_np, device=device), hist_np, logq, enc.cat_sizes(), enc.text_dim, enc.vis_dim, len(enc.quiz_vocab))
