"""GRU + multi-head self-attention over stay sequences with time-aware positional encodings (ZILN head)."""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from ..features import CAT_COLS, CAT_LEVELS, NUM_COLS, SEQ_FEATS
from .common import Inputs, NumPrep, fit_torch, ziln_loss, ziln_mean


def time_encoding(t: torch.Tensor, d: int) -> torch.Tensor:
    """Sinusoidal encoding of *continuous* time (days before cut-off), shape (B, L, d)."""
    div = torch.exp(torch.arange(0, d, 2, device=t.device) * (-math.log(1000.0) / d))
    ang = t.unsqueeze(-1) * div
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class SeqNet(nn.Module):
    def __init__(self, n_feat: int, n_num: int, cat_sizes: list[int], d: int, heads: int, drop: float) -> None:
        super().__init__()
        self.d = d
        self.proj = nn.Linear(n_feat, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.mha = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.num = nn.Linear(n_num, d)
        self.emb = nn.ModuleList([nn.Embedding(s, d) for s in cat_sizes])
        self.head = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, 3))
        self.last_attn: torch.Tensor | None = None
        with torch.no_grad():
            self.head[3].bias.copy_(torch.tensor([-0.7, 6.5, 0.5]))

    def forward(self, seq, mask, sabs, xn, xc):
        x = self.proj(seq) + time_encoding(sabs, self.d)
        h, _ = self.gru(x)
        h = h * mask.unsqueeze(-1)
        a, w = self.mha(h, h, h, key_padding_mask=~mask, need_weights=True)
        a = self.ln(h + a)
        m = mask.float()
        # importance of each stay = mean attention it receives from valid query positions
        imp = (w * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
        imp = imp * m
        self.last_attn = imp.detach()
        pooled = (a * imp.unsqueeze(-1)).sum(1) / imp.sum(1, keepdim=True).clamp_min(1e-6)
        static = self.num(xn) + sum(e(xc[:, i]) for i, e in enumerate(self.emb))
        return self.head(torch.cat([pooled, a[:, -1], static], -1))


class SeqZILN:
    name = "gru_attention_ziln"

    def __init__(self, hp: dict, device: str = "cuda") -> None:
        self.hp, self.device, self.net, self.prep = hp, device, None, NumPrep()

    def _build(self) -> SeqNet:
        h = self.hp
        return SeqNet(len(SEQ_FEATS), len(NUM_COLS), [len(CAT_LEVELS[c]) for c in CAT_COLS], h["d_model"],
                      h["n_heads"], h["dropout"]).to(self.device)

    def _t(self, inp: Inputs):
        dv = self.device
        return (torch.tensor(inp.seq, device=dv), torch.tensor(inp.mask, device=dv),
                torch.tensor(inp.sabs, device=dv),
                torch.tensor(self.prep.transform(inp.df), device=dv), torch.tensor(self.prep.codes(inp.df), device=dv))

    def fit(self, tr: Inputs, va: Inputs, out: Path, lg) -> dict:
        torch.manual_seed(42)
        self.prep.fit(tr.df)
        self.net = self._build()
        ytr = torch.tensor(tr.df["y"].values, dtype=torch.float32, device=self.device)
        yva = torch.tensor(va.df["y"].values, dtype=torch.float32, device=self.device)

        def loss_fn(m, b):
            o = m(*b[:5])
            return ziln_loss(o[:, 0], o[:, 1], o[:, 2], b[5])

        return fit_torch(self.net, (*self._t(tr), ytr), (*self._t(va), yva), loss_fn, self.hp, self.name, out, lg)

    @torch.no_grad()
    def predict_full(self, inp: Inputs, with_attention: bool = False) -> dict[str, np.ndarray]:
        self.net.eval()
        t = self._t(inp)
        res, att = [], []
        for i in range(0, len(t[0]), 4096):
            o = self.net(*[x[i:i + 4096] for x in t])
            res.append(torch.stack(ziln_mean(o[:, 0], o[:, 1], o[:, 2]), 1).cpu())
            att.append(self.net.last_attn.cpu())
        r = torch.cat(res).numpy()
        out = {"cltv": r[:, 0], "p_return": r[:, 1]}
        if with_attention:
            out["attention"] = torch.cat(att).numpy()
        return out

    def predict(self, inp: Inputs) -> np.ndarray:
        return self.predict_full(inp)["cltv"]

    def save(self, d: Path) -> None:
        torch.save({"state": self.net.state_dict(), "hp": self.hp}, d / "gru_attention_ziln.pt")
        joblib.dump(self.prep, d / "num_scaler_seq.joblib")

    def load(self, d: Path) -> "SeqZILN":
        ck = torch.load(d / "gru_attention_ziln.pt", map_location=self.device)
        self.hp = ck["hp"]
        self.net = self._build()
        self.net.load_state_dict(ck["state"])
        self.prep = joblib.load(d / "num_scaler_seq.joblib")
        return self
