"""Feature-Tokenizer Transformer (FT-Transformer) trained with Zero-Inflated Log-Normal (ZILN) loss."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from ..features import CAT_COLS, CAT_LEVELS, NUM_COLS
from .common import Inputs, NumPrep, fit_torch, ziln_loss, ziln_mean


class FTNet(nn.Module):
    """Numeric features -> per-feature affine tokens, categoricals -> embeddings, [CLS] -> (logit p, mu, raw sigma)."""

    def __init__(self, n_num: int, cat_sizes: list[int], d: int, blocks: int, heads: int, drop: float) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_num, d))
        self.emb = nn.ModuleList([nn.Embedding(s, d) for s in cat_sizes])
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 3))
        with torch.no_grad():
            self.head[2].bias.copy_(torch.tensor([-0.7, 6.5, 0.5]))

    def forward(self, xn: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        t_num = xn.unsqueeze(-1) * self.w + self.b
        t_cat = torch.stack([e(xc[:, i]) for i, e in enumerate(self.emb)], 1)
        h = self.enc(torch.cat([self.cls.expand(len(xn), -1, -1), t_num, t_cat], 1))
        return self.head(h[:, 0])


class FTZILN:
    name = "ft_transformer_ziln"

    def __init__(self, hp: dict, device: str = "cuda") -> None:
        self.hp, self.device, self.net, self.prep = hp, device, None, NumPrep()

    def _t(self, inp: Inputs) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.prep.transform(inp.df), device=self.device),
                torch.tensor(self.prep.codes(inp.df), device=self.device))

    def fit(self, tr: Inputs, va: Inputs, out: Path, lg) -> dict:
        h = self.hp
        torch.manual_seed(42)
        self.prep.fit(tr.df)
        self.net = FTNet(len(NUM_COLS), [len(CAT_LEVELS[c]) for c in CAT_COLS], h["d_token"], h["n_blocks"],
                         h["n_heads"], h["dropout"]).to(self.device)
        ytr = torch.tensor(tr.df["y"].values, dtype=torch.float32, device=self.device)
        yva = torch.tensor(va.df["y"].values, dtype=torch.float32, device=self.device)

        def loss_fn(m, b):
            o = m(b[0], b[1])
            return ziln_loss(o[:, 0], o[:, 1], o[:, 2], b[2])

        return fit_torch(self.net, (*self._t(tr), ytr), (*self._t(va), yva), loss_fn, h, self.name, out, lg)

    @torch.no_grad()
    def predict_full(self, inp: Inputs) -> dict[str, np.ndarray]:
        self.net.eval()
        xn, xc = self._t(inp)
        res = []
        for i in range(0, len(xn), 4096):
            o = self.net(xn[i:i + 4096], xc[i:i + 4096])
            res.append(torch.stack(ziln_mean(o[:, 0], o[:, 1], o[:, 2]), 1).cpu())
        r = torch.cat(res).numpy()
        return {"cltv": r[:, 0], "p_return": r[:, 1]}

    def predict(self, inp: Inputs) -> np.ndarray:
        return self.predict_full(inp)["cltv"]

    def save(self, d: Path) -> None:
        torch.save({"state": self.net.state_dict(), "hp": self.hp}, d / "ft_transformer_ziln.pt")
        joblib.dump(self.prep, d / "num_scaler.joblib")

    def load(self, d: Path) -> "FTZILN":
        ck = torch.load(d / "ft_transformer_ziln.pt", map_location=self.device)
        h = ck["hp"]
        self.net = FTNet(len(NUM_COLS), [len(CAT_LEVELS[c]) for c in CAT_COLS], h["d_token"], h["n_blocks"],
                         h["n_heads"], h["dropout"]).to(self.device)
        self.net.load_state_dict(ck["state"])
        self.prep = joblib.load(d / "num_scaler.joblib")
        return self
