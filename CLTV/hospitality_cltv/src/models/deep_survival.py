"""Neural Weibull cure-model for time-to-next-stay + log-normal stay value -> expected CLTV.

Per guest the network outputs: alive probability q, Weibull (k, lambda) for the time to next stay given alive, and a
log-normal for revenue per stay. Right-censoring at the horizon is handled in the likelihood. Expected stays in the
horizon H = q * (H/lambda)^k (non-homogeneous Poisson with Weibull intensity), CLTV = E[stays] * E[value/stay].
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features import CAT_COLS, CAT_LEVELS, NUM_COLS
from .common import Inputs, NumPrep, fit_torch


class Res(nn.Module):
    def __init__(self, d: int, drop: float = 0.1) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))

    def forward(self, x):
        return x + self.f(x)


class SurvNet(nn.Module):
    def __init__(self, n_num: int, cat_sizes: list[int], hidden: int) -> None:
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(s, 8) for s in cat_sizes])
        self.inp = nn.Linear(n_num + 8 * len(cat_sizes), hidden)
        self.trunk = nn.Sequential(Res(hidden), Res(hidden), Res(hidden))
        self.out = nn.Linear(hidden, 6)  # alive logit, log k, log lambda, mu, raw sigma, (spare)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([-0.7, 0.0, 6.0, 6.5, 0.5, 0.0]))

    def forward(self, xn, xc):
        z = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)], 1)
        return self.out(self.trunk(F.gelu(self.inp(z))))


def _params(o: torch.Tensor, H: float):
    q = torch.sigmoid(o[:, 0])
    k = (F.softplus(o[:, 1]) + 0.3).clamp(max=5.0)
    lam = torch.exp(o[:, 2].clamp(2.0, 9.0))
    lam_cum = ((H / lam) ** k).clamp(max=60.0)  # cumulative intensity over the horizon
    sigma = F.softplus(o[:, 4]) + 1e-3
    return q, k, lam, lam_cum, o[:, 3], sigma


class WeibullSurvival:
    name = "deep_weibull_survival"

    def __init__(self, hp: dict, horizon_days: float, device: str = "cuda") -> None:
        self.hp, self.H, self.device, self.net, self.prep = hp, float(horizon_days), device, None, NumPrep()

    def _t(self, inp: Inputs):
        return (torch.tensor(self.prep.transform(inp.df), device=self.device),
                torch.tensor(self.prep.codes(inp.df), device=self.device))

    def _loss(self, m: nn.Module, b) -> torch.Tensor:
        xn, xc, t, ev, n, rev = b
        o = m(xn, xc)
        q, k, lam, lam_cum, mu, sig = _params(o, self.H)
        eps = 1e-8
        z = (t / lam).clamp_min(1e-6)
        ll_event = torch.log(q + eps) + torch.log(k) - torch.log(lam) + (k - 1) * torch.log(z) - z ** k
        ll_cens = torch.logaddexp(torch.log(1 - q + eps), torch.log(q + eps) - lam_cum)
        surv = -(ev * ll_event + (1 - ev) * ll_cens).mean()
        mean_n = q * lam_cum
        pois = (mean_n - n * torch.log(mean_n + eps)).mean()
        pos = (ev > 0).float()
        lv = torch.log((rev / n.clamp_min(1)).clamp_min(1e-6))
        money = (pos * (torch.log(sig) + 0.5 * ((lv - mu) / sig) ** 2)).sum() / pos.sum().clamp_min(1)
        return surv + 0.3 * pois + money

    def fit(self, tr: Inputs, va: Inputs, out: Path, lg) -> dict:
        torch.manual_seed(42)
        self.prep.fit(tr.df)
        self.net = SurvNet(len(NUM_COLS), [len(CAT_LEVELS[c]) for c in CAT_COLS], self.hp["hidden"]).to(self.device)

        def tens(inp: Inputs):
            d = inp.df
            f = lambda c: torch.tensor(d[c].values, dtype=torch.float32, device=self.device)
            return (*self._t(inp), f("y_time"), f("y_event"), f("y_n"), f("y"))

        return fit_torch(self.net, tens(tr), tens(va), self._loss, self.hp, self.name, out, lg)

    @torch.no_grad()
    def predict_full(self, inp: Inputs) -> dict[str, np.ndarray]:
        self.net.eval()
        xn, xc = self._t(inp)
        rows = []
        for i in range(0, len(xn), 4096):
            q, k, lam, lc, mu, sig = _params(self.net(xn[i:i + 4096], xc[i:i + 4096]), self.H)
            en = q * lc
            em = torch.exp((mu + 0.5 * sig ** 2).clamp(max=14.0))
            tenure = q * lam * torch.exp(torch.lgamma(1 + 1 / k))  # E[time-to-next-stay | alive] weighted by alive prob
            rows.append(torch.stack([en * em, q * (1 - torch.exp(-lc)), en, q, tenure.clamp(max=3650.0)], 1).cpu())
        r = torch.cat(rows).numpy()
        return {"cltv": r[:, 0], "p_return": r[:, 1], "n_stays": r[:, 2], "p_alive": r[:, 3], "exp_days_to_next": r[:, 4]}

    def predict(self, inp: Inputs) -> np.ndarray:
        return self.predict_full(inp)["cltv"]

    def save(self, d: Path) -> None:
        torch.save({"state": self.net.state_dict(), "hp": self.hp, "H": self.H}, d / "deep_weibull_survival.pt")
        joblib.dump(self.prep, d / "num_scaler_surv.joblib")

    def load(self, d: Path) -> "WeibullSurvival":
        ck = torch.load(d / "deep_weibull_survival.pt", map_location=self.device)
        self.hp, self.H = ck["hp"], ck["H"]
        self.net = SurvNet(len(NUM_COLS), [len(CAT_LEVELS[c]) for c in CAT_COLS], self.hp["hidden"]).to(self.device)
        self.net.load_state_dict(ck["state"])
        self.prep = joblib.load(d / "num_scaler_surv.joblib")
        return self
