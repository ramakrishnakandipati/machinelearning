"""BG/NBD + Gamma-Gamma probabilistic CLTV baseline (own vectorised implementation, no `lifetimes` dependency)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

from .common import Inputs

WEEK = 7.0


def _rft(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x repeat count, t_x, T) in weeks; T measured from first *completed* stay when one exists."""
    x = df["repeat_x"].values.astype(float)
    tx = df["t_x"].values / WEEK
    has = df["frequency"].values > 0
    T = np.where(has, df["t_x"].values + df["recency_days"].values, df["T_days"].values) / WEEK
    return x, np.minimum(tx, T), np.maximum(T, 1e-3)


class BGNBDGammaGamma:
    """BG/NBD (purchase process) x Gamma-Gamma (monetary value); CLTV = E[#stays in horizon] x E[value/stay]."""

    name = "bgnbd_gg"

    def __init__(self, horizon_days: float = 364.8) -> None:
        self.horizon_weeks = horizon_days / WEEK
        self.bg = np.array([1.0, 10.0, 1.0, 5.0])  # r, alpha, a, b
        self.gg = np.array([2.0, 3.0, 100.0])      # p, q, gamma
        self.global_m = 500.0

    # ---- BG/NBD -------------------------------------------------------------------------------
    @staticmethod
    def _bg_ll(params: np.ndarray, x: np.ndarray, tx: np.ndarray, T: np.ndarray) -> np.ndarray:
        r, al, a, b = params
        a1 = gammaln(r + x) - gammaln(r) + r * np.log(al)
        a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        a3 = -(r + x) * np.log(al + T)
        a4 = np.log(a) - np.log(np.maximum(b + x - 1, 1e-9)) - (r + x) * np.log(al + tx)
        return a1 + a2 + np.logaddexp(a3, np.where(x > 0, a4, -np.inf))

    def _p_alive(self, x: np.ndarray, tx: np.ndarray, T: np.ndarray) -> np.ndarray:
        r, al, a, b = self.bg
        ratio = a / np.maximum(b + x - 1, 1e-9) * ((al + T) / (al + tx)) ** (r + x)
        return 1.0 / (1.0 + np.where(x > 0, ratio, 0.0))

    def _expected_n(self, x: np.ndarray, tx: np.ndarray, T: np.ndarray, t: float) -> np.ndarray:
        r, al, a, b = self.bg
        z = t / (al + T + t)
        h = hyp2f1(r + x, b + x, a + b + x - 1, z)
        first = (a + b + x - 1) / (a - 1)
        second = 1 - ((al + T) / (al + T + t)) ** (r + x) * h
        den = 1 + np.where(x > 0, a / np.maximum(b + x - 1, 1e-9) * ((al + T) / (al + tx)) ** (r + x), 0.0)
        return np.maximum(first * second / den, 0.0)

    # ---- API ----------------------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame, lg=None) -> dict:
        x, tx, T = _rft(train_df)
        obj = lambda lp: -self._bg_ll(np.exp(lp), x, tx, T).sum() / len(x)
        lb = np.log([1e-3, 1e-3, 1.02, 1e-3])
        ub = np.log([50, 500, 200, 200])
        res = minimize(obj, np.log(self.bg), method="L-BFGS-B", bounds=list(zip(lb, ub)))
        self.bg = np.exp(res.x)
        m = train_df["monetary_mean"].values
        n = train_df["frequency"].values
        k = n > 0
        mk, nk = m[k], n[k]
        self.global_m = float(mk.mean())

        def gg_nll(lp: np.ndarray) -> float:
            p, q, v = np.exp(lp)
            ll = (gammaln(p * nk + q) - gammaln(p * nk) - gammaln(q) + q * np.log(v) + (p * nk - 1) * np.log(mk)
                  + p * nk * np.log(nk) - (p * nk + q) * np.log(nk * mk + v))
            return -ll.mean()

        r2 = minimize(gg_nll, np.log(self.gg), method="L-BFGS-B",
                      bounds=[(np.log(1e-2), np.log(100)), (np.log(1.05), np.log(100)), (np.log(1e-2), np.log(1e5))])
        self.gg = np.exp(r2.x)
        corr = np.corrcoef(nk[nk > 1], mk[nk > 1])[0, 1] if (nk > 1).sum() > 10 else float("nan")
        if lg:
            lg.info(f"[bgnbd_gg] BG/NBD r,alpha,a,b={np.round(self.bg, 4)} | GG p,q,gamma={np.round(self.gg, 3)} "
                    f"| corr(freq,monetary)={corr:.3f} (independence assumption check)")
        return {"train": [float(res.fun)], "val": [float(res.fun)]}

    def predict_full(self, inp: Inputs) -> dict[str, np.ndarray]:
        df = inp.df
        x, tx, T = _rft(df)
        en = self._expected_n(x, tx, T, self.horizon_weeks)
        p, q, v = self.gg
        n = df["frequency"].values.astype(float)
        m = df["monetary_mean"].values
        em = np.where(n > 0, p * (v + n * m) / (p * n + q - 1), self.global_m)
        return {"cltv": en * em, "p_return": 1 - np.exp(-en), "n_stays": en, "p_alive": self._p_alive(x, tx, T)}

    def predict(self, inp: Inputs) -> np.ndarray:
        return self.predict_full(inp)["cltv"]

    def save(self, d: Path) -> None:
        joblib.dump({"bg": self.bg, "gg": self.gg, "gm": self.global_m, "h": self.horizon_weeks}, d / "bgnbd_gg.joblib")

    def load(self, d: Path) -> "BGNBDGammaGamma":
        s = joblib.load(d / "bgnbd_gg.joblib")
        self.bg, self.gg, self.global_m, self.horizon_weeks = s["bg"], s["gg"], s["gm"], s["h"]
        return self
