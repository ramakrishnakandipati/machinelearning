"""Common interface so evaluation / API treat every survival model uniformly.

Convention: `risk(X)` -> higher means *sooner rebooking* (higher hazard).
`survival(X, times)` -> array (n_samples, len(times)) of S(t|x), non-increasing in t.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class SurvivalModel:
    name: str = "base"
    device: str = "cpu"
    train_seconds: float = 0.0
    feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, t: np.ndarray, e: np.ndarray, X_val=None, t_val=None, e_val=None):
        raise NotImplementedError

    def risk(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # --- persistence (default: joblib pickle; subclasses override for .json / .pt) ---
    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.name}.pkl"
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path):
        return joblib.load(path)


def monotone_survival(s: np.ndarray) -> np.ndarray:
    """Clip to [0,1] and enforce non-increasing survival along the time axis."""
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return np.minimum.accumulate(s, axis=1)


def median_from_curve(times: np.ndarray, surv: np.ndarray) -> np.ndarray:
    """First time at which S(t) <= 0.5 (linear interpolation); NaN if the curve never crosses."""
    out = np.full(surv.shape[0], np.nan)
    for i, s in enumerate(surv):
        idx = np.argmax(s <= 0.5) if (s <= 0.5).any() else None
        if idx is None:
            continue
        if idx == 0:
            out[i] = times[0]
        else:
            t0, t1, s0, s1 = times[idx - 1], times[idx], s[idx - 1], s[idx]
            out[i] = t0 + (s0 - 0.5) * (t1 - t0) / max(s0 - s1, 1e-12)
    return out


# ----------------------------------------------------------------------------------------------
# Breslow baseline-hazard estimator shared by every Cox-type model (XGBoost Cox, DeepSurv)
# ----------------------------------------------------------------------------------------------
def breslow_baseline(margin: np.ndarray, t: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (event_times, H0) where H0 is the Breslow cumulative baseline hazard."""
    margin = np.asarray(margin, float)
    t, e = np.asarray(t, float), np.asarray(e, int)
    order = np.argsort(t)
    t_s, e_s, w = t[order], e[order], np.exp(margin[order] - margin.max())   # stabilised
    risk_sum = np.cumsum(w[::-1])[::-1]                                        # sum over {j: t_j >= t_i}
    uniq, first_idx = np.unique(t_s, return_index=True)
    d = np.add.reduceat(e_s, first_idx)
    rs = risk_sum[first_idx]
    h0 = d / rs * np.exp(-margin.max())                                        # undo stabilisation
    return uniq, np.cumsum(h0)


def breslow_survival(event_times: np.ndarray, H0: np.ndarray, margin: np.ndarray, times: np.ndarray) -> np.ndarray:
    """S(t|x) = exp(-H0(t) * exp(margin_x)) evaluated on `times` (step function interpolation)."""
    idx = np.searchsorted(event_times, np.asarray(times, float), side="right") - 1
    h = np.where(idx >= 0, H0[np.clip(idx, 0, None)], 0.0)
    return monotone_survival(np.exp(-np.outer(np.exp(np.clip(margin, -30, 30)), h)))
