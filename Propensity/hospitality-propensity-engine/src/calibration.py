"""Probability calibration via sklearn CalibratedClassifierCV on top of a frozen score estimator.

The wrapped estimator simply passes through a model's raw probability (X = raw_p column), so the same
calibration code works for XGBoost / LightGBM / CatBoost / the PyTorch network.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


class ScorePassThrough(ClassifierMixin, BaseEstimator):
    """X is an (n,1) array of raw probabilities; predict_proba returns them as a 2-class output."""

    classes_ = np.array([0, 1])

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        p = np.clip(np.asarray(X, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def decision_function(self, X):
        p = np.clip(np.asarray(X, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))


def _fit(method: str, p: np.ndarray, y: np.ndarray) -> CalibratedClassifierCV:
    cal = CalibratedClassifierCV(FrozenEstimator(ScorePassThrough()), method=method)
    cal.fit(np.asarray(p).reshape(-1, 1), y)
    return cal


class Calibrator:
    """Chooses isotonic vs sigmoid (Platt) by 3-fold held-out log-loss, then refits on all data."""

    def __init__(self, method: str = "auto"):
        self.method = method
        self.cal: CalibratedClassifierCV | None = None
        self.chosen: str | None = None
        self.scores: dict = {}

    def fit(self, p_raw, y):
        p_raw, y = np.asarray(p_raw, float), np.asarray(y)
        methods = ["sigmoid", "isotonic"] if self.method == "auto" else [self.method]
        if len(methods) > 1:
            for m in methods:
                losses = []
                for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(p_raw, y):
                    c = _fit(m, p_raw[tr], y[tr])
                    losses.append(log_loss(y[te], np.clip(c.predict_proba(p_raw[te].reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)))
                self.scores[m] = float(np.mean(losses))
            self.chosen = min(self.scores, key=self.scores.get)
        else:
            self.chosen = methods[0]
        self.cal = _fit(self.chosen, p_raw, y)
        return self

    def predict(self, p_raw) -> np.ndarray:
        return np.clip(self.cal.predict_proba(np.asarray(p_raw, float).reshape(-1, 1))[:, 1], 0.0, 1.0)
