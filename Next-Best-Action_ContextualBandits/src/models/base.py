"""Common interface for all contextual bandit policies.

Conventions
-----------
* Constructors are ``Model(n_features, n_actions=6, name=None, **hyperparams)``.
* Rewards are standardised internally (``r_scale``); public ``mean_scores`` are in $ units.
* ``fit`` = offline batch learning from logged bandit feedback (X, a, r, p).
* ``update`` = incremental learning (buffer + warm refit) used by the online replay and the API.
* ``select_action`` = the *acting* policy (exploration included); ``action_probs`` = its action
  distribution (Monte-Carlo for stochastic policies) used by off-policy evaluation.
"""
import time
from pathlib import Path

import joblib
import numpy as np


class BaseBandit:
    DEFAULTS: dict = {}
    stochastic = False
    default_name = None

    def __init__(self, n_features, n_actions=6, name=None, **hp):
        self.d = int(n_features)
        self.K = int(n_actions)
        self.hp = {**self.DEFAULTS, **hp}
        self.name = name or self.default_name or type(self).__name__
        self.epsilon = float(self.hp.get("epsilon", 0.0))
        self.update_every = int(self.hp.get("update_every", 1))
        self.r_scale = 1.0
        self.fitted = False
        self._buf = None
        self._rounds = 0
        self.logger = None
        self.feature_names = None
        self.rng = np.random.default_rng(int(self.hp.get("seed", 0)))

    # ------------------------------------------------------------------ hooks
    def _fit(self, X, a, r, p, warm=False):
        raise NotImplementedError

    def _mean(self, X) -> np.ndarray:
        raise NotImplementedError

    def _acting(self, X) -> np.ndarray:
        return self._mean(X)

    def _save_weights(self, stem: Path):
        raise NotImplementedError

    def _load_weights(self, stem: Path):
        raise NotImplementedError

    # --------------------------------------------------------------- utilities
    def _log(self, **rec):
        if self.logger is not None:
            self.logger({"model": self.name, **rec})

    @staticmethod
    def _prep(X, a, r, p, K):
        X = np.ascontiguousarray(X, dtype=np.float32)
        a = np.asarray(a, dtype=np.int64)
        r = np.asarray(r, dtype=np.float64)
        p = np.full(len(a), 1.0 / K) if p is None else np.asarray(p, dtype=np.float64)
        return X, a, r, p

    # ---------------------------------------------------------------- learning
    def fit(self, X, a, r, p=None):
        X, a, r, p = self._prep(X, a, r, p, self.K)
        self.r_scale = float(max(r.std(), 1e-3))
        self._buf = [X, a, r, p]
        t0 = time.time()
        self._fit(X, a, r / self.r_scale, p, warm=False)
        self.fitted = True
        self._log(event="fit_done", n=len(a), seconds=round(time.time() - t0, 3))
        return self

    def update(self, X, a, r, p=None):
        X, a, r, p = self._prep(X, a, r, p, self.K)
        if self._buf is None:
            self._buf = [X, a, r, p]
            self.r_scale = float(max(r.std(), 1e-3))
        else:
            self._buf = [np.concatenate([o, n]) for o, n in zip(self._buf, (X, a, r, p))]
        self._rounds += 1
        if (not self.fitted) or self._rounds % self.update_every == 0:
            Xb, ab, rb, pb = self._buf
            self._fit(Xb, ab, rb / self.r_scale, pb, warm=self.fitted)
            self.fitted = True
        return self

    # --------------------------------------------------------------- inference
    def mean_scores(self, X):
        """Estimated expected net margin ($) per action, shape (n, K)."""
        return self._mean(np.ascontiguousarray(X, dtype=np.float32)) * self.r_scale

    def acting_scores(self, X):
        return self._acting(np.ascontiguousarray(X, dtype=np.float32)) * self.r_scale

    def select_action(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        n = len(X)
        if not self.fitted:
            return self.rng.integers(0, self.K, n)
        a = self._acting(X).argmax(1)
        if self.epsilon > 0:
            m = self.rng.random(n) < self.epsilon
            a[m] = self.rng.integers(0, self.K, int(m.sum()))
        return a

    def action_probs(self, X, n_mc=40):
        X = np.ascontiguousarray(X, dtype=np.float32)
        n = len(X)
        if not self.fitted:
            return np.full((n, self.K), 1.0 / self.K)
        if self.stochastic:
            probs = np.zeros((n, self.K))
            for _ in range(n_mc):
                probs[np.arange(n), self._acting(X).argmax(1)] += 1.0
            probs /= n_mc
        else:
            probs = np.zeros((n, self.K))
            probs[np.arange(n), self._acting(X).argmax(1)] = 1.0
        if self.epsilon > 0:
            probs = (1 - self.epsilon) * probs + self.epsilon / self.K
        return probs

    # ------------------------------------------------------------- persistence
    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"cls": type(self).__name__, "name": self.name, "d": self.d, "K": self.K, "hp": self.hp,
                     "r_scale": self.r_scale, "feature_names": self.feature_names}, out_dir / f"{self.name}.meta.joblib")
        if self._buf is not None:
            X, a, r, p = self._buf
            np.savez_compressed(out_dir / f"{self.name}.buffer.npz", X=X, a=a, r=r, p=p)
        self._save_weights(out_dir / self.name)

    def _restore(self, meta, out_dir):
        self.r_scale = meta["r_scale"]
        self.feature_names = meta.get("feature_names")
        stem = Path(out_dir) / self.name
        self._load_weights(stem)
        buf = Path(out_dir) / f"{self.name}.buffer.npz"
        if buf.exists():
            z = np.load(buf)
            self._buf = [z["X"], z["a"], z["r"], z["p"]]
        self.fitted = True
        return self
