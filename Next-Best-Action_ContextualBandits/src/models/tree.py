"""Gradient-boosted-tree contextual bandits (XGBoost on GPU)."""
from pathlib import Path

import numpy as np
import xgboost as xgb

from ..config import XGB_DEVICE
from .base import BaseBandit


class _XGBBandit(BaseBandit):
    """One boosted regressor family on [x, onehot(a)]; ``n_models`` bootstrap members give uncertainty."""
    DEFAULTS = dict(n_models=1, n_estimators=300, max_depth=6, lr=0.08, subsample=0.8, colsample=0.8,
                    epsilon=0.0, beta=0.0, update_every=3, seed=0)

    def _make(self, i):
        return xgb.XGBRegressor(
            n_estimators=self.hp["n_estimators"], max_depth=self.hp["max_depth"], learning_rate=self.hp["lr"],
            subsample=self.hp["subsample"], colsample_bytree=self.hp["colsample"], tree_method="hist",
            device=XGB_DEVICE, random_state=self.hp["seed"] + i, reg_lambda=2.0, min_child_weight=5,
            eval_metric="rmse", verbosity=0)

    def _design(self, X, a):
        return np.hstack([X, np.eye(self.K, dtype=np.float32)[a]])

    def _fit(self, X, a, r, p, warm=False):
        n = len(a)
        rng = np.random.default_rng(self.hp["seed"] + self._rounds)
        perm = rng.permutation(n)
        n_val = max(int(0.1 * n), 1)
        val, tr = perm[:n_val], perm[n_val:]
        Zv, rv = self._design(X[val], a[val]), r[val]
        self.models = []
        for i in range(self.hp["n_models"]):
            idx = tr if self.hp["n_models"] == 1 else rng.choice(tr, size=len(tr), replace=True)
            m = self._make(i)
            m.fit(self._design(X[idx], a[idx]), r[idx], eval_set=[(self._design(X[idx], a[idx]), r[idx]), (Zv, rv)],
                  verbose=False)
            if i == 0:
                ev = m.evals_result()
                for t, (tl, vl) in enumerate(zip(ev["validation_0"]["rmse"], ev["validation_1"]["rmse"])):
                    if t % 5 == 0 or t == len(ev["validation_0"]["rmse"]) - 1:
                        self._log(event="epoch", epoch=t + 1, train_loss=tl, val_loss=vl, metric="rmse(scaled $)",
                                  kind="boost_round")
            self.models.append(m)

    def _member_preds(self, X):
        n = len(X)
        Z = np.hstack([np.repeat(X, self.K, axis=0), np.tile(np.eye(self.K, dtype=np.float32), (n, 1))])
        return np.stack([m.predict(Z).reshape(n, self.K) for m in self.models])  # (M,n,K)

    def _mean(self, X):
        return self._member_preds(X).mean(0)

    def _acting(self, X):
        P = self._member_preds(X)
        if self.hp["beta"] > 0 and len(self.models) > 1:
            return P.mean(0) + self.hp["beta"] * P.std(0)
        return P.mean(0)

    def feature_importance(self):
        """Mean gain importance per input feature, aggregated over members and actions."""
        names = (self.feature_names or [f"f{i}" for i in range(self.d)])
        gains = np.zeros(self.d)
        for m in self.models:
            sc = m.get_booster().get_score(importance_type="gain")
            for k, v in sc.items():
                j = int(k[1:])
                if j < self.d:
                    gains[j] += v / len(self.models)
        return dict(zip(names, gains))

    def _save_weights(self, stem):
        for i, m in enumerate(self.models):
            m.save_model(f"{stem}.m{i}.json")

    def _load_weights(self, stem):
        self.models = []
        for i in range(self.hp["n_models"]):
            m = self._make(i)
            m.load_model(f"{stem}.m{i}.json")
            self.models.append(m)


class XGBoostEpsilonGreedy(_XGBBandit):
    default_name = "XGBoost-EpsGreedy"
    DEFAULTS = {**_XGBBandit.DEFAULTS, "n_models": 1, "epsilon": 0.05}


class XGBoostUCB(_XGBBandit):
    """Bootstrapped-ensemble UCB: mean + beta * std across members (GPU-trained)."""
    default_name = "XGBoost-UCB"
    DEFAULTS = {**_XGBBandit.DEFAULTS, "n_models": 8, "beta": 1.0, "n_estimators": 200}
