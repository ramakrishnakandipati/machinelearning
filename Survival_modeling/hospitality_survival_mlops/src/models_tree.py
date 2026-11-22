"""Tree-based survival models: Random Survival Forest (sksurv) and GPU XGBoost (Cox / AFT)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import gumbel_l, logistic, norm
from sksurv.ensemble import RandomSurvivalForest

from .base import SurvivalModel, breslow_baseline, breslow_survival, monotone_survival
from .utils import get_logger, make_surv_array, timer

log = get_logger("tree")

_AFT_DISTS = {"normal": norm, "logistic": logistic, "extreme": gumbel_l}


class RSFModel(SurvivalModel):
    name = "random_survival_forest"

    def __init__(self, n_estimators=200, min_samples_leaf=15, max_depth=12, max_features="sqrt", seed=42):
        self.params = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                           max_depth=max_depth, max_features=max_features)
        self.seed = seed

    def fit(self, X, t, e, **_):
        self.feature_names = list(X.columns)
        with timer() as tm:
            self.model = RandomSurvivalForest(n_jobs=-1, random_state=self.seed, **self.params)
            self.model.fit(X, make_surv_array(t, e))
        self.train_seconds = tm["seconds"]
        return self

    def risk(self, X):
        return self.model.predict(X[self.feature_names])

    def survival(self, X, times):
        fns = self.model.predict_survival_function(X[self.feature_names])
        grid = np.asarray(times, float)
        lo, hi = self.model.unique_times_.min(), self.model.unique_times_.max()
        out = np.vstack([fn(np.clip(grid, lo, hi)) for fn in fns])
        return monotone_survival(out)


class XGBSurvivalModel(SurvivalModel):
    """XGBoost survival:cox or survival:aft trained with tree_method='hist' on CUDA when available."""

    def __init__(self, kind: str = "cox", device: str = "cuda", seed: int = 42, n_estimators: int = 400,
                 learning_rate: float = 0.05, max_depth: int = 4, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, min_child_weight: float = 1.0, reg_lambda: float = 1.0,
                 aft_loss_distribution: str = "normal", aft_loss_distribution_scale: float = 1.0):
        assert kind in ("cox", "aft")
        self.kind, self.device, self.seed, self.n_estimators = kind, device, seed, n_estimators
        self.dist, self.scale = aft_loss_distribution, aft_loss_distribution_scale
        self.name = f"xgboost_{kind}"
        self.params = {
            "objective": f"survival:{kind}",
            "tree_method": "hist", "device": device, "seed": seed,
            "eta": learning_rate, "max_depth": max_depth, "subsample": subsample,
            "colsample_bytree": colsample_bytree, "min_child_weight": min_child_weight,
            "lambda": reg_lambda,
        }
        if kind == "cox":
            self.params["eval_metric"] = "cox-nloglik"
        else:
            self.params.update(eval_metric="aft-nloglik", aft_loss_distribution=self.dist,
                               aft_loss_distribution_scale=self.scale)

    # -- data helpers -------------------------------------------------------------------------
    def _dmatrix(self, X, t, e):
        t, e = np.asarray(t, float), np.asarray(e, int)
        if self.kind == "cox":
            d = xgb.DMatrix(X, label=np.where(e == 1, t, -t))
        else:
            d = xgb.DMatrix(X)
            d.set_float_info("label_lower_bound", t)
            d.set_float_info("label_upper_bound", np.where(e == 1, t, np.inf))
        return d

    def fit(self, X, t, e, X_val=None, t_val=None, e_val=None):
        self.feature_names = list(X.columns)
        dtrain = self._dmatrix(X, t, e)
        evals, kw = [(dtrain, "train")], {}
        if X_val is not None:
            evals.append((self._dmatrix(X_val, t_val, e_val), "val"))
            kw["early_stopping_rounds"] = 30
        with timer() as tm:
            self.evals_result: dict = {}
            self.booster = xgb.train(self.params, dtrain, num_boost_round=self.n_estimators, evals=evals,
                                     evals_result=self.evals_result, verbose_eval=False, **kw)
            if self.kind == "cox":
                m = self.booster.predict(dtrain, output_margin=True)
                self.base_times, self.base_H0 = breslow_baseline(m, t, e)
        self.train_seconds = tm["seconds"]
        return self

    def _margin(self, X):
        d = xgb.DMatrix(X[self.feature_names])
        if self.kind == "cox":
            return self.booster.predict(d, output_margin=True)
        return np.log(np.clip(self.booster.predict(d), 1e-6, 1e9))   # AFT predicts time; margin = log(time)

    def risk(self, X):
        m = self._margin(X)
        return m if self.kind == "cox" else -m

    def survival(self, X, times):
        m = self._margin(X)
        if self.kind == "cox":
            return breslow_survival(self.base_times, self.base_H0, m, times)
        z = (np.log(np.clip(np.asarray(times, float), 1e-6, None))[None, :] - m[:, None]) / self.scale
        return monotone_survival(1.0 - _AFT_DISTS[self.dist].cdf(z))

    # -- persistence: native booster .json + sidecar metadata ---------------------------------
    def save(self, directory):
        d = Path(directory)
        self.booster.save_model(d / f"{self.name}.json")
        meta = {k: getattr(self, k) for k in ("kind", "device", "seed", "n_estimators", "dist", "scale",
                                              "name", "params", "feature_names", "train_seconds")}
        if self.kind == "cox":
            meta.update(base_times=self.base_times, base_H0=self.base_H0)
        joblib.dump(meta, d / f"{self.name}_meta.pkl")
        return d / f"{self.name}.json"

    @staticmethod
    def load(path):
        path = Path(path)
        meta = joblib.load(path.with_name(path.stem + "_meta.pkl"))
        obj = XGBSurvivalModel(kind=meta["kind"], device="cpu")   # inference on CPU is always safe
        for k, v in meta.items():
            setattr(obj, k, v)
        obj.device = "cpu"
        obj.params["device"] = "cpu"
        obj.booster = xgb.Booster()
        obj.booster.load_model(str(path))
        obj.booster.set_param({"device": "cpu"})
        return obj
