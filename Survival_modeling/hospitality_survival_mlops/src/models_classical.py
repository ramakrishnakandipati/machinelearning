"""lifelines models: Kaplan-Meier, Nelson-Aalen (baselines), CoxPH, Weibull AFT, Log-Normal AFT."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import (CoxPHFitter, KaplanMeierFitter, LogNormalAFTFitter, NelsonAalenFitter,
                       WeibullAFTFitter)
from lifelines.statistics import proportional_hazard_test

from .base import SurvivalModel, monotone_survival
from .utils import get_logger, timer

log = get_logger("classical")


def _frame(X: pd.DataFrame, t, e) -> pd.DataFrame:
    d = X.reset_index(drop=True).copy()
    d["T"] = np.asarray(t, float)
    d["E"] = np.asarray(e, int)
    return d


class KaplanMeierModel(SurvivalModel):
    """Population-level baseline: identical curve for everyone (risk is constant -> C-index 0.5)."""
    name = "kaplan_meier"

    def fit(self, X, t, e, **_):
        with timer() as tm:
            self.km = KaplanMeierFitter().fit(t, e)
        self.train_seconds = tm["seconds"]
        return self

    def risk(self, X):
        return np.zeros(len(X))

    def survival(self, X, times):
        s = self.km.survival_function_at_times(times).to_numpy()
        return monotone_survival(np.tile(s, (len(X), 1)))


class NelsonAalenModel(SurvivalModel):
    """Population-level baseline: S(t) = exp(-H_NA(t))."""
    name = "nelson_aalen"

    def fit(self, X, t, e, **_):
        with timer() as tm:
            self.na = NelsonAalenFitter().fit(t, e)
        self.train_seconds = tm["seconds"]
        return self

    def risk(self, X):
        return np.zeros(len(X))

    def survival(self, X, times):
        h = self.na.cumulative_hazard_at_times(times).to_numpy()
        return monotone_survival(np.tile(np.exp(-h), (len(X), 1)))


class CoxPHModel(SurvivalModel):
    name = "coxph"

    def __init__(self, penalizer: float = 0.05, l1_ratio: float = 0.2):
        self.penalizer, self.l1_ratio = penalizer, l1_ratio

    def fit(self, X, t, e, **_):
        self.feature_names = list(X.columns)
        with timer() as tm:
            self.model = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
            self.model.fit(_frame(X, t, e), duration_col="T", event_col="E")
            self._train = _frame(X, t, e)
        self.train_seconds = tm["seconds"]
        return self

    def risk(self, X):
        return self.model.predict_partial_hazard(X[self.feature_names]).to_numpy().ravel()

    def survival(self, X, times):
        sf = self.model.predict_survival_function(X[self.feature_names], times=np.asarray(times))
        return monotone_survival(sf.to_numpy().T)

    def ph_violations(self, alpha: float = 0.01) -> pd.DataFrame:
        """Schoenfeld-residual PH test (rank transform) on the training data."""
        res = proportional_hazard_test(self.model, self._train, time_transform="rank")
        tbl = res.summary[["test_statistic", "p"]].sort_values("p")
        tbl["violates_PH"] = tbl["p"] < alpha
        return tbl

    def save(self, directory):
        self._train = None       # keep the artefact light
        return super().save(directory)


class _AFTBase(SurvivalModel):
    fitter_cls = None

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer

    def fit(self, X, t, e, **_):
        self.feature_names = list(X.columns)
        with timer() as tm:
            self.model = self.fitter_cls(penalizer=self.penalizer)
            self.model.fit(_frame(X, t, e), duration_col="T", event_col="E")
        self.train_seconds = tm["seconds"]
        return self

    def risk(self, X):
        med = self.model.predict_median(X[self.feature_names]).to_numpy().ravel()
        return -np.log(np.clip(med, 1e-6, 1e9))        # shorter predicted time -> higher risk

    def survival(self, X, times):
        sf = self.model.predict_survival_function(X[self.feature_names], times=np.asarray(times))
        return monotone_survival(sf.to_numpy().T)


class WeibullAFTModel(_AFTBase):
    name = "weibull_aft"
    fitter_cls = WeibullAFTFitter


class LogNormalAFTModel(_AFTBase):
    name = "lognormal_aft"
    fitter_cls = LogNormalAFTFitter
