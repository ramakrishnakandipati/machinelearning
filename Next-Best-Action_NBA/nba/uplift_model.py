"""Module 3a - Multi-treatment causal uplift models (EconML DRLearner + LightGBM).

Two doubly-robust learners estimate, for each (customer, action) pair:
  * tau_margin(x, a) - incremental 30-day gross margin vs. control (action 0)
  * tau_unsub(x, a)  - incremental unsubscribe probability vs. control
Nuisance models (propensity, outcome) and the final CATE stage are all LightGBM.
Persistence uses joblib + a JSON manifest consumed by the inference service.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import econml
import joblib
import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn
from econml.dr import DRLearner
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, r2_score, roc_auc_score
from sklearn.model_selection import KFold

from .artifact_manager import ArtifactManager
from .config import (
    ACTION_COSTS, ACTION_IDS, COLD_START_SHRINKAGE, DEFAULT_LAMBDA, EXPLORATION_EPSILON,
    FEATURE_COLUMNS, SEED, TREATMENT_IDS,
)
from .features import FeatureTransformer

warnings.filterwarnings("ignore", category=UserWarning)
FloatArr = npt.NDArray[np.float64]

_COMMON: dict[str, Any] = dict(verbose=-1, n_jobs=4, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)


# --------------------------------------------------------------------------- #
# Base-learner factories (shared by training and the nuisance diagnostics)
# --------------------------------------------------------------------------- #
def make_propensity_model(seed: int = SEED) -> LGBMClassifier:
    """Multi-class generalized propensity model P(T=a | x)."""
    return LGBMClassifier(n_estimators=120, learning_rate=0.05, num_leaves=15,
                          min_child_samples=50, random_state=seed, **_COMMON)


def make_outcome_regressor(seed: int = SEED) -> LGBMRegressor:
    """Outcome nuisance E[Y | x, T]."""
    return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                         min_child_samples=50, random_state=seed, **_COMMON)


def make_final_regressor(seed: int = SEED) -> LGBMRegressor:
    """CATE stage: heavily regularised because DR pseudo-outcomes are noisy."""
    return LGBMRegressor(n_estimators=150, learning_rate=0.03, num_leaves=7,
                         min_child_samples=300, reg_lambda=10.0, random_state=seed, **_COMMON)


def make_dr_learner(seed: int = SEED, cv: int = 3) -> DRLearner:
    """Multi-treatment DR-Learner (control = treatment 0, the smallest label)."""
    return DRLearner(
        model_propensity=make_propensity_model(seed),
        model_regression=make_outcome_regressor(seed),
        model_final=make_final_regressor(seed),
        cv=cv, min_propensity=0.01, random_state=seed,
    )


def with_action_onehot(x: pd.DataFrame | npt.NDArray[np.float32], action: npt.NDArray[np.int64]) -> pd.DataFrame:
    """Append one-hot action columns (outcome-model input)."""
    df = pd.DataFrame(np.asarray(x), columns=FEATURE_COLUMNS) if not isinstance(x, pd.DataFrame) else x.reset_index(drop=True)
    out = df.copy()
    for a in ACTION_IDS:
        out[f"T_{a}"] = (np.asarray(action) == a).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# Fitted bundle
# --------------------------------------------------------------------------- #
@dataclass
class UpliftBundle:
    """Everything needed to score (customer, action) pairs, online or offline."""

    margin_est: DRLearner
    unsub_est: DRLearner
    transformer: FeatureTransformer
    baseline_model: LGBMClassifier            # P(purchase | x, no action) for "Sure Thing" detection
    global_ate_margin: FloatArr               # (4,) population ATE, used to shrink cold-start CATEs
    global_ate_unsub: FloatArr
    cv_metrics: dict[str, Any] = field(default_factory=dict)
    # Fast path: raw LightGBM final-stage models (skips EconML input validation overhead).
    _margin_fast: list[Any] = field(default_factory=list, repr=False)
    _unsub_fast: list[Any] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.build_fast_path()

    def build_fast_path(self) -> None:
        """Extract per-treatment final-stage LightGBM boosters from the DRLearners."""
        self._margin_fast = [self.margin_est.model_cate(T=t) for t in TREATMENT_IDS]
        self._unsub_fast = [self.unsub_est.model_cate(T=t) for t in TREATMENT_IDS]
        self._col = {c: i for i, c in enumerate(self.transformer.columns)}

    # ---------------------------------------------------------------- scoring
    def predict_tau(self, x: npt.NDArray[np.float32] | pd.DataFrame) -> tuple[FloatArr, FloatArr]:
        """Return ``(tau_margin, tau_unsub)``, each shaped (n, 4) for actions 1..4.

        Cold-start rows are shrunk toward the global ATE (individual CATEs learned
        from sparse history are unreliable for guests with <2 sessions).
        """
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        tm = np.column_stack([m.predict(arr) for m in self._margin_fast]).astype(np.float64)
        tu = np.column_stack([m.predict(arr) for m in self._unsub_fast]).astype(np.float64)
        cold = arr[:, self._col["is_cold_start"]] == 1
        if cold.any():
            s = COLD_START_SHRINKAGE
            tm[cold] = s * tm[cold] + (1 - s) * self.global_ate_margin
            tu[cold] = s * tu[cold] + (1 - s) * self.global_ate_unsub
        return tm, tu

    def baseline_purchase_prob(self, x: npt.NDArray[np.float32] | pd.DataFrame) -> FloatArr:
        """P(purchase | x, action=0): high values flag organic "Sure Things"."""
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        ext = np.hstack([arr, np.tile(np.array([1, 0, 0, 0, 0], dtype=np.float32), (len(arr), 1))])
        return self.baseline_model.predict_proba(ext)[:, 1]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def cross_validated_nuisance_metrics(x: pd.DataFrame, log: pd.DataFrame, seed: int = SEED, folds: int = 3) -> dict[str, float]:
    """K-fold CV quality of the nuisance stage (propensity + outcome models)."""
    t = log["action_id"].to_numpy()
    xt = with_action_onehot(x, t)
    ll, auc, r2m, aucu = [], [], [], []
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(x):
        p = make_propensity_model(seed).fit(x.iloc[tr], t[tr])
        proba = p.predict_proba(x.iloc[te])
        ll.append(log_loss(t[te], proba, labels=list(ACTION_IDS)))
        auc.append(roc_auc_score(t[te], proba, multi_class="ovr", labels=list(ACTION_IDS)))
        om = make_outcome_regressor(seed).fit(xt.iloc[tr], log["gross_margin_dollars"].to_numpy()[tr])
        r2m.append(r2_score(log["gross_margin_dollars"].to_numpy()[te], om.predict(xt.iloc[te])))
        uc = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=7, min_child_samples=100,
                            random_state=seed, **_COMMON).fit(xt.iloc[tr], log["unsubscribed_flag"].to_numpy()[tr].astype(int))
        aucu.append(roc_auc_score(log["unsubscribed_flag"].to_numpy()[te].astype(int), uc.predict_proba(xt.iloc[te])[:, 1]))
    return {
        "propensity_logloss": float(np.mean(ll)), "propensity_auc_ovr": float(np.mean(auc)),
        "margin_outcome_r2": float(np.mean(r2m)), "unsub_outcome_auc": float(np.mean(aucu)),
        "cv_folds": folds,
    }


def fit_uplift_bundle(x_train: pd.DataFrame, log_train: pd.DataFrame, transformer: FeatureTransformer,
                      seed: int = SEED, cv: int = 3, verbose: bool = True) -> UpliftBundle:
    """Fit margin + unsubscribe DR-learners and the baseline-purchase model.

    Only *mature* rewards (7d conversion + 30d return windows closed) must be
    passed in; the caller is responsible for that filtering.
    """
    t = log_train["action_id"].to_numpy()
    if verbose:
        print(f"[uplift] fitting on {len(x_train):,} mature rows; arm sizes: {dict(zip(*np.unique(t, return_counts=True)))}")
    margin_est = make_dr_learner(seed, cv).fit(log_train["gross_margin_dollars"].to_numpy(), t, X=x_train.to_numpy())
    unsub_est = make_dr_learner(seed, cv).fit(log_train["unsubscribed_flag"].to_numpy().astype(float), t, X=x_train.to_numpy())

    xt = with_action_onehot(x_train, t)
    baseline = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                              random_state=seed, **_COMMON).fit(xt.to_numpy(), log_train["purchased_flag"].to_numpy().astype(int))
    xa = x_train.to_numpy()
    ate_m = np.array([margin_est.model_cate(T=a).predict(xa).mean() for a in TREATMENT_IDS])
    ate_u = np.array([unsub_est.model_cate(T=a).predict(xa).mean() for a in TREATMENT_IDS])
    metrics = cross_validated_nuisance_metrics(x_train.reset_index(drop=True), log_train.reset_index(drop=True), seed)
    return UpliftBundle(margin_est, unsub_est, transformer, baseline, ate_m, ate_u, metrics)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_bundle(bundle: UpliftBundle, am: ArtifactManager, lam: float = DEFAULT_LAMBDA,
                extra_metrics: dict[str, Any] | None = None) -> Path:
    """Serialize models with joblib and write ``model_manifest.json``."""
    d = am.dir("saved_models")
    joblib.dump(bundle.margin_est, d / "margin_uplift_model.joblib")
    joblib.dump(bundle.unsub_est, d / "unsub_uplift_model.joblib")
    joblib.dump(bundle.transformer, d / "feature_transformer.joblib")
    joblib.dump(bundle.baseline_model, d / "baseline_purchase_model.joblib")
    manifest = {
        "model_version": datetime.now(timezone.utc).strftime("nba-%Y%m%dT%H%M%SZ"),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(bundle.transformer.columns),
        "action_costs": {str(k): v for k, v in ACTION_COSTS.items()},
        "risk_aversion_lambda": lam,
        "exploration_epsilon": EXPLORATION_EPSILON,
        "global_ate_margin": bundle.global_ate_margin.tolist(),
        "global_ate_unsub": bundle.global_ate_unsub.tolist(),
        "cross_validation_metrics": bundle.cv_metrics,
        "holdout_metrics": extra_metrics or {},
        "libraries": {"econml": econml.__version__, "lightgbm": lgb.__version__, "sklearn": sklearn.__version__},
    }
    return am.write_json("saved_models", "model_manifest.json", manifest)


def update_manifest(am: ArtifactManager, **fields: Any) -> None:
    """Merge fields (e.g. the chosen lambda, holdout metrics) into the manifest."""
    m = am.read_json("saved_models", "model_manifest.json")
    m.update(fields)
    am.write_json("saved_models", "model_manifest.json", m)


def load_bundle(models_dir: Path) -> tuple[UpliftBundle, dict[str, Any]]:
    """Load the serialized bundle + manifest (used by the FastAPI service)."""
    import json
    manifest = json.loads((models_dir / "model_manifest.json").read_text(encoding="utf-8"))
    bundle = UpliftBundle(
        margin_est=joblib.load(models_dir / "margin_uplift_model.joblib"),
        unsub_est=joblib.load(models_dir / "unsub_uplift_model.joblib"),
        transformer=joblib.load(models_dir / "feature_transformer.joblib"),
        baseline_model=joblib.load(models_dir / "baseline_purchase_model.joblib"),
        global_ate_margin=np.array(manifest["global_ate_margin"]),
        global_ate_unsub=np.array(manifest["global_ate_unsub"]),
        cv_metrics=manifest.get("cross_validation_metrics", {}),
    )
    return bundle, manifest


if __name__ == "__main__":
    from .synthetic_data import make_dataset

    ds = make_dataset(8000)
    b = fit_uplift_bundle(ds.x.iloc[ds.train_idx], ds.log.iloc[ds.train_idx], ds.transformer)
    tm, tu = b.predict_tau(ds.x.iloc[ds.test_idx])
    truth = ds.log.iloc[ds.test_idx][[f"truth_tau_margin_a{a}" for a in TREATMENT_IDS]].to_numpy()
    print("corr(pred, truth) per action:", [round(float(np.corrcoef(tm[:, j], truth[:, j])[0, 1]), 3) for j in range(4)])
    print("cv metrics:", b.cv_metrics)
