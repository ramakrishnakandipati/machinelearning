"""GPU gradient-boosting wrappers: XGBoost (CUDA hist), LightGBM (GPU with CPU fallback), CatBoost (GPU).

All wrappers share:  fit(Xtr, ytr, Xva, yva, params, trial=None) / predict_proba(X)->1d / save(path) / load(path)
X is the output of `feature_engineering.build_preprocessor(..., mode="gbdt")` (DataFrame, categorical
ordinal codes first, -1 = unknown/missing).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import optuna

log = logging.getLogger("gbdt_models")


# --------------------------------------------------------------------------- search spaces
def suggest_xgboost(trial: optuna.Trial) -> dict:
    return dict(
        max_depth=trial.suggest_int("max_depth", 3, 9),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 30, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
        gamma=trial.suggest_float("gamma", 0, 3),
    )


def suggest_lightgbm(trial: optuna.Trial) -> dict:
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 8, 255, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 150, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
        max_bin=trial.suggest_categorical("max_bin", [63, 127, 255]),
    )


def suggest_catboost(trial: optuna.Trial) -> dict:
    return dict(
        depth=trial.suggest_int("depth", 4, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        random_strength=trial.suggest_float("random_strength", 0.1, 5, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0, 3),
        border_count=trial.suggest_categorical("border_count", [64, 128, 254]),
    )


# --------------------------------------------------------------------------- XGBoost
class XGBoostModel:
    name = "xgboost"
    ext = "json"

    def __init__(self, cat_cols: list[str], use_gpu: bool = True, es_rounds: int = 30, max_rounds: int = 1500, seed: int = 42):
        self.cat_cols, self.use_gpu, self.es, self.max_rounds, self.seed = cat_cols, use_gpu, es_rounds, max_rounds, seed
        self.model = None
        self.device = "cuda" if use_gpu else "cpu"

    def fit(self, Xtr, ytr, Xva, yva, params, trial=None):
        import xgboost as xgb

        cbs = []
        if trial is not None:
            class _Prune(xgb.callback.TrainingCallback):
                def after_iteration(self_, model, epoch, evals_log):  # noqa: N805
                    if epoch % 10 == 9:
                        trial.report(float(evals_log["validation_0"]["aucpr"][-1]), epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    return False
            cbs.append(_Prune())
        p = dict(params)
        p.setdefault("learning_rate", 0.1)
        self.model = xgb.XGBClassifier(
            tree_method="hist", device=self.device, n_estimators=self.max_rounds, eval_metric="aucpr",
            early_stopping_rounds=self.es, random_state=self.seed, n_jobs=4, callbacks=cbs or None, max_bin=256, **p)
        self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        self.best_iteration = int(self.model.best_iteration) + 1
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path):
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: Path, cat_cols=(), use_gpu=True):
        import xgboost as xgb

        o = cls(list(cat_cols), use_gpu)
        o.model = xgb.XGBClassifier()
        o.model.load_model(str(path))
        o.model.set_params(device=o.device)
        return o


# --------------------------------------------------------------------------- LightGBM
_LGB_GPU: dict[str, bool] = {}


def lightgbm_device() -> str:
    """Probe LightGBM for CUDA, then OpenCL GPU, else CPU (the pip wheel is usually CPU-only)."""
    if "dev" in _LGB_GPU:
        return _LGB_GPU["dev"]
    import lightgbm as lgb

    X = np.random.rand(300, 4)
    y = (X[:, 0] > 0.5).astype(int)
    dev = "cpu"
    for cand in ("cuda", "gpu"):
        try:
            lgb.LGBMClassifier(device_type=cand, n_estimators=3, verbose=-1).fit(X, y)
            dev = cand
            break
        except Exception:
            continue
    if dev == "cpu":
        log.warning("LightGBM GPU/CUDA tree learner unavailable in this build -> falling back to CPU")
    _LGB_GPU["dev"] = dev
    return dev


class LightGBMModel:
    name = "lightgbm"
    ext = "txt"

    def __init__(self, cat_cols: list[str], use_gpu: bool = True, es_rounds: int = 30, max_rounds: int = 1500, seed: int = 42):
        self.cat_cols, self.es, self.max_rounds, self.seed = cat_cols, es_rounds, max_rounds, seed
        self.device = lightgbm_device() if use_gpu else "cpu"
        self.model = None

    def fit(self, Xtr, ytr, Xva, yva, params, trial=None):
        import lightgbm as lgb

        cbs = [lgb.early_stopping(self.es, verbose=False)]
        if trial is not None:
            def _prune(env):
                if env.iteration % 10 == 9:
                    trial.report(float(env.evaluation_result_list[0][2]), env.iteration)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            cbs.append(_prune)
        self.model = lgb.LGBMClassifier(
            n_estimators=self.max_rounds, device_type=self.device, bagging_freq=1, random_state=self.seed,
            verbose=-1, n_jobs=8, **params)
        cat = [c for c in self.cat_cols if c in Xtr.columns]
        self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                       categorical_feature=cat or "auto", callbacks=cbs)
        self.best_iteration = int(self.model.best_iteration_ or self.model.n_estimators)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path):
        self.model.booster_.save_model(str(path))

    @classmethod
    def load(cls, path: Path, cat_cols=(), use_gpu=True):
        import lightgbm as lgb

        o = cls(list(cat_cols), use_gpu)
        booster = lgb.Booster(model_file=str(path))

        class _B:  # minimal predict_proba shim around a loaded booster
            def predict_proba(self, X):
                p = booster.predict(X)
                return np.c_[1 - p, p]
        o.model = _B()
        return o


# --------------------------------------------------------------------------- CatBoost
class CatBoostModel:
    name = "catboost"
    ext = "cbm"

    def __init__(self, cat_cols: list[str], use_gpu: bool = True, es_rounds: int = 40, max_rounds: int = 800, seed: int = 42):
        self.cat_cols, self.use_gpu, self.es, self.max_rounds, self.seed = cat_cols, use_gpu, es_rounds, max_rounds, seed
        self.model = None

    def _prep(self, X):
        X = X.copy()
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = X[c].astype(int)
        return X

    def fit(self, Xtr, ytr, Xva, yva, params, trial=None):
        from catboost import CatBoostClassifier

        p = dict(params)
        kw = dict(task_type="GPU", devices="0", gpu_ram_part=0.25) if self.use_gpu else dict(task_type="CPU", thread_count=8)
        cat_idx = [Xtr.columns.get_loc(c) for c in self.cat_cols if c in Xtr.columns]
        if self.use_gpu:
            p.pop("bagging_temperature", None) if p.get("bootstrap_type") == "Bernoulli" else None
            p.setdefault("bootstrap_type", "Bayesian")
        self.model = CatBoostClassifier(
            iterations=self.max_rounds, loss_function="Logloss", eval_metric="Logloss", random_seed=self.seed,
            early_stopping_rounds=self.es, verbose=False, allow_writing_files=False, cat_features=cat_idx, **kw, **p)
        self.model.fit(self._prep(Xtr), ytr, eval_set=(self._prep(Xva), yva), use_best_model=True)
        self.best_iteration = int(self.model.get_best_iteration()) + 1
        if trial is not None:  # CatBoost has no per-iteration hook: report the final value once
            trial.report(float(self.model.get_best_score()["validation"]["Logloss"]), 0)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(self._prep(X))[:, 1]

    def save(self, path: Path):
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: Path, cat_cols=(), use_gpu=True):
        from catboost import CatBoostClassifier

        o = cls(list(cat_cols), use_gpu)
        o.model = CatBoostClassifier()
        o.model.load_model(str(path))
        return o


SUGGEST = {"xgboost": suggest_xgboost, "lightgbm": suggest_lightgbm, "catboost": suggest_catboost}
CLASSES = {"xgboost": XGBoostModel, "lightgbm": LightGBMModel, "catboost": CatBoostModel}
