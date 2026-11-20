"""GBDT CLTV models: GPU XGBoost (Tweedie), GPU CatBoost (Tweedie, native categoricals), two-stage LightGBM hurdle."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..features import CAT_COLS, NUM_COLS, tree_columns
from .common import Inputs


class XGBTweedie:
    name = "xgboost_tweedie"

    def __init__(self, hp: dict, device: str = "cuda") -> None:
        self.hp, self.device, self.model = hp, device, None
        self.cols = tree_columns()

    def fit(self, tr: Inputs, ytr: np.ndarray, va: Inputs, yva: np.ndarray, lg=None) -> dict:
        import xgboost as xgb
        hp = dict(self.hp)
        es = hp.pop("early_stopping_rounds")
        self.model = xgb.XGBRegressor(objective="reg:tweedie", tree_method="hist", device=self.device,
                                      early_stopping_rounds=es, eval_metric="tweedie-nloglik@" + str(hp["tweedie_variance_power"]),
                                      random_state=42, **hp)
        self.model.fit(tr.df[self.cols], ytr, eval_set=[(tr.df[self.cols], ytr), (va.df[self.cols], yva)], verbose=False)
        ev = self.model.evals_result()
        k = list(ev["validation_0"].keys())[0]
        if lg:
            lg.info(f"[{self.name}] device={self.device} best_iter={self.model.best_iteration}")
        return {"train": ev["validation_0"][k], "val": ev["validation_1"][k]}

    def predict(self, inp: Inputs) -> np.ndarray:
        return np.maximum(self.model.predict(inp.df[self.cols]), 0)

    def predict_full(self, inp: Inputs) -> dict[str, np.ndarray]:
        return {"cltv": self.predict(inp)}

    def contribs(self, inp: Inputs) -> np.ndarray:
        """Per-feature SHAP contributions (native TreeSHAP); last column is the bias term."""
        import xgboost as xgb
        return self.model.get_booster().predict(xgb.DMatrix(inp.df[self.cols]), pred_contribs=True)

    def save(self, d: Path) -> None:
        self.model.save_model(str(d / "xgboost_tweedie.json"))

    def load(self, d: Path) -> "XGBTweedie":
        import xgboost as xgb
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(d / "xgboost_tweedie.json"))
        self.model.set_params(device=self.device)
        return self


class CatBoostTweedie:
    name = "catboost_tweedie"

    def __init__(self, hp: dict, device: str = "cuda") -> None:
        self.hp, self.device, self.model = hp, device, None
        self.cols = NUM_COLS + CAT_COLS
        # CatBoost's GPU Tweedie starts from exp(0)=1 and stalls on $-scale targets -> train on y/1000 (deviance is scale-equivariant)
        self.scale = 1000.0

    def fit(self, tr: Inputs, ytr: np.ndarray, va: Inputs, yva: np.ndarray, lg=None) -> dict:
        from catboost import CatBoostRegressor, Pool
        ytr, yva = ytr / self.scale, yva / self.scale
        hp = dict(self.hp)
        es = hp.pop("early_stopping_rounds")
        self.model = CatBoostRegressor(task_type="GPU" if self.device == "cuda" else "CPU", random_seed=42,
                                       early_stopping_rounds=es, verbose=0, allow_writing_files=False, **hp)
        ptr = Pool(tr.df[self.cols], ytr, cat_features=CAT_COLS)
        pva = Pool(va.df[self.cols], yva, cat_features=CAT_COLS)
        self.model.fit(ptr, eval_set=pva, use_best_model=True)
        ev = self.model.get_evals_result()
        k = list(ev["learn"].keys())[0]
        if lg:
            lg.info(f"[{self.name}] task_type={'GPU' if self.device == 'cuda' else 'CPU'} best_iter={self.model.get_best_iteration()}")
        return {"train": ev["learn"][k], "val": ev["validation"][k]}

    def predict(self, inp: Inputs) -> np.ndarray:
        return np.maximum(self.model.predict(inp.df[self.cols]), 0) * self.scale

    def predict_full(self, inp: Inputs) -> dict[str, np.ndarray]:
        return {"cltv": self.predict(inp)}

    def save(self, d: Path) -> None:
        self.model.save_model(str(d / "catboost_tweedie.cbm"))

    def load(self, d: Path) -> "CatBoostTweedie":
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor()
        self.model.load_model(str(d / "catboost_tweedie.cbm"))
        return self


class LGBHurdle:
    """Two-stage hurdle: P(return) x E[spend | return]. (LightGBM pip wheels are CPU-only; logged explicitly.)"""
    name = "lightgbm_hurdle"

    def __init__(self, hp: dict) -> None:
        self.hp, self.clf, self.reg = hp, None, None
        self.cols = tree_columns()

    def fit(self, tr: Inputs, ytr: np.ndarray, va: Inputs, yva: np.ndarray, lg=None) -> dict:
        import lightgbm as lgb
        hp = dict(self.hp)
        es = hp.pop("early_stopping_rounds")
        cb = [lgb.early_stopping(es, verbose=False)]
        self.clf = lgb.LGBMClassifier(random_state=42, verbose=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, **hp)
        self.clf.fit(tr.df[self.cols], (ytr > 0).astype(int), eval_set=[(tr.df[self.cols], (ytr > 0).astype(int)),
                     (va.df[self.cols], (yva > 0).astype(int))], callbacks=cb + [lgb.record_evaluation(r1 := {})])
        mtr, mva = ytr > 0, yva > 0
        self.reg = lgb.LGBMRegressor(objective="gamma", random_state=42, verbose=-1, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, **hp)
        self.reg.fit(tr.df[self.cols][mtr], ytr[mtr], eval_set=[(tr.df[self.cols][mtr], ytr[mtr]),
                     (va.df[self.cols][mva], yva[mva])], callbacks=cb + [lgb.record_evaluation(r2 := {})])
        if lg:
            lg.info(f"[{self.name}] CPU (LightGBM pip wheel has no CUDA) | clf iters={self.clf.best_iteration_} reg iters={self.reg.best_iteration_}")
        a1, a2 = list(r1.values()), list(r2.values())
        k1, k2 = list(a1[0].keys())[0], list(a2[0].keys())[0]
        # plotted curve = stage-1 classifier logloss; stage-2 (gamma deviance) returned separately
        return {"train": a1[0][k1], "val": a1[1][k1], "train_stage2": a2[0][k2], "val_stage2": a2[1][k2]}

    def predict_full(self, inp: Inputs) -> dict[str, np.ndarray]:
        p = self.clf.predict_proba(inp.df[self.cols])[:, 1]
        m = self.reg.predict(inp.df[self.cols])
        return {"cltv": p * m, "p_return": p, "e_spend_given_return": m}

    def predict(self, inp: Inputs) -> np.ndarray:
        return self.predict_full(inp)["cltv"]

    def save(self, d: Path) -> None:
        import joblib
        joblib.dump({"clf": self.clf, "reg": self.reg}, d / "lightgbm_hurdle.joblib")

    def load(self, d: Path) -> "LGBHurdle":
        import joblib
        s = joblib.load(d / "lightgbm_hurdle.joblib")
        self.clf, self.reg = s["clf"], s["reg"]
        return self
