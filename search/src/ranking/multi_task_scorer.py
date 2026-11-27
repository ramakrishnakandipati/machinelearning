"""Multi-task funnel scorer: pCTR, pATC|click, pCVR|cart, pReturn|purchase -> expected revenue & profit.

Each head is a GPU XGBoost model trained on the (simulated-log) outcome of its funnel stage, with
conditional heads trained only on rows where the previous stage happened (weighted by event count).
"""
from __future__ import annotations

import numpy as np

from src.common.utils import get_device

HEADS = ["pctr", "patc", "pcvr", "pret"]


class MultiTaskScorer:
    def __init__(self):
        self.models = {}

    def fit(self, tr: dict, cfg: dict):
        import xgboost as xgb

        g = cfg["ranking"]["gbdt"]
        L = tr["L"]
        X = tr["X"]
        spec = {
            "pctr": (np.ones(len(X), bool), tr["n_click"] / L, np.ones(len(X))),
            "patc": (tr["n_click"] > 0, tr["n_atc"] / np.maximum(tr["n_click"], 1), tr["n_click"]),
            "pcvr": (tr["n_atc"] > 0, tr["n_pur"] / np.maximum(tr["n_atc"], 1), tr["n_atc"]),
            "pret": (tr["n_pur"] > 0, tr["y_ret"] / np.maximum(tr["n_pur"], 1), tr["n_pur"]),
        }
        for h, (mask, y, w) in spec.items():
            m = xgb.XGBRegressor(objective="binary:logistic", n_estimators=int(g["n_estimators"] * (1.0 if h == "pctr" else 0.6)),
                                 learning_rate=g["learning_rate"], max_depth=5 if h != "pret" else 4, tree_method="hist",
                                 device=get_device(), subsample=0.85, colsample_bytree=0.85, min_child_weight=3)
            m.fit(X[mask], np.clip(y[mask], 0, 1), sample_weight=w[mask])
            self.models[h] = m
        return self

    def predict(self, X: np.ndarray) -> dict:
        return {h: np.clip(self.models[h].predict(X), 1e-4, 1 - 1e-4) for h in HEADS}

    @staticmethod
    def utility(p: dict, price: np.ndarray, margin_usd: np.ndarray) -> dict:
        p_buy = p["pctr"] * p["patc"] * p["pcvr"]
        return {
            "p_purchase": p_buy,
            "expected_revenue": p_buy * price * (1 - p["pret"]),
            # E[profit] = P(click)P(cart|click)P(purchase|cart) * margin * (1 - P(return))
            "expected_profit": p_buy * margin_usd * (1 - p["pret"]),
        }
