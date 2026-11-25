"""T-Learner uplift model (GPU XGBoost) that separates Persuadables from Sure Things etc."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


class TLearnerUplift:
    """Two outcome models: mu1(x)=P(y|x,treated), mu0(x)=P(y|x,control); uplift = mu1 - mu0."""

    def __init__(self, use_gpu: bool = True, seed: int = 42, **params):
        base = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=10, reg_lambda=5.0, tree_method="hist",
                    device="cuda" if use_gpu else "cpu", random_state=seed, eval_metric="logloss",
                    early_stopping_rounds=30)
        base.update(params)
        self.params = base
        self.m1 = self.m0 = None

    def fit(self, X, y, t, Xva=None, yva=None, tva=None):
        y, t = np.asarray(y), np.asarray(t)
        self.m1, self.m0 = xgb.XGBClassifier(**self.params), xgb.XGBClassifier(**self.params)
        for m, tv in ((self.m1, 1), (self.m0, 0)):
            msk = t == tv
            ev = None
            if Xva is not None:
                mv = np.asarray(tva) == tv
                ev = [(Xva[mv], np.asarray(yva)[mv])]
            m.fit(X[msk], y[msk], eval_set=ev, verbose=False)
        return self

    def predict(self, X) -> pd.DataFrame:
        p1, p0 = self.m1.predict_proba(X)[:, 1], self.m0.predict_proba(X)[:, 1]
        return pd.DataFrame({"p_treated": p1, "p_control": p0, "uplift": p1 - p0})

    def set_thresholds(self, ref: pd.DataFrame):
        """Fix segment cut-offs on a reference (validation) score distribution so single guests can be segmented."""
        self.thr = dict(u_hi=float(ref["uplift"].quantile(0.70)), u_lo=float(min(0.0, ref["uplift"].quantile(0.10))),
                        p_hi=float(ref["p_control"].quantile(0.60)))
        return self

    def segment(self, df: pd.DataFrame) -> pd.Series:
        """Persuadable: high positive uplift | Sleeping Dog: negative uplift | Sure Thing: high baseline, little uplift | Lost Cause."""
        t = self.thr
        seg = np.select([(df["uplift"] >= t["u_hi"]) & (df["uplift"] > 0), (df["uplift"] <= t["u_lo"]) & (df["uplift"] < 0),
                         df["p_control"] >= t["p_hi"]], ["Persuadable", "Sleeping Dog", "Sure Thing"], "Lost Cause")
        return pd.Series(seg, index=df.index, name="segment")

    def save(self, path):
        joblib.dump({"m1": self.m1, "m0": self.m0, "params": self.params, "thr": getattr(self, "thr", None)}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        o = cls()
        o.m1, o.m0, o.params, o.thr = d["m1"], d["m0"], d["params"], d["thr"]
        return o


def qini_curve(y, t, score, n_bins: int = 20) -> pd.DataFrame:
    """Cumulative incremental conversions when targeting the top-k% by score (Qini)."""
    d = pd.DataFrame({"y": y, "t": t, "s": score}).sort_values("s", ascending=False).reset_index(drop=True)
    n = len(d)
    rows = []
    for q in np.linspace(0, 1, n_bins + 1):
        k = int(q * n)
        s = d.iloc[:k]
        nt, nc = max((s.t == 1).sum(), 1), max((s.t == 0).sum(), 1)
        gain = (s[s.t == 1].y.sum() - s[s.t == 0].y.sum() * (nt / nc)) if k else 0.0
        rows.append((q, gain))
    q = pd.DataFrame(rows, columns=["frac_targeted", "qini"])
    return q


def auuc(y, t, score) -> float:
    q = qini_curve(y, t, score, 50)
    return float(np.trapezoid(q["qini"], q["frac_targeted"]))
