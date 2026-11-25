"""Inference engine shared by the FastAPI service and the Streamlit dashboard.

Loads the best (by validation PR-AUC) calibrated model for each scenario, scores any number of guest
profiles across all 8 scenarios, maps scores to decile tiers and next-best actions, and gives
per-guest SHAP drivers.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .common import ROOT, cuda_info, gpu_enabled, load_config, path_of
from .feature_engineering import ZERO_FILL
from .models.deep_tabular import TorchTabularModel
from .models.gbdt_models import CLASSES
from .models.uplift_model import TLearnerUplift

TREE = ["xgboost", "lightgbm", "catboost"]


class PropensityEngine:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.use_gpu = gpu_enabled(self.cfg)
        self.mroot = path_of(self.cfg, "models")
        best = json.loads((path_of(self.cfg, "metrics") / "best_models.json").read_text())
        self.best = best
        self.scen: dict[str, dict] = {}
        for s, info in best.items():
            name = info["model"]
            d = self.mroot / s
            schema = json.loads((d / "schema.json").read_text())
            mode = "nn" if name == "pytorch_nn" else "gbdt"
            pre = joblib.load(d / f"preprocessor_{mode}.joblib")
            if name == "pytorch_nn":
                model = TorchTabularModel.load(d / "pytorch_nn.pt", use_gpu=self.use_gpu)
            else:
                cls = CLASSES[name]
                model = cls.load(d / f"{name}.{cls.ext}", schema["categorical"], self.use_gpu)
            cal = joblib.load(d / f"{name}_calibrator.joblib")
            preds = pd.read_parquet(path_of(self.cfg, "metrics") / s / f"preds_{name}.parquet")
            cuts = np.quantile(preds["p_cal"], np.linspace(0.1, 0.9, 9))
            self.scen[s] = dict(name=name, schema=schema, pre=pre, model=model, cal=cal, cuts=cuts, pre_g=None,
                                cols=schema["categorical"] + schema["numeric"], base_rate=float(preds["y"].mean()))
        self.uplift = None
        up = self.mroot / "ota_to_direct" / "uplift_tlearner.joblib"
        if up.exists():
            self.uplift = TLearnerUplift.load(up)
            self.uplift_pre = joblib.load(self.mroot / "ota_to_direct" / "preprocessor_gbdt.joblib")
        self.shap_models: dict = {}

    # ------------------------------------------------------------------ helpers
    @property
    def scenarios(self) -> list[str]:
        return list(self.scen)

    def health(self) -> dict:
        info = cuda_info()
        return {"status": "ok", "cuda_available": info["cuda_available"], "gpu_device": info["device_name"],
                "gpu_memory_used_gb": info["used_mem_gb"], "gpu_memory_total_gb": info["total_mem_gb"],
                "models_loaded": {s: v["name"] for s, v in self.scen.items()},
                "uplift_model_loaded": self.uplift is not None}

    @staticmethod
    def _derive(df: pd.DataFrame) -> pd.DataFrame:
        """Fill derived features from the simple fields a caller may supply."""
        df = df.copy()
        if "inter_stay_decay_ratio" not in df or df["inter_stay_decay_ratio"].isna().all():
            gap = df.get("inter_stay_mean_days", pd.Series(np.nan, index=df.index)).fillna(365.0).clip(lower=14)
            if "recency_days" in df:
                df["inter_stay_decay_ratio"] = df["recency_days"] / gap
        if "checkin_month" in df:
            m = pd.to_numeric(df["checkin_month"], errors="coerce")
            df["checkin_month_sin"] = np.sin(2 * np.pi * m / 12)
            df["checkin_month_cos"] = np.cos(2 * np.pi * m / 12)
            wk = (m - 1) * 4.33 + 2
            df["checkin_week_sin"] = np.sin(2 * np.pi * wk / 52)
            df["checkin_week_cos"] = np.cos(2 * np.pi * wk / 52)
            df["checkin_school_break"] = m.isin([6, 7, 8, 12]).astype(int)
        if {"adults", "children"} <= set(df.columns):
            df["party_size"] = df["adults"] + df["children"]
            df["has_children"] = (df["children"] > 0).astype(int)
        if {"adr", "los"} <= set(df.columns):
            df["rate_per_night_total"] = df["adr"] * df["los"]
        if "n_stays" in df:
            df["is_first_stay"] = (df["n_stays"].fillna(0) == 0).astype(int)
        if "checkout_reach_90d" not in df and "reached_checkout" in df:
            pass
        return df

    def _frame(self, s: str, df: pd.DataFrame) -> pd.DataFrame:
        sc = self.scen[s]
        cols = sc["cols"]
        out = pd.DataFrame(index=df.index, columns=cols, dtype=object)
        for c in cols:
            if c in df:
                out[c] = df[c]
        for c in sc["schema"]["numeric"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            if c in ZERO_FILL:
                out[c] = out[c].fillna(0)
        for c in sc["schema"]["categorical"]:
            out[c] = out[c].map(lambda v: np.nan if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)).astype(object)
        return out

    # ------------------------------------------------------------------ scoring
    def score(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        df = self._derive(df.reset_index(drop=True))
        res = {}
        for s, sc in self.scen.items():
            X = sc["pre"].transform(self._frame(s, df))
            raw = sc["model"].predict_proba(X)
            res[s] = sc["cal"].predict(raw)
        return res

    def decile(self, s: str, p: np.ndarray) -> np.ndarray:
        """1 (lowest propensity) .. 10 (top decile), relative to the held-out test score distribution."""
        return 1 + np.searchsorted(self.scen[s]["cuts"], p, side="right")

    def next_best_action(self, s: str, decile: int, segment: str | None = None) -> str:
        acts = self.cfg["next_best_action"][s]
        if s == "ota_to_direct" and segment:
            if segment == "Persuadable":
                return "Send direct-booking incentive (high incremental uplift): " + acts["high"].split(":")[-1].strip()
            if segment == "Sleeping Dog":
                return "Do NOT send promotion (negative uplift); use organic loyalty touch only"
            if segment == "Sure Thing":
                return "Skip discount (will book direct anyway); send standard member comms"
        return acts["high"] if decile >= self.cfg["nba_high_threshold_decile"] else acts["low"]

    def uplift_for(self, df: pd.DataFrame) -> pd.DataFrame | None:
        if self.uplift is None:
            return None
        df = self._derive(df.reset_index(drop=True))
        X = self.uplift_pre.transform(self._frame("ota_to_direct", df))
        u = self.uplift.predict(X)
        u["segment"] = self.uplift.segment(u)
        return u

    def predict_guests(self, df: pd.DataFrame, explain: bool = False) -> list[dict]:
        scores = self.score(df)
        upl = self.uplift_for(df)
        out = []
        for i in range(len(df)):
            r = {}
            for s in self.scen:
                p = float(scores[s][i])
                d = int(self.decile(s, np.array([p]))[0])
                seg = None
                if s == "ota_to_direct" and upl is not None:
                    seg = str(upl["segment"].iloc[i])
                r[s] = {"propensity": round(p, 5), "decile": d, "tier": "Top 10%" if d == 10 else f"Decile {d}",
                        "base_rate": round(self.scen[s]["base_rate"], 4), "model": self.scen[s]["name"],
                        "uplift_segment": seg,
                        "uplift": round(float(upl["uplift"].iloc[i]), 5) if (seg and upl is not None) else None,
                        "next_best_action": self.next_best_action(s, d, seg)}
            if explain:
                for s in self.scen:
                    r[s]["top_drivers"] = self.explain_row(s, df.iloc[[i]])
            out.append(r)
        return out

    # ------------------------------------------------------------------ explanations
    def explain_row(self, s: str, row: pd.DataFrame, k: int = 5) -> list[dict]:
        """Top-k SHAP drivers (log-odds) from the best tree model of the scenario."""
        from .explainer import contributions, load_tree_model
        sc = self.scen[s]
        if s not in self.shap_models:
            board = pd.read_csv(path_of(self.cfg, "metrics") / "master_leaderboard.csv")
            g = board[(board.scenario == s) & board.model.isin(TREE)].sort_values("val_pr_auc_raw", ascending=False)
            name = sc["name"] if sc["name"] in TREE else g.iloc[0]["model"]
            w, _ = load_tree_model(self.cfg, s, name)
            pre = joblib.load(self.mroot / s / "preprocessor_gbdt.joblib")
            self.shap_models[s] = (name, w, pre)
        name, w, pre = self.shap_models[s]
        row = self._derive(row.reset_index(drop=True))
        fr = self._frame(s, row)
        X = pre.transform(fr)
        vals, _ = contributions(name, w, X)
        order = np.argsort(-np.abs(vals[0]))[:k]
        raw = fr.iloc[0]
        return [{"feature": X.columns[j], "shap": round(float(vals[0, j]), 4), "value": None if pd.isna(raw.get(X.columns[j])) else str(raw.get(X.columns[j])),
                 "direction": "increases" if vals[0, j] > 0 else "decreases"} for j in order]

    def leaderboard(self) -> pd.DataFrame:
        return pd.read_csv(path_of(self.cfg, "metrics") / "master_leaderboard.csv")


_ENGINE: PropensityEngine | None = None


def get_engine() -> PropensityEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PropensityEngine()
    return _ENGINE
