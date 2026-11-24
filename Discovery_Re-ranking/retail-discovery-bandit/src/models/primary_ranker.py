"""Stage 3: GPU XGBoost primary relevance ranker predicting calibrated P(click) and P(buy).

History-derived item features (impression counts, smoothed CTR/CVR, buys) are structurally
missing for cold-start SKUs, producing the *cold-start bottleneck* that the discovery
re-ranker later corrects.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from src.utils import xgb_device

FEATURES: List[str] = [
    "persona", "user_log_avg_price", "user_click_rate", "user_buys",
    "category", "subcat", "log_price", "margin", "days_since_launch",
    "item_log_impr", "item_ctr_smooth", "item_cvr_smooth", "item_log_buys",
    "tt_cos", "content_cos", "price_logratio",
    "usub_click_rate", "usub_clicks", "ucat_buys", "ubrand_clicks", "usub_impr",
]


class FeatureBuilder:
    """Vectorised (user, item) -> feature matrix. Statistics are frozen on the feature window."""

    def __init__(self, world, retriever, feature_end_day: int, smooth_m: float = 20.0):
        self.world = world
        self.retriever = retriever
        it, us = world.items, world.users
        self.U, self.I = world.n_users, world.n_items
        df = world.interactions
        df = df[df.day <= feature_end_day]
        u, i, c, b = df["user"].values, df["item"].values, df["click"].values, df["buy"].values
        S, C, B = world.n_subcats, world.n_cats, world.n_brands
        self.item_impr = np.bincount(i, minlength=self.I).astype(np.float32)
        self.item_clicks = np.bincount(i, weights=c, minlength=self.I).astype(np.float32)
        self.item_buys = np.bincount(i, weights=b, minlength=self.I).astype(np.float32)
        g_ctr = c.mean()
        g_cvr = b.sum() / max(c.sum(), 1)
        m = smooth_m
        self.item_ctr = (self.item_clicks + m * g_ctr) / (self.item_impr + m)
        self.item_cvr = (self.item_buys + m * g_cvr * g_ctr) / (self.item_impr + m)
        sub, cat, brand = it["subcat"].values, it["category"].values, it["brand"].values
        self.user_impr = np.bincount(u, minlength=self.U).astype(np.float32)
        self.user_clicks = np.bincount(u, weights=c, minlength=self.U).astype(np.float32)
        self.user_buys = np.bincount(u, weights=b, minlength=self.U).astype(np.float32)
        self.usub_clicks = np.zeros((self.U, S), np.float32)
        self.usub_impr = np.zeros((self.U, S), np.float32)
        self.ucat_buys = np.zeros((self.U, C), np.float32)
        self.ubrand_clicks = np.zeros((self.U, B), np.float32)
        np.add.at(self.usub_clicks, (u, sub[i]), c)
        np.add.at(self.usub_impr, (u, sub[i]), 1)
        np.add.at(self.ucat_buys, (u, cat[i]), b)
        np.add.at(self.ubrand_clicks, (u, brand[i]), c)
        self.g_ctr = float(g_ctr)
        self.m = m
        self.persona = us["persona"].values
        self.user_log_avg_price = np.log(us["avg_item_price"].values).astype(np.float32)
        self.i_sub, self.i_cat, self.i_brand = sub, cat, brand
        self.i_logp = np.log(it["price"].values).astype(np.float32)
        self.i_margin = it["margin"].values.astype(np.float32)
        self.i_days = it["days_since_launch"].values.astype(np.float32)
        self.item_emb = world.item_emb

    @torch.no_grad()
    def build(self, u_idx: np.ndarray, i_idx: np.ndarray) -> np.ndarray:
        u_idx = np.asarray(u_idx)
        i_idx = np.asarray(i_idx)
        dev = self.retriever.device
        ut = torch.as_tensor(u_idx, dtype=torch.long, device=dev)
        it_ = torch.as_tensor(i_idx, dtype=torch.long, device=dev)
        tt = self.retriever.pair_scores(ut, it_).cpu().numpy()
        hist = torch.nn.functional.normalize(self.retriever.u_hist[ut], dim=-1)
        content = (hist * self.retriever.i_content[it_]).sum(-1).cpu().numpy()
        sub, cat, brand = self.i_sub[i_idx], self.i_cat[i_idx], self.i_brand[i_idx]
        usub_i = self.usub_impr[u_idx, sub]
        usub_c = self.usub_clicks[u_idx, sub]
        cols = {
            "persona": self.persona[u_idx],
            "user_log_avg_price": self.user_log_avg_price[u_idx],
            "user_click_rate": self.user_clicks[u_idx] / np.maximum(self.user_impr[u_idx], 1),
            "user_buys": self.user_buys[u_idx],
            "category": cat, "subcat": sub,
            "log_price": self.i_logp[i_idx], "margin": self.i_margin[i_idx],
            "days_since_launch": self.i_days[i_idx],
            "item_log_impr": np.log1p(self.item_impr[i_idx]),
            "item_ctr_smooth": self.item_ctr[i_idx], "item_cvr_smooth": self.item_cvr[i_idx],
            "item_log_buys": np.log1p(self.item_buys[i_idx]),
            "tt_cos": tt, "content_cos": content,
            "price_logratio": self.i_logp[i_idx] - self.user_log_avg_price[u_idx],
            "usub_click_rate": (usub_c + 2 * self.g_ctr) / (usub_i + 2),
            "usub_clicks": usub_c, "ucat_buys": self.ucat_buys[u_idx, cat],
            "ubrand_clicks": self.ubrand_clicks[u_idx, brand], "usub_impr": usub_i,
        }
        return np.stack([np.asarray(cols[f], dtype=np.float32) for f in FEATURES], axis=1)


class PrimaryRanker:
    def __init__(self, cfg: Dict):
        self.cfg = cfg["ranker"]
        self.click: Optional[xgb.XGBClassifier] = None
        self.buy: Optional[xgb.XGBClassifier] = None
        self.click_cal: Optional[IsotonicRegression] = None
        self.buy_cal: Optional[IsotonicRegression] = None
        self.device = xgb_device()

    def _make(self, seed: int) -> xgb.XGBClassifier:
        c = self.cfg
        return xgb.XGBClassifier(
            n_estimators=c["n_estimators"], max_depth=c["max_depth"], learning_rate=c["learning_rate"],
            subsample=c["subsample"], colsample_bytree=c["colsample_bytree"], min_child_weight=c["min_child_weight"],
            tree_method="hist", device=self.device, eval_metric="logloss",
            early_stopping_rounds=c["early_stopping_rounds"], random_state=seed,
        )

    def fit(self, X_tr, y_click_tr, y_buy_tr, X_cal, y_click_cal, y_buy_cal, logger=None) -> Dict[str, float]:
        info = {}
        self.click = self._make(1)
        self.click.fit(X_tr, y_click_tr, eval_set=[(X_cal, y_click_cal)], verbose=False)
        self.buy = self._make(2)
        self.buy.fit(X_tr, y_buy_tr, eval_set=[(X_cal, y_buy_cal)], verbose=False)
        info["click_best_iteration"] = int(self.click.best_iteration)
        info["buy_best_iteration"] = int(self.buy.best_iteration)
        # isotonic calibration on held-out calibration window
        raw_c = self.click.predict_proba(X_cal)[:, 1]
        raw_b = self.buy.predict_proba(X_cal)[:, 1]
        self.click_cal = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5).fit(raw_c, y_click_cal)
        self.buy_cal = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(raw_b, y_buy_cal)
        if logger:
            logger.info("ranker trained on %s | click iters=%d buy iters=%d", self.device,
                        info["click_best_iteration"], info["buy_best_iteration"])
        return info

    def predict(self, X: np.ndarray, calibrated: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        pc = self.click.get_booster().inplace_predict(X)
        pb = self.buy.get_booster().inplace_predict(X)
        pc, pb = np.asarray(pc, dtype=np.float64), np.asarray(pb, dtype=np.float64)
        if calibrated and self.click_cal is not None:
            pc = self.click_cal.predict(pc)
            pb = self.buy_cal.predict(pb)
        pb = np.minimum(pb, pc)
        return pc.astype(np.float32), pb.astype(np.float32)

    def feature_importance(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for name, m in (("click", self.click), ("buy", self.buy)):
            imp = m.get_booster().get_score(importance_type="gain")
            out[name] = {FEATURES[int(k[1:])]: float(v) for k, v in imp.items()}
        return out

    # -- persistence -----------------------------------------------------------------------
    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.click.save_model(str(model_dir / "xgb_click_ranker.json"))
        self.buy.save_model(str(model_dir / "xgb_buy_ranker.json"))
        with open(model_dir / "ranker_calibrators.pkl", "wb") as f:
            pickle.dump({"click": self.click_cal, "buy": self.buy_cal}, f)

    @classmethod
    def load(cls, cfg: Dict, model_dir: Path) -> "PrimaryRanker":
        r = cls(cfg)
        model_dir = Path(model_dir)
        r.click = xgb.XGBClassifier()
        r.click.load_model(str(model_dir / "xgb_click_ranker.json"))
        r.buy = xgb.XGBClassifier()
        r.buy.load_model(str(model_dir / "xgb_buy_ranker.json"))
        for m in (r.click, r.buy):
            m.get_booster().set_param({"device": r.device})
        with open(model_dir / "ranker_calibrators.pkl", "rb") as f:
            cal = pickle.load(f)
        r.click_cal, r.buy_cal = cal["click"], cal["buy"]
        return r


def cold_start_gap(world, builder: FeatureBuilder, ranker: PrimaryRanker, warm_rows: pd.DataFrame,
                   n_bins: int = 5) -> Dict[str, object]:
    """Quantify the cold-start bottleneck: do new SKUs receive systematically lower base scores
    than warm SKUs *of equal true relevance*?"""
    cold = world.cold_eval
    Xc = builder.build(cold["user"].values, cold["item"].values)
    pc_c, pb_c = ranker.predict(Xc)
    u, i = warm_rows["user"].values, warm_rows["item"].values
    Xw = builder.build(u, i)
    pc_w, pb_w = ranker.predict(Xw)
    tc_w, tb_w = world.truth.probs(u, i)
    tc_c, tb_c = cold["p_click_true"].values, cold["p_buy_true"].values
    res = {
        "mean_pred_click_cold": float(pc_c.mean()), "mean_pred_click_warm": float(pc_w.mean()),
        "mean_true_click_cold": float(tc_c.mean()), "mean_true_click_warm": float(tc_w.mean()),
        "mean_pred_buy_cold": float(pb_c.mean()), "mean_pred_buy_warm": float(pb_w.mean()),
        "mean_true_buy_cold": float(tb_c.mean()), "mean_true_buy_warm": float(tb_w.mean()),
        "calibration_ratio_click_cold": float(pc_c.mean() / tc_c.mean()),
        "calibration_ratio_click_warm": float(pc_w.mean() / tc_w.mean()),
        "calibration_ratio_buy_cold": float(pb_c.mean() / tb_c.mean()),
        "calibration_ratio_buy_warm": float(pb_w.mean() / tb_w.mean()),
    }
    # equal-relevance comparison: bin by *true* P(click) deciles pooled over cold+warm
    all_true = np.concatenate([tc_c, tc_w])
    edges = np.quantile(all_true, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    rows = []
    for b in range(n_bins):
        mc = (tc_c >= edges[b]) & (tc_c < edges[b + 1])
        mw = (tc_w >= edges[b]) & (tc_w < edges[b + 1])
        if mc.sum() < 20 or mw.sum() < 20:
            continue
        rows.append({
            "bin": b, "true_click_lo": float(edges[b]), "true_click_hi": float(edges[b + 1]),
            "n_cold": int(mc.sum()), "n_warm": int(mw.sum()),
            "pred_cold": float(pc_c[mc].mean()), "pred_warm": float(pc_w[mw].mean()),
            "true_cold": float(tc_c[mc].mean()), "true_warm": float(tc_w[mw].mean()),
            "gap_pred_cold_minus_warm": float(pc_c[mc].mean() - pc_w[mw].mean()),
            "rel_gap": float(pc_c[mc].mean() / max(pc_w[mw].mean(), 1e-9) - 1.0),
        })
    res["matched_relevance_bins"] = rows
    res["mean_relative_gap_matched"] = float(np.mean([r["rel_gap"] for r in rows])) if rows else float("nan")
    res["bottleneck_confirmed"] = bool(rows and np.mean([r["gap_pred_cold_minus_warm"] for r in rows]) < 0)
    return res
