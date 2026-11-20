"""Evaluation suite: regression + ranking + business metrics, ensemble, best-model selection, tier cut-offs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _gini_raw(y: np.ndarray, score: np.ndarray) -> float:
    """Gini from the Lorenz curve of y ordered by descending score."""
    o = np.argsort(-score, kind="stable")
    cum = np.cumsum(y[o]) / max(y.sum(), 1e-12)
    n = len(y)
    return float(cum.sum() / n - (n + 1) / (2 * n))


def normalized_gini(y: np.ndarray, p: np.ndarray) -> float:
    return _gini_raw(y, p) / _gini_raw(y, y)


def decile_capture(y: np.ndarray, p: np.ndarray, frac: float = 0.10) -> float:
    k = max(int(len(y) * frac), 1)
    top = np.argsort(-p, kind="stable")[:k]
    return float(100 * y[top].sum() / max(y.sum(), 1e-12))


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    nz = y > 0
    ss_res, ss_tot = np.sum((y - p) ** 2), np.sum((y - y.mean()) ** 2)
    return {
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))), "MAE": float(np.mean(np.abs(y - p))),
        "R2": float(1 - ss_res / ss_tot), "MAPE_nonzero": float(100 * np.mean(np.abs(y[nz] - p[nz]) / y[nz])),
        "Normalized_Gini": normalized_gini(y, p), "Spearman": float(spearmanr(y, p)[0]),
        "Top10_Revenue_Capture_pct": decile_capture(y, p),
        "Calibration_Ratio": float(p.sum() / max(y.sum(), 1e-12)),
    }


def decile_table(y: np.ndarray, p: np.ndarray, n: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), n, labels=range(1, n + 1))
    return d.groupby("decile", observed=True).agg(actual=("y", "mean"), predicted=("p", "mean"), n=("y", "size")).reset_index()


def evaluate_all(cfg: dict, pdf: pd.DataFrame, timing: dict, lg) -> dict:
    """Score every model (+ ensemble of top-3 by validation Gini) on the held-out test split."""
    out = cfg["out"]
    models = [c for c in pdf.columns if c not in ("guest_id", "split", "y")]
    va, te = pdf[pdf.split == "val"], pdf[pdf.split == "test"]
    val_gini = {m: normalized_gini(va["y"].values, va[m].values) for m in models}
    top3 = sorted(val_gini, key=val_gini.get, reverse=True)[:3]
    pdf = pdf.copy()
    pdf["ensemble_top3"] = pdf[top3].mean(1)
    te = pdf[pdf.split == "test"]
    va = pdf[pdf.split == "val"]
    rows, summary = [], {}
    for m in models + ["ensemble_top3"]:
        met = compute_metrics(te["y"].values, te[m].values)
        t = timing.get(m, {"train_time_s": sum(timing[k]["train_time_s"] for k in top3),
                           "latency_ms_per_1k": sum(timing[k]["latency_ms_per_1k"] for k in top3)})
        met.update(Train_Time_s=t["train_time_s"], Latency_ms_per_1k=t["latency_ms_per_1k"],
                   Val_Normalized_Gini=normalized_gini(va["y"].values, va[m].values))
        summary[m] = met
        rows.append({"model": m, **met})
    report = pd.DataFrame(rows)
    report.to_csv(out / "logs" / "evaluation_report.csv", index=False)
    # best single model chosen on VALIDATION (no test leakage): highest Gini, calibration as tie-break
    cand = report[report.model.isin(models)].copy()
    cand["score"] = cand["Val_Normalized_Gini"] - 0.05 * (np.log(cand["Calibration_Ratio"]).abs())
    best = str(cand.sort_values("score", ascending=False).iloc[0]["model"])
    # tier cut-offs from validation predictions of the best model
    pb = va[best].values
    cut = {"VIP": float(np.quantile(pb, 0.95)), "High": float(np.quantile(pb, 0.80)), "Medium": float(np.quantile(pb, 0.50))}
    (out / "models" / "tier_cutoffs.json").write_text(json.dumps(cut, indent=2))
    meta = {"best_model": best, "ensemble_members": top3, "n_test": int(len(te)), "tier_cutoffs": cut,
            "val_normalized_gini": val_gini, "metrics": summary}
    (out / "logs" / "metrics_summary.json").write_text(json.dumps(meta, indent=2, default=float))
    (out / "models" / "best_model.json").write_text(json.dumps({"best_model": best, "ensemble_members": top3}, indent=2))
    pdf.to_parquet(out / "data" / "predictions.parquet", index=False)
    lg.info("Evaluation report:\n" + report.round(3).to_string(index=False))
    lg.info(f"Best model (validation-selected): {best} | ensemble = {top3}")
    return {"report": report, "best": best, "top3": top3, "predictions": pdf, "meta": meta}
