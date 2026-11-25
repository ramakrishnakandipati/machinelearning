"""Metrics, leaderboards and high-DPI evaluation plots."""
from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, precision_recall_curve,  # noqa: E402
                             roc_auc_score, roc_curve)

from .common import get_logger, load_config, path_of  # noqa: E402

warnings.filterwarnings("ignore")
DPI = 220
MODELS = ["xgboost", "lightgbm", "catboost", "pytorch_nn"]
COL = {"xgboost": "#2563eb", "lightgbm": "#16a34a", "catboost": "#d97706", "pytorch_nn": "#9333ea"}
NAMES = {"xgboost": "XGBoost (CUDA)", "lightgbm": "LightGBM", "catboost": "CatBoost (GPU)", "pytorch_nn": "PyTorch NN (CUDA)"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 10, "axes.titleweight": "bold"})


# --------------------------------------------------------------------------- metrics
def decile_table(y, p, n=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False).reset_index(drop=True)
    d["decile"] = (np.arange(len(d)) * n // len(d)) + 1
    g = d.groupby("decile").agg(n=("y", "size"), positives=("y", "sum"), mean_score=("p", "mean"))
    base = d["y"].mean()
    g["rate"] = g["positives"] / g["n"]
    g["lift"] = g["rate"] / base
    g["cum_gain"] = g["positives"].cumsum() / max(d["y"].sum(), 1)
    return g.reset_index()


def top_decile_lift(y, p) -> float:
    return float(decile_table(y, p).loc[0, "lift"])


def ece(y, p, bins=10) -> float:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["b"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    g = d.groupby("b").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] * (g["y"] - g["p"]).abs()).sum() / g["n"].sum())


def compute_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "logloss": float(log_loss(y, p)), "brier": float(brier_score_loss(y, p)),
            "top_decile_lift": top_decile_lift(y, p), "ece": ece(y, p), "base_rate": float(y.mean())}


# --------------------------------------------------------------------------- leaderboard
def load_metric_files(cfg) -> pd.DataFrame:
    root = path_of(cfg, "metrics")
    rows = []
    for scen in cfg["scenarios"]:
        for m in MODELS:
            f = root / scen / f"{m}.json"
            if f.exists():
                j = json.loads(f.read_text())
                rows.append({
                    "scenario": scen, "model": m,
                    "roc_auc": j["test_calibrated"]["roc_auc"], "pr_auc": j["test_calibrated"]["pr_auc"],
                    "logloss": j["test_calibrated"]["logloss"], "brier": j["test_calibrated"]["brier"],
                    "top_decile_lift": j["test_calibrated"]["top_decile_lift"], "ece": j["test_calibrated"]["ece"],
                    "logloss_raw": j["test_raw"]["logloss"], "brier_raw": j["test_raw"]["brier"], "ece_raw": j["test_raw"]["ece"],
                    "base_rate": j["test_calibrated"]["base_rate"], "calibration": j["calibration_method"],
                    "train_time_s": j["final_fit_seconds"], "tuning_time_s": j["tuning_seconds"],
                    "cv_pr_auc_mean": j["cv"]["pr_auc_mean"], "cv_pr_auc_std": j["cv"]["pr_auc_std"],
                    "val_pr_auc_raw": j["val_pr_auc_raw"], "gpu_peak_mem_mb": j.get("gpu_peak_mem_mb"),
                    "device": j["device"], "n_trials": j["n_trials"], "n_pruned": j["n_pruned"],
                })
    return pd.DataFrame(rows)


def write_leaderboard(cfg) -> pd.DataFrame:
    df = load_metric_files(cfg)
    if df.empty:
        return df
    root = path_of(cfg, "metrics")
    df = df.sort_values(["scenario", "pr_auc"], ascending=[True, False])
    df.to_csv(root / "master_leaderboard.csv", index=False)
    best = {}
    for scen, g in df.groupby("scenario"):
        b = g.sort_values("val_pr_auc_raw", ascending=False).iloc[0]
        best[scen] = {"model": b["model"], "val_pr_auc_raw": float(b["val_pr_auc_raw"]), "test_pr_auc": float(b["pr_auc"]),
                      "test_roc_auc": float(b["roc_auc"])}
    (root / "best_models.json").write_text(json.dumps(best, indent=2))
    df.to_json(root / "master_leaderboard.json", orient="records", indent=2)
    return df


# --------------------------------------------------------------------------- plots
def _preds(cfg, scen, m):
    f = path_of(cfg, "metrics") / scen / f"preds_{m}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def plot_class_balance(cfg):
    out = path_of(cfg, "plots")
    rows = []
    for scen in cfg["scenarios"]:
        sch = path_of(cfg, "processed") / scen / "schema.json"
        if sch.exists():
            j = json.loads(sch.read_text())
            for k, v in j["pos_rate"].items():
                rows.append({"scenario": scen, "split": k, "pos_rate": v, "rows": j["rows"][k]})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    pv = d.pivot(index="scenario", columns="split", values="pos_rate")[["train", "val", "test"]]
    pv.plot.barh(ax=ax[0], color=["#94a3b8", "#60a5fa", "#2563eb"])
    ax[0].set_title("Target class balance by scenario & temporal split"); ax[0].set_xlabel("positive rate")
    pr = d.pivot(index="scenario", columns="split", values="rows")[["train", "val", "test"]]
    pr.plot.barh(ax=ax[1], color=["#94a3b8", "#60a5fa", "#2563eb"], logx=True)
    ax[1].set_title("Dataset size (rows)"); ax[1].set_xlabel("rows (log)")
    fig.tight_layout(); fig.savefig(out / "eda_class_balance.png", dpi=DPI); plt.close(fig)


def plot_eda_features(cfg, scen, k=6):
    d = path_of(cfg, "processed") / scen
    df = pd.read_parquet(d / "test.parquet")
    sch = json.loads((d / "schema.json").read_text())
    num = [c for c in sch["numeric"] if df[c].nunique() > 5]
    corr = df[num].apply(lambda s: s.corr(df["y"], method="spearman")).abs().sort_values(ascending=False).head(k)
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for a, c in zip(axs.ravel(), corr.index):
        lo, hi = df[c].quantile([0.01, 0.99])
        for v, colr in ((0, "#94a3b8"), (1, "#2563eb")):
            a.hist(df.loc[(df.y == v), c].clip(lo, hi).dropna(), bins=35, alpha=0.6, color=colr, density=True, label=f"y={v}")
        a.set_title(c, fontsize=10); a.legend(fontsize=8)
    fig.suptitle(f"EDA: top univariate signals — {cfg['scenarios'][scen]['title']} (test snapshot)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path_of(cfg, "plots") / f"eda_{scen}.png", dpi=DPI); plt.close(fig)


def plot_roc_pr(cfg, scen):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for m in MODELS:
        p = _preds(cfg, scen, m)
        if p is None:
            continue
        fpr, tpr, _ = roc_curve(p.y, p.p_cal); pr, rc, _ = precision_recall_curve(p.y, p.p_cal)
        ax[0].plot(fpr, tpr, color=COL[m], label=f"{NAMES[m]} AUC={roc_auc_score(p.y, p.p_cal):.3f}")
        ax[1].plot(rc, pr, color=COL[m], label=f"{NAMES[m]} AP={average_precision_score(p.y, p.p_cal):.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC")
    ax[1].axhline(p.y.mean(), color="k", ls="--", lw=0.8); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall")
    for a in ax: a.legend(fontsize=8)
    fig.suptitle(f"{cfg['scenarios'][scen]['title']} — held-out test", fontweight="bold")
    fig.tight_layout(); fig.savefig(path_of(cfg, "plots") / f"roc_pr_{scen}.png", dpi=DPI); plt.close(fig)


def plot_gains_lift(cfg, scen):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    best = json.loads((path_of(cfg, "metrics") / "best_models.json").read_text()).get(scen, {}).get("model")
    for m in MODELS:
        p = _preds(cfg, scen, m)
        if p is None:
            continue
        t = decile_table(p.y, p.p_cal)
        ax[0].plot(np.r_[0, t.decile * 10], np.r_[0, t.cum_gain * 100], color=COL[m], marker="o", ms=3, label=NAMES[m], lw=2.2 if m == best else 1.2)
    ax[0].plot([0, 100], [0, 100], "k--", lw=0.8, label="random")
    ax[0].set_xlabel("% of guests contacted (highest score first)"); ax[0].set_ylabel("% of positives captured"); ax[0].set_title("Cumulative gains"); ax[0].legend(fontsize=8)
    pb = _preds(cfg, scen, best) if best else None
    if pb is not None:
        t = decile_table(pb.y, pb.p_cal)
        ax[1].bar(t.decile, t.lift, color=COL[best]); ax[1].axhline(1, color="k", ls="--", lw=0.8)
        for x, v in zip(t.decile, t.lift): ax[1].text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax[1].set_xlabel("decile (1 = highest propensity)"); ax[1].set_ylabel("lift vs base rate")
        ax[1].set_title(f"Decile lift — {NAMES[best]}"); ax[1].set_xticks(range(1, 11))
    fig.suptitle(f"{cfg['scenarios'][scen]['title']}", fontweight="bold")
    fig.tight_layout(); fig.savefig(path_of(cfg, "plots") / f"gains_lift_{scen}.png", dpi=DPI); plt.close(fig)


def plot_calibration(cfg, scen):
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    for j, m in enumerate(MODELS):
        p = _preds(cfg, scen, m)
        if p is None:
            continue
        for i, (col, lab) in enumerate((("p_raw", "before"), ("p_cal", "after"))):
            a = axs[i, j]
            fr, mp = calibration_curve(p.y, p[col], n_bins=10, strategy="quantile")
            a.plot([0, 1], [0, 1], "k--", lw=0.8); a.plot(mp, fr, "o-", color=COL[m])
            a.set_title(f"{NAMES[m]} — {lab} calibration\nECE={ece(p.y, p[col]):.4f}  Brier={brier_score_loss(p.y, p[col]):.4f}", fontsize=9)
            a.set_xlabel("mean predicted"); a.set_ylabel("observed rate")
    fig.suptitle(f"Reliability diagrams — {cfg['scenarios'][scen]['title']} (test)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path_of(cfg, "plots") / f"calibration_{scen}.png", dpi=DPI); plt.close(fig)


def plot_summary(cfg, df: pd.DataFrame):
    out = path_of(cfg, "plots")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    pv = df.pivot(index="scenario", columns="model", values="pr_auc")[[m for m in MODELS if m in df.model.unique()]]
    im = ax[0].imshow(pv.values, cmap="YlGnBu", aspect="auto")
    ax[0].set_xticks(range(pv.shape[1]), [NAMES[m] for m in pv.columns], rotation=20); ax[0].set_yticks(range(pv.shape[0]), pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax[0].text(j, i, f"{pv.values[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax[0].set_title("Test PR-AUC (calibrated)"); ax[0].grid(False)
    tt = df.pivot(index="scenario", columns="model", values="train_time_s")[pv.columns]
    tt.plot.bar(ax=ax[1], color=[COL[m] for m in tt.columns]); ax[1].set_ylabel("final-fit wall-clock (s)"); ax[1].set_title("Training time per model")
    ax[1].legend([NAMES[m] for m in tt.columns], fontsize=8); ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(out / "leaderboard_summary.png", dpi=DPI); plt.close(fig)


def plot_uplift(cfg):
    from .models.uplift_model import qini_curve
    f = path_of(cfg, "metrics") / "ota_to_direct" / "uplift_preds.parquet"
    if not f.exists():
        return
    d = pd.read_parquet(f)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    q = qini_curve(d.y, d.treatment, d.uplift, 25)
    qr = qini_curve(d.y, d.treatment, np.random.default_rng(0).random(len(d)), 25)
    qp = qini_curve(d.y, d.treatment, d.p_control + 0 * d.uplift, 25)
    ax[0].plot(q.frac_targeted, q.qini, color="#2563eb", lw=2, label="T-Learner uplift")
    ax[0].plot(qp.frac_targeted, qp.qini, color="#d97706", label="Propensity (baseline p) ranking")
    ax[0].plot(qr.frac_targeted, qr.qini, "k--", label="random"); ax[0].set_title("Qini curve"); ax[0].set_xlabel("fraction targeted"); ax[0].set_ylabel("incremental conversions"); ax[0].legend(fontsize=8)
    seg = d.groupby("segment").agg(n=("y", "size"), uplift=("uplift", "mean"), p_control=("p_control", "mean"))
    order = [s for s in ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"] if s in seg.index]
    seg = seg.loc[order]
    ax[1].bar(seg.index, seg.n / seg.n.sum() * 100, color=["#16a34a", "#2563eb", "#94a3b8", "#dc2626"][:len(seg)])
    ax[1].set_title("Guest segments (share %)"); ax[1].set_ylabel("%")
    # realised uplift per segment
    rows = []
    for s, g in d.groupby("segment"):
        rows.append((s, g[g.treatment == 1].y.mean() - g[g.treatment == 0].y.mean()))
    r = pd.DataFrame(rows, columns=["segment", "realised"]).set_index("segment").loc[order]
    ax[2].bar(r.index, r.realised * 100, color=["#16a34a", "#2563eb", "#94a3b8", "#dc2626"][:len(r)])
    ax[2].axhline(0, color="k", lw=0.8); ax[2].set_title("Realised treatment effect per segment (test, pp)"); ax[2].set_ylabel("pp")
    fig.suptitle("OTA→Direct campaign: uplift T-Learner (GPU XGBoost)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path_of(cfg, "plots") / "uplift_ota_to_direct.png", dpi=DPI); plt.close(fig)


def main():
    cfg = load_config()
    log = get_logger("evaluator", cfg)
    df = write_leaderboard(cfg)
    if df.empty:
        log.warning("no metric files found"); return
    log.info("leaderboard rows=%d", len(df))
    plot_class_balance(cfg)
    for scen in cfg["scenarios"]:
        if not (path_of(cfg, "metrics") / scen).exists():
            continue
        plot_eda_features(cfg, scen); plot_roc_pr(cfg, scen); plot_gains_lift(cfg, scen); plot_calibration(cfg, scen)
        log.info("plots done: %s", scen)
    plot_summary(cfg, df); plot_uplift(cfg)


if __name__ == "__main__":
    main()
