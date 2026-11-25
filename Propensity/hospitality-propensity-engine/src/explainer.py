"""SHAP explanations (global + local) using native GPU/CPU tree contribution kernels + the shap plotting stack.

* XGBoost : booster.predict(pred_contribs=True) on CUDA  (== GPU TreeExplainer / GPUTreeShap)
* LightGBM: booster.predict(pred_contrib=True)
* CatBoost: get_feature_importance(type="ShapValues")
For scenarios whose best model is the PyTorch NN, SHAP is computed for the best *tree* model of that scenario.
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from .common import get_logger, load_config, path_of  # noqa: E402
from .models.gbdt_models import CLASSES  # noqa: E402

warnings.filterwarnings("ignore")
TREE_MODELS = ["xgboost", "lightgbm", "catboost"]


def contributions(name: str, wrapper, X: pd.DataFrame, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (shap_values [n, f], base_value [n]) in log-odds space of the raw model."""
    if name == "xgboost":
        import xgboost as xgb
        booster = wrapper.model.get_booster()
        booster.set_param({"device": device or wrapper.device})
        c = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    elif name == "lightgbm":
        booster = getattr(wrapper, "booster", None) or wrapper.model.booster_
        c = booster.predict(X, pred_contrib=True)
    else:
        from catboost import Pool
        cat_idx = [X.columns.get_loc(cc) for cc in wrapper.cat_cols if cc in X.columns]
        c = wrapper.model.get_feature_importance(Pool(wrapper._prep(X), cat_features=cat_idx), type="ShapValues")
    return c[:, :-1], c[:, -1]


def load_tree_model(cfg, scen: str, name: str, use_gpu: bool = True):
    mdir = path_of(cfg, "models") / scen
    schema = json.loads((mdir / "schema.json").read_text())
    cls = CLASSES[name]
    w = cls.load(mdir / f"{name}.{cls.ext}", schema["categorical"], use_gpu)
    if name == "lightgbm":
        import lightgbm as lgb
        w.booster = lgb.Booster(model_file=str(mdir / f"{name}.{cls.ext}"))
    return w, schema


def explain_scenario(cfg, scen: str, n: int = 3000, log=None):
    best = json.loads((path_of(cfg, "metrics") / "best_models.json").read_text())[scen]["model"]
    board = pd.read_csv(path_of(cfg, "metrics") / "master_leaderboard.csv")
    g = board[(board.scenario == scen) & board.model.isin(TREE_MODELS)].sort_values("val_pr_auc_raw", ascending=False)
    name = best if best in TREE_MODELS else g.iloc[0]["model"]
    w, schema = load_tree_model(cfg, scen, name)
    pre = joblib.load(path_of(cfg, "models") / scen / "preprocessor_gbdt.joblib")
    te = pd.read_parquet(path_of(cfg, "processed") / scen / "test.parquet")
    te = te.sample(min(n, len(te)), random_state=cfg["seed"])
    X = pre.transform(te[schema["categorical"] + schema["numeric"]])
    vals, base = contributions(name, w, X)
    exp = shap.Explanation(values=vals, base_values=base, data=X.to_numpy(), feature_names=list(X.columns))
    out = path_of(cfg, "plots")
    title = cfg["scenarios"][scen]["title"]

    plt.figure(figsize=(9, 7)); shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.title(f"SHAP beeswarm — {title} ({name})", fontweight="bold"); plt.tight_layout()
    plt.savefig(out / f"shap_beeswarm_{scen}.png", dpi=200, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(8, 6)); shap.plots.bar(exp, max_display=15, show=False)
    plt.title(f"Top drivers (mean |SHAP|) — {title}", fontweight="bold"); plt.tight_layout()
    plt.savefig(out / f"shap_bar_{scen}.png", dpi=200, bbox_inches="tight"); plt.close()

    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(vals).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(path_of(cfg, "metrics") / scen / "shap_importance.csv", index=False)
    top = [f for f in imp.feature if f not in schema["categorical"]][:3]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, f in zip(axs, top):
        shap.plots.scatter(exp[:, f], ax=a, show=False, color=exp)
        a.set_title(f"dependence: {f}", fontsize=10)
    fig.suptitle(f"SHAP dependence — {title}", fontweight="bold"); fig.tight_layout()
    fig.savefig(out / f"shap_dependence_{scen}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    if log:
        log.info("SHAP %-20s model=%-9s top drivers: %s", scen, name, ", ".join(imp.feature.head(5)))
    return name, imp


def main():
    cfg = load_config()
    log = get_logger("explainer", cfg)
    for scen in cfg["scenarios"]:
        if (path_of(cfg, "metrics") / scen).exists():
            try:
                explain_scenario(cfg, scen, log=log)
            except Exception as e:
                log.exception("SHAP failed for %s: %s", scen, e)


if __name__ == "__main__":
    main()
