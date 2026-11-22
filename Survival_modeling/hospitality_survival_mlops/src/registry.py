"""Model registry: maps a model name to its class + artefact filename so the API can reload the winner."""
from __future__ import annotations

from pathlib import Path

from .base import SurvivalModel
from .models_classical import (CoxPHModel, KaplanMeierModel, LogNormalAFTModel, NelsonAalenModel,
                               WeibullAFTModel)
from .models_deep import DeepHitModel, DeepSurvModel
from .models_tree import RSFModel, XGBSurvivalModel

ARTIFACTS = {
    "kaplan_meier": "kaplan_meier.pkl", "nelson_aalen": "nelson_aalen.pkl", "coxph": "coxph.pkl",
    "weibull_aft": "weibull_aft.pkl", "lognormal_aft": "lognormal_aft.pkl",
    "random_survival_forest": "random_survival_forest.pkl",
    "xgboost_cox": "xgboost_cox.json", "xgboost_aft": "xgboost_aft.json",
    "deepsurv": "deepsurv.pt", "deephit": "deephit.pt",
}
BASELINES = ("kaplan_meier", "nelson_aalen")


def load_model(name: str, models_dir: str | Path, device: str = "cpu") -> SurvivalModel:
    path = Path(models_dir) / ARTIFACTS[name]
    if name.startswith("xgboost"):
        return XGBSurvivalModel.load(path)
    if name == "deepsurv":
        return DeepSurvModel.load(path, device=device)
    if name == "deephit":
        return DeepHitModel.load(path, device=device)
    return SurvivalModel.load(path)
