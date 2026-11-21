from collections import OrderedDict
from pathlib import Path

import joblib

from .base import BaseBandit
from .linear import LinearThompsonSampling, LinUCB
from .neural import NeuralEpsilonGreedy, NeuralLinearBandit, NeuralTS, NeuralUCB
from .policy_gradient import DoublyRobustPolicyGradient
from .tree import XGBoostEpsilonGreedy, XGBoostUCB

MODEL_CLASSES = {c.__name__: c for c in [
    LinUCB, LinearThompsonSampling, XGBoostEpsilonGreedy, XGBoostUCB, NeuralEpsilonGreedy,
    NeuralLinearBandit, NeuralTS, NeuralUCB, DoublyRobustPolicyGradient]}


def model_zoo(d, K=6, mode="full", logger=None):
    """Ordered {name: factory()} of every bandit. mode: 'full' | 'quick' | 'replay' (lighter, for online replay)."""
    light = mode in ("quick", "replay")
    nn_kw = dict(epochs=8 if mode == "quick" else 15, replay_epochs=2) if mode != "full" else {}
    spec = OrderedDict([
        ("LinUCB-Disjoint", (LinUCB, dict(hybrid=False))),
        ("LinUCB-Hybrid", (LinUCB, dict(hybrid=True))),
        ("LinearTS", (LinearThompsonSampling, {})),
        ("XGBoost-EpsGreedy", (XGBoostEpsilonGreedy, dict(n_estimators=120) if light else {})),
        ("XGBoost-UCB", (XGBoostUCB, dict(n_estimators=80, n_models=4) if light else {})),
        ("NeuralEpsGreedy", (NeuralEpsilonGreedy, nn_kw)),
        ("NeuralLinear", (NeuralLinearBandit, nn_kw)),
        ("NeuralTS", (NeuralTS, {**nn_kw, **(dict(n_members=3) if light else {})})),
        ("NeuralUCB", (NeuralUCB, nn_kw)),
        ("DR-PolicyGradient", (DoublyRobustPolicyGradient,
                               {**nn_kw, **(dict(q_epochs=10, pg_epochs=20) if mode != "full" else {})})),
    ])

    def make(name, cls, kw):
        def factory(seed=0):
            m = cls(d, K, name=name, seed=seed, **kw)
            m.logger = logger
            return m
        return factory

    return OrderedDict((n, make(n, c, kw)) for n, (c, kw) in spec.items())


def load_model(models_dir, name):
    models_dir = Path(models_dir)
    meta = joblib.load(models_dir / f"{name}.meta.joblib")
    m = MODEL_CLASSES[meta["cls"]](meta["d"], meta["K"], name=meta["name"], **meta["hp"])
    return m._restore(meta, models_dir)
