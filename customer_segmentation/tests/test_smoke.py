"""Smoke tests: pytest tests/test_smoke.py"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generator import generate_dataset  # noqa: E402
from src.feature_engineering import FeaturePipeline  # noqa: E402
from src.models.classical_ml import GMMGPU, HDBSCANModel, KMeansGPU  # noqa: E402


@pytest.fixture(scope="module")
def data():
    g, e, s, m = generate_dataset(1500, seed=1)
    return g, e, s, m, FeaturePipeline().fit_transform(g)


def test_shapes(data):
    g, e, s, m, X = data
    assert len(g) == 1500 and s.shape == (1500, 10, 4) and m.shape == (1500, 10)
    assert X.shape[0] == 1500 and not np.isnan(X).any()
    assert len(e) > len(g)


@pytest.mark.parametrize("cls", [KMeansGPU, GMMGPU])
def test_cpu_fallback_models(data, cls):
    X = data[-1]
    model = cls(5, device=torch.device("cpu")).fit(X)
    lab = model.predict(X)
    assert lab.shape == (1500,) and len(np.unique(lab)) >= 3


def test_hdbscan(data):
    X = data[-1]
    h = HDBSCANModel(min_cluster_size=30).fit(X)
    assert h.predict(X[:10]).shape == (10,)


def test_api_health_and_predict():
    reg = Path(__file__).resolve().parent.parent / "output" / "saved_models" / "model_registry.json"
    if not reg.exists():
        pytest.skip("train the pipeline first")
    from fastapi.testclient import TestClient
    from src.api import app
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict/segment", json={"loyalty_tier": "Gold"})
        assert r.status_code == 200
        assert {"segment_id", "segment_name", "confidence_score", "latent_coordinates",
                "recommended_hospitality_action"} <= set(r.json())
        assert c.get("/segments/personas").status_code == 200
        assert c.post("/predict/batch", json={"guests": [{}] * 3}).json()["count"] == 3
