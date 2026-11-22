"""API smoke tests (require a completed training run: output/models/best_model.json)."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.skipif(not (ROOT / "output" / "models" / "best_model.json").exists(),
                                reason="run the training pipeline first")

GUEST = {"loyalty_tier": "Silver", "adr_avg": 150.0, "length_of_stay_avg": 2.5, "lead_time_days": 30.0,
         "days_since_last_stay": 90.0, "total_prior_stays": 4, "ancillary_spend_ratio": 0.1,
         "promo_email_open_rate": 0.3, "app_sessions_30d": 5, "channel_direct": 0,
         "trip_purpose_business": 0, "seasonality_index": 1.0}


@pytest.fixture(scope="module")
def client():
    from src.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["loaded_models"]


def test_predict_monotone_survival(client):
    r = client.post("/predict/survival", json=GUEST)
    assert r.status_code == 200
    p = r.json()["survival_probabilities"]
    vals = [p[f"{d}d"] for d in (30, 60, 90, 180, 365)]
    assert all(0 <= v <= 1 for v in vals) and vals == sorted(vals, reverse=True)
    assert 0 <= r.json()["risk_score"] <= 100


def test_validation_error(client):
    assert client.post("/predict/survival", json={**GUEST, "loyalty_tier": "Diamond"}).status_code == 422
