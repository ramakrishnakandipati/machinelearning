"""Pytest suite covering every cold-start scenario of the API.

Requires trained artifacts:  python run_pipeline.py --fast --skip-visual
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "outputs" / "models"
pytestmark = pytest.mark.skipif(not (MODELS / "serving_state.pkl").exists(),
                                reason="run `python run_pipeline.py` first to create model artifacts")


@pytest.fixture(scope="module")
def client():
    os.environ["COLDSTART_CONFIG"] = str(MODELS / "config_used.yaml")
    os.environ["COLDSTART_MODELS"] = str(MODELS)
    os.environ["COLDSTART_REDIS_URL"] = "redis://127.0.0.1:1/0"   # unreachable -> LRU fallback
    from fastapi.testclient import TestClient
    from src.serving.app import app

    with TestClient(app) as c:
        yield c


def _warm_and_cold_users(client):
    svc = client.app.state.svc
    warm = next(u for u, h in svc.hist.items() if len(h) >= 8)
    cold = next(u for u in svc.users if u not in svc.hist)
    return warm, cold, svc


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["faiss_index_size"] == r["n_items"] > 0
    assert all(r["models_loaded"].values())
    assert "LRU" in r["redis"] or r["redis"] == "connected"


def test_anonymous_zero_history(client):
    r = client.post("/recommend", json={"user_context": {"region": "north", "device": "ios", "hour": 20}, "k": 8}).json()
    assert r["strategy"] == "cold_zero_history" and len(r["items"]) == 8
    assert all(i["explain"] for i in r["items"])
    scores = [i["score"] for i in r["items"]]
    assert scores == sorted(scores, reverse=True)


def test_onboarding_new_user_matches_categories(client):
    body = {"user_context": {"region": "west", "device": "android", "acquisition_channel": "paid_social",
                             "onboarding_categories": ["Running Shoes", "Boots"], "hour": 10}, "k": 10}
    r = client.post("/recommend", json=body).json()
    assert r["strategy"] == "cold_zero_history"
    assert sum(i["dept"] == "Footwear" for i in r["items"]) >= 5, [i["subcat"] for i in r["items"]]
    assert any("Onboarding" in t for i in r["items"] for t in i["explain"])


def test_session_clickstream_user(client):
    svc = client.app.state.svc
    audio = [i for i, m in zip(svc.pre.item_ids, svc.router.meta) if m["subcat"] == "Audio"][:3]
    r = client.post("/recommend", json={"session_item_ids": list(audio), "k": 10}).json()
    assert r["strategy"] == "semi_cold_session" and r["n_interactions"] == 3
    assert sum(i["dept"] == "Electronics" for i in r["items"]) >= 4


def test_warm_user_routes_to_personalized(client):
    warm, _, _ = _warm_and_cold_users(client)
    r = client.post("/recommend", json={"user_id": warm, "k": 10}).json()
    assert r["strategy"] == "warm_personalized"
    assert not (set(i["item_id"] for i in r["items"]) & set(client.app.state.svc.pre.item_ids[client.app.state.svc.hist[warm]]))


def test_known_cold_user_has_zero_history(client):
    _, cold, _ = _warm_and_cold_users(client)
    r = client.post("/recommend", json={"user_id": cold, "k": 5}).json()
    assert r["strategy"] == "cold_zero_history"


def test_unknown_item_in_session_warns(client):
    r = client.post("/recommend", json={"session_item_ids": ["NOPE"], "k": 3}).json()
    assert r["warnings"] and r["strategy"] == "cold_zero_history"


def test_onboard_new_sku_is_retrievable(client):
    before = client.get("/health").json()["faiss_index_size"]
    sku = dict(item_id="SKU_NEW_TEST", title="Aero Trail Runner Pro", brand="Nova Works",
               description="Lightweight trail running shoe with grippy sole, cushion and breathable mesh.",
               dept="Footwear", subcat="Running Shoes", price=119.0, leaf="trail runner", color="teal", style="pro")
    o = client.post("/items/onboard", json=sku)
    assert o.status_code == 200 and o.json()["index_size"] == before + 1
    assert client.post("/items/onboard", json=sku).status_code == 409   # duplicate
    body = {"user_context": {"onboarding_categories": ["Running Shoes"], "region": "south"}, "k": 10,
            "include_new_items_boost": True}
    r = client.post("/recommend", json=body).json()
    new = [i for i in r["items"] if i["is_new_item"]]
    assert len(new) >= 2   # reserved injection slots
    assert new and any("New-Item Semantic Similarity" in t for i in new for t in i["explain"])
    client.app.state.svc.router.new_slots = 50      # widen the injection window: our SKU must be reachable
    client.app.state.svc.cache.clear()
    r = client.post("/recommend", json={**body, "k": 100}).json()
    assert "SKU_NEW_TEST" in {i["item_id"] for i in r["items"]}, (len(r["items"]), sum(i["is_new_item"] for i in r["items"]), r["strategy"])


def test_feedback_updates_bandit_and_session(client):
    svc = client.app.state.svc
    item = str(svc.pre.item_ids[0])
    n0 = svc.bandit.total_updates
    r = client.post("/feedback", json={"item_id": item, "event": "purchase", "session_id": "s-1",
                                       "user_context": {"region": "east"}}).json()
    assert r["accepted"] and r["reward"] == 1.0 and r["bandit_total_updates"] == n0 + 1 and r["session_length"] == 1
    assert client.post("/feedback", json={"item_id": "nope", "event": "click"}).status_code == 404


def test_feedback_shifts_bandit_scores(client):
    svc = client.app.state.svc
    ctx = {"region": "central", "device": "desktop"}
    x = svc.bandit.context_vector({"region": "central", "device": "desktop"}, {"channel": "web", "season": "winter", "hour": 9})
    arm = svc.router.subcat_idx[5]
    before = svc.bandit.arm_scores(x, explore=False)[arm]
    for _ in range(30):
        client.post("/feedback", json={"item_id": str(svc.pre.item_ids[5]), "event": "purchase",
                                       "user_context": {**ctx, "channel": "web", "season": "winter", "hour": 9}})
    assert svc.bandit.arm_scores(x, explore=False)[arm] > before


def test_cache_hit_and_lru_fallback(client):
    body = {"user_context": {"region": "north", "hour": 3}, "k": 4}
    a = client.post("/recommend", json=body).json()
    b = client.post("/recommend", json=body).json()
    assert not a["cached"] and b["cached"] and a["items"] == b["items"]
    assert client.app.state.svc.cache.backend == "in-memory-lru"


def test_validation_errors(client):
    assert client.post("/recommend", json={"k": 0}).status_code == 422
    assert client.post("/feedback", json={"item_id": "x", "event": "bogus"}).status_code == 422
