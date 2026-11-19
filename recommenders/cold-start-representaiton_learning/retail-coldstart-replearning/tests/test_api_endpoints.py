"""API tests covering every cold-start scenario and edge cases (requires trained artifacts: run_pipeline.py)."""
from __future__ import annotations

import fakeredis
import pytest
import yaml
from fastapi.testclient import TestClient

from src.serving.app import ROOT, Cache, Engine, create_app

CFG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
if not (ROOT / CFG["paths"]["outputs"] / "models" / "two_tower.pt").exists():
    pytest.skip("trained artifacts missing - run `python run_pipeline.py` first", allow_module_level=True)


@pytest.fixture(scope="module")
def engine():
    return Engine(ROOT, CFG, cache=Cache(None, 300, True))


@pytest.fixture(scope="module")
def client(engine):
    with TestClient(create_app(ROOT, CFG, engine)) as c:
        yield c


def users_of(engine, group):
    u = engine.bundle.users
    return u.user_id[u.user_group == group].tolist()


def rec(client, **body):
    return client.post("/recommend", json=body)


def test_health(client):
    j = client.get("/health").json()
    assert j["status"] in ("ok", "degraded") and j["embedding_dim"] == 128 and j["index_size"] >= 2500


def test_warm_user_uses_full_ensemble(client, engine):
    uid = next(u for u in users_of(engine, "warm") if (engine.bundle.hist_np[engine.bundle.user_index[u]] > 0).sum() > 3)
    r = rec(client, user_id=uid, k=15)
    assert r.status_code == 200
    j = r.json()
    assert j["strategy"] == "multimodal+contrastive_ensemble" and j["cold_start_type"] == "warm"
    assert len(j["items"]) == 15
    scores = [i["score"] for i in j["items"]]
    assert scores == sorted(scores, reverse=True)
    hist = {engine.item_ids[x - 1] for x in engine.bundle.hist_np[engine.bundle.user_index[uid]] if x > 0}
    assert not hist & {i["item_id"] for i in j["items"]}, "already-seen items must be excluded"


def test_zero_shot_known_user(client, engine):
    j = rec(client, user_id=users_of(engine, "zero_shot")[0]).json()
    assert j["strategy"] == "content+graph" and j["cold_start_type"] == "user_cold_zero_shot" and j["n_history_items"] == 0


def test_few_shot_known_user(client, engine):
    j = rec(client, user_id=users_of(engine, "few_shot")[0]).json()
    assert j["strategy"] == "meta_adapter+graph" and j["cold_start_type"] == "user_cold_few_shot" and 1 <= j["n_history_items"] <= 3


def test_brand_new_user_from_profile_only(client):
    j = rec(client, profile={"acquisition_channel": "social", "home_store": "S03", "quiz_tags": ["likes_audio"]}, k=5).json()
    assert j["strategy"] == "content+graph" and j["n_history_items"] == 0 and len(j["items"]) == 5
    assert sum(i["l2"] == "Audio" for i in j["items"]) >= 1, "quiz tag should steer zero-shot recommendations"


def test_system_cold_new_store(client, engine):
    store = engine.bundle.stores.store_id[engine.bundle.stores.is_system_cold].iloc[0]
    j = rec(client, profile={"home_store": store, "acquisition_channel": "in_store_qr"}, context={"store_id": store, "channel": "store"}).json()
    assert j["cold_start_type"] == "system_cold" and j["strategy"] == "content+graph"


def test_session_clicks_route_to_meta_adapter_then_ensemble(client, engine):
    ids = engine.item_ids[:6]
    one = rec(client, recent_item_ids=ids[:2], profile={"acquisition_channel": "organic"}).json()
    assert one["strategy"] == "meta_adapter+graph" and one["n_history_items"] == 2
    many = rec(client, recent_item_ids=ids, profile={"acquisition_channel": "organic"}).json()
    assert many["strategy"] == "multimodal+contrastive_ensemble"
    assert not set(ids) & {i["item_id"] for i in many["items"]}


def test_unknown_item_ids_reported_not_fatal(client):
    j = rec(client, profile={"loyalty_tier": "gold"}, recent_item_ids=["I0001", "BOGUS"]).json()
    assert j["unknown_item_ids"] == ["BOGUS"] and j["n_history_items"] == 1


def test_filters(client, engine):
    j = rec(client, profile={"loyalty_tier": "gold"}, filters={"l1": "Footwear", "max_price_tier": 3}, k=20).json()
    assert j["items"] and all(i["l1"] == "Footwear" and i["price_tier"] <= 3 for i in j["items"])
    j = rec(client, profile={"loyalty_tier": "gold"}, filters={"only_cold_items": True}, k=20).json()
    assert len(j["items"]) == 20 and all(i["is_cold_item"] for i in j["items"])
    j = rec(client, profile={"loyalty_tier": "gold"}, filters={"exclude_cold_items": True}, k=20).json()
    assert not any(i["is_cold_item"] for i in j["items"])
    assert rec(client, profile={}, filters={"only_cold_items": True, "exclude_cold_items": True}).status_code == 422


@pytest.mark.parametrize("body", [
    {}, {"user_id": "U0001", "k": 0}, {"user_id": "U0001", "k": 101}, {"user_id": "U0001", "k": "5"},
    {"user_id": "U0001", "unexpected": 1}, {"user_id": "U0001", "context": {"hour": 24}},
    {"user_id": "U0001", "filters": {"max_price_tier": 9}}, {"user_id": "U0001", "recent_item_ids": ["I0001"] * 51},
])
def test_validation_errors(client, body):
    assert client.post("/recommend", json=body).status_code == 422


def test_unknown_user_without_profile_is_404(client):
    assert rec(client, user_id="NOPE").status_code == 404


def test_unknown_user_with_profile_is_treated_as_new_user(client):
    r = rec(client, user_id="BRAND_NEW", profile={"acquisition_channel": "paid_search"})
    assert r.status_code == 200 and r.json()["user_id"] == "BRAND_NEW"


def test_cache_hit_and_bypass(client):
    body = {"profile": {"acquisition_channel": "organic", "quiz_tags": ["likes_rugs"]}, "k": 7}
    a = rec(client, **body).json()
    b = rec(client, **body).json()
    assert not a["telemetry"]["cache_hit"] and b["telemetry"]["cache_hit"]
    assert [i["item_id"] for i in a["items"]] == [i["item_id"] for i in b["items"]]
    c = rec(client, **body, use_cache=False).json()
    assert not c["telemetry"]["cache_hit"]


def test_telemetry_fields(client):
    t = rec(client, profile={"loyalty_tier": "silver"}, use_cache=False).json()["telemetry"]
    assert t["total_ms"] >= t["embed_ms"] >= 0 and t["ann_ms"] >= 0 and t["index"] == "IndexHNSWFlat"


def test_redis_backend_and_graceful_fallback():
    c = Cache(None, 60, True, client=fakeredis.FakeRedis(decode_responses=True))
    assert c.backend == "redis"
    c.set("k", {"a": 1})
    assert c.get("k") == {"a": 1}

    class Broken:
        def ping(self):
            return True

        def get(self, *_):
            raise ConnectionError("down")

        def setex(self, *_):
            raise ConnectionError("down")

    d = Cache(None, 60, True, client=Broken())
    assert d.backend == "redis"
    d.set("k", {"b": 2})                  # redis fails -> degrade to memory
    assert d.backend == "memory" and d.get("k") == {"b": 2}
    assert Cache("redis://127.0.0.1:1/0", 60, True).backend == "memory"      # unreachable redis at start-up
    assert Cache(None, 60, False).backend == "disabled"


def test_zero_shot_item_onboarding_and_retrieval(client):
    before = client.get("/health").json()["index_size"]
    body = {"title": "Acme premium black Headphone", "description": "headphones audio premium wireless over ear", "l1": "Electronics",
            "l2": "Audio", "l3": "Headphones", "brand": "Brand05", "price": 120.0, "price_tier": 3}
    r = client.post("/items/embed", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["registered"] and not j["used_visual"] and j["embedding_dim"] == 128
    assert sum(i["l2"] == "Audio" for i in j["nearest_items"]) >= 3, "new item must land near its category"
    assert client.get("/health").json()["index_size"] == before + 1
    out = rec(client, profile={"quiz_tags": ["likes_audio"]}, filters={"only_cold_items": True}, k=100, use_cache=False).json()
    assert j["item_id"] in {i["item_id"] for i in out["items"]}, "registered item must be retrievable immediately"


def test_item_embed_validation(client):
    base = {"title": "t", "description": "d", "l1": "Home", "l2": "Decor", "l3": "Rugs", "brand": "Brand01", "price": 10.0, "price_tier": 2}
    assert client.post("/items/embed", json={**base, "visual_embedding": [0.1] * 10}).status_code == 422
    assert client.post("/items/embed", json={**base, "price": -1}).status_code == 422
    ok = client.post("/items/embed", json={**base, "brand": "NeverSeenBrand", "visual_embedding": [0.0] * 64, "register": False})
    assert ok.status_code == 200 and ok.json()["used_visual"] and not ok.json()["used_graph_neighbours"]


def test_metrics_and_strategies(client):
    m = client.get("/metrics").json()
    assert m["requests"] > 0 and m["latency_ms"]["p99"] >= m["latency_ms"]["p50"]
    assert len(client.get("/strategies").json()["routes"]) == 3
