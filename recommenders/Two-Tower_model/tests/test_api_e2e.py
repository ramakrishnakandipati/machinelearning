"""End-to-end API tests against a live uvicorn process; each test writes a screenshot to outputs/screenshots/."""
from __future__ import annotations

import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd
import pytest

from src.config import DATA, METRICS, SHOTS

from .shots import capture_swagger, render_cards

LOG = logging.getLogger("e2e")


@pytest.fixture(scope="module")
def client(server):
    with httpx.Client(base_url=server, timeout=15) as c:
        yield c


def call(c: httpx.Client, method: str, url: str, body=None) -> dict:
    t0 = time.perf_counter()
    r = c.request(method, url, json=body)
    ms = (time.perf_counter() - t0) * 1000
    try:
        data = r.json()
    except ValueError:
        data = r.text
    LOG.info("%s %s body=%s -> %d %s", method, url, json.dumps(body), r.status_code, json.dumps(data)[:400])
    return {"method": method, "url": str(r.request.url), "status": r.status_code, "reason": r.reason_phrase,
            "latency_ms": ms, "request": body, "response": data, "ok": r.status_code < 400}


@pytest.fixture(scope="module")
def warm_user() -> int:
    snap = pd.read_parquet(DATA / "user_features_snapshot.parquet")
    pur = pd.read_parquet(DATA / "purchased.parquet")
    return int(pur.groupby("user_id").size().sort_values(ascending=False).index[0]) if len(pur) else int(snap.user_id.iloc[0])


def test_01_swagger_docs_screenshot(server, client):
    spec = client.get("/openapi.json").json()
    paths = set(spec["paths"])
    assert {"/health", "/recommend/user", "/recommend/similar-items", "/index/reload"} <= paths
    how = capture_swagger(server, SHOTS / "01_swagger_ui_docs.png", spec)
    LOG.info("swagger screenshot via %s", how)
    assert (SHOTS / "01_swagger_ui_docs.png").stat().st_size > 5000


def test_02_health(client):
    ex = call(client, "GET", "/health")
    h = ex["response"]
    assert ex["status"] == 200 and h["status"] == "ok"
    assert h["user_tower_loaded"] and h["faiss_ready"] and h["faiss_ntotal"] == 5000
    assert h["redis"]["connected"] and h["redis"]["backend"] in ("redis", "fakeredis")
    assert h["device"] in ("cuda", "cpu")
    render_cards(SHOTS / "02_health_check_response.png", "GET /health - GPU / FAISS / model / Redis readiness", [ex])


def test_03_user_recommendations_warm_and_cold(client, warm_user):
    purchased = set(pd.read_parquet(DATA / "purchased.parquet").query("user_id == @warm_user").item_id)
    warm = call(client, "POST", "/recommend/user", {"user_id": str(warm_user), "top_k": 10, "channel": "Mobile"})
    w = warm["response"]
    assert warm["status"] == 200 and w["strategy"] == "personalized" and not w["cold_start"]
    assert len(w["items"]) == 10
    scores = [i["score"] for i in w["items"]]
    assert scores == sorted(scores, reverse=True)
    assert not ({i["item_id"] for i in w["items"]} & purchased), "exclude_purchased violated"
    assert all(i["in_stock"] for i in w["items"])
    assert set(w["telemetry"]) == {"inference_ms", "ann_search_ms", "total_ms"}
    assert w["telemetry"]["total_ms"] >= w["telemetry"]["ann_search_ms"] > 0

    filt = call(client, "POST", "/recommend/user", {"user_id": str(warm_user), "top_k": 5, "category_filter": "Apparel"})
    assert filt["status"] == 200 and filt["response"]["items"]
    assert all(i["category_l1"] == "Apparel" for i in filt["response"]["items"])

    cold = call(client, "POST", "/recommend/user", {"user_id": "99999999", "top_k": 10, "channel": "Web"})
    c = cold["response"]
    assert cold["status"] == 200 and c["cold_start"] and c["strategy"] == "cold_start_segment" and len(c["items"]) == 10
    nonnum = call(client, "POST", "/recommend/user", {"user_id": "guest-abc", "top_k": 3})
    assert nonnum["response"]["cold_start"]
    render_cards(SHOTS / "03_user_recommendations_response.png",
                 "POST /recommend/user - warm user (personalized) vs cold-start user (segment fallback)", [warm, filt, cold])


def test_04_similar_items(client):
    ex = call(client, "POST", "/recommend/similar-items", {"item_id": 10, "top_k": 8})
    r = ex["response"]
    assert ex["status"] == 200 and len(r["items"]) == 8 and r["strategy"] == "item_similarity"
    assert 10 not in {i["item_id"] for i in r["items"]}
    s = [i["score"] for i in r["items"]]
    assert s == sorted(s, reverse=True)
    miss = call(client, "POST", "/recommend/similar-items", {"item_id": 999999, "top_k": 3})
    assert miss["status"] == 404
    render_cards(SHOTS / "04_similar_items_response.png", "POST /recommend/similar-items", [ex, miss])


def test_05_validation_errors(client):
    for body in ({"user_id": "", "top_k": 0}, {"user_id": "1", "top_k": 10, "bogus": 1}, {"user_id": "1", "channel": "Fax"}):
        ex = call(client, "POST", "/recommend/user", body)
        assert ex["status"] == 422 and ex["response"]["error"] == "validation_error" and ex["response"]["details"]


def test_06_hot_reload_keeps_serving(client, warm_user):
    before = call(client, "GET", "/health")["response"]["model_version"]
    ex = call(client, "POST", "/index/reload")
    assert ex["status"] == 200 and ex["response"]["status"] == "reloaded" and ex["response"]["n_items"] == 5000
    after = call(client, "POST", "/recommend/user", {"user_id": str(warm_user), "top_k": 5})
    assert after["status"] == 200 and len(after["response"]["items"]) == 5
    LOG.info("reload: %s -> %s", before, ex["response"]["model_version"])


def test_07_concurrency_latency_benchmark(server):
    users = pd.read_parquet(DATA / "user_features_snapshot.parquet").user_id.sample(100, random_state=1).tolist()

    def one(i):
        with httpx.Client(base_url=server, timeout=15) as c:
            t0 = time.perf_counter()
            r = c.post("/recommend/user", json={"user_id": str(users[i]), "top_k": 10})
            return r.status_code, (time.perf_counter() - t0) * 1000, r.json()["telemetry"]["total_ms"] if r.status_code == 200 else None

    with httpx.Client(base_url=server) as warm:           # warm connection / kernels
        for u in users[:5]:
            warm.post("/recommend/user", json={"user_id": str(u)})
    t0 = time.perf_counter()
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, range(100)))
    wall = time.perf_counter() - t0
    assert all(s == 200 for s, _, _ in res)
    client_ms, server_ms = sorted(r[1] for r in res), sorted(r[2] for r in res)
    pct = lambda a, p: a[min(len(a) - 1, int(p / 100 * len(a)))]
    out = {"requests": 100, "concurrency": 16, "wall_s": round(wall, 3), "throughput_rps": round(100 / wall, 1),
           "client_ms": {"p50": pct(client_ms, 50), "p95": pct(client_ms, 95), "p99": pct(client_ms, 99), "mean": statistics.mean(client_ms)},
           "server_total_ms": {"p50": pct(server_ms, 50), "p95": pct(server_ms, 95), "p99": pct(server_ms, 99), "mean": statistics.mean(server_ms)}}
    (METRICS / "api_benchmark.json").write_text(json.dumps(out, indent=2))
    LOG.info("benchmark: %s", json.dumps(out))
    assert out["server_total_ms"]["p99"] < 100, out
    assert out["client_ms"]["p99"] < 1000, out
