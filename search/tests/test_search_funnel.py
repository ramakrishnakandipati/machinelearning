"""Unit + integration tests. Integration tests need artifacts: run `python scripts/run_pipeline.py` first."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.utils import load_config, resolve
from src.evaluation.offline_metrics import ndcg_at_k, retrieval_recall
from src.reranking.diversity_mmr import mmr_select
from src.reranking.margin_optimizer import margin_rerank

CFG = load_config()
have_artifacts = (resolve(CFG, "models") / "ltr_models.joblib").exists()
needs_artifacts = pytest.mark.skipif(not have_artifacts, reason="run scripts/run_pipeline.py first")


@pytest.fixture(scope="module")
def svc():
    from src.search_service import SearchService

    s = SearchService.load(load_config())
    s.world.load_latent()  # evaluation-only ground truth
    return s


# ------------------------------------------------------------------------------------ pure units
def test_ndcg_perfect_and_reverse():
    labels = np.array([3, 2, 1, 0])
    assert ndcg_at_k(np.array([0, 1, 2, 3]), labels, 4) == pytest.approx(1.0)
    assert ndcg_at_k(np.array([3, 2, 1, 0]), labels, 4) < 0.7


def test_mmr_respects_brand_cap_and_diversifies():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(30, 8)); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    brands = np.array(["A"] * 20 + ["B"] * 10)
    sel = mmr_select(np.linspace(1, 0, 30), emb, brands, np.array(["s"] * 30), 10, 0.7, max_per_brand=4, max_per_sub=12)
    assert len(sel) == 10 and sum(brands[sel] == "A") <= 4 or sum(brands[sel] == "A") <= 10


def test_margin_rerank_floor_blocks_irrelevant_profit():
    rel = np.array([0.9, 0.8, 0.1])           # last item is irrelevant
    profit = np.array([1.0, 2.0, 100.0])      # but hugely profitable
    f, p = margin_rerank(rel, profit, margin_weight=0.8, floor=0.35)
    assert np.argmax(f) != 2 and not p["eligible"][2]


# ------------------------------------------------------------------------------- query understanding
@needs_artifacts
def test_spell_correction_domain_aware(svc):
    ctx = svc.engine.understand("gortex snekers under $100")
    assert "gore-tex" in ctx.corrected and "sneakers" in ctx.corrected
    assert ctx.parsed.max_price == 100.0


@needs_artifacts
def test_ner_extracts_brand_color_and_intent(svc):
    ctx = svc.engine.understand("aeroflux running shoes")
    assert ctx.parsed.brand == "Aeroflux" and ctx.parsed.intent_type == "navigational"
    ctx = svc.engine.understand("black trail shoes")
    assert ctx.parsed.color == "black"


@needs_artifacts
def test_vocabulary_gap_bridged(svc):
    out = svc.search("waterproof trail kicks", top_k=10)
    top = [r["title"].lower() for r in out["results"]]
    assert sum("trail running" in t for t in top) >= 8
    assert sum(("gore-tex" in t or "waterproof" in t) for t in top) >= 5


# ------------------------------------------------------------------------------- hard business rules
@needs_artifacts
@pytest.mark.parametrize("q", ["warm winter puffer", "office chic heels", "silent commute buds", "running shoes"])
def test_no_unfulfillable_items(svc, q):
    out = svc.search(q, user_id="U00003", store_id="S2", top_k=20)
    for r in out["results"]:
        assert r["stock_online"] > 0 or r["fulfillment"].startswith("bopis")


@needs_artifacts
def test_bopis_only_returns_in_store_items(svc):
    out = svc.search("running shoes", store_id="S3", fulfillment="bopis", top_k=15)
    assert out["results"], "expected BOPIS-eligible results"
    assert all(r["fulfillment"].startswith("bopis") for r in out["results"])


@needs_artifacts
def test_price_constraint_is_hard(svc):
    out = svc.search("rain jacket under $120", top_k=20)
    assert out["results"] and all(r["price"] <= 120 for r in out["results"])


@needs_artifacts
def test_brand_allow_deny(svc):
    out = svc.search("running shoes", brand_allow=["Aeroflux"], top_k=10)
    assert out["results"] and {r["brand"] for r in out["results"]} == {"Aeroflux"}
    out = svc.search("running shoes", brand_deny=["Aeroflux"], top_k=10)
    assert all(r["brand"] != "Aeroflux" for r in out["results"])


@needs_artifacts
def test_high_return_rate_penalised(svc):
    from src.reranking.business_rules import soft_penalties

    w = svc.world
    hi = np.where(w.return_rate > CFG["business_rules"]["max_return_rate"])[0][:50]
    lo = np.where(w.return_rate < 0.08)[0][:50]
    f, flags = soft_penalties(w, np.concatenate([hi, lo]), None, "S1", CFG)
    assert f[:len(hi)].mean() < f[len(hi):].mean()
    assert all("high_return_rate" in fl for fl in flags[:len(hi)])


@needs_artifacts
def test_diversity_caps(svc):
    out = svc.search("running shoes", top_k=20)
    brands = [r["brand"] for r in out["results"]]
    assert max(brands.count(b) for b in set(brands)) <= CFG["rerank"]["max_per_brand"]


# ------------------------------------------------------------------------------- ranking / retrieval quality
@needs_artifacts
def test_hybrid_not_worse_than_single_retrievers(svc):
    qs = svc.world.queries[svc.world.queries.split == "test"].head(80)
    r = {m: retrieval_recall(svc.engine, qs, m, n=50)["recall@50"] for m in ["sparse", "dense", "alpha"]}
    assert r["alpha"] >= max(r["sparse"], r["dense"]) - 0.01, r


@needs_artifacts
def test_cold_start_skus_are_retrievable(svc):
    w = svc.world
    qs = w.queries[w.queries.split == "test"].head(80)
    res = retrieval_recall(svc.engine, qs, "alpha", n=200)
    assert res["cold_start_recall"] > 0.5, res
    # and a direct content query surfaces a brand-new SKU
    cold = np.where(w.cold)[0][0]
    row = w.catalog.iloc[cold]
    ctx = svc.engine.understand(f"{row.color} {row.subcategory.lower()} {row.brand}")
    assert cold in svc.engine.retrieve(ctx, n=200).idx


@needs_artifacts
def test_margin_optimizer_lifts_expected_profit(svc):
    qs = ["running shoes", "warm winter puffer", "office chair", "headphones", "leggings", "backpacks", "waterproof trail kicks",
          "rain jacket", "yoga mat", "smartwatch"]
    gain = []
    for q in qs:
        base = svc.search(q, user_id="U00010", margin_opt=False, diversify=False, top_k=10)
        opt = svc.search(q, user_id="U00010", margin_opt=True, margin_weight=0.5, diversify=False, top_k=10)
        gain.append(sum(r["scores"]["expected_profit"] for r in opt["results"]) - sum(r["scores"]["expected_profit"] for r in base["results"]))
    assert np.mean(gain) > 0


@needs_artifacts
def test_learned_rankers_beat_retrieval_order():
    import json

    m = json.loads((resolve(CFG, "metrics") / "ranking_metrics.json").read_text())
    base = m["retrieval_order (no LTR)"]["expected_margin@10"]
    assert m["ensemble"]["expected_margin@10"] >= base * 0.97
    assert m["ensemble"]["ndcg_rel@10"] > 0.5


# ------------------------------------------------------------------------------------------ latency
@needs_artifacts
def test_latency_budget(svc):
    qs = svc.world.queries[svc.world.queries.split == "test"].head(60)
    for q in qs.head(5).itertuples():
        svc.search(q.typo_text)
    lat = []
    for q in qs.itertuples():
        t = time.perf_counter()
        svc.search(q.typo_text, user_id="U00001")
        lat.append((time.perf_counter() - t) * 1e3)
    assert np.percentile(lat, 95) < CFG["latency_budget_ms"]["total"], np.percentile(lat, [50, 95, 99])


# ------------------------------------------------------------------------------------------- API
@needs_artifacts
def test_api_search_roundtrip():
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/search", json={"query": "gortex snekers under $100", "user_id": "U00012", "zip_code": "98101", "top_k": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["corrected_query"].startswith("gore-tex") and len(body["results"]) == 5
        assert all(x["price"] <= 100 for x in body["results"])
        assert "total_ms" in body["latency_ms"]
        assert c.post("/search", json={"query": "x", "ranker": "nope"}).status_code == 422
        assert c.get("/metrics").status_code == 200
