"""FastAPI microservice exposing the four Semantic-ID cold-start approaches.

Run:  uvicorn src.api.app:app --port 8000      (after `python run_pipeline.py` has produced outputs/models)
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.api.harness_html import render_harness
from src.artifacts import Bundle, load_bundle
from src.config import DEVICE
from src.data.generate_data import (AGE_BANDS, BRANDS, CATEGORIES, CHANNELS, COLORS, DEVICES, MATERIALS, REGIONS,
                                    SUB_NAMES)
from src.recommenders.base import UserContext

Channel = Literal[tuple(CHANNELS)]
Device = Literal[tuple(DEVICES)]
Region = Literal[tuple(REGIONS)]
AgeBand = Literal[tuple(AGE_BANDS)]
Category = Literal[tuple(CATEGORIES)]
SubCategory = Literal[tuple(SUB_NAMES)]
Approach = Literal["semantic_trie", "generative_seq", "codebook_bridge", "hybrid"]
LOCK = threading.Lock()          # torch / FAISS objects are shared across worker threads


# ------------------------------------------------------------------------------------------------ request models
class ItemColdStartRequest(BaseModel):
    category: Category = Field(..., examples=["Footwear"])
    subcategory: Optional[SubCategory] = Field(None, description="Optional; inferred from the description if omitted")
    brand: Optional[str] = Field(None, examples=["Nova"], description=f"One of {BRANDS[:5]}... (unknown brands are ignored)")
    price: float = Field(..., gt=0, examples=[89.99])
    description: str = Field("", examples=["Nova black leather chelsea ankle boot, waterproof, premium"])
    color: Optional[str] = Field(None, examples=["black"])
    material: Optional[str] = Field(None, examples=["leather"])
    margin: Optional[float] = Field(None, ge=0, le=1, description="Gross margin; defaults to the category mean")
    top_k_items: int = Field(8, ge=1, le=50)
    top_k_segments: int = Field(5, ge=1, le=20)


class UserColdStartRequest(BaseModel):
    channel: Optional[Channel] = None
    device: Optional[Device] = None
    region: Optional[Region] = None
    age_band: Optional[AgeBand] = None
    intent_category: Optional[Category] = None
    intent_subcategory: Optional[SubCategory] = None
    initial_click_item_id: Optional[int] = Field(None, ge=0)
    history_item_ids: List[int] = Field(default_factory=list, max_length=2, description="0-2 prior interactions")
    k: int = Field(10, ge=1, le=50)
    approach: Approach = "hybrid"


class SystemColdStartRequest(BaseModel):
    landing_category: Optional[Category] = None
    landing_subcategory: Optional[SubCategory] = None
    channel: Optional[Channel] = None
    k: int = Field(10, ge=1, le=50)


class CompareRequest(BaseModel):
    scenario: Literal["user_cold", "item_cold", "system_cold"] = "user_cold"
    channel: Optional[Channel] = None
    device: Optional[Device] = None
    region: Optional[Region] = None
    age_band: Optional[AgeBand] = None
    intent_category: Optional[Category] = None
    intent_subcategory: Optional[SubCategory] = None
    initial_click_item_id: Optional[int] = Field(None, ge=0)
    history_item_ids: List[int] = Field(default_factory=list, max_length=8)
    k: int = Field(5, ge=1, le=30)


# ------------------------------------------------------------------------------------------------ service state
class Service:
    def __init__(self, bundle: Bundle):
        self.b = bundle
        self.cat = bundle.rq
        self.recs = bundle.main
        self.hybrid = bundle.main["hybrid"]
        self.bridge = bundle.main["codebook_bridge"]
        self.trie = bundle.main["semantic_trie"]
        self.rows = self.cat.items.to_dict("records")
        # pre-embedded customer segments for "target user segments" of new items
        self.segments = [(r, a, c) for r in REGIONS for a in AGE_BANDS for c in CATEGORIES]
        ctxs = [UserContext(region=r, age_band=a, intent_cat=c) for r, a, c in self.segments]
        u, lps = self.bridge.user_forward(ctxs)
        self.seg_u = u
        self.seg_lp = [l.cpu().numpy() for l in lps]
        self.samples = self._build_samples()

    def _build_samples(self) -> dict:
        items = self.cat.items
        warm = items[~items.is_cold]
        click = int(warm[warm.subcategory == "Headphones"].item_id.iloc[0])
        hist = [int(x) for x in warm[warm.subcategory == "Smartwatches"].item_id.iloc[:1]]
        inspect_id = int(warm[warm.subcategory == "Sneakers"].item_id.iloc[0])
        user = {"channel": "mobile_app", "device": "ios", "region": "europe", "age_band": "25-34",
                "intent_category": "Electronics", "initial_click_item_id": click, "history_item_ids": hist, "k": 8,
                "approach": "hybrid"}
        return {
            "item_cold": {"category": "Footwear", "brand": "Nova", "price": 89.99, "color": "black", "material": "leather",
                          "description": "Nova black leather chelsea ankle boot, waterproof, premium everyday boot",
                          "top_k_items": 6, "top_k_segments": 5},
            "user_cold": user,
            "system_cold": {"landing_category": "Sports", "channel": "web", "k": 8},
            "compare": {**{k: v for k, v in user.items() if k not in ("approach",)}, "scenario": "user_cold", "k": 5},
            "inspect": {"item_id": inspect_id},
        }

    # ---- presentation helpers
    def view(self, i: int, score: Optional[float] = None, explanation: Optional[str] = None) -> dict:
        r = self.rows[i]
        d = {"item_id": int(i), "title": r["title"], "category": r["category"], "subcategory": r["subcategory"],
             "brand": r["brand"], "price": round(float(r["price"]), 2), "margin": round(float(r["margin"]), 3),
             "new_launch": bool(r["is_cold"]), "semantic_id": self.cat.semantic_id(i),
             "semantic_id_str": self.cat.semantic_id_str(i)}
        if score is not None:
            d["score"] = round(float(score), 4)
        if explanation:
            d["explanation"] = explanation
        return d

    def ctx_from(self, req) -> UserContext:
        hist = list(getattr(req, "history_item_ids", []) or [])
        click = getattr(req, "initial_click_item_id", None)
        for it in hist + ([click] if click is not None else []):
            if not 0 <= it < self.cat.N:
                raise HTTPException(422, f"item id {it} is not in the catalog (0..{self.cat.N - 1})")
        return UserContext(history=hist, click_item=click, channel=getattr(req, "channel", None),
                           device=getattr(req, "device", None), region=getattr(req, "region", None),
                           age_band=getattr(req, "age_band", None), intent_cat=getattr(req, "intent_category", None),
                           intent_sub=getattr(req, "intent_subcategory", None))

    def explain(self, item: int, ctx: UserContext) -> str:
        seed, depth = self.trie.match_depth(item, ctx)
        parts = []
        if seed is not None and depth > 0:
            lvl = "(" + ",".join(f"c{d + 1}" for d in range(depth)) + ")"
            src = self.rows[seed]["title"]
            parts.append(f"shares Semantic-ID prefix {lvl}={self.cat.semantic_id(item)[:depth]} with '{src}'")
        if ctx.intent_cat and self.rows[item]["category"] == ctx.intent_cat:
            parts.append(f"matches intent category {ctx.intent_cat}")
        if ctx.region or ctx.age_band:
            parts.append("fits demographic segment prior")
        if self.rows[item]["is_cold"]:
            parts.append("new launch ranked via semantic neighbours")
        return "; ".join(parts) if parts else "popularity / margin prior"

    def timed(self, fn):
        t0 = time.perf_counter()
        out = fn()
        return out, (time.perf_counter() - t0) * 1000


# ------------------------------------------------------------------------------------------------ app
SVC: Optional[Service] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SVC
    SVC = Service(load_bundle())
    yield


app = FastAPI(
    title="Omnichannel Cold-Start Recommender (VQ-VAE / RQ-VAE Semantic IDs)",
    description="Item, user and system cold-start recommendations through hierarchical Semantic IDs "
                "(c1..c4). Four approaches: semantic trie, generative seq-to-Semantic-ID, codebook-bridged two-tower, "
                "hybrid graph + business re-ranker. Interactive test dashboard: **/ui/test-harness**.",
    version="1.0.0", lifespan=lifespan)


def svc() -> Service:
    if SVC is None:
        raise HTTPException(503, "models are still loading")
    return SVC


@app.get("/health", tags=["system"])
def health():
    s = svc()
    return {"status": "ok", "device": str(DEVICE), "cuda_available": DEVICE.type == "cuda",
            "codebook_size": s.cat.K, "n_levels": s.cat.M, "semantic_id_format": "c1-c2-c3-c4",
            "catalog_items": s.cat.N, "new_launch_items": int(s.cat.is_cold.sum()),
            "unique_semantic_ids": int(len(np.unique(s.cat.codes, axis=0))),
            "approaches": list(s.recs.keys()), "tokenizers": ["rq_vae", "vq_vae"]}


@app.post("/recommend/item-cold-start", tags=["cold-start"])
def item_cold_start(req: ItemColdStartRequest):
    """Brand-new SKU (0 interactions): encode on the fly into Semantic IDs, then return target segments + look-alikes."""
    s = svc()

    def run():
        rec = {"category": req.category, "subcategory": req.subcategory, "brand": req.brand, "price": req.price,
               "description": req.description, "color": req.color, "material": req.material, "margin": req.margin}
        X = s.b.dataset["featurizer"].transform_records([rec])
        codes, zq, _ = s.cat.encode_features(X)
        vq_codes, _, _ = s.b.vq.encode_features(X)
        codes, zq = codes[0], zq[0]
        with LOCK:
            # target segments: fused bridge score (embedding dot-product + codebook-overlap) over 240 customer segments
            v = s.bridge.embed_new_items(zq[None])[0]
            dot = (s.seg_u @ v).cpu().numpy()
            overlap = sum(s.bridge.level_w[l] * s.seg_lp[l][:, codes[l]] for l in range(s.cat.M))
            fused = s.bridge._zn(dot) + 0.5 * s.bridge._zn(np.asarray(overlap))
            order = np.argsort(-fused)[:req.top_k_segments]
            similar = s.hybrid.similar_items(zq, req.top_k_items)
        sub_c = {d: int((s.cat.codes[:, :d] == codes[:d]).all(1).sum()) for d in range(1, s.cat.M + 1)}
        sim_out = []
        for i, sc in similar:
            d = 0
            while d < s.cat.M and s.cat.codes[i, d] == codes[d]:
                d += 1
            v_ = s.view(i, sc)
            v_["shared_prefix_depth"] = d
            sim_out.append(v_)
        return {
            "input": rec,
            "semantic_id": [int(c) for c in codes], "semantic_id_str": "-".join(str(int(c)) for c in codes),
            "vq_vae_code": int(vq_codes[0][0]),
            "is_novel_full_id": bool(sub_c[s.cat.M] == 0),
            "catalog_items_sharing_prefix": {f"depth_{d}": n for d, n in sub_c.items()},
            "target_segments": [{"region": s.segments[j][0], "age_band": s.segments[j][1], "intent_category": s.segments[j][2],
                                 "affinity_z": round(float(fused[j]), 3)} for j in order],
            "similar_items": sim_out,
            "smoothed_popularity_prior": round(float(np.mean([s.hybrid.pop[i] for i, _ in similar])), 4) if similar else None,
        }

    out, ms = s.timed(run)
    out["latency_ms"] = round(ms, 2)
    return out


def _recommend(s: Service, approach: str, ctx: UserContext, k: int, candidates=None) -> List[dict]:
    rec = s.recs[approach]
    with LOCK:
        res = rec.recommend(ctx, k, candidates, set(ctx.history) | ({ctx.click_item} if ctx.click_item is not None else set()))
    return [s.view(i, sc, s.explain(i, ctx)) for i, sc in res]


@app.post("/recommend/user-cold-start", tags=["cold-start"])
def user_cold_start(req: UserColdStartRequest):
    """First-time visitor: demographic / session context and 0-2 interactions -> personalised top-K."""
    s = svc()
    ctx = s.ctx_from(req)
    if not (ctx.has_meta() or ctx.history or ctx.click_item is not None or ctx.intent_cat or ctx.intent_sub):
        raise HTTPException(422, "provide at least one context field (metadata, intent or a clicked item)")
    ctx.regime = "user_cold"
    out, ms = s.timed(lambda: _recommend(s, req.approach, ctx, req.k))
    with LOCK:
        pred = s.bridge.predicted_codes(ctx, 3)
    return {"approach": req.approach, "regime": "user_cold", "latency_ms": round(ms, 2),
            "context": {"channel": ctx.channel, "device": ctx.device, "region": ctx.region, "age_band": ctx.age_band,
                        "intent_category": ctx.intent_cat, "intent_subcategory": ctx.intent_sub,
                        "clicked_item": (s.view(ctx.click_item) if ctx.click_item is not None else None)},
            "predicted_semantic_prefix": [{"level": l + 1, "top_codes": [{"code": c, "prob": round(p, 3)} for c, p in lv]}
                                          for l, lv in enumerate(pred)],
            "recommendations": out}


@app.post("/recommend/system-cold-start", tags=["cold-start"])
def system_cold_start(req: SystemColdStartRequest):
    """Day-0 anonymous session: no logs. Diversified, high-margin Semantic-ID cluster representatives."""
    s = svc()
    ctx = UserContext(channel=req.channel, intent_cat=req.landing_category, intent_sub=req.landing_subcategory,
                      regime="system_cold")
    out, ms = s.timed(lambda: _recommend(s, "hybrid", ctx, req.k))
    clusters: Dict[int, dict] = {}
    for r in out:
        c = clusters.setdefault(r["semantic_id"][0], {"level1_code": r["semantic_id"][0], "n_items": 0, "avg_margin": 0.0,
                                                     "representative": r["title"]})
        c["n_items"] += 1
        c["avg_margin"] += r["margin"]
    for c in clusters.values():
        c["avg_margin"] = round(c["avg_margin"] / c["n_items"], 3)
    return {"approach": "hybrid", "regime": "system_cold", "latency_ms": round(ms, 2),
            "strategy": "content-semantic relevance + margin + centrality, MMR-diversified, max 2 per level-1 cluster; no popularity logs",
            "avg_margin": round(float(np.mean([r["margin"] for r in out])), 3) if out else None,
            "semantic_clusters": sorted(clusters.values(), key=lambda c: -c["n_items"]), "recommendations": out}


@app.post("/recommend/compare-approaches", tags=["cold-start"])
def compare_approaches(req: CompareRequest):
    """Run all four approaches side by side (same request) and report ranked lists, Semantic IDs and latency."""
    s = svc()
    ctx = s.ctx_from(req)
    candidates = None
    if req.scenario == "system_cold":
        ctx = UserContext(intent_cat=req.intent_category, intent_sub=req.intent_subcategory, regime="system_cold")
    elif req.scenario == "item_cold":
        candidates = s.cat.cold_ids                           # rank only newly launched SKUs
    else:
        ctx.regime = "user_cold" if len(ctx.history) < 3 else None
    results, lists = {}, {}
    for name in ("semantic_trie", "generative_seq", "codebook_bridge", "hybrid"):
        out, ms = s.timed(lambda n=name: _recommend(s, n, ctx, req.k, candidates))
        results[name] = {"latency_ms": round(ms, 2), "recommendations": out,
                         "semantic_ids": [r["semantic_id_str"] for r in out]}
        lists[name] = {r["item_id"] for r in out}
    names = list(lists)
    overlap = {f"{a} vs {b}": round(len(lists[a] & lists[b]) / max(len(lists[a] | lists[b]), 1), 3)
               for i, a in enumerate(names) for b in names[i + 1:]}
    return {"scenario": req.scenario, "k": req.k, "candidate_pool": "new launches only" if candidates is not None else "full catalog",
            "approaches": results, "jaccard_overlap": overlap}


@app.get("/semantic-id/inspect/{item_id}", tags=["semantic-ids"])
def inspect_semantic_id(item_id: int, k: int = 6):
    """Inspect an item's RQ-VAE hierarchical tokens: tree path, per-level residual energy, siblings, look-alikes."""
    s = svc()
    if not 0 <= item_id < s.cat.N:
        raise HTTPException(404, f"item {item_id} not in catalog (0..{s.cat.N - 1})")
    c = s.cat
    codes = c.codes[item_id]
    path, residual = [], c.ze[item_id].copy()
    quant = c.model.quantizers
    base_norm = float(np.linalg.norm(residual))
    for l in range(c.M):
        node_items = np.where((c.codes[:, :l + 1] == codes[:l + 1]).all(1))[0]
        cb = quant[l].embedding.detach().cpu().numpy()[codes[l]]
        residual = residual - cb
        cats = c.items["category"].values[node_items]
        vals, cnt = np.unique(cats, return_counts=True)
        path.append({"level": l + 1, "code": int(codes[l]), "node_size": int(len(node_items)),
                     "dominant_category": str(vals[np.argmax(cnt)]), "category_purity": round(float(cnt.max() / len(node_items)), 3),
                     "residual_norm_after_level": round(float(np.linalg.norm(residual)), 4),
                     "residual_pct_of_encoder_norm": round(100 * float(np.linalg.norm(residual)) / (base_norm + 1e-9), 1),
                     "sibling_examples": [s.rows[j]["title"] for j in node_items if j != item_id][:3]})
    with LOCK:
        sim = s.hybrid.similar_items(c.zq[item_id], k, exclude={item_id})
    return {"item": s.view(item_id), "semantic_id": [int(x) for x in codes], "semantic_id_str": c.semantic_id_str(item_id),
            "vq_vae_code": int(s.b.vq.codes[item_id, 0]), "tree_path": path,
            "similar_items": [s.view(i, sc) for i, sc in sim]}


@app.get("/ui/test-harness", response_class=HTMLResponse, include_in_schema=False)
def test_harness():
    return HTMLResponse(render_harness(svc().samples))


@app.get("/", include_in_schema=False)
def root():
    return {"service": app.title, "docs": "/docs", "ui": "/ui/test-harness", "health": "/health"}
