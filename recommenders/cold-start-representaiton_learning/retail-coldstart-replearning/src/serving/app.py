"""Production FastAPI microservice for cold-start personalisation.

  uvicorn src.serving.app:app --port 8000

Request flow: validate (Pydantic v2 strict) -> cache lookup (Redis, graceful in-memory fallback)
-> user representation (known user: stored profile + train history; brand-new user: inductive graph
embedding from attributes) -> cold-start router -> FAISS HNSW ANN with metadata filter -> response
with strategy / latency telemetry.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException

from src.indexing.faiss_manager import FaissManager
from src.models.common import build_bundle, l2n
from src.models.representation_fusion import STRATEGIES, RepresentationHub, load_all
from src.serving.schemas import (HealthResponse, ItemOut, NewItemRequest, NewItemResponse, RecommendRequest,
                                 RecommendResponse, Telemetry)

ROOT = Path(__file__).resolve().parents[2]


class Cache:
    """Redis cache with transparent fallback to a bounded in-process TTL dict."""

    def __init__(self, url: str | None, ttl: int = 300, enabled: bool = True, client=None):
        self.ttl, self.mem, self.client, self.enabled = ttl, {}, None, enabled
        if not enabled:
            return
        try:
            if client is None:
                import redis
                client = redis.Redis.from_url(url, socket_connect_timeout=0.25, socket_timeout=0.25, decode_responses=True)
            client.ping()
            self.client = client
        except Exception:
            self.client = None

    @property
    def backend(self) -> str:
        return "redis" if self.client is not None else ("memory" if self.enabled else "disabled")

    def get(self, key: str):
        if not self.enabled:
            return None
        if self.client is not None:
            try:
                v = self.client.get(key)
                return json.loads(v) if v else None
            except Exception:
                self.client = None            # redis went away -> degrade silently
        v = self.mem.get(key)
        if v and v[0] > time.time():
            return v[1]
        self.mem.pop(key, None)
        return None

    def set(self, key: str, value: dict):
        if not self.enabled:
            return
        if self.client is not None:
            try:
                self.client.setex(key, self.ttl, json.dumps(value))
                return
            except Exception:
                self.client = None
        if len(self.mem) > 5000:
            self.mem.pop(next(iter(self.mem)))
        self.mem[key] = (time.time() + self.ttl, value)


class Engine:
    def __init__(self, root: Path, cfg: dict, device: str = "cpu", cache: Cache | None = None):
        self.cfg, self.root = cfg, root
        dev = torch.device(device)
        self.device = dev
        self.bundle = b = build_bundle(cfg, root, dev)
        tt, gm, graph, adapter, ssl, w = load_all(b, root / cfg["paths"]["outputs"] / "models")
        self.hub = RepresentationHub(b, tt, gm, graph, adapter, ssl, weights=w if w.get("item") else None)
        fc = cfg["faiss"]
        self.items = b.items.reset_index(drop=True).copy()
        self.item_ids = self.items.item_id.tolist()
        self.item_idx = {i: k for k, i in enumerate(self.item_ids)}
        self.cold = list(self.items.is_cold_item.values)
        self.l1 = list(self.items.l1.values)
        self.tier = list(self.items.price_tier.values)
        self.fm = FaissManager(cfg["embedding_dim"]).build(self.hub.E_uni.cpu().numpy(), {}, fc["hnsw_m"], fc["ef_construction"],
                                                           fc["ef_search"], (fc["ivfpq"]["nlist"], fc["ivfpq"]["m"], fc["ivfpq"]["nbits"]))
        rc = cfg["redis"]
        self.cache = cache or Cache(rc["url"], rc["ttl_seconds"], rc["enabled"])
        self.lock = threading.Lock()
        self.lat = deque(maxlen=5000)
        self.counts = Counter()
        self.known_stores = set(b.stores.store_id[~b.stores.is_system_cold])   # stores with serving history
        self.n_new = 0

    # ------------------------------------------------------------------ helpers
    def _item_out(self, idx: int, score: float) -> ItemOut:
        r = self.items.iloc[idx]
        return ItemOut(item_id=r.item_id, title=r.title, l1=r.l1, l2=r.l2, l3=r.l3, brand=r.brand, price=float(r.price),
                       price_tier=int(r.price_tier), score=float(score), is_cold_item=bool(self.cold[idx]))

    def _allowed(self, f):
        if f is None:
            return None
        m = np.ones(len(self.item_ids), bool)
        if f.l1:
            m &= np.array([x == f.l1 for x in self.l1])
        if f.max_price_tier:
            m &= np.array(self.tier) <= f.max_price_tier
        if f.exclude_cold_items:
            m &= ~np.array(self.cold)
        if f.only_cold_items:
            m &= np.array(self.cold)
        return m

    @staticmethod
    def _key(req: RecommendRequest) -> str:
        d = req.model_dump(exclude={"use_cache"})
        return "rec:" + hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()

    # ------------------------------------------------------------------ recommend
    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        t0 = time.perf_counter()
        key = self._key(req)
        if req.use_cache:
            hit = self.cache.get(key)
            if hit:
                resp = RecommendResponse.model_validate(hit)
                resp.telemetry.cache_hit = True
                resp.telemetry.cache_backend = self.cache.backend
                resp.telemetry.total_ms = (time.perf_counter() - t0) * 1000
                self._record(resp)
                return resp
        b, hub = self.bundle, self.hub
        uidx = b.user_index.get(req.user_id) if req.user_id else None
        if req.user_id and uidx is None and req.profile is None and not req.recent_item_ids:
            raise HTTPException(404, f"unknown user_id '{req.user_id}': supply a profile or recent_item_ids to treat it as a new user")
        unknown = [i for i in req.recent_item_ids if i not in self.item_idx]
        recent = [self.item_idx[i] for i in req.recent_item_ids if i in self.item_idx and self.item_idx[i] < b.n_items]
        base = [int(x) - 1 for x in b.hist_np[uidx] if x > 0] if uidx is not None else []
        hist_items = list(dict.fromkeys(base + recent))[-b.hist.shape[1]:]
        # new (registered) items beyond the training catalogue have no history slot -> ignored for pooling
        with self.lock:
            t1 = time.perf_counter()
            prof = {"acquisition_channel": None, "loyalty_tier": None, "home_region": None, "home_store": None, "device_os": None, "quiz_tags": []}
            if uidx is not None:
                r = b.users.iloc[uidx]
                prof.update({k: r[k] for k in prof if k != "quiz_tags"}); prof["quiz_tags"] = r.quiz_tags.split("|")
            if req.profile is not None:
                prof.update({k: v for k, v in req.profile.model_dump().items() if v not in (None, [])})
            ctx = req.context.model_dump() if req.context else {}
            hour = ctx.get("hour") if ctx.get("hour") is not None else time.localtime().tm_hour
            device = ctx.get("device") or prof["device_os"] or "web"
            store = ctx.get("store_id") or prof["home_store"] or "UNKNOWN"
            channel = ctx.get("channel") or "web"
            udf = pd.DataFrame([{**{k: (v if v is not None else "UNKNOWN") for k, v in prof.items() if k != "quiz_tags"},
                                 "quiz_tags": "|".join(prof["quiz_tags"])}])
            ut = b.enc.transform_users(udf)
            dev = self.device
            ucat = torch.as_tensor(ut["cat"], dtype=torch.long, device=dev)
            quiz = torch.as_tensor(ut["quiz"], device=dev)
            c = b.enc.transform_context([hour], [device], [store], [channel])
            ctx_cat = torch.as_tensor(c["cat"], dtype=torch.long, device=dev)
            tod = torch.as_tensor(c["tod"], device=dev)
            hist = torch.zeros(1, b.hist.shape[1], dtype=torch.long, device=dev)
            if hist_items:
                hist[0, -len(hist_items):] = torch.as_tensor(hist_items, device=dev) + 1
            if uidx is not None:
                zg = hub.Zu_g[uidx:uidx + 1]
            else:
                zg = hub.gm.embed_new_user(hub.g_states, ucat, prof["home_store"], prof["acquisition_channel"], prof["quiz_tags"])
            n = (hist > 0).sum(1)
            raw = hub.raw_queries(ucat, quiz, ctx_cat, tod, hist, zg)
            q, route = hub.unified_query(raw, n)
            t2 = time.perf_counter()
            D, I = self.fm.search(q.cpu().numpy(), req.k, "hnsw", allowed=self._allowed(req.filters), exclude=set(hist_items))
            t3 = time.perf_counter()
        items = [self._item_out(i, d) for d, i in zip(D[0], I[0])]
        n_h = len(hist_items)
        given = [x for x in (ctx.get("store_id"), prof["home_store"]) if x]
        store_unknown = any(x not in self.known_stores for x in given)
        grp = b.users.user_group.iloc[uidx] if uidx is not None else None
        if grp == "system_cold" or (uidx is None and store_unknown):
            cst = "system_cold"
        elif n_h == 0:
            cst = "user_cold_zero_shot"
        elif n_h <= 3:
            cst = "user_cold_few_shot"
        else:
            cst = "warm"
        resp = RecommendResponse(
            user_id=req.user_id, strategy=STRATEGIES[int(route[0])], cold_start_type=cst, n_history_items=n_h, unknown_item_ids=unknown,
            n_cold_items_in_results=sum(i.is_cold_item for i in items), items=items,
            telemetry=Telemetry(total_ms=(time.perf_counter() - t0) * 1000, embed_ms=(t2 - t1) * 1000, ann_ms=(t3 - t2) * 1000,
                                cache_hit=False, cache_backend=self.cache.backend, index="IndexHNSWFlat"))
        if req.use_cache:
            self.cache.set(key, json.loads(resp.model_dump_json()))
        self._record(resp)
        return resp

    def _record(self, resp):
        self.lat.append(resp.telemetry.total_ms)
        self.counts["requests"] += 1
        self.counts[f"strategy:{resp.strategy}"] += 1
        self.counts[f"coldstart:{resp.cold_start_type}"] += 1
        self.counts["cache_hits"] += int(resp.telemetry.cache_hit)

    # ------------------------------------------------------------------ zero-shot item onboarding
    def add_item(self, req: NewItemRequest) -> NewItemResponse:
        t0 = time.perf_counter()
        b, hub = self.bundle, self.hub
        vd = b.vis_dim
        if req.visual_embedding is not None and len(req.visual_embedding) != vd:
            raise HTTPException(422, f"visual_embedding must have {vd} dims")
        df = pd.DataFrame([{**req.model_dump(exclude={"visual_embedding", "register", "margin"}), "margin": req.margin,
                            "launch_ts": 1735689600 + 180 * 86400}])
        vis = np.asarray(req.visual_embedding, np.float32)[None] if req.visual_embedding is not None else np.zeros((1, vd), np.float32)
        t = b.enc.transform_items(df, vis=vis)
        dev = self.device
        cat = torch.as_tensor(t["cat"], dtype=torch.long, device=dev)
        cont, text, v = (torch.as_tensor(t[k], device=dev) for k in ("cont", "text", "vis"))
        mv = torch.tensor([req.visual_embedding is None], device=dev)
        with self.lock:
            e_tt, e_ssl = hub.embed_new_items(text, v, cont, cat, None, mv)
            g = hub.graph
            used_graph = req.brand in g.bi and req.l3 in g.ci
            w = hub.w["item"]
            e = w["c"] * e_tt + w["s"] * (e_ssl @ hub.R_s)
            if used_graph:
                zg = hub.gm.embed_new_item(hub.g_states, text, v, cont, req.brand, req.l3, req.l2, req.price_tier)
                e = e + w["g"] * (zg @ hub.R_g)
            e = l2n(e)
            vec = e.cpu().numpy().astype(np.float32)
            D, I = self.fm.search(vec, 5, "hnsw")
            self.n_new += 1
            iid = f"NEW{self.n_new:04d}"
            if req.register:
                self.fm.flat.add(vec); self.fm.hnsw.add(vec); self.fm.ivfpq.add(vec)
                row = {c: None for c in self.items.columns}
                row.update({"item_id": iid, "title": req.title, "description": req.description, "l1": req.l1, "l2": req.l2, "l3": req.l3,
                            "brand": req.brand, "price_tier": req.price_tier, "price": req.price, "margin": req.margin,
                            "color": req.color, "material": req.material, "is_cold_item": True})
                self.items = pd.concat([self.items, pd.DataFrame([row])], ignore_index=True)
                self.item_ids.append(iid); self.item_idx[iid] = len(self.item_ids) - 1
                self.cold.append(True); self.l1.append(req.l1); self.tier.append(req.price_tier)
                self.cache.mem.clear()
                if self.cache.client is not None:
                    try:
                        for k in self.cache.client.scan_iter("rec:*"):
                            self.cache.client.delete(k)
                    except Exception:
                        pass
        return NewItemResponse(item_id=iid, registered=req.register, embedding_dim=int(vec.shape[1]), used_visual=req.visual_embedding is not None,
                               used_graph_neighbours=used_graph, nearest_items=[self._item_out(i, d) for d, i in zip(D[0], I[0])],
                               latency_ms=(time.perf_counter() - t0) * 1000)

    def metrics(self) -> dict:
        lat = np.array(self.lat) if self.lat else np.array([0.0])
        return {"requests": int(self.counts["requests"]), "cache_hits": int(self.counts["cache_hits"]),
                "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99))},
                "by_strategy": {k.split(":", 1)[1]: v for k, v in self.counts.items() if k.startswith("strategy:")},
                "by_cold_start_type": {k.split(":", 1)[1]: v for k, v in self.counts.items() if k.startswith("coldstart:")},
                "index_size": int(self.fm.flat.ntotal), "cache_backend": self.cache.backend}


def create_app(root: Path = ROOT, cfg: dict | None = None, engine: Engine | None = None) -> FastAPI:
    cfg = cfg or yaml.safe_load(open(root / "config" / "config.yaml"))
    state = {"engine": engine}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["engine"] is None:
            state["engine"] = Engine(root, cfg)
        yield

    app = FastAPI(title="Retail Cold-Start Representation-Learning Recommender", version="1.0.0", lifespan=lifespan,
                  description="Unified d=128 representation space across Two-Tower, inductive GraphSAGE, MAML adapter and contrastive SSL, "
                              "routed by the amount of user signal available.")
    eng = lambda: state["engine"]

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health():
        e = eng()
        return HealthResponse(status="ok" if e.cache.backend != "disabled" else "degraded", n_items=len(e.item_ids), n_users=e.bundle.n_users,
                              embedding_dim=e.fm.dim, cache_backend=e.cache.backend, device=str(e.device), index_size=int(e.fm.flat.ntotal))

    @app.post("/recommend", response_model=RecommendResponse, tags=["recommend"], summary="Personalised top-k for warm or cold-start users")
    def recommend(req: RecommendRequest):
        return eng().recommend(req)

    @app.post("/items/embed", response_model=NewItemResponse, tags=["catalog"], summary="Zero-shot onboarding of a brand-new item")
    def embed_item(req: NewItemRequest):
        return eng().add_item(req)

    @app.get("/metrics", tags=["ops"])
    def metrics():
        return eng().metrics()

    @app.get("/strategies", tags=["ops"])
    def strategies():
        return {"routes": [{"history_items": "0", "strategy": STRATEGIES[0]}, {"history_items": "1-3", "strategy": STRATEGIES[1]},
                           {"history_items": ">3", "strategy": STRATEGIES[2]}]}

    return app


app = create_app()


# ---------------------------------------------------------------------- API benchmark (used by run_pipeline)
def benchmark_api(root: Path, cfg: dict, n_requests: int = 300) -> dict:
    from fastapi.testclient import TestClient
    rng = np.random.default_rng(0)
    eng = Engine(root, cfg, cache=Cache(None, 300, True))
    b = eng.bundle
    users = b.users
    warm = users.user_id[users.user_group == "warm"].tolist()
    zero = users.user_id[users.user_group == "zero_shot"].tolist()
    few = users.user_id[users.user_group == "few_shot"].tolist()
    newitems = b.items.item_id[b.items.is_cold_item].tolist()
    scen = {"warm": lambda: {"user_id": str(rng.choice(warm)), "k": 10},
            "zero_shot_user": lambda: {"user_id": str(rng.choice(zero)), "k": 10},
            "few_shot_user": lambda: {"user_id": str(rng.choice(few)), "k": 10},
            "anonymous_session": lambda: {"recent_item_ids": [str(x) for x in rng.choice(b.items.item_id.values, 2)], "k": 10,
                                          "profile": {"acquisition_channel": "social"}}}
    out = {"n_requests": n_requests, "scenarios": {}}
    with TestClient(create_app(root, cfg, eng)) as c:
        for use_cache in (False, True):
            for name, fn in scen.items():
                lats = []
                payloads = [fn() for _ in range(n_requests // 4)]
                for p in payloads * (2 if use_cache else 1):
                    t = time.perf_counter()
                    r = c.post("/recommend", json={**p, "use_cache": use_cache})
                    lats.append((time.perf_counter() - t) * 1000)
                    assert r.status_code == 200, r.text
                out["scenarios"][f"{name}|cache={use_cache}"] = {"p50_ms": float(np.percentile(lats, 50)), "p95_ms": float(np.percentile(lats, 95)),
                                                                 "p99_ms": float(np.percentile(lats, 99)), "n": len(lats)}
        out["cache_backend"] = eng.cache.backend
        out["server_side"] = c.get("/metrics").json()
    allp = [v["p95_ms"] for k, v in out["scenarios"].items() if k.endswith("False")]
    out["max_uncached_p95_ms"] = float(max(allp))
    return out
