"""Unified representation space + cold-start router.

All four learners output L2-normalised 128-d vectors, but each lives in its own rotation of R^128.
We align them into ONE space (the two-tower content space is the reference) with orthogonal
Procrustes fitted on WARM items only (so zero-shot items never influence the alignment), then:

  item vector   e_i = norm( a_c*E_tt[i] + a_g*R_g(Z_g[i]) + a_s*R_s(E_ssl[i]) )           -> one FAISS index
  user query by interaction count n:
     n = 0   : norm( c_c*tt(profile,ctx) + c_g*R_g(graph_user) [+ c_s*R_s(quiz-tag catalogue prototype)] )  "content+graph"
     1..3    : norm( m_m*MAML_adapted(prior, support) + m_g*R_g(graph_user) [+ m_s*R_s(ssl_query)] )   "meta_adapter+graph"
     n > 3   : norm( f_c*tt(profile,ctx,history) + f_s*R_s(ssl_query) + f_g*R_g(graph_user) )
                                                                              "multimodal+contrastive"
Mixing weights are tuned on VALIDATION slices only (small grid); test is never touched.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes

from src.evaluation.offline_metrics import rank_metrics
from src.models.common import l2n
from src.models.contrastive_ssl import quiz_item_matrix, ssl_user_query
from src.models.multimodal_two_tower import MultimodalTwoTower

STRATEGIES = ["content+graph", "meta_adapter+graph", "multimodal+contrastive_ensemble"]


def route_of(nhist: torch.Tensor) -> torch.Tensor:
    return torch.where(nhist == 0, 0, torch.where(nhist <= 3, 1, 2))


class RepresentationHub:
    def __init__(self, bundle, tt, gmodel, graph, adapter, ssl, weights: dict | None = None):
        self.b, self.tt, self.gm, self.graph, self.adapter, self.ssl = bundle, tt, gmodel, graph, adapter, ssl
        ft = bundle.ft
        with torch.no_grad():
            self.E_tt = tt.embed_items(ft)
            self.Zu_g, self.Zi_g, self.g_states = gmodel.all_embeddings(ft)
            self.E_ssl = ssl.all_item_embeddings(ft)
            self.Q = quiz_item_matrix(bundle.items, bundle.enc.quiz_vocab, bundle.device)
        warm = np.where(~bundle.feats.item_cold)[0]
        ref = self.E_tt[warm].cpu().numpy()
        self.R_g = torch.as_tensor(orthogonal_procrustes(self.Zi_g[warm].cpu().numpy(), ref)[0], dtype=torch.float32, device=bundle.device)
        self.R_s = torch.as_tensor(orthogonal_procrustes(self.E_ssl[warm].cpu().numpy(), ref)[0], dtype=torch.float32, device=bundle.device)
        f = bundle.cfg["fusion"]
        self.w = weights or {"item": {"c": 1.0, "g": f["weights_graph"], "s": 0.3},
                             "r0": {"c": 1.0, "g": f["weights_graph"]},
                             "r1": {"m": 1.0, "g": f["weights_graph"]},
                             "r2": {"c": 1.0, "s": 0.3, "g": 0.3}}
        self.refresh_items()

    # ------------------------------------------------------------------ item side
    def refresh_items(self):
        w = self.w["item"]
        self.E_uni = l2n(w["c"] * self.E_tt + w["g"] * (self.Zi_g @ self.R_g) + w["s"] * (self.E_ssl @ self.R_s))

    def embed_new_items(self, text, vis, cont, cat, tmask=None, vmask=None):
        """Zero-shot items from content only: returns (unified, tt, ssl) rows. Graph part is added by the caller."""
        with torch.no_grad():
            e_tt = self.tt.item_tower(cat, cont, text, vis, tmask, vmask)
            e_ssl = self.ssl.encoder(cat, cont, text, vis, tmask, vmask)
        return e_tt, e_ssl

    # ------------------------------------------------------------------ user side (works for known AND brand-new users)
    @torch.no_grad()
    def raw_queries(self, ucat, quiz, ctx_cat, tod, hist, zg_user, which=("tt", "graph", "maml", "ssl")):
        out = {}
        n = (hist > 0).sum(1)
        if "tt" in which or "maml" in which:
            zeros = torch.zeros(len(ucat), self.tt.d, device=ucat.device)
            prior = self.tt.user_tower(ucat, quiz, ctx_cat, tod, zeros, zeros[:, 0])
            if "tt" in which:
                pooled, has = MultimodalTwoTower.pool_history(self.E_tt, hist)
                out["tt"] = self.tt.user_tower(ucat, quiz, ctx_cat, tod, pooled, has)
            if "maml" in which:
                sup = (hist[:, -3:] - 1).clamp(min=0)
                sm = (hist[:, -3:] > 0).float()
                ad = self.adapter.adapt(prior, sup, sm, self.E_tt)
                out["maml"] = torch.where((n > 0)[:, None], ad, prior)
        if "graph" in which:
            out["graph"] = zg_user
        if "ssl" in which:
            out["ssl"] = ssl_user_query(self.E_ssl, hist, quiz, self.Q)
        return out

    def unified_query(self, raw: dict, n: torch.Tensor, w: dict | None = None):
        w = w or self.w
        g = raw["graph"] @ self.R_g
        s = raw["ssl"] @ self.R_s if "ssl" in raw else None
        q0 = l2n(w["r0"]["c"] * raw["tt"] + w["r0"]["g"] * g + w["r0"].get("s", 0.0) * s)
        q1 = l2n(w["r1"]["m"] * raw["maml"] + w["r1"]["g"] * g + w["r1"].get("s", 0.0) * s)
        q2 = l2n(w["r2"]["c"] * raw["tt"] + w["r2"]["s"] * s + w["r2"]["g"] * g)
        r = route_of(n)
        return torch.where((r == 0)[:, None], q0, torch.where((r == 1)[:, None], q1, q2)), r

    # ------------------------------------------------------------------ evaluation adapters (indexed users)
    def raw_for_slice(self, sd):
        ft = self.b.ft
        return self.raw_queries(ft.user_cat[sd.u], ft.user_quiz[sd.u], sd.ctx_cat, sd.tod, sd.hist, self.Zu_g[sd.u])

    def method_table(self):
        """name -> (query_fn(SliceData), item_matrix)"""
        cache = {}

        def raw(sd):
            if id(sd) not in cache:
                cache[id(sd)] = self.raw_for_slice(sd)
            return cache[id(sd)]
        return {
            "two_tower_multimodal": (lambda sd: raw(sd)["tt"], self.E_tt),
            "inductive_graphsage": (lambda sd: raw(sd)["graph"], self.Zi_g),
            "maml_user_adapter": (lambda sd: raw(sd)["maml"], self.E_tt),
            "contrastive_ssl": (lambda sd: raw(sd)["ssl"], self.E_ssl),
            "unified_ensemble": (lambda sd: self.unified_query(raw(sd), sd.nhist)[0], self.E_uni),
        }

    # ------------------------------------------------------------------ validation-only tuning
    def tune(self, val_sets: dict, cold_mask: torch.Tensor, log=None):
        raws = {sl: self.raw_for_slice(sd) for sl, sd in val_sets.items()}

        def recall(w, route=None, item_only=False):
            tot, cnt = 0.0, 0
            E = self.E_uni
            for sl, sd in val_sets.items():
                q, r = self.unified_query(raws[sl], sd.nhist, w)
                m = torch.ones(len(sd), dtype=torch.bool, device=q.device) if route is None else (r == route)
                if m.sum() == 0:
                    continue
                per, _ = rank_metrics(q[m] @ E.T, sd.subset(m), cold_mask)
                tot += float(per["recall@20"].sum()) + float(per["ndcg@10"].sum())
                cnt += int(m.sum())
            return tot / max(cnt, 1)

        # 1) item-side mixing (macro over all val users)
        best = (-1, None)
        for ag, as_ in itertools.product([0.0, 0.25, 0.5, 0.75], [0.0, 0.25, 0.5, 1.0, 1.5]):
            self.w["item"] = {"c": 1.0, "g": ag, "s": as_}
            self.refresh_items()
            sc = recall(self.w)
            if sc > best[0]:
                best = (sc, dict(self.w["item"]))
        self.w["item"] = best[1]
        self.refresh_items()
        # 2) per-route weights
        grids = {0: [("r0", {"c": 1.0, "g": g, "s": s_}) for g, s_ in itertools.product((0.0, 0.25, 0.5, 1.0), (0.0, 0.5, 1.0, 2.0))],
                 1: [("r1", {"m": 1.0, "g": g, "s": s_}) for g, s_ in itertools.product((0.0, 0.25, 0.5, 1.0), (0.0, 0.5, 1.0, 2.0))],
                 2: [("r2", {"c": 1.0, "s": s, "g": g}) for s, g in itertools.product((0.0, 0.25, 0.5, 1.0, 2.0), (0.0, 0.25, 0.5, 1.0))]}
        for route, cands in grids.items():
            bs, bw = -1, None
            for key, cw in cands:
                trial = {**self.w, key: cw}
                sc = recall(trial, route=route)
                if sc > bs:
                    bs, bw = sc, (key, cw)
            self.w[bw[0]] = bw[1]
            if log:
                log.log("fusion", "tuned_route", route=route, weights=bw[1], val_score=round(bs, 4))
        if log:
            log.log("fusion", "tuned_items", weights=self.w["item"])
        return self.w


# ---------------------------------------------------------------------- checkpoints
def save_all(out_dir, bundle, tt, gm, adapter, ssl, weights: dict, aggregator: str = "mean"):
    import json
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cpu = lambda m: {k: v.detach().cpu() for k, v in m.state_dict().items()}
    cfg = bundle.cfg
    torch.save({"state": cpu(tt), "temperature": float(tt.tau)}, out / "two_tower.pt")
    torch.save({"state": cpu(gm), "aggregator": aggregator}, out / "inductive_graphsage.pt")
    torch.save({"state": cpu(adapter), "tau": adapter.tau}, out / "maml_user_adapter.pt")
    torch.save({"state": cpu(ssl)}, out / "contrastive_ssl.pt")
    json.dump(weights, open(out / "fusion_weights.json", "w"), indent=2)
    bundle.enc.save(out / "preprocessors.pkl")


def load_all(bundle, out_dir):
    import json
    from pathlib import Path
    from src.models.contrastive_ssl import SSLModel
    from src.models.inductive_graphsage import HeteroGraph, HeteroGraphSAGE
    from src.models.maml_user_adapter import MAMLUserAdapter
    out, dev, cfg = Path(out_dir), bundle.device, bundle.cfg
    d = cfg["embedding_dim"]
    ck = torch.load(out / "two_tower.pt", map_location=dev)
    tt = MultimodalTwoTower(bundle.cat_sizes, bundle.text_dim, bundle.vis_dim, bundle.n_quiz, d, ck["temperature"]).to(dev)
    tt.load_state_dict(ck["state"]); tt.eval()
    graph = HeteroGraph(bundle)
    ck = torch.load(out / "inductive_graphsage.pt", map_location=dev)
    gc = cfg["train"]["graphsage"]
    gm = HeteroGraphSAGE(graph, d, gc["hidden"], gc["layers"], ck["aggregator"]).to(dev)
    gm.load_state_dict(ck["state"]); gm.eval()
    ck = torch.load(out / "maml_user_adapter.pt", map_location=dev)
    mc = cfg["train"]["maml"]
    adapter = MAMLUserAdapter(d, mc["inner_lr"], mc["inner_steps"], ck["tau"]).to(dev)
    adapter.load_state_dict(ck["state"]); adapter.eval()
    ck = torch.load(out / "contrastive_ssl.pt", map_location=dev)
    ssl = SSLModel(bundle.cat_sizes, bundle.text_dim, bundle.vis_dim, d).to(dev)
    ssl.load_state_dict(ck["state"]); ssl.eval()
    weights = json.load(open(out / "fusion_weights.json"))
    return tt, gm, graph, adapter, ssl, weights
