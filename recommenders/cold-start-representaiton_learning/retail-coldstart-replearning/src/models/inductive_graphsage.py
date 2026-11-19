"""Approach 2: Inductive heterogeneous GraphSAGE (pure PyTorch, no PyG).

Multipartite graph: User, Item, Brand, Category(L1/L2/L3), Store, PriceTier, Channel, QuizTag.
Users and Items have NO ID embeddings: their initial state is a projection of side-information, and
their final state comes from relation-specific neighbourhood aggregation (mean or attention).
Hence a brand-new item attached to (Brand, Category, PriceTier) or a brand-new user attached to
(Store, Channel, QuizTags) gets an embedding without retraining (`embed_new_item/user`).

Training mimics inductive conditions: random user-item edge dropout, supervision edges removed from
message passing, and 8% of items 'cold-simulated' (all interaction edges hidden) every step.
Cold items / zero-shot users never send messages (no information leak from withheld entities).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import info_nce, l2n
from src.utils.hardware import autocast_ctx, make_scaler

T_USER, T_ITEM, T_BRAND, T_CAT, T_STORE, T_TIER, T_CHAN, T_QUIZ = range(8)
# relation id -> (src_type, dst_type)
REL = {
    0: "user->item", 1: "item->user", 2: "brand->item", 3: "item->brand", 4: "cat->item", 5: "item->cat",
    6: "tier->item", 7: "item->tier", 8: "store->user", 9: "user->store", 10: "channel->user", 11: "user->channel",
    12: "quiz->user", 13: "user->quiz", 14: "cat->cat(parent->child)", 15: "cat->cat(child->parent)",
}
N_REL = len(REL)
ITEM_IN_RELS = [2, 4, 6]            # relations that feed a (new) item node
USER_IN_RELS = [8, 10, 12]          # relations that feed a (new) user node


class HeteroGraph:
    """Index bookkeeping + static (non-interaction) edges."""

    def __init__(self, bundle):
        b = bundle
        self.b = b
        nU, nI = b.n_users, b.n_items
        self.brands = sorted(b.items.brand.unique())
        self.cats = sorted(set(b.items.l1) | set(b.items.l2) | set(b.items.l3))
        self.stores = b.stores.store_id.tolist()
        self.tiers = [1, 2, 3, 4]
        self.channels = sorted(b.users.acquisition_channel.unique())
        self.quiz = list(b.enc.quiz_vocab)
        sizes = [nU, nI, len(self.brands), len(self.cats), len(self.stores), len(self.tiers), len(self.channels), len(self.quiz)]
        self.sizes = sizes
        self.offset = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        self.N = int(self.offset[-1])
        self.type_of = torch.cat([torch.full((s,), t, dtype=torch.long) for t, s in enumerate(sizes)]).to(b.device)
        bi = {x: k for k, x in enumerate(self.brands)}
        ci = {x: k for k, x in enumerate(self.cats)}
        si = {x: k for k, x in enumerate(self.stores)}
        chi = {x: k for k, x in enumerate(self.channels)}
        qi = {x: k for k, x in enumerate(self.quiz)}
        self.bi, self.ci, self.si, self.chi, self.qi = bi, ci, si, chi, qi
        o = self.offset
        cold_item = b.feats.item_cold
        group = b.feats.user_group
        train_users = np.zeros(nU, bool)
        train_users[b.tr_df.user_id.map(b.user_index).values] = True
        self.sender_user_ok = train_users | (group == "warm")      # zero-shot / system-cold users never send messages
        edges = []  # (src, dst, rel)

        def add(src, dst, r_fwd, r_bwd, src_ok=None, dst_ok=None):
            src, dst = np.asarray(src), np.asarray(dst)
            ok_f = np.ones(len(src), bool) if src_ok is None else src_ok
            ok_b = np.ones(len(src), bool) if dst_ok is None else dst_ok
            edges.append((src[ok_f], dst[ok_f], np.full(ok_f.sum(), r_fwd)))
            edges.append((dst[ok_b], src[ok_b], np.full(ok_b.sum(), r_bwd)))

        it = b.items
        iid = np.arange(nI)
        add(o[T_BRAND] + it.brand.map(bi).values, o[T_ITEM] + iid, 2, 3, None, ~cold_item)
        add(o[T_CAT] + it.l3.map(ci).values, o[T_ITEM] + iid, 4, 5, None, ~cold_item)
        add(o[T_TIER] + (it.price_tier.values - 1), o[T_ITEM] + iid, 6, 7, None, ~cold_item)
        us = b.users
        uid = np.arange(nU)
        ok_u = self.sender_user_ok
        add(o[T_STORE] + us.home_store.map(si).values, o[T_USER] + uid, 8, 9, None, ok_u)
        add(o[T_CHAN] + us.acquisition_channel.map(chi).values, o[T_USER] + uid, 10, 11, None, ok_u)
        qs, qd = [], []
        for u, tags in enumerate(us.quiz_tags):
            for t in tags.split("|"):
                if t in qi:
                    qs.append(o[T_QUIZ] + qi[t]); qd.append(o[T_USER] + u)
        qs, qd = np.array(qs), np.array(qd)
        add(qs, qd, 12, 13, None, ok_u[qd - o[T_USER]])
        l2p = it.drop_duplicates("l3")[["l1", "l2", "l3"]]
        h_s = np.concatenate([o[T_CAT] + l2p.l1.map(ci).values, o[T_CAT] + l2p.l2.map(ci).values])
        h_d = np.concatenate([o[T_CAT] + l2p.l2.map(ci).values, o[T_CAT] + l2p.l3.map(ci).values])
        add(h_s, h_d, 14, 15)
        s, d, r = (np.concatenate(x) for x in zip(*edges))
        dev = b.device
        self.static = (torch.as_tensor(s, device=dev), torch.as_tensor(d, device=dev), torch.as_tensor(r, device=dev))
        # train user-item edges (deduplicated per pair, keep multiplicity out)
        tr = b.tr_df.drop_duplicates(["user_id", "item_id"])
        self.ui_u = torch.as_tensor(tr.user_id.map(b.user_index).values, device=dev)
        self.ui_i = torch.as_tensor(tr.item_id.map(b.item_index).values, device=dev)
        self.n_ui = len(self.ui_u)
        # inputs
        self.store_feat = torch.as_tensor(np.stack([
            np.eye(5)[[["north", "south", "east", "west", "central"].index(r) for r in b.stores.region]].T[k] for k in range(5)] +
            [b.stores.size_sqm.values / 3000.0, b.stores.is_urban.values.astype(float)], 1), dtype=torch.float32, device=dev)
        trained_stores = set(b.tr_df.store_id) | set(b.users.home_store[train_users])
        self.store_row = torch.as_tensor([self.si[s] + 1 if s in trained_stores else 0 for s in self.stores], device=dev)
        self.user_prof_cols = [1, 4, 2]   # loyalty, os, region columns in user_cat

    def ui_edges(self, keep: torch.Tensor):
        o = self.offset
        u, i = self.ui_u[keep] + o[T_USER], self.ui_i[keep] + o[T_ITEM]
        r0 = torch.zeros_like(u); r1 = torch.ones_like(u)
        return torch.cat([i, u]), torch.cat([u, i]), torch.cat([r0, r1])


class SAGELayer(nn.Module):
    def __init__(self, h: int, aggregator: str = "mean"):
        super().__init__()
        self.h, self.agg = h, aggregator
        self.W_rel = nn.Linear(h, N_REL * h, bias=False)
        self.W_self = nn.ModuleList([nn.Linear(h, h) for _ in range(8)])
        self.norm = nn.LayerNorm(h)
        if aggregator == "attention":
            self.q = nn.Linear(h, N_REL * 16, bias=False)
            self.k = nn.Linear(h, N_REL * 16, bias=False)

    def forward(self, h, graph: HeteroGraph, edges):
        s, d, r = edges
        N, H = h.shape
        proj = self.W_rel(h).view(N, N_REL, H)
        msg = proj[s, r]                                      # (E,H)
        key = d * N_REL + r
        if self.agg == "attention":
            qd = self.q(h).view(N, N_REL, 16)[d, r]
            ks = self.k(h).view(N, N_REL, 16)[s, r]
            sc = (qd * ks).sum(-1) / 4.0
            mx = torch.full((N * N_REL,), -1e9, device=h.device, dtype=sc.dtype).scatter_reduce(0, key, sc, "amax", include_self=True)
            ex = (sc - mx[key]).exp()
            den = torch.zeros(N * N_REL, device=h.device, dtype=sc.dtype).index_add_(0, key, ex)
            w = (ex / den[key].clamp(min=1e-9))[:, None]
        else:
            den = torch.zeros(N * N_REL, device=h.device, dtype=msg.dtype).index_add_(0, key, torch.ones_like(key, dtype=msg.dtype))
            w = (1.0 / den[key].clamp(min=1.0))[:, None]
        out = torch.zeros(N, H, device=h.device, dtype=msg.dtype).index_add_(0, d, (msg * w).to(msg.dtype))
        self_part = torch.cat([self.W_self[t](h[graph.offset[t]:graph.offset[t + 1]]) for t in range(8)], 0)
        return self.norm(F.relu(self_part + out)), proj

    def new_node(self, t: int, h_self, nbr_states: dict):
        """Aggregate for ONE brand-new node. nbr_states: {rel: (n,H) previous-layer states of its neighbours}."""
        out = torch.zeros_like(h_self)
        for rel, st in nbr_states.items():
            if len(st) == 0:
                continue
            if self.agg == "attention":
                W = self.W_rel.weight.view(N_REL, self.h, self.h)[rel]
                msgs = st @ W.T
                qd = (h_self @ self.q.weight.view(N_REL, 16, self.h)[rel].T)
                ks = st @ self.k.weight.view(N_REL, 16, self.h)[rel].T
                w = torch.softmax((qd * ks).sum(-1) / 4.0, 0)[:, None]
                out = out + (w * msgs).sum(0, keepdim=True)
            else:
                W = self.W_rel.weight.view(N_REL, self.h, self.h)[rel]
                out = out + (st @ W.T).mean(0, keepdim=True)
        return self.norm(F.relu(self.W_self[t](h_self) + out))


class HeteroGraphSAGE(nn.Module):
    def __init__(self, graph: HeteroGraph, d=128, hidden=128, layers=2, aggregator="mean", emb=16):
        super().__init__()
        b = graph.b
        self.g, self.hidden, self.d = graph, hidden, d
        cs = b.cat_sizes
        self.user_embs = nn.ModuleList([nn.Embedding(cs[c], emb) for c in ["loyalty_tier", "device_os", "home_region"]])
        self.user_in = nn.Linear(emb * 3, hidden)
        self.item_in = nn.Linear(b.text_dim + b.vis_dim + 3, hidden)
        self.id_embs = nn.ModuleDict({str(t): nn.Embedding(graph.sizes[t], hidden) for t in (T_BRAND, T_CAT, T_TIER, T_CHAN, T_QUIZ)})
        self.store_emb = nn.Embedding(len(graph.stores) + 1, hidden)
        self.store_in = nn.Linear(7, hidden)
        self.layers = nn.ModuleList([SAGELayer(hidden, aggregator) for _ in range(layers)])
        self.out = nn.Linear(hidden, d)
        self.store_drop = 0.0

    # --------------------------------------------------------------- inputs
    def user_input(self, ucat):
        e = torch.cat([emb(ucat[:, c]) for emb, c in zip(self.user_embs, self.g.user_prof_cols)], -1)
        return F.relu(self.user_in(e))

    def item_input(self, text, vis, cont):
        return F.relu(self.item_in(torch.cat([text, vis, cont], -1)))

    def initial(self, ft):
        g = self.g
        rows = self.g.store_row
        if self.training and self.store_drop > 0:
            rows = torch.where(torch.rand_like(rows, dtype=torch.float) < self.store_drop, torch.zeros_like(rows), rows)
        h = [self.user_input(ft.user_cat), self.item_input(ft.item_text, ft.item_vis, ft.item_cont)]
        for t in (T_BRAND, T_CAT):
            h.append(self.id_embs[str(t)].weight)
        h.append(F.relu(self.store_in(g.store_feat)) + self.store_emb(rows))
        for t in (T_TIER, T_CHAN, T_QUIZ):
            h.append(self.id_embs[str(t)].weight)
        return torch.cat(h, 0)

    def forward(self, ft, edges):
        h = self.initial(ft)
        states = [h]
        for layer in self.layers:
            h, _ = layer(h, self.g, edges)
            states.append(h)
        return h, states

    def project(self, h):
        return l2n(self.out(h))

    # --------------------------------------------------------------- inductive inference for NEW nodes
    @torch.no_grad()
    def embed_new_item(self, states, text, vis, cont, brand: str, l3: str, l2: str | None, tier: int):
        """text/vis/cont: (1,*) tensors. Aggregates ONLY from attribute neighbours (no retraining)."""
        g, o = self.g, self.g.offset
        h = self.item_input(text, vis, cont)
        nbr = {2: [o[T_BRAND] + g.bi[brand]], 4: [o[T_CAT] + g.ci[l3]], 6: [o[T_TIER] + tier - 1]}
        return self._new_node(T_ITEM, h, nbr, states)

    @torch.no_grad()
    def embed_new_user(self, states, ucat, home_store: str | None, channel: str | None, quiz_tags: list[str]):
        g, o = self.g, self.g.offset
        h = self.user_input(ucat)
        nbr = {8: [o[T_STORE] + g.si[home_store]] if home_store in g.si else [],
               10: [o[T_CHAN] + g.chi[channel]] if channel in g.chi else [],
               12: [o[T_QUIZ] + g.qi[t] for t in quiz_tags if t in g.qi]}
        return self._new_node(T_USER, h, nbr, states)

    def _new_node(self, t, h0, nbr, states):
        h = h0
        for l, layer in enumerate(self.layers):
            st = {r: states[l][torch.as_tensor(ix, dtype=torch.long, device=h.device)] if len(ix) else torch.zeros(0, self.hidden, device=h.device)
                  for r, ix in nbr.items()}
            h = layer.new_node(t, h, st)
        return self.project(h)

    # --------------------------------------------------------------- full inference
    @torch.no_grad()
    def all_embeddings(self, ft):
        self.eval()
        g = self.g
        keep = torch.ones(g.n_ui, dtype=torch.bool, device=g.ui_u.device)
        edges = tuple(torch.cat([a, b]) for a, b in zip(g.static, g.ui_edges(keep)))
        h, states = self.forward(ft, edges)
        z = self.project(h)
        o = g.offset
        return z[o[T_USER]:o[T_USER + 1]], z[o[T_ITEM]:o[T_ITEM + 1]], states


def train_graphsage(bundle, epochs: int, log=None, aggregator: str = "mean"):
    cfg = bundle.cfg["train"]["graphsage"]
    dev = bundle.device
    graph = HeteroGraph(bundle)
    model = HeteroGraphSAGE(graph, bundle.cfg["embedding_dim"], cfg["hidden"], cfg["layers"], aggregator).to(dev)
    model.store_drop = 0.15
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scaler = make_scaler(dev, bundle.cfg["hardware"]["amp"])
    ft, va = bundle.ft, bundle.va
    log_q = bundle.item_logq
    tau = 0.1
    nI = bundle.n_items
    o = graph.offset
    warm_items = torch.as_tensor(np.where(~bundle.feats.item_cold)[0], device=dev)
    history, best, best_state = [], 1e9, None
    bs = cfg["batch_size"]
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(graph.n_ui, device=dev)
        tot, nb = 0.0, 0
        for s in range(0, graph.n_ui, bs):
            e_b = perm[s:s + bs]
            if len(e_b) < 16:
                continue
            keep = torch.rand(graph.n_ui, device=dev) > 0.4
            keep[e_b] = False
            cold_sim = torch.zeros(nI, dtype=torch.bool, device=dev)
            cold_sim[warm_items[torch.randperm(len(warm_items), device=dev)[: int(0.08 * len(warm_items))]]] = True
            keep &= ~cold_sim[graph.ui_i]
            edges = tuple(torch.cat([a, b]) for a, b in zip(graph.static, graph.ui_edges(keep)))
            u, i = graph.ui_u[e_b], graph.ui_i[e_b]
            with autocast_ctx(dev, bundle.cfg["hardware"]["amp"]):
                h, _ = model(ft, edges)
                zu = model.project(h[o[T_USER] + u])
                zi = model.project(h[o[T_ITEM] + i])
                loss = info_nce(zu, zi, tau, pos_ids=i, log_q=log_q[i])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()); nb += 1
        model.eval()
        with torch.no_grad():
            Zu, Zi, _ = model.all_embeddings(ft)
            vl = []
            for s in range(0, len(va), 1024):
                sl = slice(s, s + 1024)
                vl.append(float(info_nce(Zu[va.u[sl]], Zi[va.i[sl]], tau, pos_ids=va.i[sl])))
            vl = sum(vl) / len(vl)
        history.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "val_loss": vl, "seconds": round(time.time() - t0, 2)})
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log:
            log.log("graphsage", "epoch", **history[-1])
    model.load_state_dict(best_state)
    model.eval()
    return model, graph, history
