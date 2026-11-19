"""Approach 3: MAML-inspired few-shot user representation adapter.

Given a cold-start prior e_u (content/profile user embedding, no history) and a support set
S_u = {s_1..s_k} of 1-3 clicked item embeddings, the adapter produces

  inner loop (gradient-based, differentiable, learned per-dim step size alpha):
      L_sup(e) = -1/k sum_j log softmax_i( cos(e, E_i)/tau )[s_j]
      e <- e - alpha * grad_e L_sup(e)            (inner_steps times)
  prototype residual (attention-weighted support prototype p):
      e_adapted = normalize( e + gamma * MLP([e, p, e*p]) )

Meta-training (outer loop) samples episodes from warm users' train histories:
support k in {1,2,3}, query = remaining items; outer loss = full-catalogue softmax NLL of the
query items under the adapted embedding (support items masked, as in serving).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import l2n


class MAMLUserAdapter(nn.Module):
    def __init__(self, d=128, inner_lr=0.1, inner_steps=2, tau=0.07):
        super().__init__()
        self.d, self.inner_steps, self.tau = d, inner_steps, tau
        self.log_alpha = nn.Parameter(torch.full((d,), float(np.log(inner_lr))))
        self.attn = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Linear(64, 1))
        self.res = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def inner_loss(self, e, E_all, sup_idx, sup_mask):
        logp = F.log_softmax(l2n(e) @ E_all.T / self.tau, -1)               # (B,nI)
        nll = -(logp.gather(1, sup_idx) * sup_mask).sum(1) / sup_mask.sum(1).clamp(min=1)
        return nll.sum()

    def adapt(self, prior, sup_idx, sup_mask, E_all):
        """prior (B,d) L2-normed; sup_idx (B,k) item indices; sup_mask (B,k) float. Returns (B,d) L2-normed."""
        create = torch.is_grad_enabled() and self.training
        with torch.enable_grad():
            e = prior.detach().clone().requires_grad_(True)
            alpha = self.log_alpha.exp()
            for _ in range(self.inner_steps):
                loss = self.inner_loss(e, E_all, sup_idx, sup_mask)
                g, = torch.autograd.grad(loss, e, create_graph=create)
                e = e - alpha * g
                if not create:
                    e = e.detach().requires_grad_(True)
        e = e if create else e.detach()
        S = E_all[sup_idx]                                                    # (B,k,d)
        sc = self.attn(torch.cat([e.unsqueeze(1).expand_as(S), S], -1)).squeeze(-1).masked_fill(sup_mask == 0, -1e9)
        p = (torch.softmax(sc, -1).unsqueeze(-1) * S).sum(1)
        return l2n(e + self.gamma * self.res(torch.cat([e, p, e * p], -1)))


class EpisodeSampler:
    def __init__(self, user_items: dict, device, max_len=20):
        us = [u for u, it in user_items.items() if len(it) >= 2]
        self.users = torch.as_tensor(us, device=device)
        self.items = torch.zeros(len(us), max_len, dtype=torch.long, device=device)
        self.cnt = torch.zeros(len(us), dtype=torch.long, device=device)
        for r, u in enumerate(us):
            it = user_items[u][-max_len:]
            self.items[r, :len(it)] = torch.as_tensor(it, device=device)
            self.cnt[r] = len(it)
        self.device = device

    def sample(self, T, q=4):
        r = torch.randint(0, len(self.users), (T,), device=self.device)
        it, n = self.items[r], self.cnt[r]
        L = it.shape[1]
        valid = torch.arange(L, device=self.device)[None, :] < n[:, None]
        order = torch.argsort(torch.rand(T, L, device=self.device).masked_fill(~valid, 2.0), 1)
        it = it.gather(1, order)
        valid = valid.gather(1, order)
        k = torch.randint(1, 4, (T,), device=self.device)
        k = torch.minimum(k, n - 1)
        pos = torch.arange(L, device=self.device)[None, :]
        sup_mask = (pos < k[:, None]).float()[:, :3]
        sup_idx = it[:, :3]
        qpos = (pos >= k[:, None]) & valid & (pos < k[:, None] + q)
        # gather up to q query slots
        qi = torch.zeros(T, q, dtype=torch.long, device=self.device)
        qm = torch.zeros(T, q, device=self.device)
        for j in range(q):
            idx = (k + j).clamp(max=L - 1)
            qi[:, j] = it.gather(1, idx[:, None]).squeeze(1)
            qm[:, j] = qpos.gather(1, idx[:, None]).squeeze(1).float()
        return self.users[r], sup_idx, sup_mask, qi, qm


def outer_loss(adapter, prior, E_all, sup_idx, sup_mask, qi, qm, adapted=None):
    e = adapter.adapt(prior, sup_idx, sup_mask, E_all) if adapted is None else adapted
    logits = e @ E_all.T / adapter.tau
    sm = torch.zeros_like(logits).scatter(1, sup_idx, sup_mask)       # mask support items
    logits = logits.masked_fill(sm > 0, -1e9)
    logp = F.log_softmax(logits, -1)
    nll = -(logp.gather(1, qi) * qm).sum(1) / qm.sum(1).clamp(min=1)
    return nll.mean()


@torch.no_grad()
def user_prior_embeddings(tt, bundle, E, users: torch.Tensor, ctx_cat: torch.Tensor, tod: torch.Tensor) -> torch.Tensor:
    return tt.embed_users(bundle.ft, E, users, ctx_cat, tod, bundle.hist, no_history=True)


def train_maml(bundle, tt, epochs: int, log=None):
    cfg = bundle.cfg["train"]["maml"]
    dev = bundle.device
    d = bundle.cfg["embedding_dim"]
    tt.eval()
    with torch.no_grad():
        E = tt.embed_items(bundle.ft)
    # train episodes: warm users' train items (unique, chronological)
    tr = bundle.tr_df.sort_values("ts")
    gm = bundle.feats.user_group
    ui = bundle.user_index
    user_items, last_ctx = {}, {}
    for uid, g in tr.groupby("user_id"):
        if gm[ui[uid]] != "warm":
            continue
        its = list(dict.fromkeys(g.item_id.map(bundle.item_index).tolist()))
        user_items[ui[uid]] = its
        last_ctx[ui[uid]] = g.index[-1]
    sampler = EpisodeSampler(user_items, dev)
    ctx_rows = torch.as_tensor([last_ctx[int(u)] for u in sampler.users.tolist()], device=dev)
    prior_all = user_prior_embeddings(tt, bundle, E, sampler.users, bundle.tr.ctx_cat[ctx_rows], bundle.tr.tod[ctx_rows])
    row_of = {int(u): r for r, u in enumerate(sampler.users.tolist())}
    # validation tasks: few-shot users, support = train items, query = their val items
    val = bundle.val_df[bundle.val_df["slice"] == "user_cold"]
    vt = []
    vu_ctx = {}
    for uid, g in val.groupby("user_id"):
        u = ui[uid]
        sup = bundle.hist_np[u][bundle.hist_np[u] > 0][-3:] - 1
        if len(sup) == 0:
            continue
        q = [x for x in dict.fromkeys(g.item_id.map(bundle.item_index).tolist()) if x not in set(sup.tolist())][:4]
        if not q:
            continue
        vt.append((u, sup, q, g.index[0]))
    adapter = MAMLUserAdapter(d, cfg["inner_lr"], cfg["inner_steps"], tt.tau.item() if hasattr(tt.tau, "item") else 0.07).to(dev)
    opt = torch.optim.Adam(adapter.parameters(), lr=cfg["lr"])
    # build val tensors
    def pad(rows, n, fill=0):
        out = torch.full((len(rows), n), fill, dtype=torch.long, device=dev)
        for r, x in enumerate(rows):
            out[r, :len(x)] = torch.as_tensor(list(x), device=dev)
        return out
    if vt:
        v_u = torch.as_tensor([x[0] for x in vt], device=dev)
        v_sup = pad([x[1] for x in vt], 3)
        v_sm = (v_sup > -1).float() * 0
        for r, x in enumerate(vt):
            v_sm[r, :len(x[1])] = 1
        v_q = pad([x[2] for x in vt], 4)
        v_qm = torch.zeros(len(vt), 4, device=dev)
        for r, x in enumerate(vt):
            v_qm[r, :len(x[2])] = 1
        vrows = [bundle.val_df.index.get_loc(x[3]) for x in vt]
        v_prior = user_prior_embeddings(tt, bundle, E, v_u, bundle.va.ctx_cat[vrows], bundle.va.tod[vrows])
    history, best, best_state = [], 1e9, None
    for ep in range(epochs):
        adapter.train()
        t0 = time.time()
        tot = 0.0
        for _ in range(cfg["episodes_per_epoch"]):
            users, sup, sm, qi, qm = sampler.sample(cfg["tasks_per_batch"], cfg["query_size"])
            prior = prior_all[torch.as_tensor([row_of[int(u)] for u in users.tolist()], device=dev)]
            loss = outer_loss(adapter, prior, E, sup, sm, qi, qm)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
        adapter.eval()
        if vt:
            vl = float(outer_loss(adapter, v_prior, E, v_sup, v_sm, v_q, v_qm).detach())
            with torch.no_grad():
                base = float(outer_loss(adapter, v_prior, E, v_sup, v_sm, v_q, v_qm, adapted=v_prior))
        else:
            vl = base = float("nan")
        history.append({"epoch": ep + 1, "train_loss": tot / cfg["episodes_per_epoch"], "val_loss": vl,
                        "val_loss_prior_only": base, "alpha_mean": float(adapter.log_alpha.exp().mean()), "seconds": round(time.time() - t0, 2)})
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in adapter.state_dict().items()}
        if log:
            log.log("maml", "epoch", **history[-1])
    adapter.load_state_dict(best_state)
    adapter.eval()
    return adapter, history
