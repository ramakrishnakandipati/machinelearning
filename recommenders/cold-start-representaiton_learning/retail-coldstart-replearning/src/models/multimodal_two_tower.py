"""Approach 1: Content-aligned multimodal two-tower with modality / history dropout.

Item tower  : categorical-attribute embeddings + continuous attrs + text (TF-IDF/SVD) + visual (CLIP-like)
              fused by a Gated Residual Network. Missing modalities are replaced by learned 'missing' vectors.
User tower  : static profile (categorical + quiz tags) + context (time-of-day, device, store, channel)
              + masked-mean pooled interaction history (item-tower embeddings of previously seen items).
Cold-start  : history dropout (p=0.3) zeroes the history branch during training so the user tower must
              align profile/context directly with the item *content* tower through InfoNCE.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import GRN, info_nce, l2n
from src.utils.hardware import autocast_ctx, make_scaler

ITEM_CATS = ["l1", "l2", "l3", "brand", "price_tier", "color", "material"]
USER_CATS = ["acquisition_channel", "loyalty_tier", "home_region", "home_store", "device_os"]
CTX_CATS = ["device", "store_id", "channel"]


class ItemTower(nn.Module):
    def __init__(self, cat_sizes: dict, text_dim: int, vis_dim: int, d: int = 128, emb: int = 16, dropout: float = 0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(cat_sizes[c], emb) for c in ITEM_CATS])
        self.attr = nn.Sequential(nn.Linear(emb * len(ITEM_CATS) + 3, d), nn.ReLU())
        self.text = nn.Sequential(nn.Linear(text_dim, d), nn.ReLU())
        self.vis = nn.Sequential(nn.Linear(vis_dim, d), nn.ReLU())
        self.miss_text = nn.Parameter(torch.zeros(d))
        self.miss_vis = nn.Parameter(torch.zeros(d))
        self.fuse = GRN(3 * d, d, d_hidden=2 * d, dropout=dropout)
        self.out = nn.Linear(d, d)

    def forward(self, cat, cont, text, vis, miss_text=None, miss_vis=None):
        a = self.attr(torch.cat([e(cat[:, k]) for k, e in enumerate(self.embs)] + [cont], -1))
        t, v = self.text(text), self.vis(vis)
        if miss_text is not None:
            t = torch.where(miss_text[:, None], self.miss_text.expand_as(t), t)
        if miss_vis is not None:
            v = torch.where(miss_vis[:, None], self.miss_vis.expand_as(v), v)
        return l2n(self.out(self.fuse(torch.cat([a, t, v], -1))))


class UserTower(nn.Module):
    def __init__(self, cat_sizes: dict, n_quiz: int, d: int = 128, emb: int = 16, dropout: float = 0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(cat_sizes[c], emb) for c in USER_CATS])
        self.quiz = nn.Sequential(nn.Linear(n_quiz, 64), nn.ReLU())
        self.ctx_embs = nn.ModuleList([nn.Embedding(cat_sizes[c], emb) for c in CTX_CATS])
        self.hist_proj = nn.Linear(d, d)
        d_in = emb * len(USER_CATS) + 64 + emb * len(CTX_CATS) + 2 + d + 1
        self.fuse = GRN(d_in, d, d_hidden=2 * d, dropout=dropout)
        self.out = nn.Linear(d, d)

    def forward(self, ucat, quiz, ctx_cat, tod, hist_pooled, has_hist):
        x = torch.cat([e(ucat[:, k]) for k, e in enumerate(self.embs)] + [self.quiz(quiz)]
                      + [e(ctx_cat[:, k]) for k, e in enumerate(self.ctx_embs)] + [tod]
                      + [self.hist_proj(hist_pooled) * has_hist[:, None], has_hist[:, None]], -1)
        return l2n(self.out(self.fuse(x)))


class MultimodalTwoTower(nn.Module):
    def __init__(self, cat_sizes, text_dim, vis_dim, n_quiz, d=128, temperature=0.07, mod_drop=0.15):
        super().__init__()
        self.item_tower = ItemTower(cat_sizes, text_dim, vis_dim, d)
        self.user_tower = UserTower(cat_sizes, n_quiz, d)
        self.log_tau = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(temperature)))))
        self.mod_drop = mod_drop
        self.d = d

    @property
    def tau(self):
        return self.log_tau.exp().clamp(0.02, 0.5)

    def embed_items(self, ft, idx=None, train=False):
        sel = (lambda x: x) if idx is None else (lambda x: x[idx])
        cat, cont, text, vis = map(sel, (ft.item_cat, ft.item_cont, ft.item_text, ft.item_vis))
        mt = mv = None
        if train and self.mod_drop > 0:
            n = cat.shape[0]
            r = torch.rand(n, device=cat.device)
            mt = r < self.mod_drop                                   # drop text
            mv = (r >= self.mod_drop) & (r < 2 * self.mod_drop)      # drop visual (never both)
        return self.item_tower(cat, cont, text, vis, mt, mv)

    @staticmethod
    def pool_history(E, hist, exclude=None, drop_p=0.0):
        """Masked-mean pooling of history item embeddings. hist: (B,H) item_idx+1 (0=pad)."""
        mask = hist > 0
        if exclude is not None:
            mask = mask & (hist - 1 != exclude[:, None])
        if drop_p > 0:
            mask = mask & ~(torch.rand(hist.shape[0], device=hist.device) < drop_p)[:, None]
        emb = E[(hist - 1).clamp(min=0)] * mask[..., None]
        cnt = mask.sum(1)
        pooled = emb.sum(1) / cnt.clamp(min=1)[:, None]
        return pooled, (cnt > 0).float()

    def embed_users(self, ft, E, u, ctx_cat, tod, hist, exclude=None, drop_p=0.0, no_history=False):
        if no_history:
            pooled = torch.zeros(len(u), self.d, device=E.device)
            has = torch.zeros(len(u), device=E.device)
        else:
            pooled, has = self.pool_history(E, hist[u], exclude, drop_p)
        return self.user_tower(ft.user_cat[u], ft.user_quiz[u], ctx_cat, tod, pooled, has)


def _val_loss(model, bundle, ft, hist, va, E_eval):
    if va is None or len(va) == 0:
        return float("nan")
    losses = []
    for s in range(0, len(va), 1024):
        sl = slice(s, s + 1024)
        i = va.i[sl]
        q = model.embed_users(ft, E_eval, va.u[sl], va.ctx_cat[sl], va.tod[sl], hist)
        losses.append(float(info_nce(q, E_eval[i], model.tau, pos_ids=i)))
    return sum(losses) / len(losses)


def train_two_tower(bundle, epochs: int, log=None):
    cfg = bundle.cfg["train"]["two_tower"]
    dev = bundle.device
    d = bundle.cfg["embedding_dim"]
    model = MultimodalTwoTower(bundle.cat_sizes, bundle.text_dim, bundle.vis_dim, bundle.n_quiz, d,
                               cfg["temperature"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=epochs * ((len(bundle.tr) + cfg["batch_size"] - 1) // cfg["batch_size"]))
    scaler = make_scaler(dev, bundle.cfg["hardware"]["amp"])
    ft, tr, va, hist = bundle.ft, bundle.tr, bundle.va, bundle.hist
    log_q = bundle.item_logq
    history = []
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(tr), device=dev)
        tot, nb = 0.0, 0
        for s in range(0, len(tr), cfg["batch_size"]):
            idx = perm[s:s + cfg["batch_size"]]
            if len(idx) < 8:
                continue
            u, i = tr.u[idx], tr.i[idx]
            with autocast_ctx(dev, bundle.cfg["hardware"]["amp"]):
                E = model.embed_items(ft, train=True)
                q = model.embed_users(ft, E, u, tr.ctx_cat[idx], tr.tod[idx], hist, exclude=i, drop_p=cfg["history_dropout"])
                loss = info_nce(q, E[i], model.tau, pos_ids=i, log_q=log_q[i])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss)
            nb += 1
        model.eval()
        with torch.no_grad():
            E_eval = model.embed_items(ft)
            vl = _val_loss(model, bundle, ft, hist, va, E_eval)
        history.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "val_loss": vl, "tau": float(model.tau), "seconds": round(time.time() - t0, 2)})
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log:
            log.log("two_tower", "epoch", **history[-1])
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history
