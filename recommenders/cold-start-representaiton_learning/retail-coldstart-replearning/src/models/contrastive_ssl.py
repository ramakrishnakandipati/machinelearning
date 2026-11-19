"""Approach 4: Self-supervised dual-view contrastive pretraining (SimCLR / CL4SRec style).

Item view  : catalogue attributes + text + visual of a (warm) item, augmented twice by
             - feature masking   (random dims of text / visual / continuous zeroed)
             - attribute cropping (categorical fields -> UNK, whole modalities -> 'missing')
             - semantic substitution (colour / material / brand borrowed from a random item of the same L3)
Session view: short sequences of items (train sessions + user history windows) augmented by
             item crop / item mask / same-L3 item substitution (CL4SRec).
Loss       : NT-Xent on projection-head outputs; the encoder output (pre-head) is the representation.
The encoder never sees labels and is trained on warm-catalogue items only; held-out cold items provide the
validation NT-Xent, i.e. zero-shot generalisation of the representation geometry.
Collapse monitor: uniformity (Wang & Isola) and mean per-dimension std are logged every epoch.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from src.models.common import l2n, nt_xent, uniformity
from src.models.multimodal_two_tower import ItemTower
from src.utils.hardware import autocast_ctx, make_scaler


class SSLModel(nn.Module):
    def __init__(self, cat_sizes, text_dim, vis_dim, d=128, proj=64):
        super().__init__()
        self.encoder = ItemTower(cat_sizes, text_dim, vis_dim, d)
        self.item_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, proj))
        self.sess_enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.sess_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, proj))
        self.d = d

    def encode_items(self, ft, idx=None, aug=None):
        sel = (lambda x: x) if idx is None else (lambda x: x[idx])
        cat, cont, text, vis = map(sel, (ft.item_cat, ft.item_cont, ft.item_text, ft.item_vis))
        mt = mv = None
        if aug is not None:
            cat, cont, text, vis, mt, mv = aug
        return self.encoder(cat, cont, text, vis, mt, mv)

    def encode_session(self, E, seq, mask):
        """E: (nI,d) item reps; seq: (B,L) item idx; mask (B,L) bool."""
        m = mask[..., None].float()
        pooled = (E[seq] * m).sum(1) / m.sum(1).clamp(min=1)
        return l2n(pooled + self.sess_enc(pooled))

    @torch.no_grad()
    def all_item_embeddings(self, ft):
        self.eval()
        return self.encode_items(ft)


class L3Sampler:
    """Draw a random item of the same L3 (restricted to a pool) for semantic substitution."""

    def __init__(self, l3_codes: np.ndarray, pool: np.ndarray, device):
        self.device = device
        pool_l3 = l3_codes[pool]
        order = np.argsort(pool_l3, kind="stable")
        self.sorted_items = torch.as_tensor(pool[order], device=device)
        n = l3_codes.max() + 1
        cnt = np.bincount(pool_l3, minlength=n)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        self.cnt = torch.as_tensor(np.maximum(cnt, 1), device=device)
        self.start = torch.as_tensor(start, device=device)
        self.l3 = torch.as_tensor(l3_codes, device=device)

    def sample(self, idx):
        c = self.l3[idx]
        off = (torch.rand(len(idx), device=self.device) * self.cnt[c]).long().clamp(max=self.cnt[c] - 1)
        return self.sorted_items[(self.start[c] + off).clamp(max=len(self.sorted_items) - 1)]


def augment_items(ft, idx, sampler, cfg, gen_noise=0.02):
    """Return an augmented view (cat, cont, text, vis, miss_text, miss_vis) for items idx."""
    B = len(idx)
    dev = idx.device
    cat, cont, text, vis = ft.item_cat[idx].clone(), ft.item_cont[idx].clone(), ft.item_text[idx].clone(), ft.item_vis[idx].clone()
    # semantic substitution: borrow brand/colour/material (cols 3,5,6) from a same-L3 item
    sub = sampler.sample(idx)
    swap = torch.rand(B, device=dev) < 0.5
    for c in (3, 5, 6):
        cat[:, c] = torch.where(swap & (torch.rand(B, device=dev) < 0.5), ft.item_cat[sub, c], cat[:, c])
    # attribute cropping: categorical fields -> UNK (keep l1 as anchor)
    crop = torch.rand(B, cat.shape[1], device=dev) < cfg["crop_prob"]
    crop[:, 0] = False
    cat = cat.masked_fill(crop, 0)
    # feature masking
    text = text * (torch.rand_like(text) > cfg["mask_prob"])
    vis = vis * (torch.rand_like(vis) > cfg["mask_prob"])
    cont = cont * (torch.rand_like(cont) > cfg["mask_prob"] / 2)
    text = text + gen_noise * torch.randn_like(text)
    vis = vis + gen_noise * torch.randn_like(vis)
    r = torch.rand(B, device=dev)
    return cat, cont, text, vis, r < 0.12, (r >= 0.12) & (r < 0.24)


def build_sequences(bundle, max_len=6):
    """Short sequences: train sessions with >=2 items + sliding windows over user histories."""
    ii = bundle.item_index
    seqs = []
    tr = bundle.tr_df.sort_values("ts")
    for _, g in tr.groupby("session_id"):
        it = g.item_id.map(ii).values
        if len(it) >= 2:
            seqs.append(it[:max_len])
    for _, g in tr.groupby("user_id"):
        it = g.item_id.map(ii).values
        for s in range(0, max(len(it) - 2, 0), 3):
            w = it[s:s + max_len]
            if len(w) >= 3:
                seqs.append(w)
    L = max_len
    S = np.zeros((len(seqs), L), np.int64)
    M = np.zeros((len(seqs), L), bool)
    for k, s in enumerate(seqs):
        S[k, :len(s)] = s
        M[k, :len(s)] = True
    return S, M


def augment_sequences(S, M, sampler, crop_p=0.3, mask_p=0.2, sub_p=0.2):
    dev = S.device
    keep = M & (torch.rand_like(S, dtype=torch.float) > crop_p)
    # always keep at least one item
    first = M.float().argmax(1)
    none = keep.sum(1) == 0
    keep[none, first[none]] = True
    sub = sampler.sample(S.reshape(-1)).view_as(S)
    S2 = torch.where(torch.rand_like(S, dtype=torch.float) < sub_p, sub, S)
    keep &= ~((torch.rand_like(S, dtype=torch.float) < mask_p) & (keep.sum(1, keepdim=True) > 1))
    return S2, keep


def train_ssl(bundle, epochs: int, log=None):
    cfg = bundle.cfg["train"]["ssl"]
    dev = bundle.device
    d = bundle.cfg["embedding_dim"]
    model = SSLModel(bundle.cat_sizes, bundle.text_dim, bundle.vis_dim, d).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scaler = make_scaler(dev, bundle.cfg["hardware"]["amp"])
    ft = bundle.ft
    cold = bundle.feats.item_cold
    warm_idx = np.where(~cold)[0]
    cold_idx = np.where(cold)[0]
    l3_codes = bundle.items.l3.astype("category").cat.codes.values.astype(np.int64)
    sampler = L3Sampler(l3_codes, warm_idx, dev)
    warm_t = torch.as_tensor(warm_idx, device=dev)
    cold_t = torch.as_tensor(cold_idx, device=dev)
    S_np, M_np = build_sequences(bundle)
    S, M = torch.as_tensor(S_np, device=dev), torch.as_tensor(M_np, device=dev)
    # sequences must only contain warm items (they come from train, which is cold-item free)
    bs, tau = cfg["batch_size"], cfg["temperature"]
    history, best, best_state = [], 1e9, None
    g = torch.Generator(device=dev)
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = warm_t[torch.randperm(len(warm_t), device=dev)]
        sperm = torch.randperm(len(S), device=dev)
        tot, nb = 0.0, 0
        for k, s in enumerate(range(0, len(perm), bs)):
            idx = perm[s:s + bs]
            if len(idx) < 16:
                continue
            si = sperm[(k * bs) % max(len(S) - bs, 1):][:bs]
            with autocast_ctx(dev, bundle.cfg["hardware"]["amp"]):
                v1 = model.encode_items(ft, idx, augment_items(ft, idx, sampler, cfg))
                v2 = model.encode_items(ft, idx, augment_items(ft, idx, sampler, cfg))
                loss_item = nt_xent(model.item_head(v1), model.item_head(v2), tau)
                E = model.encode_items(ft, None, None)           # clean catalogue reps for session pooling
                a_s, a_m = augment_sequences(S[si], M[si], sampler)
                b_s, b_m = augment_sequences(S[si], M[si], sampler)
                z1 = model.sess_head(model.encode_session(E, a_s, a_m))
                z2 = model.sess_head(model.encode_session(E, b_s, b_m))
                loss_sess = nt_xent(z1, z2, tau)
                loss = loss_item + 0.5 * loss_sess
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()); nb += 1
        model.eval()
        with torch.no_grad():
            g.manual_seed(1234)
            torch.manual_seed(1234)
            ci = cold_t[:512]
            w1 = model.encode_items(ft, ci, augment_items(ft, ci, sampler, cfg))
            w2 = model.encode_items(ft, ci, augment_items(ft, ci, sampler, cfg))
            vl = float(nt_xent(model.item_head(w1), model.item_head(w2), tau))
            Eall = model.encode_items(ft)
            unif = uniformity(Eall)
            dim_std = float(Eall.std(0).mean())
        history.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "val_loss": vl, "uniformity": unif,
                        "mean_dim_std": dim_std, "seconds": round(time.time() - t0, 2)})
        if vl < best:
            best, best_state = vl, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        if log:
            log.log("ssl", "epoch", **history[-1])
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def quiz_item_matrix(items, quiz_vocab, device) -> torch.Tensor:
    """(n_quiz, nI) membership: quiz tag 'likes_<l2>' / 'cat_<l1>' -> items of that category (row-normalised)."""
    l2 = items.l2.str.lower().str.replace(" ", "_").values
    l1 = items.l1.str.lower().values
    Q = np.zeros((len(quiz_vocab), len(items)), np.float32)
    for k, t in enumerate(quiz_vocab):
        if t.startswith("likes_"):
            Q[k] = (l2 == t[6:])
        elif t.startswith("cat_"):
            Q[k] = (l1 == t[4:])
    Q /= np.maximum(Q.sum(1, keepdims=True), 1)
    return torch.as_tensor(Q, device=device)


@torch.no_grad()
def ssl_user_query(E: torch.Tensor, hist: torch.Tensor, quiz: torch.Tensor, Q: torch.Tensor, alpha: float = 0.6) -> torch.Tensor:
    """User query in SSL item space. hist (B,H) item_idx+1; quiz (B,n_quiz).
    history prototype (if any) blended with a quiz-tag catalogue prototype (zero-shot prior)."""
    mask = (hist > 0).float()
    cnt = mask.sum(1, keepdim=True)
    hp = (E[(hist - 1).clamp(min=0)] * mask[..., None]).sum(1) / cnt.clamp(min=1)
    qp = (quiz @ Q) @ E / quiz.sum(1, keepdim=True).clamp(min=1)
    w = torch.where(cnt > 0, torch.full_like(cnt, alpha), torch.zeros_like(cnt))
    return l2n(w * l2n(hp) + (1 - w) * l2n(qp))
