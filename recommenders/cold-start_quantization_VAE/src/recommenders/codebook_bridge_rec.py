"""Approach 3: codebook-bridged two-tower / cross-modal alignment.

A user tower maps *metadata + session context* (channel, device, region, age band, intent, initial click, optional
history) into the item latent space spanned by the VQ-/RQ-VAE quantized embeddings. It is trained with InfoNCE and an
auxiliary per-level code-prediction loss, which enables
  * zero-shot dot-product retrieval against item z_q vectors (works for brand-new items), and
  * codebook-overlap (Hamming-style) retrieval from the predicted code distributions p_l(c_l | user).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DEVICE, USE_AMP, BridgeConfig
from .base import Catalog, ColdStartRecommender, UserContext, META_FIELDS


class UserTower(nn.Module):
    def __init__(self, vocab_sizes: List[int], latent_dim: int, cfg: BridgeConfig, n_levels: int, num_codes: int):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v + 1, cfg.emb_dim) for v in vocab_sizes])   # 0 = unknown
        in_dim = cfg.emb_dim * len(vocab_sizes) + 2 * latent_dim + 2
        self.mlp = nn.Sequential(nn.Linear(in_dim, cfg.hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(cfg.hidden, cfg.hidden), nn.GELU())
        self.out = nn.Linear(cfg.hidden, latent_dim)
        self.code_heads = nn.ModuleList([nn.Linear(cfg.hidden, num_codes) for _ in range(n_levels)])

    def forward(self, meta: torch.Tensor, hist_vec: torch.Tensor, hist_flag: torch.Tensor, click_vec: torch.Tensor,
                click_flag: torch.Tensor):
        e = [emb(meta[:, j]) for j, emb in enumerate(self.embs)]
        h = self.mlp(torch.cat(e + [hist_vec, click_vec, hist_flag[:, None], click_flag[:, None]], -1))
        return F.normalize(self.out(h), dim=-1), [head(h) for head in self.code_heads]


class ItemTower(nn.Module):
    """Light residual head on top of the frozen quantized embedding z_q."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, z):
        return F.normalize(z + self.net(z), dim=-1)


class CodebookBridgeRecommender(ColdStartRecommender):
    name = "codebook_bridge"

    def __init__(self, catalog: Catalog, cfg: Optional[BridgeConfig] = None, mode: str = "fused"):
        super().__init__(catalog)
        self.cfg, self.mode = cfg or BridgeConfig(), mode
        mv = catalog.dataset["meta_vocab"]
        self.vocab = {f: list(mv[f]) for f in META_FIELDS}
        D = catalog.zq.shape[1]
        self.D = D
        self.user_tower = UserTower([len(self.vocab[f]) for f in META_FIELDS], D, self.cfg, catalog.M, catalog.K).to(DEVICE)
        self.item_tower = ItemTower(D).to(DEVICE)
        zq = catalog.zq
        self.zq_scale = float(np.linalg.norm(zq, axis=1).mean() + 1e-9)
        self.zq_t = torch.tensor(zq / self.zq_scale, device=DEVICE)
        self._z_np = (zq / self.zq_scale).astype(np.float32)
        self.codes_t = torch.tensor(catalog.codes, device=DEVICE)
        self.level_w = np.array([1.0 / 2 ** l for l in range(catalog.M)])
        self._item_emb: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ feature encoding
    def meta_index(self, ctx: UserContext) -> List[int]:
        out = []
        for f in META_FIELDS:
            v = getattr(ctx, f)
            out.append(self.vocab[f].index(v) + 1 if v in self.vocab[f] else 0)
        return out

    def _ctx_tensors(self, ctxs: List[UserContext]):
        meta = torch.tensor([self.meta_index(c) for c in ctxs], device=DEVICE)
        hv = np.zeros((len(ctxs), self.D), np.float32); hf = np.zeros(len(ctxs), np.float32)
        cv = np.zeros((len(ctxs), self.D), np.float32); cf = np.zeros(len(ctxs), np.float32)
        z = self._z_np
        for j, c in enumerate(ctxs):
            if c.history:
                hv[j] = z[np.asarray(c.history[-8:])].mean(0); hf[j] = 1
            if c.click_item is not None:
                cv[j] = z[c.click_item]; cf[j] = 1
        t = lambda a: torch.tensor(a, device=DEVICE)
        return meta, t(hv), t(hf), t(cv), t(cf)

    # ------------------------------------------------------------------ training
    def fit(self, dataset: dict, seed: int = 0, verbose: bool = True) -> List[dict]:
        rng = np.random.default_rng(seed)
        inter = dataset["interactions"]
        users = dataset["users"].set_index("user_id")
        tr = inter[inter["split"] == "train"].sort_values(["user_id", "ts"])
        z = self.zq_t.cpu().numpy()
        meta_rows, hist_vec, hist_flag, click_vec, click_flag, targets = [], [], [], [], [], []
        for u, g in tr.groupby("user_id"):
            r = users.loc[u]
            ctx = UserContext(channel=r.channel, device=r.device, region=r.region, age_band=r.age_band,
                              intent_cat=r.intent_cat, intent_sub=r.intent_sub)
            m = self.meta_index(ctx)
            items = g["item_id"].tolist()
            for t, tgt in enumerate(items):
                hist = items[max(0, t - 8):t]
                meta_rows.append(m)
                if hist:
                    hist_vec.append(z[hist].mean(0)); hist_flag.append(1.0)
                else:
                    hist_vec.append(np.zeros(self.D, np.float32)); hist_flag.append(0.0)
                click_vec.append(z[int(r.click_item)]); click_flag.append(1.0)
                targets.append(tgt)
        T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=DEVICE)
        meta, hv, hf, cv, cf = T(meta_rows, torch.long), T(hist_vec), T(hist_flag), T(click_vec), T(click_flag)
        tgt = T(targets, torch.long)
        n = len(tgt)
        perm0 = rng.permutation(n)
        n_val = max(500, n // 20)
        va_idx, tr_idx = torch.tensor(perm0[:n_val], device=DEVICE), torch.tensor(perm0[n_val:], device=DEVICE)

        params = list(self.user_tower.parameters()) + list(self.item_tower.parameters())
        opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.cfg.epochs)
        logs = []
        best_val, best_state = float('inf'), None
        ncodes = self.cat.M

        def step(idx, train: bool):
            m = meta[idx].clone()
            h, hfl, c, cfl = hv[idx], hf[idx], cv[idx], cf[idx]
            if train:
                B = len(idx)
                keep_h = (torch.rand(B, device=DEVICE) > self.cfg.hist_dropout).float()      # simulate cold users
                keep_c = (torch.rand(B, device=DEVICE) > self.cfg.click_dropout).float()
                h, hfl = h * keep_h[:, None], hfl * keep_h
                c, cfl = c * keep_c[:, None], cfl * keep_c
                drop = torch.rand(m.shape, device=DEVICE) < 0.1                             # metadata dropout
                m = torch.where(drop, torch.zeros_like(m), m)
            u, code_logits = self.user_tower(m, h, hfl, c, cfl)
            tgt_b = tgt[idx]
            v = self.item_tower(self.zq_t[tgt_b])
            logits = (u @ v.t()) / self.cfg.temperature
            same = tgt_b[:, None] == tgt_b[None, :]
            eye = torch.eye(len(idx), dtype=torch.bool, device=DEVICE)
            logits = logits.masked_fill(same & ~eye, float("-inf"))          # same item is not a negative
            labels = torch.arange(len(idx), device=DEVICE)
            nce = F.cross_entropy(logits, labels)
            aux = sum(float(self.level_w[l]) * F.cross_entropy(code_logits[l].float(), self.codes_t[tgt_b, l])
                      for l in range(ncodes)) / self.level_w.sum()
            return nce, aux

        for ep in range(1, self.cfg.epochs + 1):
            self.user_tower.train(); self.item_tower.train()
            p = tr_idx[torch.randperm(len(tr_idx), device=DEVICE)]
            tot_n = tot_a = 0.0; nb = 0
            for i in range(0, len(p), self.cfg.batch_size):
                idx = p[i:i + self.cfg.batch_size]
                if len(idx) < 16:
                    continue
                nce, aux = step(idx, True)
                loss = nce + self.cfg.aux_weight * aux
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                tot_n += float(nce.detach()); tot_a += float(aux.detach()); nb += 1
            sched.step()
            self.user_tower.eval(); self.item_tower.eval()
            with torch.no_grad():
                vn, va = step(va_idx[:512], False)
            logs.append({"epoch": ep, "train_infonce": tot_n / nb, "train_code_ce": tot_a / nb,
                         "val_infonce": float(vn), "val_code_ce": float(va)})
            if float(vn) < best_val:
                best_val = float(vn)
                best_state = ({k: v.detach().clone() for k, v in self.user_tower.state_dict().items()},
                              {k: v.detach().clone() for k, v in self.item_tower.state_dict().items()})
            if verbose and (ep % 10 == 0 or ep == 1):
                print(f"[bridge] ep {ep:2d} infonce {tot_n / nb:.3f} code_ce {tot_a / nb:.3f} | val infonce {float(vn):.3f}")
        if best_state is not None:
            self.user_tower.load_state_dict(best_state[0]); self.item_tower.load_state_dict(best_state[1])
        self.user_tower.eval(); self.item_tower.eval()
        self._item_emb = None
        return logs

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def item_embeddings(self) -> torch.Tensor:
        if self._item_emb is None:
            self.item_tower.eval()
            self._item_emb = self.item_tower(self.zq_t)
        return self._item_emb

    @torch.no_grad()
    def user_forward(self, ctxs: List[UserContext]):
        self.user_tower.eval()
        u, code_logits = self.user_tower(*self._ctx_tensors(ctxs))
        return u, [F.log_softmax(l.float(), -1) for l in code_logits]

    @torch.no_grad()
    def embed_new_items(self, zq: np.ndarray) -> torch.Tensor:
        """Embed items that are not in the catalog (e.g. a brand-new SKU) with the item tower."""
        self.item_tower.eval()
        return self.item_tower(torch.tensor(zq / self.zq_scale, dtype=torch.float32, device=DEVICE))

    @staticmethod
    def _zn(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-9)

    def score_components(self, ctx: UserContext) -> Tuple[np.ndarray, np.ndarray]:
        u, lps = self.user_forward([ctx])
        dot = (self.item_embeddings() @ u[0]).cpu().numpy()
        overlap = np.zeros(self.cat.N)
        for l in range(self.cat.M):
            overlap += self.level_w[l] * lps[l][0].cpu().numpy()[self.cat.codes[:, l]]
        return dot, overlap

    def score_all(self, ctx: UserContext) -> np.ndarray:
        dot, overlap = self.score_components(ctx)
        if self.mode == "dot":
            return dot
        if self.mode == "overlap":
            return overlap
        return self._zn(dot) + 0.5 * self._zn(overlap)

    def predicted_codes(self, ctx: UserContext, topn: int = 3) -> List[List[Tuple[int, float]]]:
        """Top predicted codes per level for explanations."""
        _, lps = self.user_forward([ctx])
        out = []
        for l in range(self.cat.M):
            p = lps[l][0].exp().cpu().numpy()
            idx = np.argsort(-p)[:topn]
            out.append([(int(i), float(p[i])) for i in idx])
        return out

    def with_mode(self, mode: str) -> "CodebookBridgeRecommender":
        clone = object.__new__(CodebookBridgeRecommender)
        clone.__dict__.update(self.__dict__)
        clone.mode = mode
        return clone

    def save(self, path) -> None:
        torch.save({"user": self.user_tower.state_dict(), "item": self.item_tower.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path) -> None:
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        self.user_tower.load_state_dict(ck["user"]); self.item_tower.load_state_dict(ck["item"])
        self.user_tower.eval(); self.item_tower.eval(); self._item_emb = None
