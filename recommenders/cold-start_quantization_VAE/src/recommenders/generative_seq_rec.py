"""Approach 2: generative sequence-to-Semantic-ID retrieval (TIGER-style).

A small causal Transformer reads  [BOS, user-context tokens, Semantic-ID tokens of history items ...]  and predicts
the next item's tokens (c1..cM). Inference uses beam search constrained to valid catalog Semantic IDs (trie).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DEVICE, USE_AMP, GenConfig
from .base import Catalog, ColdStartRecommender, UserContext
from .semantic_trie_rec import SemanticTrie

CTX_FIELDS = ["channel", "device", "region", "age_band", "intent_cat"]
PAD, BOS = 0, 1


class TokenSpace:
    """Vocabulary layout: [PAD, BOS] + context tokens (0 = unknown per field) + M*K code tokens."""

    def __init__(self, meta_vocab: Dict[str, list], M: int, K: int):
        self.M, self.K = M, K
        self.field_off, off = {}, 2
        self.field_vals = {}
        for f in CTX_FIELDS:
            vals = list(meta_vocab[f])
            self.field_vals[f] = vals
            self.field_off[f] = off
            off += len(vals) + 1
        self.code_off = off
        self.vocab = off + M * K

    def ctx_tokens(self, ctx: UserContext, drop: Optional[np.ndarray] = None) -> List[int]:
        toks = []
        for j, f in enumerate(CTX_FIELDS):
            v = getattr(ctx, f)
            idx = self.field_vals[f].index(v) + 1 if (v in self.field_vals[f]) else 0
            if drop is not None and drop[j]:
                idx = 0
            toks.append(self.field_off[f] + idx)
        return toks

    def item_tokens(self, codes_row: np.ndarray) -> List[int]:
        return [self.code_off + l * self.K + int(c) for l, c in enumerate(codes_row)]


class SeqTransformer(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_layers: int, n_heads: int, max_len: int, dropout: float):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(L, device=ids.device))[None]
        causal = torch.triu(torch.ones(L, L, device=ids.device, dtype=torch.bool), 1)
        x = self.enc(x, mask=causal, src_key_padding_mask=(ids == PAD))
        return self.head(self.ln(x))


class GenerativeSeqRecommender(ColdStartRecommender):
    name = "generative_seq"

    def __init__(self, catalog: Catalog, cfg: Optional[GenConfig] = None):
        super().__init__(catalog)
        self.cfg = cfg or GenConfig()
        self.ts = TokenSpace(catalog.dataset["meta_vocab"], catalog.M, catalog.K)
        self.max_len = 1 + len(CTX_FIELDS) + (self.cfg.max_hist + 1) * catalog.M + catalog.M
        self.model = SeqTransformer(self.ts.vocab, self.cfg.d_model, self.cfg.n_layers, self.cfg.n_heads,
                                    self.max_len + 2, self.cfg.dropout).to(DEVICE)
        self._trie_cache: Dict[bytes, SemanticTrie] = {}
        self._item_tokens = [self.ts.item_tokens(catalog.codes[i]) for i in range(catalog.N)]
        # level masks for constrained logits
        self._level_ranges = [(self.ts.code_off + l * catalog.K, self.ts.code_off + (l + 1) * catalog.K) for l in range(catalog.M)]

    # ------------------------------------------------------------------ training data
    def _build_prefix(self, ctx_tokens: List[int], click: Optional[int], hist: Sequence[int]) -> List[int]:
        seq = [BOS] + ctx_tokens
        items = ([click] if click is not None else []) + list(hist)[-self.cfg.max_hist:]
        for it in items:
            seq += self._item_tokens[int(it)]
        return seq

    def build_training_set(self, dataset: dict, rng: np.random.Generator):
        inter = dataset["interactions"]
        users = dataset["users"].set_index("user_id")
        click_of = users["click_item"].to_dict()
        tr = inter[inter["split"] == "train"].sort_values(["user_id", "ts"])
        samples = []
        for u, g in tr.groupby("user_id"):
            items = g["item_id"].tolist()
            for t, tgt in enumerate(items):
                samples.append((int(u), items[max(0, t - self.cfg.max_hist):t], tgt, int(click_of[u])))
        return samples, users

    def _make_ctx(self, urow, drop: Optional[np.ndarray] = None) -> List[int]:
        """urow: precomputed (5,) token array for the user; `drop` replaces fields by their <unknown> token."""
        toks = urow.copy()
        if drop is not None:
            toks[drop] = self._ctx_unk[drop]
        return toks.tolist()

    def fit(self, dataset: dict, seed: int = 0, verbose: bool = True) -> List[dict]:
        rng = np.random.default_rng(seed)
        samples, users = self.build_training_set(dataset, rng)
        self._ctx_unk = np.array([self.ts.field_off[f] for f in CTX_FIELDS])
        users = {int(u): np.array(self.ts.ctx_tokens(UserContext(channel=r.channel, device=r.device, region=r.region,
                                                                   age_band=r.age_band, intent_cat=r.intent_cat)))
                 for u, r in dataset["users"].set_index("user_id").iterrows()}
        n_val = max(500, len(samples) // 20)
        order = rng.permutation(len(samples))
        val_idx, tr_idx = order[:n_val], order[n_val:]
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.01)
        steps = self.cfg.epochs * int(np.ceil(len(tr_idx) / self.cfg.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.cfg.lr, total_steps=steps, pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
        M = self.cat.M
        logs = []
        best_val, best_state = float('inf'), None

        def batchify(idx, train: bool):
            seqs, masks = [], []
            for si in idx:
                u, hist, tgt, click = samples[si]
                urow = users[u]
                drop = rng.random(len(CTX_FIELDS)) < self.cfg.ctx_dropout if train else None
                h = hist
                if train and rng.random() < 0.3:                     # simulate cold users with truncated history
                    n_keep = int(rng.integers(0, 3))
                    h = hist[len(hist) - n_keep:] if n_keep > 0 else []
                use_click = (not train) or rng.random() > 0.2
                prefix = self._build_prefix(self._make_ctx(urow, drop), click if use_click else None, h)
                seq = prefix + self._item_tokens[tgt]
                mask = [0] * len(prefix) + [1] * M
                seqs.append(seq); masks.append(mask)
            L = max(len(s) for s in seqs)
            ids = np.zeros((len(seqs), L), np.int64)
            msk = np.zeros((len(seqs), L), np.bool_)
            for j, (s, m) in enumerate(zip(seqs, masks)):
                ids[j, :len(s)] = s; msk[j, :len(m)] = m
            return torch.tensor(ids, device=DEVICE), torch.tensor(msk, device=DEVICE)

        def loss_fn(ids, msk):
            logits = self.model(ids[:, :-1]).float()
            tgt, m = ids[:, 1:], msk[:, 1:]
            # restrict each target position to its own level's token range
            lvl_mask = torch.zeros_like(logits)
            pos_level = (m.cumsum(1) - 1).clamp(min=0)               # level index of each target token
            lvl_mask[m] = -1e9
            for l, (a, b) in enumerate(self._level_ranges):
                sel = m & (pos_level == l)
                if sel.any():
                    lvl_mask[sel, a:b] = 0.0
            lp = F.log_softmax(logits + lvl_mask, -1)
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            return (nll * m).sum() / m.sum()

        for ep in range(1, self.cfg.epochs + 1):
            self.model.train()
            perm = rng.permutation(tr_idx)
            tot, nb = 0.0, 0
            for i in range(0, len(perm), self.cfg.batch_size):
                ids, msk = batchify(perm[i:i + self.cfg.batch_size], True)
                with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    loss = loss_fn(ids, msk)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                tot += float(loss.detach()); nb += 1
            self.model.eval()
            with torch.no_grad():
                vl = np.mean([float(loss_fn(*batchify(val_idx[j:j + 512], False))) for j in range(0, len(val_idx), 512)])
            logs.append({"epoch": ep, "train_loss": tot / nb, "val_loss": float(vl)})
            if vl < best_val:
                best_val, best_state = float(vl), {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            if verbose and (ep % 5 == 0 or ep == 1):
                print(f"[generative] ep {ep:2d} train nll {tot / nb:.3f} val nll {vl:.3f}")
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return logs

    # ------------------------------------------------------------------ constrained beam search
    def _trie_for(self, candidates: Optional[Sequence[int]]) -> SemanticTrie:
        if candidates is None:
            key = b"all"
        else:
            arr = np.unique(np.asarray(candidates, dtype=np.int64))
            key = arr.tobytes()
        if key not in self._trie_cache:
            ids = None if candidates is None else np.unique(np.asarray(candidates, dtype=np.int64))
            if len(self._trie_cache) > 16:
                self._trie_cache.clear()
            self._trie_cache[key] = SemanticTrie(self.cat.codes, ids)
        return self._trie_cache[key]

    @torch.no_grad()
    def beam_search(self, prefix: List[int], trie: SemanticTrie, beam_width: int) -> List[Tuple[tuple, float]]:
        """Beam search over valid Semantic IDs only. Returns [(codes_tuple, logprob)] best first."""
        self.model.eval()
        K, M = self.cat.K, self.cat.M
        beams: List[Tuple[tuple, float]] = [((), 0.0)]
        for level in range(M):
            seqs = torch.tensor([prefix + [self.ts.code_off + l * K + c for l, c in enumerate(path)] for path, _ in beams],
                                device=DEVICE)
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = self.model(seqs)[:, -1].float()
            a, _ = self._level_ranges[level]
            cand: List[Tuple[tuple, float]] = []
            for b, (path, score) in enumerate(beams):
                children = trie.child_codes(path)
                if not children:
                    continue
                idx = torch.tensor(children, device=DEVICE) + a
                lp = F.log_softmax(logits[b, idx], -1).cpu().numpy()      # distribution over *valid* children
                cand.extend((path + (c,), score + float(l)) for c, l in zip(children, lp))
            cand.sort(key=lambda x: -x[1])
            beams = cand[:beam_width]
        return beams

    def recommend(self, ctx: UserContext, k: int = 10, candidates: Optional[Sequence[int]] = None,
                  exclude: Optional[set] = None) -> List[Tuple[int, float]]:
        exclude = exclude if exclude is not None else set(ctx.history)
        trie = self._trie_for(candidates)
        prefix = self._build_prefix(self.ts.ctx_tokens(ctx), ctx.click_item, ctx.history)
        if ctx.click_item is None and not ctx.history and ctx.intent_sub is None and ctx.intent_cat:
            # Day-0 style request: seed the sequence with content prototypes of the landing category
            for p in self.cat.prototypes(cat=ctx.intent_cat, n=2):
                prefix += self._item_tokens[int(p)]
        beams = self.beam_search(prefix, trie, max(self.cfg.beam_width, k + len(exclude) + 8))
        out, seen = [], set()
        for path, score in beams:
            leaf = [i for i in trie.items_under(path) if i not in exclude and i not in seen]
            leaf.sort(key=lambda i: -self.cat.pop[i])
            for i in leaf:
                seen.add(i)
                out.append((i, score))
                if len(out) >= k:
                    return out
        return out

    def score_all(self, ctx: UserContext) -> np.ndarray:
        res = self.recommend(ctx, k=self.cat.N)
        s = np.full(self.cat.N, -1e9)
        for i, sc in res:
            s[i] = sc
        return s

    def save(self, path) -> None:
        torch.save({"state": self.model.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path) -> None:
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ck["state"])
        self.model.eval()
