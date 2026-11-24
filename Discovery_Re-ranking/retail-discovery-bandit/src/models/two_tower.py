"""GPU Two-Tower retrieval model (in-batch negatives + logQ correction) with a cosine-similarity index."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, n_users, n_personas, content_dim, cfg):
        super().__init__()
        self.id_emb = nn.Embedding(n_users, cfg["id_dim"])
        self.persona_emb = nn.Embedding(n_personas, 8)
        h, out = cfg["hidden"], cfg["tower_dim"]
        self.mlp = nn.Sequential(
            nn.Linear(cfg["id_dim"] + 8 + 1 + content_dim, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, out),
        )

    def forward(self, uid, persona, log_price, hist_vec):
        x = torch.cat([self.id_emb(uid), self.persona_emb(persona), log_price[:, None], hist_vec], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, n_items, n_subcats, n_cats, content_dim, cfg):
        super().__init__()
        self.n_items = n_items
        self.id_emb = nn.Embedding(n_items + 1, cfg["id_dim"], padding_idx=n_items)  # pad = cold-start / unknown
        self.sub_emb = nn.Embedding(n_subcats, 16)
        self.cat_emb = nn.Embedding(n_cats, 8)
        self.id_dropout = cfg["id_dropout"]
        h, out = cfg["hidden"], cfg["tower_dim"]
        self.mlp = nn.Sequential(
            nn.Linear(cfg["id_dim"] + 16 + 8 + 2 + content_dim, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, out),
        )

    def forward(self, iid, subcat, cat, log_price, margin, content, drop_ids: bool = False):
        if drop_ids and self.id_dropout > 0:
            mask = torch.rand(iid.shape, device=iid.device) < self.id_dropout
            iid = torch.where(mask, torch.full_like(iid, self.n_items), iid)
        x = torch.cat([self.id_emb(iid), self.sub_emb(subcat), self.cat_emb(cat),
                       log_price[:, None], margin[:, None], content], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTowerRetriever(nn.Module):
    def __init__(self, world, cfg: Dict, device: torch.device):
        super().__init__()
        self.cfg = cfg["retrieval"]
        self.device = device
        I, U = world.n_items, world.n_users
        D = world.item_emb.shape[1]
        self.user_tower = UserTower(U, len(world.meta["personas"]), D, self.cfg)
        self.item_tower = ItemTower(I, world.n_subcats, world.n_cats, D, self.cfg)
        it = world.items
        self.register_buffer("i_sub", torch.as_tensor(it["subcat"].values, dtype=torch.long))
        self.register_buffer("i_cat", torch.as_tensor(it["category"].values, dtype=torch.long))
        self.register_buffer("i_logp", torch.as_tensor(np.log(it["price"].values), dtype=torch.float32))
        self.register_buffer("i_margin", torch.as_tensor(it["margin"].values, dtype=torch.float32))
        self.register_buffer("i_content", torch.as_tensor(world.item_emb, dtype=torch.float32))
        self.register_buffer("i_is_cold", torch.as_tensor(it["is_cold"].values, dtype=torch.bool))
        self.register_buffer("u_persona", torch.as_tensor(world.users["persona"].values, dtype=torch.long))
        self.register_buffer("u_logp", torch.as_tensor(np.log(world.users["avg_item_price"].values), dtype=torch.float32))
        self.register_buffer("u_hist", torch.zeros(U, D))
        self.register_buffer("item_matrix", torch.zeros(I, self.cfg["tower_dim"]))
        self.register_buffer("user_matrix", torch.zeros(U, self.cfg["tower_dim"]))
        self.to(device)

    # ----- embeddings ------------------------------------------------------------------
    def set_user_history(self, world, train_end_day: int) -> None:
        """User content-history vector = mean observed embedding of clicked items in the window."""
        df = world.interactions
        df = df[(df.day <= train_end_day) & (df.click == 1)]
        U, D = world.n_users, world.item_emb.shape[1]
        h = np.zeros((U, D), dtype=np.float32)
        np.add.at(h, df["user"].values, world.item_emb[df["item"].values])
        n = np.bincount(df["user"].values, minlength=U).astype(np.float32)[:, None]
        h = h / np.maximum(n, 1.0)
        self.u_hist.copy_(torch.as_tensor(h, device=self.device))

    def encode_users(self, uids: torch.Tensor) -> torch.Tensor:
        return self.user_tower(uids, self.u_persona[uids], self.u_logp[uids], self.u_hist[uids])

    def encode_items(self, iids: torch.Tensor, cold_mask_ids: bool = True, drop_ids: bool = False) -> torch.Tensor:
        ids = iids
        if cold_mask_ids:  # cold-start items have no learned ID embedding -> pad (content-only)
            ids = torch.where(self.i_is_cold[iids], torch.full_like(iids, self.item_tower.n_items), iids)
        return self.item_tower(ids, self.i_sub[iids], self.i_cat[iids], self.i_logp[iids],
                               self.i_margin[iids], self.i_content[iids], drop_ids=drop_ids)

    @torch.no_grad()
    def build_index(self) -> None:
        """Pre-compute the (normalised) item/user matrices used for GPU cosine retrieval."""
        self.eval()
        all_i = torch.arange(self.item_matrix.shape[0], device=self.device)
        self.item_matrix.copy_(self.encode_items(all_i))
        all_u = torch.arange(self.user_matrix.shape[0], device=self.device)
        self.user_matrix.copy_(self.encode_users(all_u))

    # ----- retrieval -------------------------------------------------------------------
    @torch.no_grad()
    def retrieve(self, uids: torch.Tensor, k: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k cosine retrieval: one GPU matmul + top-k. uids: (B,) -> (scores, item ids) each (B,k)."""
        sims = self.user_matrix[uids] @ self.item_matrix.T
        return torch.topk(sims, k, dim=1)

    @torch.no_grad()
    def pair_scores(self, uids: torch.Tensor, iids: torch.Tensor) -> torch.Tensor:
        return (self.user_matrix[uids] * self.item_matrix[iids]).sum(-1)


# ----------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------
def train_two_tower(model: TwoTowerRetriever, world, train_pairs: np.ndarray, val_pairs: Optional[np.ndarray],
                    logger=None, jsonl=None) -> Dict[str, List]:
    """train_pairs/val_pairs: (N, 2) int arrays of positive (user, item)."""
    cfg = model.cfg
    dev = model.device
    I = world.n_items
    pu = torch.as_tensor(train_pairs[:, 0], dtype=torch.long, device=dev)
    pi = torch.as_tensor(train_pairs[:, 1], dtype=torch.long, device=dev)
    freq = torch.bincount(pi, minlength=I).float() + 1.0
    log_q = torch.log(freq / freq.sum())   # sampling probability of items in batch (logQ correction)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = pu.shape[0]
    bs = cfg["batch_size"]
    hist = {"epoch": [], "loss": [], "val_recall@200": [], "val_recall@50": []}
    best, best_state = -1.0, None
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot, nb = 0.0, 0
        for s in range(0, n - bs + 1, bs):
            idx = perm[s:s + bs]
            u, i = pu[idx], pi[idx]
            ue = model.encode_users(u)
            ie = model.encode_items(i, drop_ids=True)
            logits = ue @ ie.T / cfg["temperature"] - log_q[i][None, :]
            # mask other in-batch positives of the same user/item pair (false negatives)
            same = (i[None, :] == i[:, None]) & ~torch.eye(len(i), dtype=torch.bool, device=dev)
            logits = logits.masked_fill(same, -1e9)
            labels = torch.arange(len(i), device=dev)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        rec = {"recall@50": float("nan"), "recall@200": float("nan")}
        if val_pairs is not None and (ep % 3 == 2 or ep == cfg["epochs"] - 1):
            model.build_index()
            rec = recall_at_k(model, val_pairs, ks=(50, 200))
            if rec["recall@200"] > best:
                best = rec["recall@200"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        hist["epoch"].append(ep)
        hist["loss"].append(tot / max(nb, 1))
        hist["val_recall@200"].append(rec["recall@200"])
        hist["val_recall@50"].append(rec["recall@50"])
        msg = {"stage": "two_tower_train", "epoch": ep, "loss": tot / max(nb, 1), **rec, "elapsed_s": time.time() - t0}
        if logger:
            logger.info("two-tower epoch %02d loss=%.4f val R@50=%.4f R@200=%.4f", ep, msg["loss"], rec["recall@50"], rec["recall@200"])
        if jsonl:
            jsonl.log(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.build_index()
    return hist


@torch.no_grad()
def recall_at_k(model: TwoTowerRetriever, pairs: np.ndarray, ks=(50, 200), cold_only: bool = False,
                warm_only: bool = False) -> Dict[str, float]:
    """Recall@K / HitRate@K per user over the positives in `pairs` (user, item)."""
    if len(pairs) == 0:
        return {f"recall@{k}": float("nan") for k in ks}
    dev = model.device
    users = np.unique(pairs[:, 0])
    kmax = max(ks)
    out_rec = {k: [] for k in ks}
    out_hit = {k: [] for k in ks}
    pos_by_user: Dict[int, set] = {}
    for u, i in pairs:
        pos_by_user.setdefault(int(u), set()).add(int(i))
    for s in range(0, len(users), 512):
        ub = users[s:s + 512]
        _, top = model.retrieve(torch.as_tensor(ub, device=dev), kmax)
        top = top.cpu().numpy()
        for row, u in zip(top, ub):
            pos = pos_by_user[int(u)]
            for k in ks:
                hits = len(pos.intersection(row[:k].tolist()))
                out_rec[k].append(hits / len(pos))
                out_hit[k].append(1.0 if hits > 0 else 0.0)
    res = {}
    for k in ks:
        res[f"recall@{k}"] = float(np.mean(out_rec[k]))
        res[f"hitrate@{k}"] = float(np.mean(out_hit[k]))
    return res


@torch.no_grad()
def benchmark_retrieval_latency(model: TwoTowerRetriever, n: int = 300, k: int = 200) -> Dict[str, float]:
    dev = model.device
    U = model.user_matrix.shape[0]
    ts = []
    for _ in range(20):  # warm-up
        model.retrieve(torch.randint(0, U, (1,), device=dev), k)
    for _ in range(n):
        u = torch.randint(0, U, (1,), device=dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        model.retrieve(u, k)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    a = np.asarray(ts)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "target_ms": 10.0, "meets_target": bool(np.percentile(a, 99) < 10.0)}
