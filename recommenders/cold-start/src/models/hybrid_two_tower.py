"""Strategy 3 - attribute-enriched Two-Tower network with ID / modality dropout (DropoutNet style)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.preprocessor import Preprocessor
from src.utils.hardware import amp_enabled, autocast_ctx, make_grad_scaler


def _mlp(i: int, h: int, o: int, p: float = 0.1) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, o))


class TwoTower(nn.Module):
    """User tower (ID + demographics + session context + history/onboarding) and item tower (ID + catalog attrs).

    Index 0 of every ID embedding is a *learned UNK vector*; ID dropout replaces IDs by 0 during training so
    unseen users/items are served from side-features alone.
    """

    def __init__(self, pre: Preprocessor, tt_cfg: Dict[str, Any]):
        super().__init__()
        d, h, idd = int(tt_cfg["embed_dim"]), int(tt_cfg["hidden_dim"]), int(tt_cfg["id_dim"])
        self.out_dim = d
        self.uid = nn.Embedding(pre.n_uid, idd)
        self.region, self.device_e = nn.Embedding(len(pre.v_region), 8), nn.Embedding(len(pre.v_device), 8)
        self.acq, self.age = nn.Embedding(len(pre.v_acq), 8), nn.Embedding(len(pre.v_age), 8)
        self.channel, self.season = nn.Embedding(len(pre.v_channel), 4), nn.Embedding(len(pre.v_season), 4)
        self.hour = nn.Embedding(len(pre.v_hour), 4)
        self.onboard = nn.Linear(pre.n_sub, 32)
        self.hist = nn.Linear(pre.content_dim, 48)
        self.user_mlp = _mlp(idd + 32 + 12 + 32 + 48 + 1, h, d)
        self.iid = nn.Embedding(pre.n_iid, idd)
        self.dept, self.sub = nn.Embedding(len(pre.v_dept), 8), nn.Embedding(len(pre.v_sub), 16)
        self.brand, self.tier = nn.Embedding(len(pre.v_brand), 16), nn.Embedding(len(pre.v_tier) + 1, 4)
        self.text = nn.Linear(pre.embedder.dim, 48)
        self.item_mlp = _mlp(idd + 8 + 16 + 16 + 4 + 1 + 48, h, d)

    def user_forward(self, uid, ucats, onboard, ctx, hist, hist_n) -> torch.Tensor:
        x = torch.cat([self.uid(uid), self.region(ucats[:, 0]), self.device_e(ucats[:, 1]), self.acq(ucats[:, 2]),
                       self.age(ucats[:, 3]), self.channel(ctx[:, 0]), self.season(ctx[:, 1]), self.hour(ctx[:, 2]),
                       F.relu(self.onboard(onboard)), F.relu(self.hist(hist)), hist_n.unsqueeze(1)], 1)
        return F.normalize(self.user_mlp(x), dim=-1)

    def item_forward(self, iid, cats, text, price) -> torch.Tensor:
        x = torch.cat([self.iid(iid), self.dept(cats[:, 0]), self.sub(cats[:, 1]), self.brand(cats[:, 2]),
                       self.tier(cats[:, 3]), price.unsqueeze(1), F.relu(self.text(text))], 1)
        return F.normalize(self.item_mlp(x), dim=-1)


class TwoTowerRecommender:
    """Training, inference and live item-injection wrapper around :class:`TwoTower`."""

    def __init__(self, pre: Preprocessor, cfg: Dict[str, Any], device: torch.device):
        self.pre, self.cfg, self.device = pre, cfg, device
        self.tt_cfg = cfg["two_tower"]
        self.model = TwoTower(pre, self.tt_cfg).to(device)
        self.item_emb: Optional[torch.Tensor] = None       # (N, d) on device
        self.history: List[Dict[str, float]] = []
        self.amp_used = False

    # ------------------------------------------------------------------ helpers
    def _item_tensors(self, idx: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        p, dev = self.pre, self.device
        sl = slice(None) if idx is None else idx
        return dict(iid=torch.as_tensor(p.item_id_idx[sl], device=dev), cats=torch.as_tensor(p.item_cats[sl], device=dev),
                    text=torch.as_tensor(p.item_text[sl], device=dev), price=torch.as_tensor(p.item_price[sl], device=dev))

    @torch.no_grad()
    def refresh_item_embeddings(self) -> torch.Tensor:
        self.model.eval()
        t = self._item_tensors()
        outs = [self.model.item_forward(t["iid"][i:i + 4096], t["cats"][i:i + 4096], t["text"][i:i + 4096],
                                        t["price"][i:i + 4096]) for i in range(0, len(t["iid"]), 4096)]
        self.item_emb = torch.cat(outs).float()
        return self.item_emb

    @torch.no_grad()
    def add_item(self, enc: Dict[str, Any]) -> int:
        """Embed a brand-new SKU from metadata only (ID = UNK) and append it. Returns its position."""
        self.model.eval()
        dev = self.device
        e = self.model.item_forward(torch.zeros(1, dtype=torch.long, device=dev),
                                    torch.as_tensor(enc["cats"][None], device=dev),
                                    torch.as_tensor(enc["text"][None], device=dev),
                                    torch.as_tensor([enc["price"]], device=dev, dtype=torch.float32)).float()
        self.item_emb = torch.cat([self.item_emb, e]) if self.item_emb is not None else e
        return int(self.item_emb.shape[0] - 1)

    # ------------------------------------------------------------------ training
    def fit(self, train: Dict[str, np.ndarray], val: Dict[str, np.ndarray], log=None) -> List[Dict[str, float]]:
        c, dev = self.tt_cfg, self.device
        use_amp = amp_enabled(dev, bool(self.cfg["hardware"]["use_amp"]))
        self.amp_used = use_amp
        scaler = make_grad_scaler(dev, use_amp)
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(c["lr"]), weight_decay=float(c["weight_decay"]))
        n_items = len(self.pre.item_ids)
        cnt = np.bincount(train["item_pos"], minlength=n_items).astype(np.float64) + 1.0
        logq = torch.as_tensor(np.log(cnt / cnt.sum()), dtype=torch.float32, device=dev)
        tau = float(c["temperature"])
        T = {k: torch.as_tensor(v, device=dev) for k, v in train.items()}
        V = {k: torch.as_tensor(v, device=dev) for k, v in val.items()}
        items_all = self._item_tensors()
        n, bs = len(T["item_pos"]), int(c["batch_size"])
        g = torch.Generator(device="cpu").manual_seed(int(self.cfg["seed"]))
        best, best_state, bad = -1.0, None, 0
        p_u, p_i = float(c["id_dropout_user"]), float(c["id_dropout_item"])

        def forward(B: Dict[str, torch.Tensor], idx: torch.Tensor, train_mode: bool):
            uid, hist, hist_n, onboard = B["uid"][idx], B["hist"][idx], B["hist_n"][idx], B["onboard"][idx]
            ipos = B["item_pos"][idx]
            iid = items_all["iid"][ipos]
            if train_mode:  # ID + modality dropout -> network must work from side-features alone
                m = lambda p: (torch.rand(len(idx), device=dev) < p)
                uid = torch.where(m(p_u), torch.zeros_like(uid), uid)
                iid = torch.where(m(p_i), torch.zeros_like(iid), iid)
                dh = m(0.35)
                hist = torch.where(dh[:, None], torch.zeros_like(hist), hist)
                hist_n = torch.where(dh, torch.zeros_like(hist_n), hist_n)
                onboard = torch.where(m(0.15)[:, None], torch.zeros_like(onboard), onboard)
            with autocast_ctx(dev, use_amp):
                u = self.model.user_forward(uid, B["ucats"][idx], onboard, B["ctx"][idx], hist, hist_n)
                v = self.model.item_forward(iid, items_all["cats"][ipos], items_all["text"][ipos], items_all["price"][ipos])
            logits = (u.float() @ v.float().T) / tau - logq[ipos][None, :]    # logQ correction (sampling bias)
            same = ipos[:, None] == ipos[None, :]                             # mask duplicate-item false negatives
            logits = logits.masked_fill(same & ~torch.eye(len(idx), dtype=torch.bool, device=dev), -1e9)
            return F.cross_entropy(logits, torch.arange(len(idx), device=dev))

        for epoch in range(1, int(c["epochs"]) + 1):
            self.model.train()
            t0, perm, losses = time.perf_counter(), torch.randperm(n, generator=g).to(dev), []
            for s in range(0, n - bs + 1, bs):
                opt.zero_grad(set_to_none=True)
                loss = forward(T, perm[s:s + bs], True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
            self.model.eval()
            with torch.no_grad():
                vl = float(np.mean([forward(V, torch.arange(s, min(s + bs, len(V["item_pos"])), device=dev), False).item()
                                    for s in range(0, len(V["item_pos"]), bs) if min(s + bs, len(V["item_pos"])) - s > 8]))
                rec = self._val_recall(V, items_all, 10)
            row = dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=vl, val_recall_at_10=rec,
                       seconds=round(time.perf_counter() - t0, 2))
            self.history.append(row)
            if log:
                log(f"  epoch {epoch:02d} train_loss={row['train_loss']:.4f} val_loss={vl:.4f} val_R@10={rec:.4f} "
                    f"({row['seconds']}s, amp={use_amp})")
            if rec > best:
                best, bad = rec, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= int(c["patience"]):
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.refresh_item_embeddings()
        return self.history

    @torch.no_grad()
    def _val_recall(self, V: Dict[str, torch.Tensor], items_all: Dict[str, torch.Tensor], k: int) -> float:
        """Recall@k of the held-out item among all warm items (user tower at inference: no dropout)."""
        emb = self.refresh_item_embeddings()
        n = min(len(V["item_pos"]), 6000)
        idx = torch.arange(n, device=self.device)
        u = self.model.user_forward(V["uid"][idx], V["ucats"][idx], V["onboard"][idx], V["ctx"][idx], V["hist"][idx],
                                    V["hist_n"][idx]).float()
        scores = u @ emb.T
        scores[:, torch.as_tensor(self.pre.is_cold_item, device=self.device)] = -1e9
        top = scores.topk(k, dim=1).indices
        return float((top == V["item_pos"][idx][:, None]).any(1).float().mean().item())

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def user_embedding(self, user_row: Dict[str, Any], ctx: Dict[str, Any], history: List[int],
                       drop_id: bool = False) -> torch.Tensor:
        """Embed one request. ``user_row`` may be sparse (anonymous users) - missing fields map to UNK."""
        self.model.eval()
        p, dev = self.pre, self.device
        enc = p.encode_user_row(user_row)
        hv, hn = p.history_vec(history)
        cx = p.encode_ctx(ctx.get("channel"), ctx.get("season"), ctx.get("hour"))
        uid = 0 if drop_id else enc["uid"]
        return self.model.user_forward(
            torch.tensor([uid], device=dev), torch.as_tensor(enc["cats"][None], device=dev),
            torch.as_tensor(enc["onboard"][None], device=dev), torch.as_tensor(cx[None], device=dev),
            torch.as_tensor(hv[None], device=dev), torch.tensor([hn], device=dev, dtype=torch.float32)).float()

    @torch.no_grad()
    def score_all(self, user_row: Dict[str, Any], ctx: Dict[str, Any], history: List[int]) -> np.ndarray:
        u = self.user_embedding(user_row, ctx, history)
        return (u @ self.item_emb.T)[0].cpu().numpy()

    # ----------------------------------------------------------------------- IO
    def save(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.model.state_dict(), "history": self.history}, Path(directory) / "two_tower.pt")

    def load(self, directory: str | Path) -> "TwoTowerRecommender":
        ck = torch.load(Path(directory) / "two_tower.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["state"])
        self.history = ck.get("history", [])
        self.refresh_item_embeddings()
        return self
