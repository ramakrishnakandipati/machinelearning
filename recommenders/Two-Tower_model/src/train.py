"""GPU training: in-batch InfoNCE + logQ correction + hard negatives, AMP, early stopping, export."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import EMB_DIM, METRICS, MODELS, SEED, TrainCfg
from .logger import get_logger, stage_timer
from .model import TwoTower

log = get_logger("train", "03_model_training.log")
U_KEYS = ["uid", "tier", "chan", "region", "dense", "hist", "dow", "hour", "device"]
I_KEYS = ["iid", "cat1", "cat2", "brand", "season", "dense"]


def device_of() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_dev(d: dict, keys, dev) -> dict:
    return {k: torch.as_tensor(d[k]).to(dev) for k in keys}


def u_in(t: dict, idx) -> tuple:
    return tuple(t[k][idx] for k in U_KEYS)


def i_in(t: dict, idx) -> tuple:
    return tuple(t[k][idx] for k in I_KEYS)


def info_nce(u, v, tau, log_q, item_ids, v_hard=None, hard_ids=None, hard_valid=None, use_logq=True):
    """Softmax CE over in-batch items. logQ subtracts log sampling prob (popular items are over-represented as
    negatives); same-item collisions are masked; optional per-row hard negative appended as an extra column."""
    logits = (u @ v.T) / tau
    if use_logq:
        logits = logits - log_q[item_ids][None, :]
    same = item_ids[:, None] == item_ids[None, :]
    eye = torch.eye(len(item_ids), dtype=torch.bool, device=u.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    if v_hard is not None:
        hl = (u * v_hard).sum(1) / tau
        if use_logq:
            hl = hl - log_q[hard_ids.clamp(min=0)]
        logits = torch.cat([logits, hl.masked_fill(~hard_valid, float("-inf")).unsqueeze(1)], 1)
    return F.cross_entropy(logits, torch.arange(len(item_ids), device=u.device))


@torch.no_grad()
def all_item_emb(model, item_t: dict, bs: int = 8192) -> torch.Tensor:
    n = len(item_t["iid"])
    return torch.cat([model.item_tower(*i_in(item_t, slice(s, min(s + bs, n)))) for s in range(0, n, bs)])


@torch.no_grad()
def ranks_of_positive(model, split_t: dict, idx: torch.Tensor, item_emb: torch.Tensor, bs: int = 4096) -> torch.Tensor:
    out = []
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        sc = model.user_tower(*u_in(split_t, b)) @ item_emb.T
        pos = sc.gather(1, split_t["item"][b][:, None])
        out.append((sc > pos).sum(1))
    return torch.cat(out)


def metrics_from_ranks(r: torch.Tensor) -> dict:
    r = r.float()
    return {"recall_10": float((r < 10).float().mean()), "recall_50": float((r < 50).float().mean()),
            "ndcg_10": float(torch.where(r < 10, 1 / torch.log2(r + 2), torch.zeros_like(r)).mean()),
            "mrr_10": float(torch.where(r < 10, 1 / (r + 1), torch.zeros_like(r)).mean())}


def eval_loss(model, split_t, item_t, log_q, cfg, bs=4096) -> float:
    model.eval()
    n, tot, cnt = len(split_t["item"]), 0.0, 0
    g = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(n, generator=g).to(split_t["item"].device)
    with torch.no_grad():
        for s in range(0, n - bs + 1, bs):
            b = perm[s:s + bs]
            u, v = model.user_tower(*u_in(split_t, b)), model.item_tower(*i_in(item_t, split_t["item"][b]))
            tot += float(info_nce(u, v, model.tau, log_q, split_t["item"][b], use_logq=cfg.logq))
            cnt += 1
    return tot / max(cnt, 1)


def train(prep, cfg: TrainCfg | None = None):
    cfg = cfg or TrainCfg()
    dev = device_of()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log.info("device=%s amp=%s cfg=%s", dev, dev.type == "cuda", cfg)
    sizes = prep.pre.sizes()
    model = TwoTower(sizes, EMB_DIM, cfg.dropout, cfg.init_temperature).to(dev)
    log.info("params=%d sizes=%s", sum(p.numel() for p in model.parameters()), sizes)

    tr = to_dev(prep.splits["train"], U_KEYS + ["item", "hard_item"], dev)
    va = to_dev(prep.splits["val"], U_KEYS + ["item"], dev)
    item_t = to_dev(prep.item_table, I_KEYS, dev)
    log_q = torch.as_tensor(prep.log_q, dtype=torch.float32, device=dev)
    n = len(tr["item"])
    steps_per_epoch = n // cfg.batch_size
    total = steps_per_epoch * cfg.epochs
    warm = max(1, steps_per_epoch // 2)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))) * 0.98 + 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    g = torch.Generator(device="cpu").manual_seed(SEED)
    val_idx = torch.randperm(len(va["item"]), generator=g)[:20000].to(dev)
    best, best_state, best_epoch, bad, hist = -1.0, None, -1, 0, []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        run = 0.0
        for s in range(steps_per_epoch):
            b = perm[s * cfg.batch_size:(s + 1) * cfg.batch_size]
            items = tr["item"][b]
            with torch.amp.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                u = model.user_tower(*u_in(tr, b))
                v = model.item_tower(*i_in(item_t, items))
                vh = hid = hv = None
                if cfg.hard_neg:
                    hid = tr["hard_item"][b]
                    hv = hid >= 0
                    vh = model.item_tower(*i_in(item_t, hid.clamp(min=0)))
            loss = info_nce(u.float(), v.float(), model.tau, log_q, items, None if vh is None else vh.float(), hid, hv, cfg.logq)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss)
        model.eval()
        val_loss = eval_loss(model, va, item_t, log_q, cfg)
        m = metrics_from_ranks(ranks_of_positive(model, va, val_idx, all_item_emb(model, item_t)))
        rec = {"epoch": ep, "train_loss": run / steps_per_epoch, "val_loss": val_loss,
               **{f"val_{k}": v for k, v in m.items()}, "lr": opt.param_groups[0]["lr"], "temperature": float(model.tau),
               "amp_scale": float(scaler.get_scale()) if dev.type == "cuda" else None,
               "epoch_duration_sec": time.perf_counter() - t0,
               "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0}
        hist.append(rec)
        log.info("epoch %02d | train_loss=%.4f val_loss=%.4f R@10=%.4f R@50=%.4f NDCG@10=%.4f MRR@10=%.4f | lr=%.2e tau=%.4f "
                 "amp_scale=%s | %.1fs peak_vram=%.0fMB", ep, rec["train_loss"], val_loss, m["recall_10"], m["recall_50"],
                 m["ndcg_10"], m["mrr_10"], rec["lr"], rec["temperature"], rec["amp_scale"], rec["epoch_duration_sec"],
                 rec["peak_gpu_memory_mb"])
        if m["recall_50"] > best:
            best, best_epoch, bad = m["recall_50"], ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "sizes": sizes, "epoch": ep, "cfg": cfg.__dict__},
                       MODELS / "two_tower_best.pt")
            log.info("checkpoint saved (best val Recall@50=%.4f @ epoch %d)", best, ep)
        else:
            bad += 1
            if bad >= cfg.patience:
                log.info("early stopping at epoch %d (best epoch %d)", ep, best_epoch)
                break
    model.load_state_dict(best_state)
    model.eval()
    df = pd.DataFrame(hist)
    df["best_epoch"] = best_epoch
    df.to_csv(METRICS / "training_history.csv", index=False)
    (METRICS / "training_history.json").write_text(json.dumps({"best_epoch": best_epoch, "history": hist}, indent=2))
    return model, hist, best_epoch


def export(model, prep, dev):
    """item_embeddings.npy, TorchScript user tower, per-channel segment embeddings for cold-start."""
    item_t = to_dev(prep.item_table, I_KEYS, dev)
    emb = all_item_emb(model, item_t).cpu().numpy().astype(np.float32)
    np.save(MODELS / "item_embeddings.npy", emb)

    cpu_tower = model.user_tower.to("cpu").eval()
    ex = {k: torch.as_tensor(v[:4]) for k, v in prep.splits["val"].items() if k in U_KEYS}
    traced = torch.jit.trace(cpu_tower, tuple(ex[k] for k in U_KEYS), check_trace=False)
    big = {k: torch.as_tensor(v[:37]) for k, v in prep.splits["val"].items() if k in U_KEYS}   # different batch size
    with torch.no_grad():
        diff = (traced(*[big[k] for k in U_KEYS]) - cpu_tower(*[big[k] for k in U_KEYS])).abs().max().item()
    assert diff < 1e-4, f"TorchScript mismatch {diff}"
    traced.save(str(MODELS / "user_tower_serving.pt"))
    log.info("exported item_embeddings %s, user_tower_serving.pt (max diff vs eager %.2e)", emb.shape, diff)
    model.user_tower.to(dev)

    # segment embeddings: mean user embedding of warm users per preferred channel (cold-start fallback)
    from .data_generator import CHANNEL_DEVICE
    snap = prep.snapshot
    n = len(snap)
    enc = prep.pre.encode_users(snap.user_id, snap.loyalty_tier, snap.preferred_channel, snap.home_region,
                                snap[["recency_days", "frequency_90d", "monetary_value_90d", "avg_basket_size"]].to_numpy(),
                                np.stack(snap.recent_item_history.to_numpy()), np.full(n, 3), np.full(n, 14),
                                [CHANNEL_DEVICE[c][0] for c in snap.preferred_channel])
    with torch.no_grad():
        ue = model.user_tower(*[torch.as_tensor(enc[k]).to(dev) for k in U_KEYS]).cpu().numpy()
    seg = {"global": ue.mean(0)}
    for c in snap.preferred_channel.unique():
        seg[c] = ue[(snap.preferred_channel == c).to_numpy()].mean(0)
    seg = {k: (v / np.linalg.norm(v)).astype(np.float32) for k, v in seg.items()}
    np.savez(MODELS / "segment_embeddings.npz", **seg)
    return emb


def main(prep=None):
    from . import dataset
    prep = prep or dataset.main()
    with stage_timer(log, "training"):
        model, hist, best_epoch = train(prep)
    with stage_timer(log, "export"):
        export(model, prep, device_of())
    return prep, model, hist, best_epoch


if __name__ == "__main__":
    main()
