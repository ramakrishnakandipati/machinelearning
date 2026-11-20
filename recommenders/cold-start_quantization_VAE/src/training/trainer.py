"""Training loops (AMP w/ CPU fallback), early stopping and JSON/CSV loggers for the VQ-/RQ-VAE tokenizers."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.config import DEVICE, LOGS_DIR, USE_AMP, QuantConfig
from src.training.evaluator import codebook_stats


@torch.no_grad()
def _eval_split(model, X: torch.Tensor) -> Dict[str, float]:
    model.eval()
    out = model(X)
    return {"recon": float(out["recon_loss"]), "commit": float(out["commit_loss"]), "loss": float(out["loss"]),
            "perplexity": float(out["perplexity"]), "level_perplexity": [float(v) for v in out["level_perplexity"]]}


@torch.no_grad()
def encode_all(model, X: torch.Tensor, bs: int = 2048) -> np.ndarray:
    model.eval()
    return torch.cat([model.encode_to_codes(X[i:i + bs]) for i in range(0, len(X), bs)]).cpu().numpy()


def train_quantizer(model, X_train: np.ndarray, X_val: np.ndarray, X_cold: np.ndarray, cfg: QuantConfig, name: str,
                    verbose: bool = True) -> Dict[str, list]:
    """Returns {'train': [...], 'val': [...], 'final': {...}}; restores the best-val checkpoint."""
    model.to(DEVICE)
    Xt = torch.tensor(X_train, device=DEVICE)
    Xv = torch.tensor(X_val, device=DEVICE)
    Xc = torch.tensor(X_cold, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    train_log, val_log = [], []
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        agg = {"recon": 0.0, "commit": 0.0, "loss": 0.0, "perplexity": 0.0}
        lvl_ppl = None
        nb = 0
        for i in range(0, len(Xt), cfg.batch_size):
            xb = Xt[perm[i:i + cfg.batch_size]]
            if len(xb) < 32:
                continue
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                out = model(xb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            agg["recon"] += float(out["recon_loss"].detach()); agg["commit"] += float(out["commit_loss"].detach())
            agg["loss"] += float(out["loss"].detach()); agg["perplexity"] += float(out["perplexity"].detach())
            lp = out["level_perplexity"].detach().cpu().numpy()
            lvl_ppl = lp if lvl_ppl is None else lvl_ppl + lp
            nb += 1
        sched.step()
        rec = {k: v / nb for k, v in agg.items()}
        rec["level_perplexity"] = (lvl_ppl / nb).tolist()
        quants = [model.quantizer] if hasattr(model, "quantizer") else list(model.quantizers)
        rec["dead_code_resets"] = int(sum(int(q.n_resets) for q in quants))
        rec.update(epoch=epoch, lr=sched.get_last_lr()[0])
        train_log.append(rec)

        v = _eval_split(model, Xv)
        codes_tr = encode_all(model, Xt)
        st = codebook_stats(codes_tr, cfg.codebook_size)
        v.update(epoch=epoch, codebook_utilization_pct=st["mean_utilization_pct"],
                 level_utilization_pct=st["level_utilization_pct"], collision_rate=st["collision_rate"])
        val_log.append(v)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"[{name}] ep {epoch:3d} train recon {rec['recon']:.4f} commit {rec['commit']:.4f} "
                  f"ppl {rec['perplexity']:.1f} | val recon {v['recon']:.4f} util {v['codebook_utilization_pct']:.1f}% "
                  f"coll {v['collision_rate']:.3f}")
        if v["recon"] < best - 1e-5:
            best, bad = v["recon"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"[{name}] early stop at epoch {epoch} (best val recon {best:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    final = {"best_val_recon": best, "epochs_run": len(train_log), "train_seconds": time.time() - t0,
             "cold_item_recon": _eval_split(model, Xc)["recon"], "val_recon": _eval_split(model, Xv)["recon"]}
    return {"train": train_log, "val": val_log, "final": final}


def write_logs(logs: Dict[str, dict], extra: Optional[dict] = None, out_dir: Path = LOGS_DIR) -> None:
    """Write training_logs.json / validation_logs.json (+ per-epoch CSV for convenience)."""
    train = {k: v["train"] for k, v in logs.items()}
    val = {k: {"epochs": v["val"], "final": v["final"]} for k, v in logs.items()}
    if extra:
        val["tokenizer_summary"] = extra
    (out_dir / "training_logs.json").write_text(json.dumps(train, indent=2))
    (out_dir / "validation_logs.json").write_text(json.dumps(val, indent=2))
    rows = []
    for k, v in logs.items():
        for t, vv in zip(v["train"], v["val"]):
            rows.append({"model": k, "epoch": t["epoch"], "train_recon": t["recon"], "train_commit": t["commit"],
                         "train_perplexity": t["perplexity"], "val_recon": vv["recon"],
                         "val_utilization_pct": vv["codebook_utilization_pct"], "val_collision_rate": vv["collision_rate"]})
    pd.DataFrame(rows).to_csv(out_dir / "tokenizer_epoch_metrics.csv", index=False)
