"""PyTorch (CUDA) residual tabular MLP with entity embeddings, LayerNorm + SiLU + Dropout and AMP."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, log_loss


def suggest_nn(trial: optuna.Trial) -> dict:
    return dict(
        hidden=trial.suggest_categorical("hidden", [128, 256, 512]),
        n_blocks=trial.suggest_int("n_blocks", 1, 4),
        dropout=trial.suggest_float("dropout", 0.05, 0.4),
        lr=trial.suggest_float("lr", 3e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048]),
    )


class ResBlock(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, x):
        return x + self.net(x)


class TabNet(nn.Module):
    def __init__(self, cat_cards: list[int], n_num: int, hidden=256, n_blocks=2, dropout=0.15):
        super().__init__()
        self.embs = nn.ModuleList()
        emb_total = 0
        for card in cat_cards:  # +2: index 0 reserved for unknown/missing (-1 code)
            dim = int(min(16, max(2, round(1.6 * (card + 1) ** 0.56))))
            self.embs.append(nn.Embedding(card + 2, dim))
            emb_total += dim
        self.inp = nn.Linear(emb_total + n_num, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, xc, xn):
        parts = [e(xc[:, i]) for i, e in enumerate(self.embs)]
        x = torch.cat(parts + [xn], 1) if parts else xn
        return self.head(self.blocks(self.inp(x))).squeeze(-1)


class TorchTabularModel:
    name = "pytorch_nn"
    ext = "pt"

    def __init__(self, cat_cols: list[str], cat_cards: list[int], use_gpu: bool = True, max_epochs: int = 25,
                 patience: int = 5, seed: int = 42):
        self.cat_cols, self.cat_cards = list(cat_cols), list(cat_cards)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed
        self.model: TabNet | None = None
        self.params: dict = {}
        self.n_num = 0

    # --- tensors -------------------------------------------------------------------
    def _split(self, X):
        k = len(self.cat_cols)
        arr = X.to_numpy(np.float32) if hasattr(X, "to_numpy") else np.asarray(X, np.float32)
        xc = torch.as_tensor(arr[:, :k].astype(np.int64) + 1, device=self.device).clamp_(min=0)  # -1 -> 0
        xn = torch.as_tensor(np.nan_to_num(arr[:, k:], nan=0.0), device=self.device)
        return xc, xn

    @torch.no_grad()
    def _predict(self, xc, xn, bs=16384) -> np.ndarray:
        self.model.eval()
        out = []
        for i in range(0, len(xn), bs):
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                out.append(torch.sigmoid(self.model(xc[i:i + bs], xn[i:i + bs]).float()))
        return torch.cat(out).cpu().numpy()

    def fit(self, Xtr, ytr, Xva, yva, params, trial=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        p = dict(hidden=256, n_blocks=2, dropout=0.15, lr=1e-3, weight_decay=1e-4, batch_size=1024)
        p.update(params)
        self.params = p
        xc, xn = self._split(Xtr)
        vc, vn = self._split(Xva)
        y = torch.as_tensor(np.asarray(ytr, np.float32), device=self.device)
        yv = np.asarray(yva)
        self.n_num = xn.shape[1]
        self.model = TabNet(self.cat_cards, self.n_num, p["hidden"], p["n_blocks"], p["dropout"]).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        n, bs = len(y), p["batch_size"]
        steps = self.max_epochs * ((n + bs - 1) // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=steps, pct_start=0.2)
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        lossf = nn.BCEWithLogitsLoss()
        best, best_state, bad = -1.0, None, 0
        for ep in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                    logit = self.model(xc[idx], xn[idx])
                loss = lossf(logit.float(), y[idx])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            pv = self._predict(vc, vn)
            ap = average_precision_score(yv, pv)
            if trial is not None:
                trial.report(float(ap), ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if ap > best + 1e-5:
                best, bad, best_state = ap, 0, copy.deepcopy(self.model.state_dict())
                self.best_iteration = ep + 1
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if self.device.type == "cuda" else 0.0
        return self

    def predict_proba(self, X) -> np.ndarray:
        xc, xn = self._split(X)
        return self._predict(xc, xn)

    def save(self, path: Path):
        torch.save({"state": self.model.state_dict(), "params": self.params, "cat_cols": self.cat_cols,
                    "cat_cards": self.cat_cards, "n_num": self.n_num}, str(path))

    @classmethod
    def load(cls, path: Path, cat_cols=(), use_gpu=True):
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        o = cls(ck["cat_cols"], ck["cat_cards"], use_gpu)
        p = ck["params"]
        o.params, o.n_num = p, ck["n_num"]
        o.model = TabNet(o.cat_cards, o.n_num, p["hidden"], p["n_blocks"], p["dropout"]).to(o.device)
        o.model.load_state_dict(ck["state"])
        o.model.eval()
        return o
