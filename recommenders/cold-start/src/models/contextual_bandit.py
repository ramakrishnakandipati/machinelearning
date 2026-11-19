"""Strategy 4 - contextual bandit (LinUCB / Thompson Sampling) for exploration on new users and items.

Arms are catalog sub-categories (shared by every SKU in them, so brand-new SKUs inherit category priors
immediately). A per-item exposure bonus ``alpha_item / sqrt(1 + impressions)`` pushes under-exposed (new)
SKUs. Rewards are updated online through ``update`` (used by the ``/feedback`` endpoint).
"""
from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.data.preprocessor import Preprocessor

REWARD = {"view": 0.0, "skip": 0.0, "click": 0.4, "cart": 0.7, "purchase": 1.0}


class ContextualBandit:
    """Disjoint LinUCB with Sherman-Morrison inverse updates (O(d^2) per feedback event)."""

    def __init__(self, pre: Preprocessor, cfg: Dict[str, Any], seed: int = 42):
        b = cfg["bandit"]
        self.pre = pre
        self.alpha, self.ridge, self.policy = float(b["alpha"]), float(b["ridge"]), str(b["policy"])
        self.item_alpha = 0.5 * self.alpha
        self.arms = list(pre.v_sub.itos)            # includes UNK arm 0
        self.arm_idx = {a: i for i, a in enumerate(self.arms)}
        self.d = (len(pre.v_region) + len(pre.v_device) + len(pre.v_acq) + len(pre.v_age) + pre.n_sub
                  + len(pre.v_channel) + len(pre.v_season) + len(pre.v_hour) + 1)
        self.A_inv = np.stack([np.eye(self.d) / self.ridge for _ in self.arms])
        self.b = np.zeros((len(self.arms), self.d))
        self.n_updates = np.zeros(len(self.arms), dtype=np.int64)
        self.impressions = np.zeros(len(pre.item_ids), dtype=np.float64)
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self.total_updates = 0

    # ------------------------------------------------------------- context
    def context_vector(self, user_row: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
        p = self.pre
        enc = p.encode_user_row(user_row)
        cx = p.encode_ctx((ctx or {}).get("channel"), (ctx or {}).get("season"), (ctx or {}).get("hour"))
        sizes = [len(p.v_region), len(p.v_device), len(p.v_acq), len(p.v_age)]
        parts = [np.eye(s)[i] for s, i in zip(sizes, enc["cats"])]
        parts.append(enc["onboard"])
        parts += [np.eye(s)[i] for s, i in zip([len(p.v_channel), len(p.v_season), len(p.v_hour)], cx)]
        parts.append(np.ones(1))
        x = np.concatenate(parts).astype(np.float64)
        return x / max(np.linalg.norm(x), 1e-9)

    # -------------------------------------------------------------- scoring
    def arm_scores(self, x: np.ndarray, explore: bool = True) -> np.ndarray:
        """Per-arm expected reward (+ UCB bonus, or a Thompson sample)."""
        with self._lock:
            theta = np.einsum("aij,aj->ai", self.A_inv, self.b)
            mean = theta @ x
            var = np.einsum("i,aij,j->a", x, self.A_inv, x)
            if not explore:
                return mean
            if self.policy == "thompson":
                z = self._rng.standard_normal(len(self.arms))
                return mean + self.alpha * np.sqrt(np.maximum(var, 0)) * z
            return mean + self.alpha * np.sqrt(np.maximum(var, 0))

    def score_items(self, x: np.ndarray, item_subcat_idx: np.ndarray, explore: bool = True) -> np.ndarray:
        """Score all items: arm score + per-item exposure bonus (new SKUs have fewer impressions)."""
        arm = self.arm_scores(x, explore)[item_subcat_idx]
        imp = self.impressions
        if len(imp) < len(item_subcat_idx):
            imp = np.pad(imp, (0, len(item_subcat_idx) - len(imp)))
        bonus = self.item_alpha / np.sqrt(1.0 + imp[: len(item_subcat_idx)]) if explore else 0.0
        return arm + bonus

    # --------------------------------------------------------------- update
    def update(self, x: np.ndarray, subcat_idx: int, reward: float, item_pos: Optional[int] = None) -> None:
        with self._lock:
            Ai = self.A_inv[subcat_idx]
            Ax = Ai @ x
            self.A_inv[subcat_idx] = Ai - np.outer(Ax, Ax) / (1.0 + x @ Ax)
            self.b[subcat_idx] += reward * x
            self.n_updates[subcat_idx] += 1
            self.total_updates += 1
            if item_pos is not None:
                if item_pos >= len(self.impressions):
                    self.impressions = np.pad(self.impressions, (0, item_pos + 1 - len(self.impressions)))
                self.impressions[item_pos] += 1

    def record_impressions(self, item_positions: Sequence[int]) -> None:
        with self._lock:
            for i in item_positions:
                if i >= len(self.impressions):
                    self.impressions = np.pad(self.impressions, (0, i + 1 - len(self.impressions)))
                self.impressions[i] += 1

    def fit_replay(self, train_rows, users_df, items_subcat: np.ndarray, max_rows: int = 60000) -> int:
        """Warm-start the category priors by replaying logged (context, arm, reward) tuples."""
        df = train_rows.sample(n=min(max_rows, len(train_rows)), random_state=0)
        urows = users_df.set_index("user_id").to_dict("index")
        n = 0
        for r in df.itertuples():
            u = dict(urows[r.user_id]); u["user_id"] = r.user_id
            x = self.context_vector(u, dict(channel=r.channel, season=r.season, hour=r.hour))
            ip = self.pre.item_pos[r.item_id]
            self.update(x, int(items_subcat[ip]), REWARD.get(r.event, 0.0))
            n += 1
        self.impressions[:] = 0   # replay should not mark items as "explored"
        return n

    # ------------------------------------------------------------------- IO
    def save(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        with self._lock, open(Path(directory) / "bandit.pkl", "wb") as f:
            pickle.dump({k: v for k, v in self.__dict__.items() if k not in ("_lock", "pre", "_rng")}, f)

    def load(self, directory: str | Path) -> "ContextualBandit":
        with open(Path(directory) / "bandit.pkl", "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self
