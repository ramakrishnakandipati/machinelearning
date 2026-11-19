"""Strategy 1 - system / zero-history cold-start: contextual, Bayesian-smoothed, time-decayed popularity."""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data.generator import hour_bucket

EVENT_WEIGHT = {"view": 0.1, "click": 1.0, "cart": 2.0, "purchase": 3.0}
SEGMENT_DIMS = ["region", "device", "hour_bucket", "season", "acq_channel", "channel"]


def wilson_lower_bound(successes: np.ndarray, n: np.ndarray, z: float = 1.64) -> np.ndarray:
    """Wilson score lower confidence bound of a Bernoulli rate (vectorised, zero-safe)."""
    n = np.maximum(n, 1e-9)
    p = np.clip(successes / n, 0, 1)
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return np.where(n > 1e-6, (centre - margin) / denom, 0.0)


class ContextualPopularity:
    """Segment popularity (region, device, time-of-day, season, acquisition channel, touchpoint).

    Score_i(segment) = Bayesian-dampened share (shrunk towards the global share with ``m`` pseudo-events)
    x (0.7 + 0.3 * Wilson conversion bound), blended with a *trending* score from a short recent window.
    Every event is exponentially time-decayed so that fresh items can surface.
    """

    def __init__(self, cfg: Dict[str, Any]):
        p = cfg["popularity"]
        self.half_life = float(p["half_life_days"])
        self.m = float(p["bayes_prior_strength"])
        self.min_support = float(p["min_segment_support"])
        self.trend_days = float(p["trending_window_days"])
        self.trend_w = float(p["trending_weight"])
        self.n_items = 0
        self.seg_share: Dict[tuple, np.ndarray] = {}
        self.seg_support: Dict[tuple, float] = {}
        self.global_score = np.zeros(0)
        self._live = np.zeros(0)
        self._lock = threading.Lock()

    def __getstate__(self) -> Dict[str, Any]:
        s = self.__dict__.copy()
        s.pop("_lock", None)
        return s

    def __setstate__(self, s: Dict[str, Any]) -> None:
        self.__dict__.update(s)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ fit
    def fit(self, inter: pd.DataFrame, users: pd.DataFrame, item_pos: Dict[str, int], now_day: float) -> "ContextualPopularity":
        n = len(item_pos)
        self.n_items = n
        df = inter[["user_id", "item_id", "event", "day", "hour", "channel", "season"]].copy()
        df = df.merge(users[["user_id", "region", "device", "acq_channel"]], on="user_id", how="left")
        df["hour_bucket"] = df.hour.map(hour_bucket)
        df["ipos"] = df.item_id.map(item_pos)
        decay = 0.5 ** ((now_day - df.day.values) / self.half_life)
        df["w"] = df.event.map(EVENT_WEIGHT).values * decay
        df["buy"] = (df.event == "purchase").values * decay
        df["recent"] = (now_day - df.day.values <= self.trend_days).astype(float) * df.w
        df["ev"] = decay

        def vec(frame: pd.DataFrame, col: str) -> np.ndarray:
            return np.bincount(frame.ipos.values, weights=frame[col].values, minlength=n).astype(np.float64)

        g_w, g_buy, g_ev, g_recent = vec(df, "w"), vec(df, "buy"), vec(df, "ev"), vec(df, "recent")
        self.g_share = g_w / max(g_w.sum(), 1e-9)
        self.g_recent_share = g_recent / max(g_recent.sum(), 1e-9)
        wil = wilson_lower_bound(g_buy, g_ev)
        self.conv = wil / max(wil.max(), 1e-9)
        self.global_score = self._combine(self.g_share, self.g_recent_share, self.conv)
        for dim in SEGMENT_DIMS:
            for val, sub in df.groupby(dim):
                w = vec(sub, "w")
                tot = w.sum()
                if tot < self.min_support * 0.1:
                    continue
                share = (w + self.m * self.g_share) / (tot + self.m)   # Bayesian dampening toward global
                rec = vec(sub, "recent")
                rshare = (rec + self.m * self.g_recent_share) / (rec.sum() + self.m)
                self.seg_share[(dim, str(val))] = self._combine(share, rshare, self.conv)
                self.seg_support[(dim, str(val))] = float(tot)
        self._live = np.zeros(n)
        return self

    def _combine(self, share: np.ndarray, recent: np.ndarray, conv: np.ndarray) -> np.ndarray:
        base = share / max(share.max(), 1e-12) * (0.7 + 0.3 * conv)
        trend = recent / max(recent.max(), 1e-12)
        return (1 - self.trend_w) * base + self.trend_w * trend

    # --------------------------------------------------------------- scoring
    def score_all(self, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Score every item for a context dict (missing keys back off to the global ranking)."""
        ctx = dict(ctx or {})
        if "hour" in ctx and ctx.get("hour") is not None and "hour_bucket" not in ctx:
            ctx["hour_bucket"] = hour_bucket(int(ctx["hour"]))
        acc, wsum = self.global_score.copy(), 1.0
        for dim in SEGMENT_DIMS:
            key = (dim, str(ctx.get(dim)))
            if ctx.get(dim) is not None and key in self.seg_share:
                sup = self.seg_support[key]
                wt = min(1.0, sup / max(self.min_support, 1.0)) * 1.0
                s = self.seg_share[key]
                if len(s) < len(acc):
                    s = np.pad(s, (0, len(acc) - len(s)))
                acc[: len(s)] += wt * s[: len(acc)]
                wsum += wt
        out = acc / wsum
        live = self._live
        if live.size and live.max() > 0:
            out = out + 0.1 * live / live.max()
        return out

    def observe(self, item_pos: int, weight: float = 1.0) -> None:
        """Online feedback: bump a live trending counter (new items surface quickly)."""
        with self._lock:
            if item_pos >= len(self._live):
                self._live = np.pad(self._live, (0, item_pos + 1 - len(self._live)))
            self._live[item_pos] += weight

    def grow(self, new_n: int) -> None:
        """Extend per-item arrays for items onboarded after fit (zero history)."""
        with self._lock:
            pad = new_n - len(self.global_score)
            if pad > 0:
                self.global_score = np.pad(self.global_score, (0, pad))
                self._live = np.pad(self._live, (0, max(0, new_n - len(self._live))))
                self.n_items = new_n
