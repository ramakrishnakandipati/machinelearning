"""Mutable per-user shopping state used for filtering, novelty and guardrails."""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

TIER_NAMES = ["previously_bought", "new_sku", "unseen_brand", "unseen_subcategory"]


class UserState:
    def __init__(self, world, today: int):
        self.n_subcats, self.n_brands = world.n_subcats, world.n_brands
        U = world.n_users
        self.items_subcat = world.items["subcat"].values
        self.items_brand = world.items["brand"].values
        self.seen_sub = np.zeros((U, self.n_subcats), dtype=bool)       # interacted (click/buy)
        self.seen_brand = np.zeros((U, self.n_brands), dtype=bool)
        self.last_buy: List[Dict[int, int]] = [dict() for _ in range(U)]  # item -> last purchase day
        self.impressions: List[List[Tuple[int, int, int]]] = [[] for _ in range(U)]  # (day, item, clicked)
        self.today = today

        df = world.interactions
        pos = df[df.click == 1]
        self.seen_sub[pos["user"].values, self.items_subcat[pos["item"].values]] = True
        self.seen_brand[pos["user"].values, self.items_brand[pos["item"].values]] = True
        for u, i, dday in df[df.buy == 1][["user", "item", "day"]].itertuples(index=False):
            prev = self.last_buy[u].get(i, -1)
            if dday > prev:
                self.last_buy[u][i] = int(dday)
        recent = df[df.day >= today - 14]
        for u, i, dday, c in recent[["user", "item", "day", "click"]].itertuples(index=False):
            self.impressions[u].append((int(dday), int(i), int(c)))

    def clone(self) -> "UserState":
        return copy.deepcopy(self)

    # -- queries ----------------------------------------------------------------------------
    def recently_bought(self, u: int, items: np.ndarray, day: int, window: int) -> np.ndarray:
        lb = self.last_buy[u]
        return np.array([(i in lb) and (day - lb[i] <= window) for i in items.tolist()], dtype=bool)

    def unclicked_counts(self, u: int, items: np.ndarray, day: int, window: int) -> np.ndarray:
        counts: Dict[int, int] = {}
        for dday, it, c in self.impressions[u]:
            if c == 0 and day - dday <= window:
                counts[it] = counts.get(it, 0) + 1
        return np.array([counts.get(i, 0) for i in items.tolist()], dtype=np.int16)

    def novelty_tier(self, u: int, items: np.ndarray) -> np.ndarray:
        """3 = unseen subcategory, 2 = unseen brand, 1 = new SKU (known subcat+brand), 0 = previously bought."""
        lb = self.last_buy[u]
        bought = np.array([i in lb for i in items.tolist()], dtype=bool)
        unseen_sub = ~self.seen_sub[u, self.items_subcat[items]]
        unseen_brand = ~self.seen_brand[u, self.items_brand[items]]
        tier = np.ones(len(items), dtype=np.int8)
        tier[unseen_brand] = 2
        tier[unseen_sub] = 3
        tier[bought] = 0
        return tier

    def unseen_subcat(self, u: int, items: np.ndarray) -> np.ndarray:
        return ~self.seen_sub[u, self.items_subcat[items]]

    # -- updates ----------------------------------------------------------------------------
    def record(self, u: int, items: np.ndarray, clicks: np.ndarray, buys: np.ndarray, day: int) -> None:
        for i, c, b in zip(items.tolist(), clicks.tolist(), buys.tolist()):
            self.impressions[u].append((day, i, int(c)))
            if c:
                self.seen_sub[u, self.items_subcat[i]] = True
                self.seen_brand[u, self.items_brand[i]] = True
            if b:
                self.last_buy[u][i] = day
        # keep the list bounded (only the last 14 days matter)
        if len(self.impressions[u]) > 400:
            self.impressions[u] = [t for t in self.impressions[u] if day - t[0] <= 14]
        self.today = max(self.today, day)


def novelty_tier_pairs(state: "UserState", u_arr: np.ndarray, i_arr: np.ndarray) -> np.ndarray:
    """Vectorised novelty tier for arbitrary (user, item) pairs."""
    bought = np.fromiter((i in state.last_buy[u] for u, i in zip(u_arr.tolist(), i_arr.tolist())), dtype=bool, count=len(u_arr))
    unseen_sub = ~state.seen_sub[u_arr, state.items_subcat[i_arr]]
    unseen_brand = ~state.seen_brand[u_arr, state.items_brand[i_arr]]
    tier = np.ones(len(u_arr), dtype=np.int8)
    tier[unseen_brand] = 2
    tier[unseen_sub] = 3
    tier[bought] = 0
    return tier
