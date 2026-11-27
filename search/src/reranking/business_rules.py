"""Hard business filters (stock, regional/BOPIS availability, brand/price constraints) and soft penalties."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Constraints:
    store_id: str | None = None
    fulfillment: str = "any"            # any | ship | bopis
    brand_allow: list = field(default_factory=list)
    brand_deny: list = field(default_factory=list)
    min_price: float = 0.0
    max_price: float = 0.0
    in_stock_only: bool = True


def availability(world, idx: np.ndarray, store_id: str | None, fulfillment: str, bopis_max_km: float):
    """Return boolean fulfillable mask + per-candidate fulfillment route ('ship', 'bopis:S1', '')."""
    ship = world.stock["WH"][idx] > 0
    bopis = np.zeros(len(idx), dtype=bool)
    route = np.array([""] * len(idx), dtype=object)
    if store_id and store_id in world.stock:
        near = [s for s in world.store_ids if s != "WH" and world.distance_km(store_id, s) <= bopis_max_km]
        for s in near:
            ok = (world.stock[s][idx] > 0) & ~bopis
            route[ok] = f"bopis:{s}"
            bopis |= ok
    mask = {"ship": ship, "bopis": bopis}.get(fulfillment, ship | bopis)
    route = np.where(ship & (fulfillment != "bopis"), "ship", route)
    return mask, route


def apply_hard_filters(world, idx: np.ndarray, c: Constraints, parsed_max_price: float, cfg: dict):
    """Returns keep mask and a reason-count dict explaining what was dropped."""
    keep = np.ones(len(idx), dtype=bool)
    reasons: dict[str, int] = {}

    def drop(mask, name):
        nonlocal keep
        n = int((keep & mask).sum())
        if n:
            reasons[name] = reasons.get(name, 0) + n
        keep &= ~mask

    avail, route = availability(world, idx, c.store_id, c.fulfillment, cfg["business_rules"]["bopis_max_km"])
    if c.in_stock_only:
        drop(~avail, "out_of_stock_or_unavailable_in_region")
    price = world.price[idx]
    cap = c.max_price if c.max_price > 0 else parsed_max_price
    if cap and cap > 0:
        drop(price > cap, "over_price_constraint")
    if c.min_price > 0:
        drop(price < c.min_price, "under_min_price")
    if c.brand_allow:
        allow = {b.lower() for b in c.brand_allow}
        drop(np.array([world.brand[i].lower() not in allow for i in idx]), "brand_not_allowed")
    if c.brand_deny:
        deny = {b.lower() for b in c.brand_deny}
        drop(np.array([world.brand[i].lower() in deny for i in idx]), "brand_denied")
    return keep, reasons, route


def soft_penalties(world, idx: np.ndarray, user_idx: int, store_id: str | None, cfg: dict):
    """Multiplicative factor in (0,1]: high return-rate SKUs, low stock, recently bought durables."""
    br = cfg["business_rules"]
    f = np.ones(len(idx), dtype=np.float32)
    flags: list[list[str]] = [[] for _ in idx]
    rr = world.return_rate[idx]
    hi = rr > br["max_return_rate"]
    f[hi] *= 1.0 - np.minimum(0.7, br["return_penalty"] + 2.0 * (rr[hi] - br["max_return_rate"]))
    stock_here = world.stock["WH"][idx] + (world.stock[store_id][idx] if store_id in world.stock else 0)
    low = stock_here <= br["low_stock_threshold"]
    f[low] *= 1.0 - br["low_stock_penalty"]
    recent = np.zeros(len(idx), dtype=bool) if user_idx is None else np.array([world.subcat[i] in world.recent_sub[user_idx] for i in idx])
    f[recent] *= 0.8
    for j in np.where(hi)[0]:
        flags[j].append("high_return_rate")
    for j in np.where(low)[0]:
        flags[j].append("low_stock")
    for j in np.where(recent)[0]:
        flags[j].append("recently_purchased_durable")
    return f, flags
