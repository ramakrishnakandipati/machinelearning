"""Synthetic omnichannel retail data with a learnable latent-factor ground truth.

Users and items live in a shared latent space; users prefer a few categories. Interactions are sampled
from softmax(affinity * scale + popularity prior). Events are `view` (exposure) or a positive event
(`click`, `add_to_cart`, `purchase`) whose probability grows with affinity, so "viewed but not purchased"
items are genuinely harder negatives. A fraction of users/items only appear after the training window
(cold-start).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DataCfg, DATA, SEED
from .logger import get_logger, stage_timer

log = get_logger("data", "02_data_pipeline.log")

L1 = ["Apparel", "Footwear", "Grocery", "Beauty", "Home", "Electronics", "Sports", "Toys"]
TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
CHANNELS = ["Web", "Mobile", "In-Store"]
REGIONS = ["North", "South", "East", "West", "Central", "Islands"]
SEASONS = ["all_year", "spring_summer", "fall_winter", "holiday"]
DEVICES = ["desktop", "ios", "android", "kiosk"]
CHANNEL_DEVICE = {"Web": ["desktop"], "Mobile": ["ios", "android"], "In-Store": ["kiosk"]}
BASE_PRICE = [35, 60, 6, 18, 45, 250, 40, 22]
POSITIVE = ("click", "add_to_cart", "purchase")


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def generate(cfg: DataCfg | None = None, seed: int = SEED):
    cfg = cfg or DataCfg()
    rng = np.random.default_rng(seed)
    nU, nI, D = cfg.n_users, cfg.n_items, cfg.latent_dim

    # ---- items
    l1 = rng.choice(len(L1), nI, p=np.array([.18, .1, .2, .1, .12, .1, .1, .1]))
    l2 = l1 * 3 + rng.integers(0, 3, nI)
    brand = l1 * 5 + rng.integers(0, 5, nI)
    price = np.array(BASE_PRICE)[l1] * rng.lognormal(0, 0.45, nI)
    stock = rng.beta(8, 1.5, nI)
    stock[rng.random(nI) < 0.06] = 0.0
    items = pd.DataFrame({
        "item_id": np.arange(nI), "category_l1": np.array(L1)[l1], "category_l2": [f"{L1[a]}-{b % 3 + 1}" for a, b in zip(l1, l2)],
        "brand": [f"brand_{b:02d}" for b in brand], "seasonality_tag": rng.choice(SEASONS, nI, p=[.55, .15, .2, .1]),
        "price": price.round(2), "margin_score": rng.beta(2, 4, nI).round(4), "in_stock_ratio": stock.round(3)})
    centroid = _unit(rng.normal(size=(len(L1) * 3, D)))
    item_vec = _unit(centroid[l2] + 0.45 * rng.normal(size=(nI, D)))
    pop = (np.arange(1, nI + 1) ** -0.7)[rng.permutation(nI)]
    log_pop = np.log(pop / pop.max())
    cold_items = np.zeros(nI, bool)
    cold_items[rng.choice(nI, int(nI * cfg.cold_item_frac), replace=False)] = True

    # ---- users
    tier = rng.choice(TIERS, nU, p=[.5, .3, .15, .05])
    pref_ch = rng.choice(CHANNELS, nU, p=[.45, .35, .2])
    users = pd.DataFrame({"user_id": np.arange(nU), "loyalty_tier": tier, "preferred_channel": pref_ch,
                          "home_region": rng.choice(REGIONS, nU)})
    user_vec = np.empty((nU, D))
    for u in range(nU):
        cats = rng.choice(len(L1), rng.integers(1, 4), replace=False)
        l2s = np.concatenate([c * 3 + rng.choice(3, 2, replace=False) for c in cats])
        user_vec[u] = centroid[l2s].mean(0)
    user_vec = _unit(user_vec + 0.25 * rng.normal(size=(nU, D)))
    cold_users = np.zeros(nU, bool)
    cold_users[rng.choice(nU, int(nU * cfg.cold_user_frac), replace=False)] = True

    # ---- interactions
    act = rng.lognormal(0, 0.7, nU)
    n_ev = np.maximum(3, np.round(act / act.sum() * cfg.n_interactions)).astype(int)
    rows = []
    for u in range(nU):
        k = n_ev[u]
        lo = cfg.train_end if cold_users[u] else 0
        day = rng.integers(lo, cfg.n_days, k)
        hour = np.clip(np.round(rng.normal(15, 4.5, k)), 0, 23).astype(int)
        aff = item_vec @ user_vec[u]
        logits = 6.0 * aff + 0.6 * log_pop
        p_all = np.exp(logits - logits.max()); p_all /= p_all.sum()
        p_warm = np.where(cold_items, 0.0, p_all); p_warm /= p_warm.sum()
        pre = day < cfg.train_end
        it = np.where(pre, rng.choice(nI, k, p=p_warm), rng.choice(nI, k, p=p_all))
        p_pos = np.clip(1 / (1 + np.exp(-6 * (aff[it] - 0.35))), 0.08, 0.92)
        positive = rng.random(k) < p_pos
        et = np.where(positive, rng.choice(POSITIVE, k, p=[.6, .25, .15]), "view")
        stay = rng.random(k) < 0.75
        ch = np.where(stay, pref_ch[u], rng.choice(CHANNELS, k))
        rows.append(pd.DataFrame({"user_id": u, "item_id": it, "day": day, "hour": hour, "channel": ch, "event_type": et}))
    ev = pd.concat(rows, ignore_index=True)
    ev["dow"] = ev["day"] % 7
    ev["device_type"] = [rng.choice(CHANNEL_DEVICE[c]) for c in ev["channel"].to_numpy()]
    ev["ts"] = ev["day"] + ev["hour"] / 24.0
    ev = ev.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)
    users["is_cold_user"] = cold_users
    items["is_cold_item"] = cold_items
    return users, items, ev


def main(cfg: DataCfg | None = None):
    cfg = cfg or DataCfg()
    with stage_timer(log, "generate synthetic data"):
        users, items, ev = generate(cfg)
    users.to_parquet(DATA / "users.parquet", index=False)
    items.to_parquet(DATA / "items.parquet", index=False)
    ev.to_parquet(DATA / "events.parquet", index=False)
    log.info("users=%d items=%d events=%d positive_rate=%.3f purchases=%d", len(users), len(items), len(ev),
             (ev.event_type != "view").mean(), (ev.event_type == "purchase").sum())
    return users, items, ev


if __name__ == "__main__":
    main()
