"""Synthetic omnichannel (web / app / in-store POS) retail dataset generator.

The generator produces users, a catalog and an interaction log with *real signal* in
the side-features (region, device, acquisition channel, onboarding categories, item
text/brand/price) so that cold-start strategies have something genuine to learn.

Partitions (column ``split`` in the interaction table):
    train            warm users x warm items, before the cutoff day (fit data)
    val              warm users x warm items, the last days before the cutoff
    test_warm        warm users x warm items, after the cutoff
    test_cold_user   users with ZERO train interactions (new customers)
    test_cold_item   interactions on newly launched SKUs (ZERO train interactions)
    session_ctx      the 2-3 real-time clickstream events of anonymous session users
    test_session     the held-out follow-up events of those session users
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# department -> sub-category -> leaf product nouns (3-level hierarchy)
CATALOG: Dict[str, Dict[str, List[str]]] = {
    "Apparel": {
        "Outerwear": ["parka", "windbreaker", "trench coat", "puffer jacket"],
        "Tops": ["t-shirt", "blouse", "sweater", "hoodie"],
        "Denim": ["skinny jeans", "denim jacket", "wide-leg jeans", "denim shorts"],
        "Activewear": ["yoga leggings", "sports bra", "track jacket", "running shorts"],
    },
    "Footwear": {
        "Running Shoes": ["trail runner", "road racer", "stability trainer", "marathon flat"],
        "Boots": ["chelsea boot", "hiking boot", "combat boot", "rain boot"],
        "Sandals": ["slide sandal", "strappy sandal", "sport sandal", "flip flop"],
        "Casual Sneakers": ["canvas sneaker", "retro sneaker", "slip-on", "platform sneaker"],
    },
    "Electronics": {
        "Audio": ["wireless earbuds", "over-ear headphones", "bluetooth speaker", "soundbar"],
        "Smartphones": ["flagship phone", "budget phone", "phone case", "wireless charger"],
        "Computing": ["ultrabook laptop", "gaming mouse", "mechanical keyboard", "usb hub"],
        "Wearables": ["fitness tracker", "smartwatch", "sleep ring", "heart monitor"],
    },
    "Home": {
        "Kitchen": ["air fryer", "chef knife", "espresso machine", "cast iron pan"],
        "Bedding": ["duvet cover", "memory pillow", "cotton sheets", "weighted blanket"],
        "Decor": ["table lamp", "wall art", "scented candle", "area rug"],
        "Storage": ["shelf unit", "storage bin", "closet organizer", "shoe rack"],
    },
    "Beauty": {
        "Skincare": ["vitamin c serum", "daily moisturizer", "sunscreen", "retinol cream"],
        "Makeup": ["matte lipstick", "foundation", "mascara", "eyeshadow palette"],
        "Haircare": ["repair shampoo", "leave-in conditioner", "hair oil", "curl cream"],
        "Fragrance": ["eau de parfum", "body mist", "cologne", "perfume oil"],
    },
    "Sports": {
        "Fitness Gear": ["dumbbell set", "resistance bands", "yoga mat", "kettlebell"],
        "Outdoor": ["camping tent", "sleeping bag", "hiking backpack", "trekking poles"],
        "Cycling": ["road helmet", "bike lights", "cycling jersey", "bike lock"],
        "Team Sports": ["soccer ball", "basketball", "tennis racket", "baseball glove"],
    },
    "Grocery": {
        "Snacks": ["protein bar", "trail mix", "dark chocolate", "sea salt chips"],
        "Beverages": ["cold brew", "sparkling water", "green tea", "oat milk"],
        "Pantry": ["olive oil", "pasta", "basmati rice", "hot sauce"],
        "Organic": ["organic oats", "organic honey", "organic granola", "organic nut butter"],
    },
    "Toys": {
        "Board Games": ["strategy game", "party game", "puzzle 1000pc", "card game"],
        "Building Sets": ["brick set", "robot kit", "model train", "magnetic tiles"],
        "Outdoor Play": ["water blaster", "kite", "scooter", "trampoline"],
        "Plush": ["teddy bear", "plush dino", "bunny plush", "plush octopus"],
    },
}
DEPT_VOCAB: Dict[str, List[str]] = {
    "Apparel": ["fabric", "stitch", "fit", "wardrobe", "layer", "seasonal", "wash"],
    "Footwear": ["sole", "cushion", "grip", "stride", "lace", "arch", "traction"],
    "Electronics": ["battery", "bluetooth", "latency", "sync", "firmware", "charge", "app"],
    "Home": ["cozy", "household", "durable", "kitchen", "room", "easy-clean", "storage"],
    "Beauty": ["glow", "hydrating", "dermatologist", "formula", "scent", "skin", "finish"],
    "Sports": ["performance", "training", "endurance", "gear", "workout", "outdoor", "grip"],
    "Grocery": ["flavor", "fresh", "pantry", "snack", "ingredients", "nutrition", "taste"],
    "Toys": ["play", "kids", "creative", "family", "fun", "imagination", "safe"],
}
COLORS = ["black", "white", "navy", "olive", "red", "sand", "grey", "teal", "pink", "tan"]
MATERIALS = ["cotton", "recycled", "leather", "mesh", "wool", "steel", "silicone", "bamboo"]
STYLES = ["classic", "modern", "minimal", "bold", "vintage", "eco", "pro", "casual"]
REGIONS = ["north", "south", "east", "west", "central"]
DEVICES = ["ios", "android", "desktop", "mobile_web"]
ACQ_CHANNELS = ["organic_search", "paid_social", "email", "referral", "in_store_signup"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
SEASONS = ["winter", "spring", "summer", "autumn"]
WEATHER = {
    "winter": ["cold", "snowy", "rainy"],
    "spring": ["rainy", "mild", "sunny"],
    "summer": ["hot", "sunny", "mild"],
    "autumn": ["rainy", "mild", "cold"],
}
PRICE_TIERS = ["budget", "mid", "premium", "luxury"]
HOUR_BUCKETS = ["night", "morning", "afternoon", "evening"]
N_DAYS = 120
CUTOFF_DAY = 100
VAL_START_DAY = 90

EVENT_TYPES = ["view", "click", "cart", "purchase"]


def hour_bucket(hour: int) -> str:
    """Map hour-of-day to a coarse bucket."""
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def season_of_day(day: int) -> str:
    """Map a day index to a season."""
    return SEASONS[min(3, int(day) // (N_DAYS // 4))]


@dataclass
class RetailDataset:
    """Container for the generated tables and metadata."""

    users: pd.DataFrame
    items: pd.DataFrame
    interactions: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)


def _build_items(rng: np.random.Generator, n_items: int, pop_exp: float, cold_frac: float) -> pd.DataFrame:
    subcats = [(d, s, leaves) for d, subs in CATALOG.items() for s, leaves in subs.items()]
    brands_by_dept: Dict[str, List[str]] = {}
    brand_prefix = ["Nova", "Aria", "Terra", "Volt", "Luma", "Zen", "Apex", "Orbit", "Kite", "Maple",
                    "Atlas", "Ember", "Pixel", "Sol", "Fjord", "Crest", "Echo", "Halo", "Iris", "Jade"]
    brand_suffix = ["Co", "Labs", "Works", "Studio", "Goods", "&Sons"]
    for d in CATALOG:
        k = 6
        brands_by_dept[d] = [
            f"{rng.choice(brand_prefix)} {rng.choice(brand_suffix)}" + f" {d[:2].upper()}{j}" for j in range(k)
        ]
    base_price = {"Apparel": 45, "Footwear": 80, "Electronics": 150, "Home": 60, "Beauty": 28,
                  "Sports": 55, "Grocery": 9, "Toys": 24}
    rows = []
    per = np.full(len(subcats), n_items // len(subcats))
    per[: n_items - per.sum()] += 1
    for (dept, sub, leaves), cnt in zip(subcats, per):
        for _ in range(int(cnt)):
            leaf = str(rng.choice(leaves))
            brand = str(rng.choice(brands_by_dept[dept]))
            btier = int(brand[-1]) % 4
            price = float(np.round(base_price[dept] * np.exp(rng.normal(0, 0.35)) * (0.7 + 0.35 * btier), 2))
            color, mat, style = str(rng.choice(COLORS)), str(rng.choice(MATERIALS)), str(rng.choice(STYLES))
            words = rng.choice(DEPT_VOCAB[dept], size=3, replace=False)
            title = f"{brand.rsplit(' ', 1)[0]} {style.title()} {color.title()} {leaf.title()}"
            desc = (f"{style.capitalize()} {leaf} in {color} {mat}. Part of our {sub.lower()} range in "
                    f"{dept.lower()}. Known for {words[0]}, {words[1]} and {words[2]}.")
            rows.append(dict(title=title, description=desc, brand=brand, dept=dept, subcat=sub, leaf=leaf,
                             price=price, color=color, material=mat, style=style))
    items = pd.DataFrame(rows).sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    items.insert(0, "item_id", [f"SKU{i:05d}" for i in range(len(items))])
    items["price_tier"] = pd.qcut(items["price"].rank(method="first"), 4, labels=False).astype(int)
    # power-law popularity weights (shuffled across catalog)
    w = (np.arange(1, len(items) + 1) ** -pop_exp)
    rng.shuffle(w)
    items["pop_weight"] = w / w.sum() * len(items)
    items["conv_rate"] = rng.beta(2, 12, size=len(items)) + 0.02
    # launch day: warm items launched early; cold items launch late (after cutoff)
    n_cold = int(round(cold_frac * len(items)))
    cold_idx = rng.choice(len(items), size=n_cold, replace=False)
    launch = rng.integers(0, 80, size=len(items))
    launch[cold_idx] = rng.integers(CUTOFF_DAY, N_DAYS - 4, size=n_cold)
    items["launch_day"] = launch
    items["is_cold_item"] = False
    items.loc[cold_idx, "is_cold_item"] = True
    # newly launched SKUs get a mild novelty boost so they have enough test interactions
    items.loc[items.is_cold_item, "pop_weight"] *= 2.5
    return items


def generate_dataset(cfg: Dict[str, Any], seed: int = 42) -> RetailDataset:
    """Generate the full synthetic dataset according to ``cfg['data']``."""
    d = cfg["data"]
    rng = np.random.default_rng(seed)
    n_users, n_items, n_inter = int(d["n_users"]), int(d["n_items"]), int(d["n_interactions"])
    items = _build_items(rng, n_items, float(d["popularity_exponent"]), float(d["cold_item_frac"]))
    subcats = sorted(items["subcat"].unique())
    sub_idx = {s: i for i, s in enumerate(subcats)}
    sub_dept = {s: dept for dept, subs in CATALOG.items() for s in subs}
    depts = list(CATALOG)
    dept_idx = {dp: i for i, dp in enumerate(depts)}
    sc_dept = np.array([dept_idx[sub_dept[s]] for s in subcats])
    items_by_sub = [np.where(items["subcat"].values == s)[0] for s in subcats]
    launch = items["launch_day"].values
    price_tier = items["price_tier"].values
    brand_codes, brand_names = pd.factorize(items["brand"])
    pop_w = items["pop_weight"].values
    conv = items["conv_rate"].values

    # ---- context multipliers (dept level) --------------------------------------------
    def mult(n_rows: int) -> np.ndarray:
        return np.exp(rng.normal(0, 0.45, size=(n_rows, len(depts))))

    region_m, season_m = mult(len(REGIONS)), mult(len(SEASONS))
    hour_m, acq_m = mult(len(HOUR_BUCKETS)), mult(len(ACQ_CHANNELS))
    # hand-written seasonal structure to make seasonality real
    season_m[SEASONS.index("winter"), dept_idx["Apparel"]] *= 2.0
    season_m[SEASONS.index("summer"), dept_idx["Sports"]] *= 1.8
    season_m[SEASONS.index("summer"), dept_idx["Grocery"]] *= 1.3
    hour_m[HOUR_BUCKETS.index("evening"), dept_idx["Electronics"]] *= 1.5

    # ---- users ---------------------------------------------------------------------------
    n_cold_u = int(round(d["cold_user_frac"] * n_users))
    n_sess_u = int(round(d["session_user_frac"] * n_users))
    user_type = np.array(["warm"] * n_users, dtype=object)
    perm = rng.permutation(n_users)
    user_type[perm[:n_cold_u]] = "cold"
    user_type[perm[n_cold_u:n_cold_u + n_sess_u]] = "session"
    user_rows, tastes = [], []
    for u in range(n_users):
        fav_dept = rng.choice(len(depts), size=2, replace=False, p=None)
        taste = np.full(len(subcats), 0.03)
        fav_subs = []
        for fd in fav_dept:
            cand = np.where(sc_dept == fd)[0]
            pick = rng.choice(cand, size=2, replace=False)
            taste[pick] += rng.uniform(0.6, 1.0)
            fav_subs.extend(pick.tolist())
        taste /= taste.sum()
        onboard = [subcats[i] for i in fav_subs[: int(rng.integers(2, 4))]]
        if rng.random() < 0.2:  # noisy onboarding answer
            onboard[0] = str(rng.choice(subcats))
        user_rows.append(dict(
            user_id=f"U{u:05d}", region=str(rng.choice(REGIONS)), device=str(rng.choice(DEVICES, p=[.3, .3, .2, .2])),
            acq_channel=str(rng.choice(ACQ_CHANNELS, p=[.3, .25, .15, .15, .15])), age_band=str(rng.choice(AGE_BANDS)),
            price_pref=int(rng.integers(0, 4)), onboarding_categories=list(dict.fromkeys(onboard)),
            user_type=str(user_type[u])))
        tastes.append(taste)
    users = pd.DataFrame(user_rows)
    tastes_arr = np.vstack(tastes)

    # ---- interaction counts -----------------------------------------------------------------
    counts = np.zeros(n_users, dtype=int)
    cold_mask, sess_mask = user_type == "cold", user_type == "session"
    warm_mask = ~(cold_mask | sess_mask)
    counts[cold_mask] = rng.integers(5, 13, size=cold_mask.sum())
    counts[sess_mask] = rng.integers(5, 7, size=sess_mask.sum())   # 2-3 session clicks + 3 targets
    budget = max(n_inter - counts.sum(), warm_mask.sum())
    raw = np.minimum(rng.pareto(1.25, size=warm_mask.sum()) + 1.0, 60.0)
    counts[warm_mask] = np.maximum(1, np.round(raw / raw.sum() * budget)).astype(int)

    dev_channel = {"ios": "app", "android": "app", "desktop": "web", "mobile_web": "web"}
    recs: List[tuple] = []
    for u in range(n_users):
        ur = users.iloc[u]
        utype = user_type[u]
        ridx, aidx = REGIONS.index(ur.region), ACQ_CHANNELS.index(ur.acq_channel)
        brand_aff = np.exp(rng.normal(0, 0.5, size=len(brand_names)))
        pt = np.abs(price_tier - ur.price_pref)
        w_u = pop_w * brand_aff[brand_codes] * np.exp(-0.45 * pt)
        n_u = int(counts[u])
        if utype == "cold":
            days = rng.integers(CUTOFF_DAY, N_DAYS, size=n_u)
        elif utype == "session":
            days = np.full(n_u, rng.integers(CUTOFF_DAY, N_DAYS))
        else:
            days = rng.integers(0, N_DAYS, size=n_u)
        days = np.sort(days)
        hours = np.clip(np.round(np.where(rng.random(n_u) < .5, rng.normal(13, 3, n_u), rng.normal(20, 2, n_u))), 0, 23).astype(int)
        if utype == "session":
            hours = np.sort(hours)
        for j in range(n_u):
            day, hour = int(days[j]), int(hours[j])
            s = season_of_day(day)
            dm = region_m[ridx] * season_m[SEASONS.index(s)] * hour_m[HOUR_BUCKETS.index(hour_bucket(hour))] * acq_m[aidx]
            p_sub = tastes_arr[u] * dm[sc_dept]
            p_sub = p_sub / p_sub.sum()
            for _try in range(6):
                si = int(rng.choice(len(subcats), p=p_sub))
                cand = items_by_sub[si]
                cand = cand[launch[cand] <= day]
                if len(cand):
                    break
            else:
                cand = np.where(launch <= day)[0]
            pw = w_u[cand]
            it = int(cand[rng.choice(len(cand), p=pw / pw.sum())])
            pr = float(min(0.35, 0.10 * conv[it] / 0.1))
            ev = str(rng.choice(EVENT_TYPES, p=[1 - 0.35 - 0.10 - pr, 0.35, 0.10, pr]))
            if utype == "session" and j < (n_u - 3):
                ev = "click"
            channel = "pos" if (rng.random() < 0.18 or ur.acq_channel == "in_store_signup" and rng.random() < 0.3) else dev_channel[ur.device]
            weather = str(rng.choice(WEATHER[s]))
            recs.append((u, it, ev, day, hour, channel, s, weather, j))
    inter = pd.DataFrame(recs, columns=["u", "i", "event", "day", "hour", "channel", "season", "weather", "seq"])
    inter["user_id"] = users["user_id"].values[inter.u.values]
    inter["item_id"] = items["item_id"].values[inter.i.values]
    inter["ts"] = inter.day + inter.hour / 24.0 + inter.seq * 1e-4
    inter["positive"] = (inter.event != "view").astype(int)
    inter["user_type"] = user_type[inter.u.values]
    inter["item_cold"] = items["is_cold_item"].values[inter.i.values]
    # ---- split assignment ----------------------------------------------------------------------
    split = np.full(len(inter), "train", dtype=object)
    is_warm_u = inter.user_type.values == "warm"
    split[is_warm_u & (inter.day.values >= VAL_START_DAY) & (inter.day.values < CUTOFF_DAY)] = "val"
    split[is_warm_u & (inter.day.values >= CUTOFF_DAY)] = "test_warm"
    split[is_warm_u & inter.item_cold.values] = "test_cold_item"
    split[inter.user_type.values == "cold"] = "test_cold_user"
    sess = inter.user_type.values == "session"
    n_per = inter.groupby("u")["seq"].transform("max").values + 1
    split[sess] = np.where(inter.seq.values[sess] < (n_per[sess] - 3), "session_ctx", "test_session")
    inter["split"] = split
    inter = inter.drop(columns=["u", "i"]).sort_values("ts").reset_index(drop=True)
    # items that appear in train must not be flagged cold; guard invariant
    assert not inter[(inter.split == "train") & inter.item_cold].shape[0], "cold item leaked into train"
    meta = dict(departments=depts, subcategories=subcats, regions=REGIONS, devices=DEVICES,
                acq_channels=ACQ_CHANNELS, age_bands=AGE_BANDS, seasons=SEASONS, hour_buckets=HOUR_BUCKETS,
                price_tiers=PRICE_TIERS, channels=["web", "app", "pos"], n_days=N_DAYS, cutoff_day=CUTOFF_DAY)
    return RetailDataset(users=users, items=items, interactions=inter, meta=meta)


def summarize(ds: RetailDataset) -> Dict[str, Any]:
    """Return a dictionary of partition sizes for logging."""
    s = ds.interactions.split.value_counts().to_dict()
    return {
        "n_users": len(ds.users), "n_items": len(ds.items), "n_interactions": len(ds.interactions),
        "user_types": ds.users.user_type.value_counts().to_dict(), "n_cold_items": int(ds.items.is_cold_item.sum()),
        "splits": {k: int(v) for k, v in s.items()},
    }
