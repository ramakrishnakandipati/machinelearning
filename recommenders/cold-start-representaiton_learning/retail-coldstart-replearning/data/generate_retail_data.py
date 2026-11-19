"""Synthetic omnichannel retail dataset with strict, leakage-free cold-start splits.

Behaviour is generated from latent user preferences (category / brand / price / region)
that are *observable through side-information* (quiz tags, region, channel, item text,
visual features), so content-based representation learning has real signal to find.

Outputs (data/generated/):
  items.parquet, users.parquet, stores.parquet, interactions.parquet, split_manifest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_TS = 1735689600  # 2025-01-01
DAY = 86400
HORIZON_DAYS = 180

TAXONOMY = {
    "Apparel": {
        "Tops": ["T-Shirts", "Shirts", "Sweaters"],
        "Bottoms": ["Jeans", "Chinos", "Shorts"],
        "Outerwear": ["Jackets", "Coats", "Vests"],
    },
    "Footwear": {
        "Casual Shoes": ["Sneakers", "Loafers", "Sandals"],
        "Athletic Shoes": ["Running Shoes", "Training Shoes", "Trail Shoes"],
        "Formal Shoes": ["Oxfords", "Heels", "Boots"],
    },
    "Home": {
        "Kitchen": ["Cookware", "Cutlery", "Appliances"],
        "Bedroom": ["Bedding", "Pillows", "Lamps"],
        "Decor": ["Wall Art", "Rugs", "Candles"],
    },
    "Electronics": {
        "Audio": ["Headphones", "Speakers", "Earbuds"],
        "Mobile": ["Phone Cases", "Chargers", "Power Banks"],
        "Wearables": ["Smartwatches", "Fitness Bands", "Trackers"],
    },
    "Beauty": {
        "Skincare": ["Moisturizers", "Serums", "Sunscreen"],
        "Makeup": ["Lipsticks", "Foundation", "Mascara"],
        "Haircare": ["Shampoo", "Conditioner", "Styling"],
    },
    "Sports": {
        "Fitness": ["Yoga Mats", "Dumbbells", "Resistance Bands"],
        "Outdoor": ["Backpacks", "Tents", "Water Bottles"],
        "Team Sports": ["Footballs", "Basketballs", "Rackets"],
    },
}
COLORS = ["black", "white", "navy", "grey", "red", "green", "beige", "brown", "blue", "pink"]
MATERIALS = ["cotton", "leather", "denim", "wool", "steel", "plastic", "silicone", "ceramic", "mesh", "linen"]
ADJ = ["classic", "premium", "essential", "modern", "lightweight", "durable", "signature", "everyday", "compact", "ergonomic"]
COMMON = ["quality", "design", "comfort", "style", "value", "daily", "perfect", "great", "new", "popular", "season", "gift"]
CHANNELS = ["organic", "paid_search", "in_store_qr", "social"]
LOYALTY = ["none", "bronze", "silver", "gold", "platinum"]
OS = ["ios", "android", "windows", "macos"]
REGIONS = ["north", "south", "east", "west", "central"]
EVENTS = ["view", "add_to_cart", "purchase", "pos_scan"]


def build_taxonomy_table():
    rows = []
    for l1, d in TAXONOMY.items():
        for l2, l3s in d.items():
            for l3 in l3s:
                rows.append((l1, l2, l3))
    return pd.DataFrame(rows, columns=["l1", "l2", "l3"])


def generate(cfg: dict, out_dir: Path, seed: int = 42, verbose: bool = True) -> dict:
    dc = cfg["data"]
    rng = np.random.default_rng(seed)
    nU, nI, nS, nX = dc["n_users"], dc["n_items"], dc["n_stores"], dc["n_interactions"]
    vdim = dc["visual_dim"]
    tax = build_taxonomy_table()
    n_l3, l2_names = len(tax), list(dict.fromkeys(tax.l2))
    l1_names = list(TAXONOMY)

    # ------------------------------------------------------------------ stores
    stores = pd.DataFrame({
        "store_id": [f"S{j:02d}" for j in range(nS)],
        "region": [REGIONS[j % len(REGIONS)] for j in range(nS)],
        "size_sqm": rng.integers(300, 3000, nS),
        "is_urban": rng.integers(0, 2, nS),
    })

    # ------------------------------------------------------------------ items
    l3_idx = rng.choice(n_l3, size=nI, p=rng.dirichlet(np.ones(n_l3) * 6))
    brands = [f"Brand{b:02d}" for b in range(40)]
    brand_l1 = {b: rng.choice(len(l1_names), size=2, replace=False) for b in brands}
    brand_quality = rng.uniform(0, 1, len(brands))
    item_brand = []
    for li in l3_idx:
        l1_i = l1_names.index(tax.l1[li])
        cands = [b for b in brands if l1_i in brand_l1[b]] or brands
        item_brand.append(rng.choice(cands))
    item_brand = np.array(item_brand)
    price_tier = np.clip(np.round(brand_quality[[brands.index(b) for b in item_brand]] * 3 + rng.normal(0, 0.6, nI)).astype(int) + 1, 1, 4)
    price = np.round(np.exp(rng.normal(3.0 + 0.7 * price_tier, 0.25)), 2)
    margin = np.round(rng.beta(2, 5, nI) * 0.6 + 0.1, 3)
    color = rng.choice(COLORS, nI)
    material = rng.choice(MATERIALS, nI)
    launch_day = rng.uniform(0, 100, nI)
    launch_ts = (BASE_TS + launch_day * DAY).astype(np.int64)

    l3_kw = {}
    for li in range(n_l3):
        pool = [tax.l3[li].lower(), tax.l2[li].lower(), tax.l1[li].lower()] + [f"{tax.l3[li][:4].lower()}{k}" for k in range(3)]
        l3_kw[li] = pool
    titles, descs = [], []
    for i in range(nI):
        li = l3_idx[i]
        adj = rng.choice(ADJ)
        titles.append(f"{item_brand[i]} {adj} {color[i]} {tax.l3[li][:-1] if tax.l3[li].endswith('s') else tax.l3[li]}")
        words = list(rng.choice(l3_kw[li], 6)) + list(rng.choice(COMMON, 4)) + [material[i], color[i], adj, tax.l2[li].lower(), tax.l1[li].lower()]
        rng.shuffle(words)
        descs.append(f"{titles[i]}. Made of {material[i]}. " + " ".join(words))

    # visual: hierarchical centroids + brand offset + noise (simulated CLIP/ResNet embedding)
    l1_c = rng.normal(0, 1, (len(l1_names), vdim))
    l2_c = rng.normal(0, 0.7, (len(l2_names), vdim))
    l3_c = rng.normal(0, 0.5, (n_l3, vdim))
    br_c = rng.normal(0, 0.25, (len(brands), vdim))
    vis = np.stack([
        l1_c[l1_names.index(tax.l1[l3_idx[i]])] + l2_c[l2_names.index(tax.l2[l3_idx[i]])] + l3_c[l3_idx[i]]
        + br_c[brands.index(item_brand[i])] + rng.normal(0, 0.6, vdim) for i in range(nI)
    ])
    vis = vis / np.linalg.norm(vis, axis=1, keepdims=True)
    items = pd.DataFrame({
        "item_id": [f"I{i:04d}" for i in range(nI)],
        "title": titles, "description": descs,
        "l1": tax.l1.values[l3_idx], "l2": tax.l2.values[l3_idx], "l3": tax.l3.values[l3_idx],
        "brand": item_brand, "price_tier": price_tier, "price": price, "margin": margin,
        "color": color, "material": material, "launch_ts": launch_ts,
    })
    for d in range(vdim):
        items[f"vis_{d}"] = vis[:, d].astype(np.float32)

    # ------------------------------------------------------------------ cold-start assignments (BEFORE behaviour)
    item_cold = np.zeros(nI, bool)
    item_cold[rng.choice(nI, int(round(dc["item_cold_frac"] * nI)), replace=False)] = True
    items["is_cold_item"] = item_cold

    # ------------------------------------------------------------------ users
    region = rng.choice(REGIONS, nU)
    store_by_region = {r: stores.store_id[stores.region == r].tolist() for r in REGIONS}
    home_store = np.array([rng.choice(store_by_region[r]) for r in region])
    channel = rng.choice(CHANNELS, nU, p=[0.4, 0.25, 0.15, 0.2])
    loyalty = rng.choice(LOYALTY, nU, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    os_ = rng.choice(OS, nU, p=[0.4, 0.4, 0.1, 0.1])

    aff_l2 = rng.dirichlet(np.ones(len(l2_names)) * 0.25, nU)
    reg_boost = rng.dirichlet(np.ones(len(l2_names)) * 2.0, len(REGIONS))
    aff_l2 = aff_l2 * 0.85 + 0.15 * reg_boost[[REGIONS.index(r) for r in region]]
    aff_l2 /= aff_l2.sum(1, keepdims=True)
    brand_pref = rng.dirichlet(np.ones(len(brands)) * 0.6, nU)
    tier_pref = np.clip(rng.normal(2.2 + 0.25 * np.array([LOYALTY.index(x) for x in loyalty]), 0.7), 1, 4)

    quiz_vocab = [f"likes_{n.lower().replace(' ', '_')}" for n in l2_names] + [f"cat_{n.lower()}" for n in l1_names]
    quiz = []
    for u in range(nU):
        k = rng.integers(1, 4)
        top = np.argsort(-aff_l2[u])[:k]
        tags = []
        for t in top:
            if rng.random() < 0.8:
                tags.append(quiz_vocab[t])
            else:
                tags.append(quiz_vocab[rng.integers(len(quiz_vocab))])
        quiz.append("|".join(sorted(set(tags))))

    # system cold-start: entire stores withheld => their users are zero-shot and have never been served
    sys_stores = list(rng.choice(stores.store_id, dc["system_cold_stores"], replace=False))
    sys_user = np.isin(home_store, sys_stores)
    other = np.where(~sys_user)[0]
    cold_users = rng.choice(other, int(round(dc["user_cold_frac"] * nU)), replace=False)
    zero_cnt = int(round(len(cold_users) * dc["zero_shot_user_share"]))
    group = np.array(["warm"] * nU, dtype=object)
    group[cold_users[:zero_cnt]] = "zero_shot"
    group[cold_users[zero_cnt:]] = "few_shot"
    group[sys_user] = "system_cold"
    users = pd.DataFrame({
        "user_id": [f"U{u:04d}" for u in range(nU)],
        "acquisition_channel": channel, "loyalty_tier": loyalty, "home_region": region,
        "home_store": home_store, "device_os": os_, "quiz_tags": quiz, "user_group": group,
    })
    stores["is_system_cold"] = stores.store_id.isin(sys_stores)

    # ------------------------------------------------------------------ behaviour
    pop = rng.zipf(1.6, nI).astype(float)
    pop = np.clip(pop, 1, 60)
    log_pop = np.log(pop)
    l2_of_item = np.array([l2_names.index(x) for x in items.l2])
    brand_of_item = np.array([brands.index(b) for b in item_brand])
    score = (
        0.4 * log_pop[None, :]
        + 2.2 * np.log(aff_l2[:, l2_of_item] + 1e-3)
        + 1.0 * np.log(brand_pref[:, brand_of_item] + 1e-3)
        - 0.8 * np.abs(tier_pref[:, None] - price_tier[None, :])
    )
    activity = 3 + rng.negative_binomial(2, 2 / (2 + (nX / nU - 3)), nU)
    activity = np.maximum(3, np.round(activity * nX / activity.sum())).astype(int)
    rows = []
    for u in range(nU):
        n = int(activity[u])
        ts = np.sort(BASE_TS + rng.uniform(0, HORIZON_DAYS, n) * DAY)
        s_u = score[u].copy()
        if group[u] == "system_cold":
            s_u[item_cold] += np.log(3.0)   # new-store launch pushes newly launched items
        # session structure: every 1-3 consecutive events share a session / visit day
        sess = 0
        sess_of = np.zeros(n, int)
        for k in range(1, n):
            if rng.random() < 0.45 and ts[k] - ts[k - 1] < DAY:
                sess_of[k] = sess_of[k - 1]
            else:
                sess += 1
                sess_of[k] = sess
        # users drift slightly inside a session: session intent amplifies one category
        for k in range(n):
            avail = launch_ts <= ts[k]
            if not avail.any():
                avail[:] = True
            p = np.exp(s_u - s_u[avail].max()) * avail
            p /= p.sum()
            it = rng.choice(nI, p=p)
            instore = rng.random() < (0.35 if channel[u] == "in_store_qr" else 0.15)
            if instore:
                ev, ch = ("pos_scan" if rng.random() < 0.7 else "view"), "store"
            else:
                ev = rng.choice(["view", "add_to_cart", "purchase"], p=[0.65, 0.2, 0.15])
                ch = "app" if os_[u] in ("ios", "android") and rng.random() < 0.7 else "web"
            st = home_store[u] if (ch == "store" or rng.random() < 0.6) else rng.choice(stores.store_id)
            hr = int((ts[k] % DAY) // 3600)
            rows.append((users.user_id[u], items.item_id[it], ev, ch, st, f"{users.user_id[u]}_S{sess_of[k]}", int(ts[k]), hr,
                         os_[u] if ch != "store" else "pos"))
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "event_type", "channel", "store_id", "session_id", "ts", "hour", "device"])
    inter = inter.sort_values("ts").reset_index(drop=True)

    # ------------------------------------------------------------------ splits
    inter["user_group"] = inter.user_id.map(users.set_index("user_id").user_group)
    inter["item_cold"] = inter.item_id.map(items.set_index("item_id").is_cold_item)
    t_val = BASE_TS + int(0.70 * HORIZON_DAYS) * DAY
    t_test = BASE_TS + int(0.85 * HORIZON_DAYS) * DAY
    split = np.full(len(inter), "drop", dtype=object)
    slice_ = np.full(len(inter), "", dtype=object)

    g = inter.user_group.values
    ic = inter.item_cold.values
    ts_ = inter.ts.values
    rank_in_user = inter.groupby("user_id").cumcount().values      # chronological rank (sorted by ts)
    nU_total = inter.groupby("user_id").user_id.transform("size").values

    # warm users: global temporal cut
    w = g == "warm"
    split[w & ~ic & (ts_ < t_val)] = "train"
    ev = w & (ts_ >= t_val)
    split[ev & (ts_ < t_test)] = "val"
    split[ev & (ts_ >= t_test)] = "test"
    slice_[ev & ~ic] = "warm"
    slice_[ev & ic] = "item_cold"
    # few-shot: first k (1..3) *warm-item* interactions go to train, rest to eval
    k_user = pd.Series(rng.integers(1, 4, nU), index=users.user_id)
    fs = g == "few_shot"
    kk = inter.user_id.map(k_user).values
    cum_warm = (fs & ~ic).astype(int)
    cum_warm = pd.Series(cum_warm).groupby(inter.user_id.values).cumsum().values
    is_hist = fs & ~ic & (cum_warm <= kk)
    split[is_hist] = "train"
    rest = fs & ~is_hist
    # eval items for few-shot / zero-shot / system users: chronological split by per-user median
    def half_split(mask):
        idx = np.where(mask)[0]
        df = pd.DataFrame({"u": inter.user_id.values[idx], "r": np.arange(len(idx))})
        df["rk"] = df.groupby("u").cumcount()
        df["n"] = df.groupby("u").u.transform("size")
        return idx, (df.rk.values < np.ceil(df.n.values / 2)), df
    for m in (rest, g == "zero_shot", g == "system_cold"):
        idx, first_half, _ = half_split(m)
        split[idx] = np.where(first_half, "val", "test")
    uc = (fs & ~is_hist) | (g == "zero_shot")
    slice_[uc & ~ic] = "user_cold"
    # user-cold x item-cold and few-shot/zero-shot with cold items are not an official slice
    split[uc & ic] = np.where(split[uc & ic] == "train", "train", split[uc & ic])
    slice_[uc & ic] = "user_item_cold_unassigned"
    sysm = g == "system_cold"
    slice_[sysm & ic] = "system_cold"
    slice_[sysm & ~ic] = "system_user_warm_item_unassigned"
    inter["split"] = split
    inter["slice"] = slice_
    inter = inter.drop(columns=["user_group", "item_cold"])

    # warm users with zero train interactions cannot be "warm": drop their eval rows
    ntr = inter[inter.split == "train"].groupby("user_id").size()
    users["n_train_interactions"] = users.user_id.map(ntr).fillna(0).astype(int)
    bad = users.user_id[(users.user_group == "warm") & (users.n_train_interactions == 0)]
    inter.loc[inter.user_id.isin(bad) & (inter.split != "train"), "split"] = "drop"

    # ------------------------------------------------------------------ leakage assertions
    tr = inter[inter.split == "train"]
    assert not tr.item_id.isin(items.item_id[items.is_cold_item]).any(), "cold item leaked into train"
    assert not tr.user_id.isin(users.user_id[users.user_group.isin(["zero_shot", "system_cold"])]).any(), "zero-shot user leaked"
    assert not tr.user_id.isin(users.user_id[users.user_group == "system_cold"]).any()
    fs_cnt = tr[tr.user_id.isin(users.user_id[users.user_group == "few_shot"])].groupby("user_id").size()
    assert fs_cnt.max() <= 3, "few-shot user has >3 train interactions"
    # temporal sanity for warm users
    assert tr[tr.user_id.isin(users.user_id[users.user_group == "warm"])].ts.max() < t_val
    ev_df = inter[inter.split.isin(["val", "test"]) & inter["slice"].isin(["warm", "item_cold", "user_cold", "system_cold"])]
    manifest = {
        "n_users": nU, "n_items": nI, "n_stores": nS, "n_interactions": int(len(inter)),
        "train_rows": int((inter.split == "train").sum()),
        "val_rows": int((inter.split == "val").sum()), "test_rows": int((inter.split == "test").sum()),
        "slices": {s: {sp: int(((ev_df["slice"] == s) & (ev_df.split == sp)).sum()) for sp in ("val", "test")}
                   for s in ["warm", "item_cold", "user_cold", "system_cold"]},
        "user_groups": users.user_group.value_counts().to_dict(),
        "cold_items": int(items.is_cold_item.sum()),
        "system_cold_stores": sys_stores,
        "t_val": int(t_val), "t_test": int(t_test),
        "leakage_checks": "passed",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    items.to_parquet(out_dir / "items.parquet")
    users.to_parquet(out_dir / "users.parquet")
    stores.to_parquet(out_dir / "stores.parquet")
    inter.to_parquet(out_dir / "interactions.parquet")
    json.dump(manifest, open(out_dir / "split_manifest.json", "w"), indent=2)
    if verbose:
        print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    generate(cfg, ROOT / cfg["paths"]["data_dir"], cfg["seed"])
