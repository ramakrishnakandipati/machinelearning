"""Synthetic omnichannel retail dataset with strict warm / cold-start splits.

Splits
------
* warm items + warm users: temporal train / val / test interactions
* item cold-start: 10% of items never appear in training interactions (only content features)
* user cold-start: 10% of users never appear in training; keep 0-2 history interactions + context
* system cold-start (Day-0): evaluated with *no* interaction logs (landing category only)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DataConfig, MODELS_DIR, SEED

# --------------------------------------------------------------------------- taxonomy
TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "Apparel": {
        "Jeans": ["denim", "slim", "straight", "indigo", "stretch"],
        "Jackets": ["bomber", "parka", "windproof", "zip", "outerwear"],
        "TShirts": ["tee", "crewneck", "graphic", "jersey", "tshirt"],
        "Dresses": ["maxi", "floral", "evening", "wrap", "dress"],
    },
    "Footwear": {
        "Sneakers": ["sneaker", "lowtop", "cushioned", "trainer", "laces"],
        "Boots": ["boot", "chelsea", "waterproof", "ankle", "hiking"],
        "Sandals": ["sandal", "strappy", "slides", "flipflop", "summer"],
        "Loafers": ["loafer", "slipon", "formal", "moccasin", "dress-shoe"],
    },
    "Electronics": {
        "Headphones": ["headphones", "wireless", "noise-cancelling", "earbuds", "bass"],
        "Smartphones": ["smartphone", "5g", "oled", "camera", "unlocked"],
        "Laptops": ["laptop", "ultrabook", "ssd", "gaming-laptop", "processor"],
        "Smartwatches": ["smartwatch", "fitness-tracker", "heartrate", "amoled", "wearable"],
    },
    "HomeKitchen": {
        "Cookware": ["skillet", "nonstick", "saucepan", "cast-iron", "wok"],
        "Cutlery": ["knife", "chef", "serrated", "flatware", "cleaver"],
        "Appliances": ["blender", "airfryer", "toaster", "espresso", "kettle"],
        "Storage": ["container", "organizer", "airtight", "shelf", "basket"],
    },
    "Beauty": {
        "Skincare": ["serum", "moisturizer", "retinol", "sunscreen", "hydrating"],
        "Makeup": ["lipstick", "mascara", "foundation", "eyeshadow", "blush"],
        "Haircare": ["shampoo", "conditioner", "keratin", "hairmask", "curl"],
        "Fragrance": ["perfume", "eau-de-parfum", "cologne", "musk", "floral-scent"],
    },
    "Sports": {
        "Yoga": ["yoga", "mat", "pilates", "stretch-band", "meditation"],
        "Running": ["running", "marathon", "hydration-vest", "reflective", "pace"],
        "Cycling": ["bike", "helmet", "cycling", "pedal", "derailleur"],
        "Camping": ["tent", "sleeping-bag", "camping", "lantern", "backpacking"],
    },
    "Grocery": {
        "Snacks": ["chips", "crackers", "granola-bar", "popcorn", "trailmix"],
        "Beverages": ["juice", "sparkling", "cold-brew", "soda", "kombucha"],
        "Organic": ["organic", "non-gmo", "farm", "vegan", "plant-based"],
        "Bakery": ["sourdough", "croissant", "bagel", "pastry", "baguette"],
    },
    "Toys": {
        "Puzzles": ["puzzle", "jigsaw", "brainteaser", "cube", "riddle"],
        "BoardGames": ["boardgame", "strategy", "dice", "cards", "tabletop"],
        "Dolls": ["doll", "plush", "dollhouse", "figurine", "teddy"],
        "BuildingSets": ["bricks", "construction", "robotics-kit", "blocks", "stem-kit"],
    },
}
CATEGORIES = list(TAXONOMY)
SUBCATS = [(c, s) for c in CATEGORIES for s in TAXONOMY[c]]  # global sub index = position
SUB_NAMES = [s for _, s in SUBCATS]
SUB_TO_CAT = np.array([CATEGORIES.index(c) for c, _ in SUBCATS])
CAT_MATERIALS = {
    "Apparel": ["cotton", "leather"], "Footwear": ["leather", "plastic"],
    "Electronics": ["metal", "plastic", "glass"], "HomeKitchen": ["metal", "ceramic", "wood", "glass"],
    "Beauty": ["glass", "plastic"], "Sports": ["plastic", "metal", "cotton"],
    "Grocery": ["paper", "plastic", "glass"], "Toys": ["wood", "plastic", "paper"],
}
BRANDS = ["Nova", "Aster", "Brio", "Cedar", "Delta", "Ember", "Flux", "Gaia", "Helio", "Ionic",
          "Juno", "Kite", "Lumen", "Mosaic", "Nimbus", "Onyx", "Pulse", "Quill", "Raven", "Solis",
          "Terra", "Umbra", "Vertex", "Willow", "Xenon", "Yonder", "Zephyr", "Atlas", "Bloom", "Crest"]
COLORS = ["black", "white", "navy", "red", "green", "beige", "grey", "blue", "pink", "brown"]
MATERIALS = ["cotton", "leather", "metal", "plastic", "wood", "glass", "ceramic", "paper"]
GENERIC = ["premium", "new", "classic", "everyday", "durable", "lightweight", "modern", "eco",
           "value", "signature", "essential", "pro"]
CHANNELS = ["web", "mobile_app", "in_store"]
DEVICES = ["desktop", "mobile_web", "ios", "android", "pos_terminal"]
CHANNEL_DEVICES = {"web": ["desktop", "mobile_web"], "mobile_app": ["ios", "android"], "in_store": ["pos_terminal"]}
REGIONS = ["north_america", "europe", "apac", "latam", "mea", "oceania"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
PRICE_TIERS = ["budget", "mid", "premium", "luxury"]

META_VOCAB = {
    "channel": CHANNELS, "device": DEVICES, "region": REGIONS, "age_band": AGE_BANDS,
    "intent_cat": CATEGORIES, "intent_sub": SUB_NAMES,
}


# --------------------------------------------------------------------------- featurizer
@dataclass
class ItemFeaturizer:
    """Turns raw item attributes into the multimodal feature vector. Pickled with the dataset so the
    API can encode brand-new SKUs on the fly."""
    tfidf: TfidfVectorizer
    svd: TruncatedSVD
    W_img: np.ndarray
    num_mean: np.ndarray
    num_std: np.ndarray
    txt_mean: np.ndarray
    txt_std: np.ndarray
    img_mean: np.ndarray
    img_std: np.ndarray
    price_edges: np.ndarray
    cat_margin: Dict[str, float]

    # layout: cat(8) sub(32) brand(30) tier(4) color(10) material(8) | num(2) | text | img
    @property
    def dim(self) -> int:
        return len(CATEGORIES) + len(SUB_NAMES) + len(BRANDS) + 4 + len(COLORS) + len(MATERIALS) + 2 + \
            self.svd.n_components + self.W_img.shape[0]

    def price_tier(self, price: float) -> int:
        return int(np.searchsorted(self.price_edges, price))

    def infer_sub_probs(self, category: str, description: str) -> np.ndarray:
        """Soft sub-category distribution from description keywords (uniform within category if none)."""
        p = np.zeros(len(SUB_NAMES), dtype=np.float32)
        subs = TAXONOMY[category]
        text = (description or "").lower()
        scores = {s: sum(kw in text for kw in kws) for s, kws in subs.items()}
        tot = sum(scores.values())
        for s in subs:
            idx = SUB_NAMES.index(s)
            p[idx] = scores[s] / tot if tot > 0 else 1.0 / len(subs)
        return p

    def transform_records(self, recs: List[dict]) -> np.ndarray:
        n = len(recs)
        X_cat = np.zeros((n, len(CATEGORIES)), np.float32)
        X_sub = np.zeros((n, len(SUB_NAMES)), np.float32)
        X_brand = np.zeros((n, len(BRANDS)), np.float32)
        X_tier = np.zeros((n, 4), np.float32)
        X_col = np.zeros((n, len(COLORS)), np.float32)
        X_mat = np.zeros((n, len(MATERIALS)), np.float32)
        num = np.zeros((n, 2), np.float32)
        img_in = np.zeros((n, self.W_img.shape[1]), np.float32)
        for i, r in enumerate(recs):
            cat = r["category"]
            X_cat[i, CATEGORIES.index(cat)] = 1
            if r.get("subcategory"):
                X_sub[i, SUB_NAMES.index(r["subcategory"])] = 1
            else:
                X_sub[i] = self.infer_sub_probs(cat, r.get("description", ""))
            if r.get("brand") in BRANDS:
                X_brand[i, BRANDS.index(r["brand"])] = 1
            X_tier[i, self.price_tier(float(r["price"]))] = 1
            if r.get("color") in COLORS:
                X_col[i, COLORS.index(r["color"])] = 1
            if r.get("material") in MATERIALS:
                X_mat[i, MATERIALS.index(r["material"])] = 1
            margin = r.get("margin")
            if margin is None:
                margin = self.cat_margin.get(cat, 0.3)
            num[i] = [np.log1p(float(r["price"])), float(margin)]
            img_in[i, :len(SUB_NAMES)] = X_sub[i]
            img_in[i, len(SUB_NAMES):len(SUB_NAMES) + len(COLORS)] = X_col[i]
            img_in[i, len(SUB_NAMES) + len(COLORS):] = X_mat[i]
        txt = self.svd.transform(self.tfidf.transform([r.get("description", "") for r in recs]))
        img = img_in @ self.W_img.T
        num = (num - self.num_mean) / self.num_std
        txt = (txt - self.txt_mean) / self.txt_std
        img = (img - self.img_mean) / self.img_std
        return np.concatenate([X_cat, X_sub, X_brand, X_tier, X_col, X_mat, num, txt, img], 1).astype(np.float32)


def _build_description(rng, brand, color, material, sub_kws, cat) -> str:
    kws = list(rng.choice(sub_kws, size=3, replace=False))
    gen = list(rng.choice(GENERIC, size=2, replace=False))
    words = [brand.lower(), color, material] + kws + gen
    rng.shuffle(words)
    return " ".join(words)


# --------------------------------------------------------------------------- main generator
def generate_dataset(cfg: Optional[DataConfig] = None, seed: int = SEED) -> dict:
    cfg = cfg or DataConfig()
    rng = np.random.default_rng(seed)
    N, U = cfg.n_items, cfg.n_users
    n_sub, n_cat = len(SUB_NAMES), len(CATEGORIES)

    # ---------------- items
    sub_w = rng.dirichlet(np.full(n_sub, 2.0))            # category skew
    sub_w = 0.5 * sub_w + 0.5 / n_sub
    sub_idx = rng.choice(n_sub, size=N, p=sub_w)
    cat_idx = SUB_TO_CAT[sub_idx]
    # brands: each category has 8 'home' brands
    cat_brands = {c: rng.choice(len(BRANDS), size=8, replace=False) for c in range(n_cat)}
    brand_idx = np.array([rng.choice(cat_brands[c]) if rng.random() < 0.75 else rng.integers(len(BRANDS)) for c in cat_idx])
    sub_base_price = np.exp(rng.normal(3.6, 0.8, size=n_sub))
    brand_prem = np.exp(rng.normal(0, 0.25, size=len(BRANDS)))
    price = np.round(sub_base_price[sub_idx] * brand_prem[brand_idx] * np.exp(rng.normal(0, 0.35, N)), 2) + 0.99
    price_edges = np.quantile(price, [0.25, 0.6, 0.9])
    tier_idx = np.searchsorted(price_edges, price)
    color_pref = rng.dirichlet(np.full(len(COLORS), 1.0), size=n_sub)
    color_idx = np.array([rng.choice(len(COLORS), p=color_pref[s]) for s in sub_idx])
    mat_idx = np.array([MATERIALS.index(rng.choice(CAT_MATERIALS[CATEGORIES[c]])) if rng.random() < 0.85
                        else rng.integers(len(MATERIALS)) for c in cat_idx])
    cat_margin_base = rng.uniform(0.15, 0.45, size=n_cat)
    margin = np.clip(cat_margin_base[cat_idx] + 0.04 * tier_idx + rng.normal(0, 0.04, N), 0.05, 0.65).round(3)
    pop_w = np.exp(rng.normal(0, 0.9, size=N))

    desc, title = [], []
    for i in range(N):
        cat = CATEGORIES[cat_idx[i]]
        sub = SUB_NAMES[sub_idx[i]]
        desc.append(_build_description(rng, BRANDS[brand_idx[i]], COLORS[color_idx[i]], MATERIALS[mat_idx[i]],
                                       TAXONOMY[cat][sub], cat))
        title.append(f"{BRANDS[brand_idx[i]]} {COLORS[color_idx[i]].title()} {sub}")

    items = pd.DataFrame({
        "item_id": np.arange(N), "title": title, "category": [CATEGORIES[c] for c in cat_idx],
        "subcategory": [SUB_NAMES[s] for s in sub_idx], "cat_idx": cat_idx, "sub_idx": sub_idx,
        "brand": [BRANDS[b] for b in brand_idx], "price": price, "price_tier": [PRICE_TIERS[t] for t in tier_idx],
        "color": [COLORS[c] for c in color_idx], "material": [MATERIALS[m] for m in mat_idx],
        "margin": margin, "description": desc,
    })
    cold_items = rng.choice(N, size=int(cfg.item_cold_frac * N), replace=False)
    items["is_cold"] = False
    items.loc[cold_items, "is_cold"] = True
    warm_item_mask = ~items["is_cold"].values

    # ---------------- text / image encoders (fit on warm items only) + feature matrix
    tfidf = TfidfVectorizer(min_df=2, token_pattern=r"[A-Za-z0-9\-]+")
    tf = tfidf.fit_transform(items.loc[warm_item_mask, "description"])
    svd = TruncatedSVD(cfg.text_dim, random_state=seed).fit(tf)
    W_img = rng.normal(0, 1 / np.sqrt(8), size=(cfg.img_dim, n_sub + len(COLORS) + len(MATERIALS))).astype(np.float32)

    # first pass with identity scalers to compute statistics on warm items
    ident = ItemFeaturizer(tfidf, svd, W_img, np.zeros(2), np.ones(2), np.zeros(cfg.text_dim), np.ones(cfg.text_dim),
                           np.zeros(cfg.img_dim), np.ones(cfg.img_dim), price_edges, {})
    recs = items.to_dict("records")
    raw = ident.transform_records(recs)
    o = n_cat + n_sub + len(BRANDS) + 4 + len(COLORS) + len(MATERIALS)
    num, txt, img = raw[:, o:o + 2], raw[:, o + 2:o + 2 + cfg.text_dim], raw[:, o + 2 + cfg.text_dim:]
    w = warm_item_mask
    feat = ItemFeaturizer(
        tfidf, svd, W_img,
        num[w].mean(0), num[w].std(0) + 1e-6, txt[w].mean(0), txt[w].std(0) + 1e-6,
        img[w].mean(0), img[w].std(0) + 1e-6, price_edges,
        {CATEGORIES[c]: float(margin[cat_idx == c].mean()) for c in range(n_cat)},
    )
    item_feats = feat.transform_records(recs)

    # ---------------- users
    cold_users = rng.choice(U, size=int(cfg.user_cold_frac * U), replace=False)
    channel = rng.choice(CHANNELS, size=U, p=[0.45, 0.35, 0.20])
    device = np.array([rng.choice(CHANNEL_DEVICES[c]) for c in channel])
    region = rng.choice(REGIONS, size=U, p=[0.3, 0.25, 0.2, 0.1, 0.07, 0.08])
    age = rng.choice(AGE_BANDS, size=U, p=[0.2, 0.3, 0.22, 0.16, 0.12])
    Rg = rng.normal(0, 0.9, size=(len(REGIONS), n_sub))
    Ag = rng.normal(0, 0.9, size=(len(AGE_BANDS), n_sub))
    Ch = rng.normal(0, 0.4, size=(len(CHANNELS), n_sub))
    style_dim = 8
    st_brand, st_color, st_mat = (rng.normal(0, 1, size=(k, style_dim)) for k in (len(BRANDS), len(COLORS), len(MATERIALS)))
    item_style = 0.8 * st_brand[brand_idx] + 0.6 * st_color[color_idx] + 0.5 * st_mat[mat_idx] + 0.2 * rng.normal(size=(N, style_dim))
    item_style /= np.linalg.norm(item_style, axis=1, keepdims=True)

    sub_pref = np.zeros((U, n_sub))
    for u in range(U):
        logits = (Rg[REGIONS.index(region[u])] + Ag[AGE_BANDS.index(age[u])] + Ch[CHANNELS.index(channel[u])]
                  + rng.normal(0, 1.2, n_sub))
        p = np.exp(logits - logits.max()); p /= p.sum()
        prim = rng.choice(n_sub, size=2, replace=False, p=p)
        sub_pref[u] = 0.2 / n_sub
        sub_pref[u, prim[0]] += 0.55
        sub_pref[u, prim[1]] += 0.25
    user_style = rng.normal(size=(U, style_dim)); user_style /= np.linalg.norm(user_style, axis=1, keepdims=True)

    # interaction counts (mean ~ n_interactions/U)
    raw_n = np.exp(rng.normal(0, 0.55, size=U))
    n_u = np.maximum(np.round(raw_n / raw_n.sum() * cfg.n_interactions).astype(int), 4)
    n_u[cold_users] = np.maximum(n_u[cold_users], 7)

    rows = []
    click_item = np.zeros(U, int)
    warm_ids = np.where(warm_item_mask)[0]
    for u in range(U):
        score = sub_pref[u, sub_idx] * pop_w ** 0.6 * np.exp(2.0 * item_style @ user_style[u])
        p = score / score.sum()
        # initial click (warm items only): 65% from user's taste, else random
        pw = p[warm_ids] / p[warm_ids].sum()
        click_item[u] = rng.choice(warm_ids, p=pw) if rng.random() < 0.65 else rng.choice(warm_ids)
        items_u = rng.choice(N, size=min(n_u[u], N), replace=False, p=p)
        ts = np.sort(rng.uniform(0, 90, size=len(items_u)))
        ch = np.where(rng.random(len(items_u)) < 0.6, channel[u], rng.choice(CHANNELS, size=len(items_u)))
        for it, t, c in zip(items_u, ts, ch):
            rows.append((u, int(it), float(t), c))
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "ts", "channel"])
    inter["split"] = ""
    is_cold_item = items["is_cold"].values
    is_cold_user = np.zeros(U, bool)
    is_cold_user[cold_users] = True
    inter["item_cold"] = is_cold_item[inter["item_id"].values]
    inter["user_cold"] = is_cold_user[inter["user_id"].values]

    # both cold -> dropped; warm user x cold item -> item_cold_test
    inter = inter[~(inter["item_cold"] & inter["user_cold"])].copy()
    inter.loc[inter["item_cold"], "split"] = "item_cold_test"

    # cold users: first 0-2 interactions are context history, the rest is test
    cu_mask = inter["user_cold"] & ~inter["item_cold"]
    for u, g in inter[cu_mask].groupby("user_id"):
        g = g.sort_values("ts")
        n_hist = int(rng.integers(0, 3))
        inter.loc[g.index[:n_hist], "split"] = "user_cold_history"
        inter.loc[g.index[n_hist:], "split"] = "user_cold_test"

    # warm x warm: temporal split per user
    ww = inter[(~inter["item_cold"]) & (~inter["user_cold"])]
    for u, g in ww.groupby("user_id"):
        g = g.sort_values("ts")
        n = len(g)
        n_te = max(1, int(round(0.1 * n)))
        n_va = max(1, int(round(0.1 * n))) if n >= 5 else 0
        n_tr = n - n_te - n_va
        inter.loc[g.index[:n_tr], "split"] = "train"
        if n_va:
            inter.loc[g.index[n_tr:n_tr + n_va], "split"] = "val"
        inter.loc[g.index[n_tr + n_va:], "split"] = "test"
    inter = inter.drop(columns=["item_cold", "user_cold"]).reset_index(drop=True)

    users = pd.DataFrame({
        "user_id": np.arange(U), "channel": channel, "device": device, "region": region, "age_band": age,
        "click_item": click_item, "intent_cat": items["category"].values[click_item],
        "intent_sub": items["subcategory"].values[click_item], "is_cold": is_cold_user,
    })

    ds = {
        "items": items, "users": users, "interactions": inter, "item_feats": item_feats, "featurizer": feat,
        "meta_vocab": META_VOCAB, "categories": CATEGORIES, "sub_names": SUB_NAMES, "sub_to_cat": SUB_TO_CAT,
        "taxonomy": TAXONOMY, "brands": BRANDS, "colors": COLORS, "materials": MATERIALS,
        "feature_blocks": {
            "category": (0, n_cat), "subcategory": (n_cat, n_cat + n_sub),
            "brand": (n_cat + n_sub, n_cat + n_sub + len(BRANDS)),
            "numeric": (o, o + 2), "text": (o + 2, o + 2 + cfg.text_dim), "image": (o + 2 + cfg.text_dim, item_feats.shape[1]),
        },
    }
    return ds


def summarize(ds: dict) -> dict:
    inter, users, items = ds["interactions"], ds["users"], ds["items"]
    return {
        "n_users": int(len(users)), "n_items": int(len(items)), "n_interactions": int(len(inter)),
        "n_cold_items": int(items["is_cold"].sum()), "n_cold_users": int(users["is_cold"].sum()),
        "feature_dim": int(ds["item_feats"].shape[1]),
        "split_counts": {k: int(v) for k, v in inter["split"].value_counts().items()},
    }


def save_dataset(ds: dict, path: Path = MODELS_DIR / "dataset.pkl") -> Path:
    with open(path, "wb") as f:
        pickle.dump(ds, f)
    return path


def load_dataset(path: Path = MODELS_DIR / "dataset.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    d = generate_dataset()
    print(summarize(d))
    save_dataset(d)
