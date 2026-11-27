"""Synthetic retail world: catalog (head/tail + cold-start), regional inventory, users,
vocabulary-gap queries and an Impression->Click->Cart->Purchase->Return clickstream.

All ground truth (graded relevance, latent click/cart/purchase/return probabilities) is produced
by a *latent* model that is NOT exposed to the serving stack: `latent_*` columns live in a
separate file that only the evaluation code loads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from loguru import logger  # noqa: E402

from src.common.utils import load_config, resolve, setup_logging, sigmoid  # noqa: E402
from src.data_gen import taxonomy as T  # noqa: E402

SUBS = [t["sub"] for t in T.TYPES]
CATS = sorted({t["cat"] for t in T.TYPES})
DURABLE_SUBS = {"Office Chairs", "Smartwatches", "Headphones", "Backpacks", "Winter Boots", "Bedding Sets"}


# --------------------------------------------------------------------------------------
# Latent behavioural model (shared by the generator and the offline evaluator)
# --------------------------------------------------------------------------------------
def price_fit(price: np.ndarray, median_spend: float) -> np.ndarray:
    z = (np.log(np.asarray(price) + 1.0) - np.log(median_spend + 1.0)) / 0.7
    return np.exp(-0.5 * z * z)


def latent_probs(rel_true, rel_claimed, quality, brand_match, pfit, ret_base, exam=1.0):
    """Funnel probabilities. Clicks react to what the *title claims*; purchases/returns to truth."""
    p_attr = sigmoid(-2.7 + 0.95 * rel_claimed + 0.7 * quality + 0.8 * brand_match + 0.9 * pfit)
    p_click = exam * p_attr
    p_atc = sigmoid(-1.9 + 0.55 * rel_true + 0.5 * quality + 0.6 * pfit)
    p_cvr = sigmoid(-0.8 + 0.45 * rel_true + 0.4 * pfit + 0.3 * brand_match)
    p_ret = np.clip(ret_base + 0.12 * (rel_true <= 1), 0.005, 0.85)
    return p_click, p_atc, p_cvr, p_ret


def relevance_for_query(q: dict, sub_idx: np.ndarray, feat_arr: dict, brand: np.ndarray,
                        color: np.ndarray, price: np.ndarray, subs_index: dict) -> np.ndarray:
    """Graded relevance 0..3 for every SKU given a query record and a feature-array dict."""
    sub_ids = [subs_index[s] for s in json.loads(q["subs"]) ] if isinstance(q["subs"], str) else [subs_index[s] for s in q["subs"]]
    in_sub = np.isin(sub_idx, sub_ids)
    rel = np.zeros(len(sub_idx), dtype=np.int8)
    if q["feat"]:
        has = feat_arr.get(q["feat"], np.zeros(len(sub_idx), dtype=bool))
        rel[in_sub & has] = 3
        rel[in_sub & ~has] = 1
    else:
        rel[in_sub] = 3
    pen = np.zeros(len(sub_idx), dtype=np.int8)
    if q["color"]:
        pen += (color != q["color"]).astype(np.int8)
    if q["brand"]:
        pen += 2 * (brand != q["brand"]).astype(np.int8)
    if q["max_price"] and q["max_price"] > 0:
        pen += (price > q["max_price"]).astype(np.int8)
    return np.clip(rel - pen, 0, 3).astype(np.int8)


# --------------------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------------------
def build_catalog(cfg: dict, rng: np.random.Generator):
    n = cfg["data"]["n_skus"]
    cold_frac = cfg["data"]["cold_start_frac"]
    mis_frac = cfg["data"]["misleading_frac"]
    type_probs = np.array([1.3 if t["cat"] in ("Footwear", "Apparel", "Outerwear") else 1.0 for t in T.TYPES])
    type_probs /= type_probs.sum()
    type_idx = rng.choice(len(T.TYPES), size=n, p=type_probs)
    margin_base = {"Footwear": 0.48, "Apparel": 0.55, "Outerwear": 0.50, "Accessories": 0.52,
                   "Home": 0.42, "Electronics": 0.24, "Fitness": 0.50}
    ret_cat = {"Footwear": 0.17, "Apparel": 0.19, "Outerwear": 0.14, "Accessories": 0.07,
               "Home": 0.08, "Electronics": 0.06, "Fitness": 0.06}
    rows, true_feat_sets, claimed_feat_sets = [], [], []
    now = pd.Timestamp("2026-11-01")
    for i in range(n):
        t = T.TYPES[type_idx[i]]
        brand = rng.choice(T.CATEGORY_BRANDS[t["cat"]])
        tier = T.BRAND_TIERS[brand]
        color = rng.choice(T.COLORS)
        feat_names = list(t["feats"])
        k = min(len(feat_names), int(rng.choice([1, 2, 2, 3])))
        true_feats = list(rng.choice(feat_names, size=k, replace=False))
        misleading = bool(rng.random() < mis_frac)
        claimed = list(true_feats)
        fake = [f for f in feat_names if f not in true_feats]
        if misleading and fake:
            claimed.append(str(rng.choice(fake)))
        else:
            misleading = False
        noun = str(rng.choice(t["nouns"]))
        material = str(rng.choice(t["materials"]))
        model = f"{''.join(w[0] for w in t['sub'].split()[:2]).upper()}-{rng.integers(1000, 9999)}"
        title_feats = " ".join(claimed[:2]).title()
        title = f"{brand} {model} {title_feats} {noun.title()} - {color.title()}"
        sizes = T.SIZES[t["sizes"]]
        size = ",".join(sizes)
        attributes = f"{color} {material} " + " ".join(true_feats) + f" {t['cat'].lower()} {t['sub'].lower()}"
        desc_parts = [f"The {brand} {model} is a {noun} built from {material}."]
        for f in true_feats:
            desc_parts.append(t["feats"][f])
        if misleading:
            desc_parts.append("Features a light water-resistant treatment and standard construction.")
        desc_parts.append(f"Available in {color}.")
        lo, hi = t["price"]
        mult = {1: 1.25, 2: 1.0, 3: 0.72}[tier]
        price = float(np.clip(lo + (hi - lo) * rng.beta(2.0, 3.2) * mult, lo * 0.8, hi * 1.2))
        price = round(price, 2)
        gm_pct = float(np.clip(margin_base[t["cat"]] + {1: 0.06, 2: 0.0, 3: -0.07}[tier] + rng.normal(0, 0.05), 0.08, 0.75))
        cost = round(price * (1 - gm_pct), 2)
        quality = float(rng.normal(0, 0.8) + (3 - tier) * 0.28)
        ret_base = float(np.clip(rng.beta(2, 8) * ret_cat[t["cat"]] / 0.2 + (0.13 if misleading else 0.0)
                                 + {1: -0.01, 2: 0.0, 3: 0.03}[tier], 0.01, 0.6))
        cold = bool(rng.random() < cold_frac)
        age_days = int(rng.integers(0, 45)) if cold else int(rng.integers(60, 1500))
        rows.append(dict(
            sku_id=f"SKU{i:06d}", title=title, description=" ".join(desc_parts), brand=brand,
            category=t["cat"], subcategory=t["sub"], color=color, size=size, material=material,
            attributes=attributes, price=price, cost=cost,
            gross_margin=round(price - cost, 2), gross_margin_pct=round(gm_pct, 4),
            is_cold_start=cold, created_at=(now - pd.Timedelta(days=age_days)).strftime("%Y-%m-%d"),
            brand_tier=tier, sub_idx=int(type_idx[i]),
        ))
        true_feat_sets.append(set(true_feats))
        claimed_feat_sets.append(set(claimed))
        rows[-1]["_q"] = quality
        rows[-1]["_ret"] = ret_base
        rows[-1]["_mis"] = misleading
    df = pd.DataFrame(rows)
    latent = df[["sku_id", "_q", "_ret", "_mis"]].rename(
        columns={"_q": "latent_quality", "_ret": "latent_return_base", "_mis": "is_misleading"})
    df = df.drop(columns=["_q", "_ret", "_mis"])
    return df, latent, true_feat_sets, claimed_feat_sets


def feature_arrays(sets: list[set]) -> dict:
    names = {f for s in sets for f in s}
    return {f: np.array([f in s for s in sets]) for f in names}


def build_image_embeddings(df: pd.DataFrame, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic 'visual' embeddings: aligned with subcategory, color and material (+noise)."""
    sub_dim, col_dim, mat_dim = dim // 2, dim // 4, dim - dim // 2 - dim // 4
    tables = {
        "sub": {s: rng.normal(size=sub_dim) for s in df.subcategory.unique()},
        "col": {c: rng.normal(size=col_dim) for c in df.color.unique()},
        "mat": {m: rng.normal(size=mat_dim) for m in df.material.unique()},
    }
    emb = np.stack([np.concatenate([tables["sub"][s], tables["col"][c], tables["mat"][m]])
                    for s, c, m in zip(df.subcategory, df.color, df.material)])
    emb += rng.normal(scale=0.35, size=emb.shape)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return emb.astype(np.float32)


# --------------------------------------------------------------------------------------
# Stores & inventory
# --------------------------------------------------------------------------------------
def build_inventory(cfg: dict, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    stores = cfg["data"]["stores"]
    n = len(df)
    inv = {"sku_id": df.sku_id.values}
    cold = df.is_cold_start.values
    oos_all = rng.random(n) < 0.03
    for s in stores:
        if s["store_id"] == "WH":
            stock = rng.poisson(70, n)
            stock = np.where(rng.random(n) < 0.04, 0, stock)
        else:
            stock = rng.poisson(9, n)
            stock = np.where(rng.random(n) < 0.28, 0, stock)
        stock = np.where(cold, (stock * 0.5).astype(int), stock)
        stock = np.where(oos_all, 0, stock)
        inv[f"stock_{s['store_id']}"] = stock.astype(int)
    return pd.DataFrame(inv)


# --------------------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------------------
def build_users(cfg: dict, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg["data"]["n_users"]
    stores = [s for s in cfg["data"]["stores"] if s["store_id"] != "WH"]
    durable_idx = np.where(df.subcategory.isin(DURABLE_SUBS))[0]
    users = []
    for u in range(n):
        tier = int(rng.choice([0, 1, 2], p=[0.3, 0.45, 0.25]))
        median = float(np.exp(rng.normal({0: 3.5, 1: 4.3, 2: 5.0}[tier], 0.35)))
        aff = rng.dirichlet(np.ones(len(CATS)) * 0.6)
        prefs = list(rng.choice(T.BRANDS, size=3, replace=False))
        recent = [df.sku_id.values[j] for j in rng.choice(durable_idx, size=int(rng.choice([0, 0, 1, 2])), replace=False)]
        st = stores[int(rng.integers(0, len(stores)))]
        users.append(dict(
            user_id=f"U{u:05d}", preferred_brands=json.dumps(prefs), price_tier=tier, median_spend=round(median, 2),
            size_profile=str(rng.choice(["XS", "S", "M", "L", "XL"], p=[0.08, 0.22, 0.35, 0.25, 0.10])),
            shoe_size=str(rng.choice(T.SIZES["shoe"])),
            category_affinity=json.dumps({c: round(float(a), 4) for c, a in zip(CATS, aff)}),
            recent_purchases=json.dumps(recent), home_store=st["store_id"], zip_code=st["zip_code"]))
    return pd.DataFrame(users)


# --------------------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------------------
TYPO_MAP = {"gore-tex": "gortex", "sneakers": "snekers", "waterproof": "watrproof", "breathable": "breathble",
            "running": "runing", "jacket": "jackit", "headphones": "headfones", "leggings": "legings",
            "backpack": "bakpack", "puffer": "pufer", "insulated": "insulatd", "wireless": "wirless",
            "ergonomic": "ergonmic", "trail": "tral", "yoga": "yogga", "shirt": "shrit", "boots": "bots"}


def inject_typo(text: str, rng: np.random.Generator) -> str:
    words = text.split()
    cand = [i for i, w in enumerate(words) if len(w) > 4 and not w.startswith("$")]
    if not cand:
        return text
    i = int(rng.choice(cand))
    w = words[i]
    lw = w.lower()
    if lw in TYPO_MAP and rng.random() < 0.6:
        words[i] = TYPO_MAP[lw]
    else:
        op = rng.integers(0, 3)
        j = int(rng.integers(1, len(w) - 1))
        if op == 0:
            words[i] = w[:j] + w[j + 1:]
        elif op == 1:
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        else:
            words[i] = w[:j] + w[j] + w[j:]
    return " ".join(words)


def build_queries(cfg: dict, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    d = cfg["data"]
    total = d["n_train_queries"] + d["n_val_queries"] + d["n_test_queries"]
    price_by_sub = {s: df.price[df.subcategory == s].values for s in SUBS}
    recs, seen = [], set()
    attempts = 0
    while len(recs) < total and attempts < total * 40:
        attempts += 1
        kind = rng.choice(["colloquial", "category", "navigational", "formal"], p=[0.5, 0.15, 0.15, 0.2])
        brand = color = None
        feat = None
        if kind == "colloquial":
            phrase, sub, feat = T.INTENTS[int(rng.integers(len(T.INTENTS)))]
            subs = [sub]
            formal = f"{feat} {rng.choice(T.TYPE_BY_SUB[sub]['nouns'])}" if feat else str(rng.choice(T.TYPE_BY_SUB[sub]["nouns"]))
            text = phrase
        elif kind == "category":
            phrase = str(rng.choice(list(T.CATEGORY_TERMS)))
            subs = T.CATEGORY_TERMS[phrase]
            text = phrase
            formal = phrase
        elif kind == "navigational":
            phrase = str(rng.choice(list(T.CATEGORY_TERMS)))
            subs = T.CATEGORY_TERMS[phrase]
            cat = T.TYPE_BY_SUB[subs[0]]["cat"]
            brand = str(rng.choice(T.CATEGORY_BRANDS[cat]))
            text = f"{brand} {phrase}".lower()
            formal = text
        else:
            _, sub, feat = T.INTENTS[int(rng.integers(len(T.INTENTS)))]
            subs = [sub]
            noun = str(rng.choice(T.TYPE_BY_SUB[sub]["nouns"]))
            text = f"{feat} {noun}" if feat else noun
            formal = text
        t0 = T.TYPE_BY_SUB[subs[0]]
        if rng.random() < 0.22 and t0["cat"] in ("Footwear", "Apparel", "Outerwear", "Accessories", "Home", "Fitness"):
            color = str(rng.choice(T.COLORS))
            text = f"{color} {text}"
            formal = f"{color} {formal}"
        if kind in ("colloquial", "formal") and rng.random() < 0.10:
            brand = str(rng.choice(T.CATEGORY_BRANDS[t0["cat"]]))
            text = f"{brand.lower()} {text}"
            formal = f"{brand.lower()} {formal}"
        max_price = 0.0
        if rng.random() < 0.25:
            pr = np.concatenate([price_by_sub[s] for s in subs])
            max_price = float(max(10, round(np.percentile(pr, rng.uniform(40, 75)) / 10) * 10))
            word = rng.choice(["under", "below", "less than", "max"])
            text = f"{text} {word} ${int(max_price)}"
            formal = f"{formal} under ${int(max_price)}"
        if text in seen:
            continue
        seen.add(text)
        if brand and not feat and not color and not max_price:
            itype = "navigational"
        elif feat or color or max_price:
            itype = "attribute_specific"
        else:
            itype = "category_browsing"
        typo = inject_typo(text, rng) if rng.random() < 0.18 else text
        recs.append(dict(text=text, typo_text=typo, formal_text=formal, intent_type=itype,
                         subs=json.dumps(subs), feat=feat or "", color=color or "", brand=brand or "",
                         max_price=max_price))
    qdf = pd.DataFrame(recs)
    qdf.insert(0, "query_id", [f"Q{i:05d}" for i in range(len(qdf))])
    n_tr, n_va = d["n_train_queries"], d["n_val_queries"]
    perm = rng.permutation(len(qdf))
    split = np.empty(len(qdf), dtype=object)
    split[perm[:n_tr]] = "train"
    split[perm[n_tr:n_tr + n_va]] = "val"
    split[perm[n_tr + n_va:]] = "test"
    qdf["split"] = split
    return qdf


# --------------------------------------------------------------------------------------
# Clickstream
# --------------------------------------------------------------------------------------
def simulate_sessions(cfg, df, latent, users, qdf, true_arr, claimed_arr, rng):
    k = cfg["data"]["impressions_per_session"]
    n_sessions = cfg["data"]["n_sessions"]
    subs_index = {s: i for i, s in enumerate(SUBS)}
    sub_idx = df.sub_idx.values
    brand, color, price = df.brand.values, df.color.values, df.price.values
    quality = latent.latent_quality.values
    cold = df.is_cold_start.values
    train_q = qdf[qdf.split == "train"].reset_index(drop=True)
    pools = []
    for _, q in train_q.iterrows():
        rel_t = relevance_for_query(q, sub_idx, true_arr, brand, color, price, subs_index)
        rel_c = relevance_for_query(q, sub_idx, claimed_arr, brand, color, price, subs_index)
        cand = np.where((rel_c >= 1) & ~cold)[0]
        if len(cand) < k:
            cand = np.where(~cold)[0][: 200]
        w = (rel_c[cand].astype(float) + 0.4) * np.exp(0.5 * quality[cand])
        pools.append((cand, w / w.sum(), rel_t, rel_c))
    uidx = rng.integers(0, len(users), n_sessions)
    qi = rng.integers(0, len(train_q), n_sessions)
    all_sku, all_sess, all_pos, all_rel_t, all_rel_c = [], [], [], [], []
    for s in range(n_sessions):
        cand, w, rel_t, rel_c = pools[qi[s]]
        n_pop = k - 2
        pick = rng.choice(cand, size=min(n_pop, len(cand)), replace=False, p=w)
        rand = rng.integers(0, len(df), k - len(pick))
        imp = np.concatenate([pick, rand])
        score = rel_c[imp] + 0.3 * quality[imp] + rng.normal(0, 0.8, len(imp))  # legacy ranker (noisy)
        imp = imp[np.argsort(-score)]
        all_sku.append(imp)
        all_sess.append(np.full(len(imp), s))
        all_pos.append(np.arange(len(imp)))
        all_rel_t.append(rel_t[imp])
        all_rel_c.append(rel_c[imp])
    sku = np.concatenate(all_sku); sess = np.concatenate(all_sess); pos = np.concatenate(all_pos)
    rel_t = np.concatenate(all_rel_t).astype(float); rel_c = np.concatenate(all_rel_c).astype(float)
    # user dependent terms
    u_rows = uidx[sess]
    med = users.median_spend.values[u_rows]
    prefs = [set(json.loads(p)) for p in users.preferred_brands.values]
    bm = np.array([brand[sku[i]] in prefs[u_rows[i]] for i in range(len(sku))], dtype=float)
    pf = price_fit(price[sku], med)
    exam = 1.0 / (1.0 + 0.35 * pos)
    pc, pa, pv, pr = latent_probs(rel_t, rel_c, quality[sku], bm, pf, latent.latent_return_base.values[sku], exam)
    click = rng.random(len(sku)) < pc
    atc = click & (rng.random(len(sku)) < pa)
    purchase = atc & (rng.random(len(sku)) < pv)
    ret = purchase & (rng.random(len(sku)) < pr)
    sess_df = pd.DataFrame(dict(
        session_id=sess, user_id=users.user_id.values[u_rows], query_id=train_q.query_id.values[qi[sess]],
        sku_id=df.sku_id.values[sku], position=pos, click=click.astype(np.int8), add_to_cart=atc.astype(np.int8),
        purchase=purchase.astype(np.int8), returned=ret.astype(np.int8)))
    return sess_df


def attach_history(df: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    agg = sessions.groupby("sku_id").agg(impressions=("click", "size"), clicks=("click", "sum"),
                                           carts=("add_to_cart", "sum"), purchases=("purchase", "sum"),
                                           returns=("returned", "sum"))
    df = df.merge(agg, left_on="sku_id", right_index=True, how="left")
    for c in ["impressions", "clicks", "carts", "purchases", "returns"]:
        df[c] = df[c].fillna(0).astype(int)
    # cold-start SKUs: zero interaction history by construction
    cold = df.is_cold_start.values
    for c in ["impressions", "clicks", "carts", "purchases", "returns"]:
        df.loc[cold, c] = 0
    df["historical_ctr"] = np.where(df.impressions > 0, df.clicks / df.impressions.clip(lower=1), 0.0)
    df["historical_cvr"] = np.where(df.clicks > 0, df.purchases / df.clicks.clip(lower=1), 0.0)
    cat_prior = df[df.purchases > 0].groupby("category").apply(lambda g: g.returns.sum() / max(g.purchases.sum(), 1))
    prior = df.category.map(cat_prior).fillna(0.1)
    df["return_rate"] = ((df.returns + 10 * prior) / (df.purchases + 10)).round(4)
    return df


def main(cfg=None, small: bool = False):
    cfg = cfg or load_config()
    if small:
        cfg = json.loads(json.dumps(cfg))
        cfg["data"].update(n_skus=3000, n_users=500, n_sessions=20000, n_train_queries=500,
                           n_val_queries=100, n_test_queries=100)
    setup_logging(cfg, "data_gen")
    rng = np.random.default_rng(cfg["seed"])
    out = resolve(cfg, "data")
    logger.info("Generating catalog ({} SKUs)", cfg["data"]["n_skus"])
    df, latent, true_sets, claimed_sets = build_catalog(cfg, rng)
    img = build_image_embeddings(df, cfg["retrieval"]["image_dim"], rng)
    inv = build_inventory(cfg, df, rng)
    users = build_users(cfg, df, rng)
    qdf = build_queries(cfg, df, rng)
    logger.info("{} queries, {} users", len(qdf), len(users))
    true_arr, claimed_arr = feature_arrays(true_sets), feature_arrays(claimed_sets)
    logger.info("Simulating {} sessions", cfg["data"]["n_sessions"])
    sessions = simulate_sessions(cfg, df, latent, users, qdf, true_arr, claimed_arr, rng)
    df = attach_history(df, sessions)
    pd.DataFrame(cfg["data"]["stores"]).to_parquet(out / "stores.parquet", index=False)
    df.to_parquet(out / "catalog.parquet", index=False)
    latent.to_parquet(out / "catalog_latent.parquet", index=False)
    inv.to_parquet(out / "inventory.parquet", index=False)
    users.to_parquet(out / "users.parquet", index=False)
    qdf.to_parquet(out / "queries.parquet", index=False)
    sessions.to_parquet(out / "sessions.parquet", index=False)
    np.save(out / "image_embeddings.npy", img)
    with open(out / "true_features.json", "w") as f:
        json.dump([sorted(s) for s in true_sets], f)
    with open(out / "claimed_features.json", "w") as f:
        json.dump([sorted(s) for s in claimed_sets], f)
    with open(out / "synonyms.json", "w") as f:
        json.dump(T.WORD_SYNONYMS, f, indent=1)
    logger.info("catalog={} cold={} sessions={} ctr={:.3f} cvr={:.3f}",
                len(df), int(df.is_cold_start.sum()), len(sessions), sessions.click.mean(),
                sessions.purchase.sum() / max(sessions.click.sum(), 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    main(small=ap.parse_args().small)
