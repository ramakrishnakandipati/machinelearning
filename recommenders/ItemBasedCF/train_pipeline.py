"""Offline batch pipeline for the Item-Based CF (IBCF) retail recommender.

Run:  python train_pipeline.py

Stages: synthetic omnichannel data -> time-decayed implicit matrix -> BM25 cosine +
shrinkage -> temporal split + grid search (beta, K) -> artifacts -> verification
report -> diagnostic charts. Everything lands under ./output/.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import sparse  # noqa: E402

try:
    import pyarrow  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyarrow is required: pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
ART = OUT / "artifacts"
CHARTS = OUT / "charts"

SEED = 42
N_USERS = 3000
N_DAYS = 180
START = pd.Timestamp("2025-01-01")
EVENT_W0 = {"view": 1.0, "add_to_cart": 3.0, "purchase": 5.0}
HALF_LIFE_DAYS = 30.0
LAMBDA = np.log(2) / HALF_LIFE_DAYS
BM25_K1, BM25_B = 1.2, 0.75
BETAS = [0, 5, 10, 25, 50, 100]
KS = [10, 20, 50, 100]
TOPN = 10
VAL_FRACTION = 0.2

CATEGORIES = ["Grocery", "Beverages", "Personal Care", "Home Care", "Electronics",
              "Apparel", "Footwear", "Toys", "Sports", "Books"]
CONSUMABLE_CATS = {"Grocery", "Beverages", "Personal Care", "Home Care"}
BASE_PRICE = {"Grocery": 4, "Beverages": 3, "Personal Care": 8, "Home Care": 9, "Electronics": 120,
              "Apparel": 35, "Footwear": 60, "Toys": 22, "Sports": 40, "Books": 15}


# --------------------------------------------------------------------------- data
def generate_data(rng: np.random.Generator):
    """Synthetic omnichannel retail data with category block structure + staple items."""
    rows, item_pop, is_staple = [], [], []
    n = 0
    for c_i, cat in enumerate(CATEGORIES):
        for s in range(3):
            sub = f"{cat}-S{s + 1}"
            ranks = rng.permutation(16) + 1
            for r in ranks:
                n += 1
                price = BASE_PRICE[cat] * rng.lognormal(0, 0.4)
                rows.append((f"SKU{n:04d}", cat, sub, round(price, 2), cat in CONSUMABLE_CATS,
                             bool(rng.random() > 0.08), rng.uniform(0.2, 0.5)))
                item_pop.append(float(r) ** -1.1)
                is_staple.append(False)
    # staples: bought by almost everybody, low margin, consumable
    for k, cat in enumerate(["Grocery", "Grocery", "Grocery", "Grocery", "Beverages", "Personal Care",
                             "Home Care", "Home Care"]):
        n += 1
        rows.append((f"SKU{n:04d}", cat, f"{cat}-S1", round(rng.uniform(1.5, 4), 2), True, True,
                     rng.uniform(0.04, 0.09)))
        item_pop.append(1.0)
        is_staple.append(True)
    catalog = pd.DataFrame(rows, columns=["item_id", "category", "sub_category", "price", "is_consumable",
                                          "in_stock", "margin_rate"])
    item_pop = np.array(item_pop)
    is_staple = np.array(is_staple)
    cat_arr = catalog["category"].to_numpy()
    sub_arr = catalog["sub_category"].to_numpy()
    items = catalog["item_id"].to_numpy()
    cat_items = {c: np.where((cat_arr == c) & ~is_staple)[0] for c in CATEGORIES}
    sub_items = {s: np.where((sub_arr == s) & ~is_staple)[0] for s in np.unique(sub_arr)}
    staple_idx = np.where(is_staple)[0]
    nonstaple = np.where(~is_staple)[0]
    gp = item_pop[nonstaple] / item_pop[nonstaple].sum()
    sp = item_pop[staple_idx] / item_pop[staple_idx].sum()

    ev_u, ev_i, ev_t, ev_type = [], [], [], []
    tr_u, tr_i, tr_t, tr_q, tr_c = [], [], [], [], []
    channels = np.array(["web", "app", "pos"])
    price = catalog["price"].to_numpy()
    margin = catalog["margin_rate"].to_numpy()
    for u in range(N_USERS):
        uid = f"U{u:05d}"
        n_cats = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        cats = rng.choice(CATEGORIES, n_cats, replace=False)
        aff = rng.dirichlet(np.ones(n_cats) * 2)
        fav_sub = {c: f"{c}-S{rng.integers(1, 4)}" for c in cats}
        n_int = rng.poisson(40) + 5
        days = (N_DAYS - 1) * rng.beta(1.6, 1.0, n_int)
        for d in days:
            x = rng.random()
            if x < 0.12:
                it = rng.choice(staple_idx, p=sp)
            elif x < 0.18:
                it = rng.choice(nonstaple, p=gp)
            else:
                c = rng.choice(cats, p=aff)
                pool = sub_items[fav_sub[c]] if rng.random() < 0.6 else cat_items[c]
                w = item_pop[pool]
                it = rng.choice(pool, p=w / w.sum())
            t = START + pd.Timedelta(days=float(d))
            ev_u.append(uid); ev_i.append(items[it]); ev_t.append(t); ev_type.append("view")
            if rng.random() < 0.4:
                t2 = t + pd.Timedelta(minutes=float(rng.integers(1, 120)))
                ev_u.append(uid); ev_i.append(items[it]); ev_t.append(t2); ev_type.append("add_to_cart")
                if rng.random() < 0.6:
                    t3 = t2 + pd.Timedelta(minutes=float(rng.integers(1, 240)))
                    ev_u.append(uid); ev_i.append(items[it]); ev_t.append(t3); ev_type.append("purchase")
                    q = 1 + (rng.poisson(1.0) if is_staple[it] else 0)
                    tr_u.append(uid); tr_i.append(items[it]); tr_t.append(t3); tr_q.append(q)
                    tr_c.append(rng.choice(channels))
    events = pd.DataFrame({"user_id": ev_u, "item_id": ev_i, "event_type": ev_type, "timestamp": ev_t})
    tr = pd.DataFrame({"user_id": tr_u, "item_id": tr_i, "timestamp": tr_t, "quantity": tr_q, "channel": tr_c})
    rate = catalog.set_index("item_id")["margin_rate"]
    pr = catalog.set_index("item_id")["price"]
    tr["gross_margin"] = (tr["item_id"].map(pr) * tr["quantity"] * tr["item_id"].map(rate)).round(2)
    staple_ids = items[staple_idx].tolist()
    catalog_out = catalog.drop(columns=["margin_rate"])
    return tr, events, catalog_out, staple_ids


# ----------------------------------------------------------------------- features
def build_matrix(events, trans, user_idx, item_idx, t_ref, t_max, lam=LAMBDA):
    """Time-decayed implicit user x item matrix using events with t <= t_max (days)."""
    ev = events[(events["event_type"] != "purchase") & (events["t"] <= t_max)]
    w_ev = ev["event_type"].map(EVENT_W0).to_numpy() * np.exp(-lam * (t_ref - ev["t"].to_numpy()))
    tr = trans[trans["t"] <= t_max]
    w_tr = EVENT_W0["purchase"] * (1 + np.log(tr["quantity"].to_numpy())) * np.exp(-lam * (t_ref - tr["t"].to_numpy()))
    u = np.concatenate([ev["user_id"].map(user_idx).to_numpy(), tr["user_id"].map(user_idx).to_numpy()])
    i = np.concatenate([ev["item_id"].map(item_idx).to_numpy(), tr["item_id"].map(item_idx).to_numpy()])
    w = np.concatenate([w_ev, w_tr])
    return sparse.csr_matrix((w, (u, i)), shape=(len(user_idx), len(item_idx))).toarray()


# ---------------------------------------------------------------------- similarity
def bm25_cosine(X, k1=BM25_K1, b=BM25_B):
    """BM25-weighted item-item cosine + co-occurrence counts. Dense (fine for ~500 items;
    swap for sparse X.T @ X + chunked top-K at larger catalog sizes)."""
    B = (X > 0).astype(np.float32)
    n_users = X.shape[0]
    df = B.sum(0)
    idf = np.log1p((n_users - df + 0.5) / (df + 0.5))          # damps popular (staple) items
    ulen = X.sum(1)
    avg = ulen[ulen > 0].mean()
    denom = X + k1 * (1 - b + b * ulen / avg)[:, None]          # damps heavy users
    W = X * (k1 + 1) / np.where(denom == 0, 1, denom) * idf[None, :]
    W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
    S = W.T @ W
    np.fill_diagonal(S, 0)
    nij = B.T @ B
    np.fill_diagonal(nij, 0)
    return S, nij


def shrink(S, nij, beta):
    return S * np.where(nij > 0, nij / (nij + beta + 1e-12), 0.0)


def prune_topk(S, k):
    out = np.zeros_like(S)
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    rows = np.arange(S.shape[0])[:, None]
    out[rows, idx] = S[rows, idx]
    return out


# ------------------------------------------------------------------------ metrics
def evaluate(scores, G, mask, n=TOPN):
    """NDCG@n, Recall@n, coverage. scores: users x items; G: binary truth; mask: bool (excluded)."""
    sc = np.where(mask, -np.inf, scores)
    top = np.argpartition(-sc, n - 1, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(sc, top, 1), axis=1)
    top = np.take_along_axis(top, order, 1)
    hits = np.take_along_axis(G, top, 1)
    disc = 1 / np.log2(np.arange(n) + 2)
    dcg = (hits * disc).sum(1)
    nrel = G.sum(1)
    idcg = np.array([disc[: int(min(n, r))].sum() for r in nrel])
    recall = hits.sum(1) / np.maximum(nrel, 1)
    return {"ndcg": float((dcg / np.maximum(idcg, 1e-12)).mean()), "recall": float(recall.mean()),
            "coverage": float(len(np.unique(top)) / scores.shape[1])}


def neighbor_stats(S_topk, cat_arr, staple_mask):
    """Share of neighbor slots filled by staples and score-weighted intra-category share (non-staple seeds)."""
    nz = S_topk > 0
    seeds = ~staple_mask
    slots = nz[seeds].sum()
    staple_share = (nz[seeds][:, staple_mask].sum() / max(slots, 1))
    same = cat_arr[:, None] == cat_arr[None, :]
    intra = (S_topk * same)[seeds].sum() / max(S_topk[seeds].sum(), 1e-12)
    return float(staple_share), float(intra)


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    for d in (ART, CHARTS, OUT / "api_tests"):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/6] Generating synthetic omnichannel data ...")
    trans, events, catalog, staple_ids = generate_data(rng)
    for df in (trans, events):
        df["t"] = (df["timestamp"] - START).dt.total_seconds() / 86400.0
    item_ids = catalog["item_id"].tolist()
    item_idx = {k: i for i, k in enumerate(item_ids)}
    user_ids = sorted(events["user_id"].unique())
    user_idx = {k: i for i, k in enumerate(user_ids)}
    n_items = len(item_ids)
    cat_arr = catalog["category"].to_numpy()
    staple_mask = np.isin(item_ids, staple_ids)
    consumable = catalog["is_consumable"].to_numpy()
    print(f"      users={len(user_ids)} items={n_items} events={len(events)} transactions={len(trans)}")

    print("[2/6] Temporal split + hyperparameter grid search ...")
    split_day = N_DAYS * (1 - VAL_FRACTION)
    X_tr = build_matrix(events, trans, user_idx, item_idx, t_ref=split_day, t_max=split_day)
    val_tr = trans[trans["t"] > split_day]
    train_tr = trans[trans["t"] <= split_day]
    assert train_tr["t"].max() <= split_day < val_tr["t"].min(), "temporal leakage"
    G = np.zeros((len(user_ids), n_items), dtype=np.float32)
    G[val_tr["user_id"].map(user_idx).to_numpy(), val_tr["item_id"].map(item_idx).to_numpy()] = 1
    P = np.zeros_like(G)
    P[train_tr["user_id"].map(user_idx).to_numpy(), train_tr["item_id"].map(item_idx).to_numpy()] = 1
    mask = (P > 0) & (~consumable[None, :])        # durable repeat suppression
    G = G * (~mask)                                # cannot recommend -> not ground truth
    keep = (G.sum(1) > 0) & (X_tr.sum(1) > 0)
    Xe, Ge, maske = X_tr[keep], G[keep], mask[keep]
    print(f"      eval users={int(keep.sum())}")

    S_raw, nij = bm25_cosine(X_tr)
    grid = []
    for beta in BETAS:
        Ss = shrink(S_raw, nij, beta)
        for k in KS:
            m = evaluate(Xe @ prune_topk(Ss, k), Ge, maske)
            grid.append({"beta": beta, "K": k, **m})
            print(f"      beta={beta:>3} K={k:>3} ndcg={m['ndcg']:.4f} recall={m['recall']:.4f} cov={m['coverage']:.3f}")
    gdf = pd.DataFrame(grid).sort_values(["ndcg", "recall"], ascending=False)
    best = gdf.iloc[0].to_dict()
    best_beta, best_k = int(best["beta"]), int(best["K"])
    print(f"      best: beta={best_beta} K={best_k} ndcg={best['ndcg']:.4f} recall={best['recall']:.4f}")

    # baselines on the same split
    pop = np.asarray(P.sum(0))
    pop_m = evaluate(np.tile(pop, (Xe.shape[0], 1)).astype(float), Ge, maske)
    cooc_k = prune_topk(nij.astype(float), best_k)
    raw_m = evaluate(Xe @ cooc_k, Ge, maske)

    print("[3/6] Refitting on full data + writing artifacts ...")
    t_end = float(max(events["t"].max(), trans["t"].max()))
    X_full = build_matrix(events, trans, user_idx, item_idx, t_ref=t_end, t_max=t_end)
    S_full_raw, nij_full = bm25_cosine(X_full)
    S_full = shrink(S_full_raw, nij_full, best_beta)
    S_top = prune_topk(S_full, best_k)
    rows = []
    for i in range(n_items):
        nb = np.where(S_top[i] > 0)[0]
        nb = nb[np.argsort(-S_top[i, nb])]
        rows += [(item_ids[i], item_ids[j], float(S_top[i, j]), r + 1) for r, j in enumerate(nb)]
    sim_df = pd.DataFrame(rows, columns=["item_id", "neighbor_id", "score", "rank"])
    sim_df.to_parquet(ART / "item_sim_map.parquet", index=False)
    catalog.to_parquet(ART / "catalog.parquet", index=False)

    # user history: top decayed items + purchase recency (for durable / consumable suppression)
    lp = trans.groupby(["user_id", "item_id"]).agg(n_purchases=("quantity", "size"), last_purchase_ts=("timestamp", "max")).reset_index()
    hist = []
    for u_i, uid in enumerate(user_ids):
        row = X_full[u_i]
        top = np.argsort(-row)[:50]
        top = top[row[top] > 0]
        hist += [(uid, item_ids[j], float(row[j])) for j in top]
    hist_df = pd.DataFrame(hist, columns=["user_id", "item_id", "score"])
    hist_df = hist_df.merge(lp, on=["user_id", "item_id"], how="outer")
    hist_df["score"] = hist_df["score"].fillna(0.0)
    hist_df["n_purchases"] = hist_df["n_purchases"].fillna(0).astype(int)
    hist_df.to_parquet(ART / "user_history.parquet", index=False)

    tr_full = trans.copy()
    tr_full["w"] = np.exp(-LAMBDA * (t_end - tr_full["t"])) * tr_full["gross_margin"]
    fb = tr_full.groupby("item_id")["w"].sum().sort_values(ascending=False)
    cat_i = catalog.set_index("item_id")
    fallback = [{"item_id": i, "score": round(float(s), 4), "category": cat_i.loc[i, "category"],
                 "sub_category": cat_i.loc[i, "sub_category"]} for i, s in fb.items()]
    (ART / "fallback_popular.json").write_text(json.dumps(fallback, indent=1))

    version = "ibcf-" + hashlib.sha256(sim_df.to_csv(index=False).encode()).hexdigest()[:8]
    meta = {"model_version": version, "reference_time": (START + pd.Timedelta(days=t_end)).isoformat(),
            "beta": best_beta, "K": best_k, "half_life_days": HALF_LIFE_DAYS, "staple_item_ids": staple_ids,
            "consumable_repurchase_days": 14, "n_items": n_items, "n_users": len(user_ids)}
    (ART / "model_meta.json").write_text(json.dumps(meta, indent=1))

    print("[4/6] Verification report ...")
    s_final_stats = neighbor_stats(prune_topk(shrink(S_raw, nij, best_beta), best_k), cat_arr, staple_mask)
    s_raw_stats = neighbor_stats(cooc_k, cat_arr, staple_mask)
    checks = {
        "no_temporal_leakage": bool(train_tr["t"].max() <= split_day < val_tr["t"].min()),
        "beats_popularity_ndcg": bool(best["ndcg"] > pop_m["ndcg"]),
        "beats_raw_cooccurrence_ndcg": bool(best["ndcg"] > raw_m["ndcg"]),
        "intra_category_share_gt_0.5": bool(s_final_stats[1] > 0.5),
        "staple_bias_reduced_vs_raw": bool(s_final_stats[0] < s_raw_stats[0]),
        "every_item_has_neighbors": bool(sim_df["item_id"].nunique() == n_items),
    }
    report = {
        "model_version": version, "generated_at": pd.Timestamp.now().isoformat(),
        "best_params": {"beta": best_beta, "K": best_k},
        "validation_metrics": {"ndcg@10": best["ndcg"], "recall@10": best["recall"], "coverage": best["coverage"]},
        "baselines": {"popularity": pop_m, "raw_cooccurrence_topK": raw_m},
        "neighbor_quality": {
            "final": {"staple_slot_share": s_final_stats[0], "intra_category_score_share": s_final_stats[1]},
            "raw_cooccurrence": {"staple_slot_share": s_raw_stats[0], "intra_category_score_share": s_raw_stats[1]}},
        "split": {"split_day": split_day, "train_max_day": float(train_tr["t"].max()), "val_min_day": float(val_tr["t"].min()),
                  "eval_users": int(keep.sum())},
        "grid": grid, "checks": checks, "all_checks_passed": bool(all(checks.values())),
        "artifacts": {p.name: sha(p) for p in sorted(ART.glob("*")) if p.name != "verification_report.json"},
    }
    (ART / "verification_report.json").write_text(json.dumps(report, indent=1))
    print("      checks:", checks)

    print("[5/6] Charts ...")
    sns.set_theme(style="whitegrid", context="notebook")
    make_charts(events, trans, X_full, catalog, grid, best_beta, best_k, S_full_raw, nij_full, S_full,
                cat_arr, staple_mask, report)
    print(f"[6/6] Done in {time.time() - t0:.1f}s -> {OUT}")


# -------------------------------------------------------------------------- charts
def make_charts(events, trans, X, catalog, grid, best_beta, best_k, S_raw, nij, S_shr, cat_arr, staple_mask, report):
    # 01 sparsity / long tail
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    freq = np.sort((X > 0).sum(0))[::-1]
    ax[0].loglog(np.arange(1, len(freq) + 1), freq, "o-", ms=3, color="#2a6fbb")
    ax[0].set(title="Item interaction frequency (long tail)", xlabel="item rank (log)", ylabel="# users (log)")
    ufreq = (X > 0).sum(1)
    sns.histplot(ufreq, bins=40, ax=ax[1], color="#2a9d8f")
    ax[1].set(title="Distinct items per user", xlabel="items", ylabel="users")
    order_i = np.argsort(-(X > 0).sum(0))[:120]
    order_u = np.argsort(-(X > 0).sum(1))[:120]
    sns.heatmap((X[np.ix_(order_u, order_i)] > 0).astype(int), cmap="Blues", cbar=False, ax=ax[2],
                xticklabels=False, yticklabels=False)
    dens = (X > 0).mean()
    ax[2].set(title=f"Top-120x120 interaction map | full matrix {X.shape[0]}x{X.shape[1]}, sparsity {100 * (1 - dens):.2f}%",
              xlabel="items (by popularity)", ylabel="users (by activity)")
    fig.tight_layout(); fig.savefig(CHARTS / "01_data_sparsity_and_long_tail.png", dpi=200); plt.close(fig)

    # 02 funnel + decay
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    counts = events["event_type"].value_counts().reindex(["view", "add_to_cart", "purchase"])
    bars = ax[0].barh(counts.index[::-1], counts.values[::-1], color=["#e76f51", "#f4a261", "#2a6fbb"])
    for b, v, nm in zip(bars, counts.values[::-1], counts.index[::-1]):
        pct = 100 * v / counts.iloc[0]
        ax[0].text(v, b.get_y() + b.get_height() / 2, f" {v:,} ({pct:.0f}% of views)", va="center")
    ax[0].set(title="Event funnel: view -> add_to_cart -> purchase", xlabel="events", xlim=(0, counts.max() * 1.3))
    d = np.arange(0, 121)
    for nm, w0 in EVENT_W0.items():
        ax[1].plot(d, w0 * np.exp(-LAMBDA * d), label=f"{nm} (w0={w0:g})", lw=2)
    ax[1].axvline(HALF_LIFE_DAYS, color="gray", ls="--"); ax[1].text(HALF_LIFE_DAYS + 1, 4.5, f"half-life {HALF_LIFE_DAYS:g}d")
    ax[1].set(title=r"Exponential time decay $w(t)=w_0 e^{-\lambda \Delta t}$", xlabel="age (days)", ylabel="weight")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(CHARTS / "02_feature_time_decay_and_funnel.png", dpi=200); plt.close(fig)

    # 03 tuning curves
    g = pd.DataFrame(grid)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for a, (col, title) in zip(ax, [("ndcg", "Validation NDCG@10"), ("recall", "Validation Recall@10"), ("coverage", "Catalog coverage")]):
        for k, sub in g.groupby("K"):
            a.plot(sub["beta"], sub[col], "o-", label=f"K={k}")
        row = g[(g.beta == best_beta) & (g.K == best_k)].iloc[0]
        a.scatter([best_beta], [row[col]], marker="*", s=380, color="gold", edgecolor="k", zorder=5, label="selected")
        a.set(title=title, xlabel=r"shrinkage $\beta$", ylabel=col)
        a.legend(fontsize=8)
    fig.suptitle(f"Hyperparameter tuning (selected beta={best_beta}, K={best_k})")
    fig.tight_layout(); fig.savefig(CHARTS / "03_tuning_and_validation_curves.png", dpi=200); plt.close(fig)

    # 04 verification heatmaps
    order = np.argsort(cat_arr, kind="stable")
    bounds = np.cumsum([np.sum(cat_arr == c) for c in CATEGORIES])[:-1]
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for a, M, title, cm in [(ax[0], np.log1p(nij), "Raw co-occurrence (log1p) - staples bleed across blocks", "mako_r"),
                             (ax[1], S_shr, "BM25 cosine + shrinkage - block-diagonal", "rocket_r")]:
        sns.heatmap(M[np.ix_(order, order)], ax=a, cmap=cm, xticklabels=False, yticklabels=False, cbar=True,
                    vmax=np.quantile(M, 0.995))
        for bnd in bounds:
            a.axhline(bnd, color="w", lw=0.6); a.axvline(bnd, color="w", lw=0.6)
        a.set_title(title, fontsize=11)
    q = report["neighbor_quality"]
    labels = ["staple slot share", "intra-category share"]
    x = np.arange(2); w = 0.35
    ax[2].bar(x - w / 2, [q["raw_cooccurrence"]["staple_slot_share"], q["raw_cooccurrence"]["intra_category_score_share"]], w, label="raw co-occurrence", color="#e76f51")
    ax[2].bar(x + w / 2, [q["final"]["staple_slot_share"], q["final"]["intra_category_score_share"]], w, label="BM25 + shrinkage", color="#2a6fbb")
    ax[2].set_xticks(x, labels); ax[2].set_ylim(0, 1); ax[2].legend()
    ax[2].set_title("Staple-bias removal & category purity (top-K neighbors)", fontsize=11)
    fig.tight_layout(); fig.savefig(CHARTS / "04_model_verification_similarity_heatmap.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
