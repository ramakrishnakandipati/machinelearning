"""Feature engineering, temporal split, preprocessors, hard negatives, dataset summary + EDA chart."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .config import CHARTS, DATA, HIST_LEN, METRICS, MODELS, SEED, DataCfg  # noqa: E402
from .logger import get_logger, stage_timer  # noqa: E402

log = get_logger("data", "02_data_pipeline.log")

USER_DENSE = ["recency_days", "frequency_90d", "monetary_value_90d", "avg_basket_size"]
ITEM_DENSE = ["price_normalized", "margin_score", "in_stock_ratio", "popularity_score_7d"]
CAT_COLS = {"tier": "loyalty_tier", "chan": "preferred_channel", "region": "home_region", "device": "device_type"}


class Preprocessors:
    """Fitted encoders/scalers. Index 0 is reserved for OOV / padding in every vocabulary."""

    def __init__(self, user_map, item_map, vocabs, user_scaler, item_scaler, hist_len=HIST_LEN):
        self.user_map, self.item_map, self.vocabs = user_map, item_map, vocabs
        self.user_scaler, self.item_scaler, self.hist_len = user_scaler, item_scaler, hist_len

    def sizes(self) -> dict:
        s = {k: len(v) + 1 for k, v in self.vocabs.items()}
        s["user"], s["item"] = int(self.user_map.max()) + 2, int(self.item_map.max()) + 2
        return s

    def _cat(self, name, values):
        v = self.vocabs[name]
        return np.fromiter((v.get(x, 0) for x in values), dtype=np.int64, count=len(values))

    def encode_users(self, user_id, tier, chan, region, dense_raw, hist_raw, dow, hour, device) -> dict:
        user_id = np.asarray(user_id, dtype=np.int64)
        uid = np.where((user_id >= 0) & (user_id < len(self.user_map)), self.user_map[np.clip(user_id, 0, len(self.user_map) - 1)], 0)
        hist_raw = np.asarray(hist_raw, dtype=np.int64)
        hist = np.where(hist_raw >= 0, self.item_map[np.clip(hist_raw, 0, len(self.item_map) - 1)], 0)
        dense = self.user_scaler.transform(np.log1p(np.asarray(dense_raw, dtype=np.float64))).astype(np.float32)
        return {"uid": uid.astype(np.int64), "tier": self._cat("tier", tier), "chan": self._cat("chan", chan),
                "region": self._cat("region", region), "dense": dense, "hist": hist.astype(np.int64),
                "dow": np.asarray(dow, dtype=np.int64), "hour": np.asarray(hour, dtype=np.int64),
                "device": self._cat("device", device)}

    def encode_items(self, items: pd.DataFrame, pop7d: np.ndarray) -> dict:
        d = np.stack([np.log(items["price"].to_numpy()), items["margin_score"].to_numpy(),
                      items["in_stock_ratio"].to_numpy(), np.log1p(pop7d)], 1)
        return {"iid": self.item_map[items["item_id"].to_numpy()].astype(np.int64),
                "cat1": self._cat("cat1", items["category_l1"]), "cat2": self._cat("cat2", items["category_l2"]),
                "brand": self._cat("brand", items["brand"]), "season": self._cat("season", items["seasonality_tag"]),
                "dense": self.item_scaler.transform(d).astype(np.float32)}


@dataclass
class Prepared:
    pre: Preprocessors
    splits: dict                      # name -> dict of numpy arrays (+ 'item','user_raw','unseen_user','pref_channel')
    item_table: dict                  # encoded item features, row == item_id
    item_pop7d: np.ndarray
    log_q: np.ndarray                 # log sampling probability of each item (train positives)
    snapshot: pd.DataFrame            # as-of-now user features (online feature store source)
    purchased: pd.DataFrame           # (user_id, item_id) all purchases
    items: pd.DataFrame
    users: pd.DataFrame
    summary: dict = field(default_factory=dict)


def _rfm_and_history(pos: pd.DataFrame, item_price: np.ndarray, n_days: int, hist_len: int):
    """Point-in-time RFM + last-N history for every positive event (events strictly before it), plus an
    as-of-end snapshot per user. `pos` must be sorted by (user_id, ts)."""
    uid, t, day = pos.user_id.to_numpy(), pos.ts.to_numpy(), pos.day.to_numpy()
    items = pos.item_id.to_numpy()
    purch = (pos.event_type == "purchase").to_numpy()
    price = item_price[items] * purch
    n = len(pos)
    feats, hist = np.zeros((n, 4)), np.full((n, hist_len), -1, dtype=np.int64)
    bounds = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1], True])
    snap_u, snap_f, snap_h = [], [], []
    for s, e in zip(bounds[:-1], bounds[1:]):
        k, tt = e - s, t[s:e]
        pm = purch[s:e]
        cum_m, cum_p = np.r_[0, np.cumsum(price[s:e])], np.r_[0, np.cumsum(pm)]
        flag = np.zeros(k)
        pdays = day[s:e][pm]
        if pdays.size:
            flag[pm] = np.r_[1, pdays[1:] != pdays[:-1]]
        cum_o = np.r_[0, np.cumsum(flag)]

        def feat(q, pre):
            lo = np.minimum(np.searchsorted(tt, q - 90, "left"), pre)
            freq, money = pre - lo, cum_m[pre] - cum_m[lo]
            npur, nor = cum_p[pre] - cum_p[lo], cum_o[pre] - cum_o[lo]
            rec = np.clip(np.where(pre > 0, q - tt[np.maximum(pre - 1, 0)], 90.0), 0, 90)
            return np.stack([rec, freq, money, npur / np.maximum(nor, 1)], 1)

        feats[s:e] = feat(tt, np.arange(k))
        padded = np.r_[np.full(hist_len, -1), items[s:e]]
        hist[s:e] = sliding_window_view(padded, hist_len)[:k]
        snap_u.append(uid[s])
        snap_f.append(feat(np.array([float(n_days)]), np.array([k]))[0])
        snap_h.append(padded[-hist_len:])
    return feats, hist, np.array(snap_u), np.array(snap_f), np.array(snap_h)


def prepare(users: pd.DataFrame, items: pd.DataFrame, ev: pd.DataFrame, cfg: DataCfg | None = None,
            hist_len: int = HIST_LEN, seed: int = SEED) -> Prepared:
    cfg = cfg or DataCfg()
    rng = np.random.default_rng(seed)
    nU, nI = len(users), len(items)
    pos = ev[ev.event_type != "view"].reset_index(drop=True)
    pos = pos.merge(users[["user_id", "loyalty_tier", "preferred_channel", "home_region"]], on="user_id", how="left")
    feats, hist, snap_u, snap_f, snap_h = _rfm_and_history(pos, items["price"].to_numpy(), cfg.n_days, hist_len)

    tr_mask = (pos.day < cfg.train_end).to_numpy()
    split_of = np.where(tr_mask, 0, np.where(pos.day < cfg.val_end, 1, 2))

    # ---- vocabularies / scalers fit on TRAIN only (no leakage)
    seen_u = np.unique(pos.user_id[tr_mask])
    seen_i = np.unique(pos.item_id[tr_mask])
    user_map, item_map = np.zeros(nU, dtype=np.int64), np.zeros(nI, dtype=np.int64)
    user_map[seen_u], item_map[seen_i] = np.arange(1, len(seen_u) + 1), np.arange(1, len(seen_i) + 1)
    vocabs = {"tier": {v: i + 1 for i, v in enumerate(sorted(users.loyalty_tier.unique()))},
              "chan": {v: i + 1 for i, v in enumerate(sorted(users.preferred_channel.unique()))},
              "region": {v: i + 1 for i, v in enumerate(sorted(users.home_region.unique()))},
              "device": {v: i + 1 for i, v in enumerate(sorted(ev.device_type.unique()))},
              "cat1": {v: i + 1 for i, v in enumerate(sorted(items.category_l1.unique()))},
              "cat2": {v: i + 1 for i, v in enumerate(sorted(items.category_l2.unique()))},
              "brand": {v: i + 1 for i, v in enumerate(sorted(items.brand.unique()))},
              "season": {v: i + 1 for i, v in enumerate(sorted(items.seasonality_tag.unique()))}}
    u_scaler = StandardScaler().fit(np.log1p(feats[tr_mask]))
    pop7 = np.bincount(pos.item_id[(pos.day >= cfg.train_end - 7) & tr_mask], minlength=nI).astype(float)
    tmp_items = np.stack([np.log(items.price), items.margin_score, items.in_stock_ratio, np.log1p(pop7)], 1)
    i_scaler = StandardScaler().fit(tmp_items)
    pre = Preprocessors(user_map, item_map, vocabs, u_scaler, i_scaler, hist_len)
    item_table = pre.encode_items(items, pop7)

    # ---- per-split encoded examples
    enc = pre.encode_users(pos.user_id, pos.loyalty_tier, pos.preferred_channel, pos.home_region, feats, hist,
                           pos.dow, pos.hour, pos.device_type)
    enc["item"] = pos.item_id.to_numpy().astype(np.int64)
    enc["user_raw"] = pos.user_id.to_numpy()
    enc["unseen_user"] = (enc["uid"] == 0)
    enc["pref_channel"] = pos.preferred_channel.to_numpy()
    enc["event_type"] = pos.event_type.to_numpy()
    enc["day"] = pos.day.to_numpy()

    # ---- hard negatives for train: same category_l1, viewed (train window) but never positively engaged
    cat1_of = item_table["cat1"]
    views = ev[(ev.event_type == "view") & (ev.day < cfg.train_end)][["user_id", "item_id"]].drop_duplicates()
    posp = pos[tr_mask][["user_id", "item_id"]].drop_duplicates()
    hn = views.merge(posp, how="left", on=["user_id", "item_id"], indicator=True)
    hn = hn[hn["_merge"] == "left_only"]
    pool: dict[int, dict[int, np.ndarray]] = {}
    for u, g in hn.groupby("user_id"):
        c = cat1_of[g.item_id.to_numpy()]
        pool[u] = {int(k): g.item_id.to_numpy()[c == k] for k in np.unique(c)}
    hard = np.full(len(pos), -1, dtype=np.int64)
    tr_idx = np.flatnonzero(tr_mask)
    for i in tr_idx:
        p = pool.get(enc["user_raw"][i])
        if p:
            arr = p.get(int(cat1_of[enc["item"][i]]))
            if arr is not None and len(arr):
                hard[i] = arr[rng.integers(len(arr))]
    enc["hard_item"] = hard

    splits = {}
    for name, code in (("train", 0), ("val", 1), ("test", 2)):
        m = split_of == code
        splits[name] = {k: v[m] for k, v in enc.items()}
    cnt = np.bincount(splits["train"]["item"], minlength=nI).astype(float)
    log_q = np.log((cnt + 1) / (cnt + 1).sum())

    snap = pd.DataFrame({"user_id": snap_u})
    snap = snap.merge(users[["user_id", "loyalty_tier", "preferred_channel", "home_region"]], on="user_id")
    for j, c in enumerate(USER_DENSE):
        snap[c] = snap_f[:, j]
    snap["recent_item_history"] = list(snap_h)
    purchased = pos[pos.event_type == "purchase"][["user_id", "item_id"]].drop_duplicates()

    te_u = splits["test"]["user_raw"]
    summary = {
        "rows": {"users": nU, "items": nI, "events_total": int(len(ev)), "positive_events": int(len(pos)),
                 "views": int((ev.event_type == "view").sum()), "purchases": int((ev.event_type == "purchase").sum())},
        "sparsity_pct": float(100 * (1 - len(pos[["user_id", "item_id"]].drop_duplicates()) / (nU * nI))),
        "split_boundaries_day": {"train": [0, cfg.train_end], "val": [cfg.train_end, cfg.val_end], "test": [cfg.val_end, cfg.n_days]},
        "split_examples": {k: int(len(v["item"])) for k, v in splits.items()},
        "cold_start": {
            "users_unseen_in_train_pct": float(100 * (1 - len(seen_u) / nU)),
            "items_unseen_in_train_pct": float(100 * (1 - len(seen_i) / nI)),
            "test_examples_from_unseen_users_pct": float(100 * splits["test"]["unseen_user"].mean()),
            "test_examples_on_unseen_items_pct": float(100 * (item_map[splits["test"]["item"]] == 0).mean()),
            "test_unique_users": int(len(np.unique(te_u)))},
        "train_examples_with_hard_negative_pct": float(100 * (hard[tr_idx] >= 0).mean()),
        "history_len": hist_len,
        "leakage_check": {"train_max_day": float(pos.day[tr_mask].max()), "val_min_day": float(pos.day[split_of == 1].min()),
                          "passed": bool(pos.day[tr_mask].max() < pos.day[split_of == 1].min())},
    }
    return Prepared(pre, splits, item_table, pop7, log_q, snap, purchased, items, users, summary)


def eda_chart(users, items, ev, path):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    sns.countplot(x=users.preferred_channel, order=["Web", "Mobile", "In-Store"], ax=ax[0, 0], color="#2a6fbb")
    ax[0, 0].set(title="User preferred channel mix", xlabel="")
    c = ev.merge(items[["item_id", "category_l1"]], on="item_id").category_l1.value_counts()
    ax[0, 1].barh(c.index[::-1], c.values[::-1], color="#2a9d8f")
    ax[0, 1].set(title="Interactions by category_l1")
    f = np.sort(ev.item_id.value_counts().to_numpy())[::-1]
    ax[1, 0].loglog(np.arange(1, len(f) + 1), f, color="#e76f51")
    ax[1, 0].set(title="Item interaction long tail (log-log)", xlabel="item rank", ylabel="# interactions")
    sns.countplot(x=users.loyalty_tier, order=["Bronze", "Silver", "Gold", "Platinum"], ax=ax[1, 1], color="#f4a261")
    ax[1, 1].set(title="Loyalty tier distribution", xlabel="")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(cfg: DataCfg | None = None) -> Prepared:
    from . import data_generator
    cfg = cfg or DataCfg()
    users, items, ev = data_generator.main(cfg)
    with stage_timer(log, "feature engineering + temporal split"):
        prep = prepare(users, items, ev, cfg)
    log.info("dataset summary: %s", json.dumps(prep.summary))
    assert prep.summary["leakage_check"]["passed"], "temporal leakage"
    (METRICS / "dataset_summary.json").write_text(json.dumps(prep.summary, indent=2))
    joblib.dump({"pre": prep.pre, "log_q": prep.log_q, "item_pop7d": prep.item_pop7d}, MODELS / "preprocessors.joblib")
    prep.snapshot.to_parquet(DATA / "user_features_snapshot.parquet", index=False)
    prep.purchased.to_parquet(DATA / "purchased.parquet", index=False)
    eda_chart(users, items, ev, CHARTS / "eda_feature_distributions.png")
    log.info("saved preprocessors, snapshot (%d users), EDA chart", len(prep.snapshot))
    return prep


if __name__ == "__main__":
    main()
