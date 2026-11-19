"""Text (TF-IDF -> SVD), visual, tabular and temporal encoders.

All statistics (vocabularies, TF-IDF, scalers) are fitted on TRAIN-visible entities only:
  * text / vocab / scaler fit on warm (non cold) items
  * user vocab on users that appear in train
Unseen categorical values map to a reserved UNK index (0), which makes the encoders usable
for zero-shot items / users / stores at serving time.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ITEM_CAT_COLS = ["l1", "l2", "l3", "brand", "price_tier", "color", "material"]
USER_CAT_COLS = ["acquisition_channel", "loyalty_tier", "home_region", "home_store", "device_os"]
CTX_CAT_COLS = ["device", "store_id", "channel"]
BASE_TS = 1735689600
DAY = 86400
HORIZON = 180 * DAY


@dataclass
class Features:
    item_ids: list
    user_ids: list
    item_cat: np.ndarray        # (nI, 7) int64
    item_cont: np.ndarray       # (nI, 3) float32
    item_text: np.ndarray       # (nI, text_dim)
    item_vis: np.ndarray        # (nI, vis_dim)
    user_cat: np.ndarray        # (nU, 5) int64
    user_quiz: np.ndarray       # (nU, n_quiz) float32
    item_cold: np.ndarray       # (nI,) bool
    user_group: np.ndarray      # (nU,) object
    item_l3: np.ndarray
    item_l2: np.ndarray
    item_l1: np.ndarray
    item_brand: np.ndarray
    item_tier: np.ndarray
    cat_sizes: dict = field(default_factory=dict)


class RepresentationEncoders:
    """Fit / transform wrapper. `transform_*` accept raw dict records (used by the serving layer)."""

    def __init__(self, text_dim: int = 64, tfidf_max_features: int = 4000, vis_dim: int = 64, seed: int = 42):
        self.text_dim, self.vis_dim, self.seed = text_dim, vis_dim, seed
        self.tfidf = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        self.svd = TruncatedSVD(n_components=text_dim, random_state=seed)
        self.vocab: dict[str, dict] = {}
        self.quiz_vocab: list[str] = []
        self.cont_stats: dict = {}
        self.text_scale = 1.0

    # ------------------------------------------------------------------ fit
    @staticmethod
    def _build_vocab(values) -> dict:
        vocab = {"<UNK>": 0}
        for v in sorted(set(map(str, values))):
            vocab[v] = len(vocab)
        return vocab

    def fit(self, items: pd.DataFrame, users: pd.DataFrame, stores: pd.DataFrame, inter: pd.DataFrame) -> "RepresentationEncoders":
        warm_items = items[~items.is_cold_item]
        for c in ITEM_CAT_COLS:
            self.vocab[c] = self._build_vocab(warm_items[c])
        train_users = set(inter.loc[inter.split == "train", "user_id"])
        tu = users[users.user_id.isin(train_users)]
        for c in USER_CAT_COLS:
            self.vocab[c] = self._build_vocab(tu[c])
        # contextual vocab from training interactions
        tr = inter[inter.split == "train"]
        self.vocab["device"] = self._build_vocab(tr.device)
        self.vocab["store_id"] = self._build_vocab(tr.store_id)
        self.vocab["channel"] = self._build_vocab(tr.channel)
        tags = sorted({t for q in users.quiz_tags for t in q.split("|")})  # fixed questionnaire vocabulary
        self.quiz_vocab = tags
        text = (warm_items.title + " " + warm_items.description).tolist()
        X = self.tfidf.fit_transform(text)
        Z = self.svd.fit_transform(X)
        self.text_scale = float(np.linalg.norm(Z, axis=1).mean() + 1e-8)
        lp = np.log1p(warm_items.price.values)
        self.cont_stats = {"logp_mean": float(lp.mean()), "logp_std": float(lp.std() + 1e-6),
                           "m_mean": float(warm_items.margin.mean()), "m_std": float(warm_items.margin.std() + 1e-6)}
        self.stores = stores.set_index("store_id")
        return self

    # ------------------------------------------------------------------ transform
    def _cat(self, col: str, values) -> np.ndarray:
        v = self.vocab[col]
        return np.array([v.get(str(x), 0) for x in values], dtype=np.int64)

    def text_embed(self, titles, descs) -> np.ndarray:
        X = self.tfidf.transform([f"{t} {d}" for t, d in zip(titles, descs)])
        return (self.svd.transform(X) / self.text_scale).astype(np.float32)

    def transform_items(self, df: pd.DataFrame, vis: np.ndarray | None = None) -> dict:
        cat = np.stack([self._cat(c, df[c]) for c in ITEM_CAT_COLS], 1)
        s = self.cont_stats
        launch = ((df.launch_ts.values - BASE_TS) / HORIZON).astype(np.float32) if "launch_ts" in df else np.zeros(len(df), np.float32)
        cont = np.stack([(np.log1p(df.price.values) - s["logp_mean"]) / s["logp_std"],
                         (df.margin.values - s["m_mean"]) / s["m_std"], launch], 1).astype(np.float32)
        if vis is None:
            vis = df[[f"vis_{d}" for d in range(self.vis_dim)]].values.astype(np.float32)
        return {"cat": cat, "cont": cont, "text": self.text_embed(df.title, df.description), "vis": vis.astype(np.float32)}

    def transform_users(self, df: pd.DataFrame) -> dict:
        cat = np.stack([self._cat(c, df[c]) for c in USER_CAT_COLS], 1)
        q = np.zeros((len(df), len(self.quiz_vocab)), np.float32)
        idx = {t: i for i, t in enumerate(self.quiz_vocab)}
        for r, tags in enumerate(df.quiz_tags):
            for t in str(tags).split("|"):
                if t in idx:
                    q[r, idx[t]] = 1.0
        return {"cat": cat, "quiz": q}

    def transform_context(self, hour, device, store_id, channel) -> dict:
        hour = np.asarray(hour, dtype=np.float32)
        tod = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)], 1).astype(np.float32)
        cat = np.stack([self._cat("device", device), self._cat("store_id", store_id), self._cat("channel", channel)], 1)
        return {"cat": cat, "tod": tod}

    def cat_sizes(self) -> dict:
        return {k: len(v) for k, v in self.vocab.items()}

    def build_features(self, items: pd.DataFrame, users: pd.DataFrame) -> Features:
        it = self.transform_items(items)
        us = self.transform_users(users)
        return Features(
            item_ids=items.item_id.tolist(), user_ids=users.user_id.tolist(),
            item_cat=it["cat"], item_cont=it["cont"], item_text=it["text"], item_vis=it["vis"],
            user_cat=us["cat"], user_quiz=us["quiz"],
            item_cold=items.is_cold_item.values.astype(bool), user_group=users.user_group.values,
            item_l3=items.l3.values, item_l2=items.l2.values, item_l1=items.l1.values,
            item_brand=items.brand.values, item_tier=items.price_tier.values, cat_sizes=self.cat_sizes(),
        )

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "RepresentationEncoders":
        with open(path, "rb") as f:
            return pickle.load(f)


def load_data(data_dir: str | Path):
    d = Path(data_dir)
    return (pd.read_parquet(d / "items.parquet"), pd.read_parquet(d / "users.parquet"),
            pd.read_parquet(d / "stores.parquet"), pd.read_parquet(d / "interactions.parquet"))
