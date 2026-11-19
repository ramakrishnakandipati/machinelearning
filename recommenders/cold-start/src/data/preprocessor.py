"""Feature engineering, text embeddings and cold-start aware split/case builders."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.generator import RetailDataset, hour_bucket

HIST_WINDOW = 5


def l2norm(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (safe for zero rows)."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


class TextEmbedder:
    """Sentence-Transformer embedder with an automatic offline TF-IDF + SVD fallback."""

    def __init__(self, prefer_transformer: bool = True, name: str = "all-MiniLM-L6-v2", dim: int = 64, seed: int = 42):
        self.prefer_transformer, self.name, self.dim, self.seed = prefer_transformer, name, dim, seed
        self.backend = "tfidf_svd"
        self._st = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None

    def fit(self, texts: Sequence[str]) -> "TextEmbedder":
        if self.prefer_transformer:
            try:  # pragma: no cover - depends on optional dependency / network
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer(self.name)
                self.backend = f"sentence-transformer:{self.name}"
                self.dim = int(self._st.get_sentence_embedding_dimension())
                return self
            except Exception:
                self._st = None
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        mat = self.tfidf.fit_transform(texts)
        k = max(2, min(self.dim, mat.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(mat)
        self.dim = k
        self.backend = "tfidf_svd"
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._st is not None:  # pragma: no cover
            return l2norm(np.asarray(self._st.encode(list(texts), show_progress_bar=False), dtype=np.float32))
        assert self.tfidf is not None and self.svd is not None, "embedder not fitted"
        return l2norm(self.svd.transform(self.tfidf.transform(texts)).astype(np.float32))

    def __getstate__(self) -> Dict[str, Any]:
        s = self.__dict__.copy()
        s["_st"] = None  # transformer is reloaded lazily by name
        return s

    def __setstate__(self, s: Dict[str, Any]) -> None:
        self.__dict__.update(s)
        if self.backend.startswith("sentence-transformer"):  # pragma: no cover
            try:
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer(self.name)
            except Exception:
                raise RuntimeError("Saved model used a Sentence-Transformer which can no longer be loaded")


def item_text(row: Dict[str, Any]) -> str:
    """Concatenate searchable item text fields."""
    return " . ".join(str(row.get(k, "")) for k in ("title", "description", "brand", "dept", "subcat", "leaf",
                                                      "color", "material", "style"))


class Vocab:
    """String vocabulary with reserved UNK=0."""

    def __init__(self, values: Sequence[str]):
        self.itos = ["<UNK>"] + sorted({str(v) for v in values})
        self.stoi = {v: i for i, v in enumerate(self.itos)}

    def __call__(self, v: Optional[str]) -> int:
        return self.stoi.get(str(v), 0) if v is not None else 0

    def __len__(self) -> int:
        return len(self.itos)


@dataclass
class EvalCase:
    """One evaluation request for a cold-start tier."""

    split: str
    user_id: str
    user_row: Dict[str, Any]
    ctx: Dict[str, Any]
    history: List[int]            # item positions (time ordered) visible at request time
    targets: List[int]            # held-out positive item positions
    candidates: Optional[np.ndarray]  # None = all warm items
    exclude: set


class Preprocessor:
    """Fits encoders on the *warm* part of the data and encodes users, items, contexts."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        e = cfg["embeddings"]
        self.embedder = TextEmbedder(e["prefer_transformer"], e["transformer_name"], int(e["text_dim"]), cfg["seed"])

    # ------------------------------------------------------------------ fit
    def fit(self, ds: RetailDataset) -> "Preprocessor":
        items, users, inter = ds.items, ds.users, ds.interactions
        self.meta = ds.meta
        warm_items = items[~items.is_cold_item]
        self.embedder.fit([item_text(r) for r in warm_items.to_dict("records")])
        self.v_region, self.v_device = Vocab(users.region), Vocab(users.device)
        self.v_acq, self.v_age = Vocab(users.acq_channel), Vocab(users.age_band)
        self.v_channel, self.v_season = Vocab(ds.meta["channels"]), Vocab(ds.meta["seasons"])
        self.v_hour = Vocab(ds.meta["hour_buckets"])
        self.v_dept, self.v_sub = Vocab(items.dept), Vocab(items.subcat)
        self.v_brand = Vocab(warm_items.brand)
        self.v_tier = Vocab(range(4))
        self.n_sub = len(self.v_sub)
        lp = np.log1p(items.price.values)
        self.price_mu, self.price_sd = float(lp.mean()), float(lp.std() + 1e-6)

        self.item_ids = items.item_id.values.astype(str)
        self.item_pos = {iid: p for p, iid in enumerate(self.item_ids)}
        self.is_cold_item = items.is_cold_item.values.copy()
        self.item_subcat = items.subcat.values.astype(str)
        self.item_dept = items.dept.values.astype(str)
        self.users_ids = users.user_id.values.astype(str)
        self.user_pos = {u: p for p, u in enumerate(self.users_ids)}

        # ID vocabularies: only entities with >=1 TRAIN positive get an ID embedding; the rest map to UNK=0
        tr = inter[(inter.split == "train") & (inter.positive == 1)]
        self.train_users = sorted(tr.user_id.unique())
        self.train_items = sorted(tr.item_id.unique())
        self.uid_map = {u: i + 1 for i, u in enumerate(self.train_users)}
        self.iid_map = {u: i + 1 for i, u in enumerate(self.train_items)}
        self.n_uid, self.n_iid = len(self.uid_map) + 1, len(self.iid_map) + 1

        recs = items.to_dict("records")
        self.item_text = self.embedder.transform([item_text(r) for r in recs])
        self.item_cats = np.array([[self.v_dept(r["dept"]), self.v_sub(r["subcat"]), self.v_brand(r["brand"]),
                                    int(r["price_tier"]) + 1] for r in recs], dtype=np.int64)
        self.item_price = ((np.log1p(items.price.values) - self.price_mu) / self.price_sd).astype(np.float32)
        self.item_id_idx = np.array([self.iid_map.get(i, 0) for i in self.item_ids], dtype=np.int64)
        self.item_content = np.vstack([self.content_vector(r, self.item_text[k]) for k, r in enumerate(recs)])
        self.content_dim = self.item_content.shape[1]
        # sub-category centroids (from warm items) for onboarding profiles
        self.sub_centroid: Dict[str, np.ndarray] = {}
        for s in sorted(set(self.item_subcat)):
            m = (self.item_subcat == s) & ~self.is_cold_item
            self.sub_centroid[s] = l2norm(self.item_content[m].mean(0, keepdims=True))[0]
        self.users_static = self.encode_users(users)
        return self

    # ------------------------------------------------------------ item side
    def content_vector(self, item: Dict[str, Any], text_vec: np.ndarray) -> np.ndarray:
        """Dense multi-modal item vector: [text | dept | subcat | brand | price-tier], L2-normalised."""
        dept = np.zeros(len(self.v_dept), np.float32); dept[self.v_dept(item["dept"])] = 0.5
        sub = np.zeros(len(self.v_sub), np.float32); sub[self.v_sub(item["subcat"])] = 0.8
        br = np.zeros(len(self.v_brand), np.float32); br[self.v_brand(item["brand"])] = 0.3
        tier = np.zeros(5, np.float32); tier[int(item["price_tier"]) + 1] = 0.2
        dept[0] = sub[0] = br[0] = 0.0  # UNK slots carry no information
        return l2norm(np.concatenate([text_vec, dept, sub, br, tier]).astype(np.float32)[None])[0]

    def encode_new_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Encode a brand-new SKU (not in the catalog) from metadata only. Computes price tier if absent."""
        item = dict(item)
        if "price_tier" not in item:
            q = np.expm1(self.price_mu + self.price_sd * np.array([-0.67, 0.0, 0.67]))
            item["price_tier"] = int(np.searchsorted(q, float(item["price"])))
        tv = self.embedder.transform([item_text(item)])[0]
        return dict(
            text=tv.astype(np.float32),
            cats=np.array([self.v_dept(item["dept"]), self.v_sub(item["subcat"]), self.v_brand(item["brand"]),
                           int(item["price_tier"]) + 1], dtype=np.int64),
            price=np.float32((np.log1p(float(item["price"])) - self.price_mu) / self.price_sd),
            content=self.content_vector(item, tv), item=item)

    # ------------------------------------------------------------ user side
    def onboarding_vec(self, cats: Optional[Sequence[str]]) -> np.ndarray:
        v = np.zeros(self.n_sub, np.float32)
        for c in cats or []:
            i = self.v_sub(c)
            if i:
                v[i] = 1.0
        return v

    def encode_user_row(self, u: Dict[str, Any]) -> Dict[str, Any]:
        """Encode user static features from a dict (works with missing fields -> UNK)."""
        return dict(
            uid=self.uid_map.get(str(u.get("user_id")), 0),
            cats=np.array([self.v_region(u.get("region")), self.v_device(u.get("device")),
                           self.v_acq(u.get("acq_channel")), self.v_age(u.get("age_band"))], dtype=np.int64),
            onboard=self.onboarding_vec(u.get("onboarding_categories")))

    def encode_users(self, users: pd.DataFrame) -> Dict[str, np.ndarray]:
        rows = [self.encode_user_row(r) for r in users.to_dict("records")]
        return dict(uid=np.array([r["uid"] for r in rows]), cats=np.vstack([r["cats"] for r in rows]),
                    onboard=np.vstack([r["onboard"] for r in rows]))

    def encode_ctx(self, channel: Optional[str], season: Optional[str], hour: Optional[int]) -> np.ndarray:
        hb = hour_bucket(int(hour)) if hour is not None else None
        return np.array([self.v_channel(channel), self.v_season(season), self.v_hour(hb)], dtype=np.int64)

    def history_vec(self, positions: Sequence[int]) -> tuple[np.ndarray, float]:
        """Mean content vector of the last ``HIST_WINDOW`` items and log-count feature."""
        if len(positions) == 0:
            return np.zeros(self.content_dim, np.float32), 0.0
        w = list(positions)[-HIST_WINDOW:]
        return self.item_content[w].mean(0).astype(np.float32), float(np.log1p(min(len(positions), 50)))

    def profile_vec(self, history: Sequence[int], onboarding: Optional[Sequence[str]]) -> Optional[np.ndarray]:
        """Content-based user profile: recency-weighted history centroid, else onboarding centroid."""
        if len(history):
            w = np.arange(1, len(history[-10:]) + 1, dtype=np.float32)
            v = (self.item_content[list(history)[-10:]] * w[:, None]).sum(0)
            return l2norm(v[None])[0]
        cs = [self.sub_centroid[c] for c in (onboarding or []) if c in self.sub_centroid]
        return l2norm(np.mean(cs, 0, keepdims=True))[0] if cs else None

    # ------------------------------------------------------ training arrays
    def build_rows(self, ds: RetailDataset, splits: Sequence[str]) -> Dict[str, np.ndarray]:
        """Training/validation arrays for positive interactions, with strictly-prior history features."""
        inter = ds.interactions
        base = inter[(inter.split.isin(["train", "val"])) & (inter.positive == 1)].sort_values(["user_id", "ts"])
        upos = base.user_id.map(self.user_pos).values
        ipos = base.item_id.map(self.item_pos).values
        hist = np.zeros((len(base), self.content_dim), np.float32)
        hist_n = np.zeros(len(base), np.float32)
        start = 0
        uvals = base.user_id.values
        bounds = np.flatnonzero(np.r_[True, uvals[1:] != uvals[:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            V = self.item_content[ipos[a:b]]
            cs = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V, 0)])
            k = np.arange(b - a)
            lo = np.maximum(0, k - HIST_WINDOW)
            cnt = (k - lo).astype(np.float32)
            h = (cs[k] - cs[lo]) / np.maximum(cnt, 1)[:, None]
            hist[a:b] = h
            hist_n[a:b] = np.log1p(np.minimum(k, 50))
        keep = base.split.isin(splits).values
        ctx = np.vstack([self.encode_ctx(c, s, h) for c, s, h in zip(base.channel.values[keep], base.season.values[keep],
                                                                     base.hour.values[keep])])
        u = upos[keep]
        return dict(user_pos=u, item_pos=ipos[keep], uid=self.users_static["uid"][u], ucats=self.users_static["cats"][u],
                    onboard=self.users_static["onboard"][u], ctx=ctx, hist=hist[keep], hist_n=hist_n[keep])

    # ----------------------------------------------------------- eval cases
    def build_eval_cases(self, ds: RetailDataset, max_users: int, seed: int) -> Dict[str, List[EvalCase]]:
        """Create evaluation cases for warm / cold_user / cold_item / session tiers."""
        rng = np.random.default_rng(seed)
        inter, users = ds.interactions, ds.users.set_index("user_id")
        pos = inter[inter.positive == 1]
        hist_df = pos[pos.split.isin(["train", "val"])].sort_values("ts")
        hist_by_user = hist_df.groupby("user_id")["item_id"].apply(lambda s: [self.item_pos[i] for i in s]).to_dict()
        warm_cands = np.where(~self.is_cold_item)[0]
        cold_cands = np.where(self.is_cold_item)[0]
        out: Dict[str, List[EvalCase]] = {"warm": [], "cold_user": [], "cold_item": [], "session": []}

        def mk(split: str, uid: str, g: pd.DataFrame, history: List[int], cands: Optional[np.ndarray], key: str) -> None:
            first = g.sort_values("ts").iloc[0]
            urow = users.loc[uid].to_dict(); urow["user_id"] = uid
            ctx = dict(channel=first.channel, season=first.season, hour=int(first.hour))
            tg = [self.item_pos[i] for i in g.item_id.values]
            out[key].append(EvalCase(split, uid, urow, ctx, history, sorted(set(tg)), cands, set(history)))

        def sample(groups: Dict[str, pd.DataFrame]) -> List[str]:
            keys = list(groups)
            return [keys[i] for i in rng.permutation(len(keys))[:max_users]]

        g = {u: d for u, d in pos[pos.split == "test_warm"].groupby("user_id") if u in hist_by_user}
        for u in sample(g):
            mk("test_warm", u, g[u], hist_by_user[u], warm_cands, "warm")
        g = {u: d for u, d in pos[pos.split == "test_cold_user"].groupby("user_id")}
        for u in sample(g):
            mk("test_cold_user", u, g[u], [], None, "cold_user")
        g = {u: d for u, d in pos[pos.split == "test_cold_item"].groupby("user_id") if u in hist_by_user}
        for u in sample(g):
            mk("test_cold_item", u, g[u], hist_by_user[u], cold_cands, "cold_item")
        sc = pos[pos.split == "session_ctx"].sort_values("ts").groupby("user_id")["item_id"].apply(
            lambda s: [self.item_pos[i] for i in s]).to_dict()
        g = {u: d for u, d in pos[pos.split == "test_session"].groupby("user_id") if u in sc}
        for u in sample(g):
            mk("test_session", u, g[u], sc[u], None, "session")
        return out

    # ----------------------------------------------------------------- IO
    def save(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(Path(directory) / "preprocessor.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(directory: str | Path) -> "Preprocessor":
        with open(Path(directory) / "preprocessor.pkl", "rb") as f:
            return pickle.load(f)
