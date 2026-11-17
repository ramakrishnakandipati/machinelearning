"""Module 2b - Point-in-time retail feature engineering + serving transformer.

``build_features`` only ever looks at events strictly *before* each customer's
``as_of`` decision timestamp, so there is no look-ahead leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import COLD_START_MIN_SESSIONS, FEATURE_COLUMNS, FULL_PRICE_LOYALIST_MARKDOWN_MAX, N_CATEGORIES


# --------------------------------------------------------------------------- #
# Point-in-time feature builder
# --------------------------------------------------------------------------- #
def build_features(
    transactions: pd.DataFrame,
    sessions: pd.DataFrame,
    as_of: pd.Series,
    profile: pd.DataFrame,
    category_stock: pd.Series,
) -> pd.DataFrame:
    """Compute retail predictors for each customer as of its own decision time.

    Parameters
    ----------
    transactions : columns ``customer_id, ts, amount, is_markdown, is_online, category``
    sessions     : columns ``customer_id, ts, cart_abandon``
    as_of        : decision timestamp indexed by ``customer_id``
    profile      : static dimension with ``customer_id, size_run_factor, app_active_30d``
    category_stock : in-stock size availability (0-1) indexed by category id

    Returns a frame indexed by ``customer_id``. Missing history stays NaN (it is
    imputed by :class:`FeatureTransformer` with population priors, which is how
    cold-start shoppers are handled) except counts, which are 0.
    """
    ids = as_of.index
    asof_df = as_of.rename("as_of").reset_index().rename(columns={"index": "customer_id"})
    asof_df.columns = ["customer_id", "as_of"]

    # ---- transactions: strictly before as_of (point-in-time) --------------
    tx = transactions.merge(asof_df, on="customer_id", how="inner")
    tx = tx[tx["ts"] < tx["as_of"]].copy()
    tx["age"] = (tx["as_of"] - tx["ts"]).dt.total_seconds() / 86400.0
    tx = tx[tx["age"] <= 365]
    tx["online_amt"] = tx["amount"] * tx["is_online"]
    feats = pd.DataFrame(index=ids)
    feats.index.name = "customer_id"

    feats["recency_days"] = tx.groupby("customer_id")["age"].min()
    for w in (30, 90, 365):
        sub = tx[tx["age"] <= w].groupby("customer_id")
        feats[f"freq_{w}d"] = sub.size()
        feats[f"monetary_{w}d"] = sub["amount"].sum()
    feats[["freq_30d", "freq_90d", "freq_365d", "monetary_30d", "monetary_90d", "monetary_365d"]] = (
        feats[["freq_30d", "freq_90d", "freq_365d", "monetary_30d", "monetary_90d", "monetary_365d"]].fillna(0.0)
    )
    feats["aov_365d"] = feats["monetary_365d"] / feats["freq_365d"].replace(0, np.nan)

    # Discount sensitivity: share of orders bought on markdown vs full price.
    g = tx.groupby("customer_id")
    feats["markdown_share"] = g["is_markdown"].mean()
    feats["discount_tier"] = pd.cut(
        feats["markdown_share"], [-0.01, 0.30, 0.60, 1.01], labels=[0, 1, 2]
    ).astype(float)

    # Category affinity (spend share of top category) + size availability.
    cat_spend = tx.groupby(["customer_id", "category"])["amount"].sum().unstack(fill_value=0.0)
    cat_spend = cat_spend.reindex(columns=range(N_CATEGORIES), fill_value=0.0)
    total = cat_spend.sum(axis=1).replace(0, np.nan)
    feats["top_affinity_cat"] = cat_spend.idxmax(axis=1).where(total.notna())
    feats["affinity_top_share"] = cat_spend.max(axis=1) / total
    prof = profile.set_index("customer_id")
    stock = feats["top_affinity_cat"].map(category_stock)
    feats["size_availability_score"] = (stock * prof["size_run_factor"].reindex(ids)).clip(0, 1)

    # Omnichannel mix: online share of spend.
    feats["online_spend_ratio"] = g["online_amt"].sum() / g["amount"].sum().replace(0, np.nan)

    # ---- sessions ----------------------------------------------------------
    ss = sessions.merge(asof_df, on="customer_id", how="inner")
    ss = ss[ss["ts"] < ss["as_of"]].copy()
    ss["age"] = (ss["as_of"] - ss["ts"]).dt.total_seconds() / 86400.0
    ss = ss[ss["age"] <= 365]
    feats["n_sessions_90d"] = ss[ss["age"] <= 90].groupby("customer_id").size()
    feats["n_sessions_90d"] = feats["n_sessions_90d"].fillna(0.0)
    n_sess_total = ss.groupby("customer_id").size().reindex(ids).fillna(0)
    feats["cart_abandon_recency_days"] = ss[ss["cart_abandon"] == 1].groupby("customer_id")["age"].min()
    feats["app_active_30d"] = prof["app_active_30d"].reindex(ids).astype(float)

    # Cold start = guest / new shopper with fewer than 2 sessions of history.
    feats["is_cold_start"] = (n_sess_total < COLD_START_MIN_SESSIONS).astype(float)
    return feats[FEATURE_COLUMNS]


# --------------------------------------------------------------------------- #
# Segments (observable, rule-based) used for plots & legacy CRM logic
# --------------------------------------------------------------------------- #
def assign_segment(df: pd.DataFrame) -> pd.Series:
    """Map raw (un-imputed or imputed) features to a business segment label."""
    md = df["markdown_share"]
    rec = df["recency_days"]
    seg = pd.Series("Mainstream", index=df.index, dtype=object)
    seg[md > 0.55] = "Discount Hunters"
    seg[md < FULL_PRICE_LOYALIST_MARKDOWN_MAX] = "Full-Price Loyalists"
    seg[rec > 120] = "At-Risk Churners"
    seg[df["is_cold_start"] == 1] = "Cold-Start Guests"
    return seg


# --------------------------------------------------------------------------- #
# Fitted transformer (serialized with joblib, reused verbatim at serving time)
# --------------------------------------------------------------------------- #
@dataclass
class FeatureTransformer:
    """Imputes missing history with population priors and fixes column order.

    Priors are medians of *established* (non cold-start) customers; cold-start
    shoppers therefore receive "typical customer" values plus ``is_cold_start=1``
    so the model (and the shrinkage layer) can treat them differently.
    """

    columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    priors: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        base = df[df["is_cold_start"] == 0] if (df["is_cold_start"] == 0).any() else df
        self.priors = {c: float(base[c].median()) for c in self.columns}
        # Sensible hard defaults when a prior is undefined.
        self.priors["recency_days"] = float(base["recency_days"].median())
        self.priors["cart_abandon_recency_days"] = 400.0  # "never abandoned" sentinel
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.columns].copy()
        return out.fillna(value=self.priors).astype(np.float32)

    def transform_dict(self, row: dict[str, float | None]) -> np.ndarray:
        """Hot-path single-row transform without building a DataFrame."""
        vals = np.empty(len(self.columns), dtype=np.float32)
        for i, c in enumerate(self.columns):
            v = row.get(c)
            vals[i] = self.priors[c] if v is None or (isinstance(v, float) and np.isnan(v)) else v
        return vals
