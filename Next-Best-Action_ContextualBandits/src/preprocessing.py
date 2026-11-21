"""Feature engineering + ColumnTransformer shared by training and serving."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ARCHETYPES, CHANNELS, DAYS, TIERS

NUM_COLS = ["log_lead_time", "los", "log_hist_spend", "ancillary_rate", "party_size", "month_sin", "month_cos"]
CAT_COLS = ["loyalty_tier", "booking_channel", "guest_archetype", "day_of_week"]
RAW_COLS = ["loyalty_tier", "lead_time", "los", "booking_channel", "hist_spend", "ancillary_rate",
            "month", "day_of_week", "party_size", "guest_archetype"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df[RAW_COLS].copy()
    out["log_lead_time"] = np.log1p(out["lead_time"].astype(float))
    out["log_hist_spend"] = np.log1p(out["hist_spend"].astype(float))
    ang = 2 * np.pi * out["month"].astype(float) / 12.0
    out["month_sin"] = np.sin(ang)
    out["month_cos"] = np.cos(ang)
    return out


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("num", StandardScaler(), NUM_COLS),
         ("cat", OneHotEncoder(categories=[TIERS, CHANNELS, ARCHETYPES, DAYS], sparse_output=False,
                               handle_unknown="ignore"), CAT_COLS)],
        verbose_feature_names_out=False,
    )


def feature_names(pre: ColumnTransformer):
    return list(pre.get_feature_names_out())


def transform(pre, df: pd.DataFrame) -> np.ndarray:
    return pre.transform(add_derived(df)).astype(np.float32)


def save_preprocessor(pre, path):
    joblib.dump(pre, path)


def load_preprocessor(path):
    return joblib.load(path)


def fit_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    pre = build_preprocessor()
    pre.fit(add_derived(df))
    return pre
