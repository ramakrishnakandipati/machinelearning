"""Feature engineering (RFM, lead time, loyalty, seasonality) and train/test split."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import FEATURE_COLUMNS, TIERS, make_surv_array

BINARY = ["tier_Silver", "tier_Gold", "tier_Platinum", "channel_direct", "trip_purpose_business"]
CONTINUOUS = [
    "log_adr", "length_of_stay_avg", "log_lead_time", "log_days_since_last_stay",
    "log_prior_stays", "ancillary_spend_ratio", "promo_email_open_rate", "log_app_sessions",
    "seasonality_index", "log_monetary", "lead_x_business", "engagement_score",
]
MODEL_FEATURES = BINARY + CONTINUOUS


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw guest profile -> unscaled engineered feature frame (column order = MODEL_FEATURES)."""
    d = raw[FEATURE_COLUMNS].copy()
    out = pd.DataFrame(index=raw.index)
    for t in TIERS[1:]:
        out[f"tier_{t}"] = (d["loyalty_tier"] == t).astype(float)
    out["channel_direct"] = d["channel_direct"].astype(float)
    out["trip_purpose_business"] = d["trip_purpose_business"].astype(float)

    out["log_adr"] = np.log(d["adr_avg"].astype(float))
    out["length_of_stay_avg"] = d["length_of_stay_avg"].astype(float)
    out["log_lead_time"] = np.log1p(d["lead_time_days"].astype(float))
    out["log_days_since_last_stay"] = np.log(d["days_since_last_stay"].astype(float).clip(lower=1))  # Recency
    out["log_prior_stays"] = np.log1p(d["total_prior_stays"].astype(float))                          # Frequency
    out["ancillary_spend_ratio"] = d["ancillary_spend_ratio"].astype(float)
    out["promo_email_open_rate"] = d["promo_email_open_rate"].astype(float)
    out["log_app_sessions"] = np.log1p(d["app_sessions_30d"].astype(float))
    out["seasonality_index"] = d["seasonality_index"].astype(float)
    out["log_monetary"] = np.log1p(d["adr_avg"] * d["length_of_stay_avg"] * d["total_prior_stays"])  # Monetary
    out["lead_x_business"] = out["log_lead_time"] * out["trip_purpose_business"]
    out["engagement_score"] = out["log_app_sessions"] + 2.0 * out["promo_email_open_rate"]
    return out[MODEL_FEATURES]


class Preprocessor:
    """Engineers features and standardises the continuous block (binary flags stay 0/1)."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.feature_names = MODEL_FEATURES

    def fit(self, raw: pd.DataFrame) -> "Preprocessor":
        self.scaler.fit(engineer(raw)[CONTINUOUS])
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        x = engineer(raw)
        x[CONTINUOUS] = self.scaler.transform(x[CONTINUOUS])
        return x

    def fit_transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self.fit(raw).transform(raw)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(path)


def load_and_split(csv_path: str | Path, test_size: float, seed: int) -> dict:
    """Stratified (by event indicator and duration quartile) hold-out split."""
    df = pd.read_csv(csv_path)
    strat = df["event_observed"].astype(str) + "_" + pd.qcut(df["duration_days"], 4, labels=False).astype(str)
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)

    pre = Preprocessor().fit(tr)
    X_tr, X_te = pre.transform(tr), pre.transform(te)
    return {
        "raw_train": tr, "raw_test": te, "X_train": X_tr, "X_test": X_te,
        "t_train": tr["duration_days"].to_numpy(float), "e_train": tr["event_observed"].to_numpy(int),
        "t_test": te["duration_days"].to_numpy(float), "e_test": te["event_observed"].to_numpy(int),
        "y_train": make_surv_array(tr["duration_days"], tr["event_observed"]),
        "y_test": make_surv_array(te["duration_days"], te["event_observed"]),
        "preprocessor": pre,
    }
