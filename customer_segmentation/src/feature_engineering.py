"""Feature engineering: RFM / behavioural / ancillary / channel / temporal features + preprocessing."""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data_generator import LOYALTY

LOG_COLS = ["monetary_total_spend", "recency_days", "avg_daily_rate", "booking_lead_time_days",
            "avg_length_of_stay", "tenure_months", "frequency_stays_12m"]
NUM_COLS = LOG_COLS + [
    "weekend_vs_weekday_ratio", "cancellation_rate", "last_minute_booking_ratio",
    "spa_spend_ratio", "dining_fb_spend_ratio", "golf_recreation_spend_ratio", "business_center_usage",
    "direct_app_booking_ratio", "ota_booking_ratio", "email_open_rate", "mobile_key_usage_rate",
    "points_redemption_rate", "spend_per_stay", "ancillary_total_ratio", "stay_velocity",
    "digital_engagement", "other_channel_ratio"]
ORD_COLS = ["loyalty_tier"]
OHE_COLS = ["room_tier_preference", "property_type_affinity", "age_group", "home_region"]


def engineer_features(g: pd.DataFrame) -> pd.DataFrame:
    df = g.copy()
    df["spend_per_stay"] = df["monetary_total_spend"] / df["total_stays"].clip(lower=1)
    df["ancillary_total_ratio"] = df[["spa_spend_ratio", "dining_fb_spend_ratio",
                                      "golf_recreation_spend_ratio"]].sum(axis=1)
    df["stay_velocity"] = df["total_stays"] / (df["tenure_months"].clip(lower=1) / 12.0)
    df["digital_engagement"] = (df["email_open_rate"] + df["mobile_key_usage_rate"]
                                + df["direct_app_booking_ratio"]) / 3
    df["other_channel_ratio"] = (1 - df["direct_app_booking_ratio"] - df["ota_booking_ratio"]).clip(lower=0)
    return df


def _log(X):
    return np.log1p(np.clip(X, 0, None))


class FeaturePipeline:
    """Fit/transform wrapper that exposes `scaler` and `encoder` for joblib persistence."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.ord_enc = OrdinalEncoder(categories=[LOYALTY])
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.extra_log = ["spend_per_stay", "stay_velocity"]
        self.feature_names_ = None

    def _numeric(self, df):
        X = df[NUM_COLS].copy()
        for c in LOG_COLS + self.extra_log:
            X[c] = _log(X[c].values)
        return X

    def fit_transform(self, df):
        df = engineer_features(df) if "spend_per_stay" not in df else df
        num = self.scaler.fit_transform(self._numeric(df))
        ordv = self.ord_enc.fit_transform(df[ORD_COLS]) / 4.0
        ohe = self.encoder.fit_transform(df[OHE_COLS])
        self.feature_names_ = NUM_COLS + ORD_COLS + list(self.encoder.get_feature_names_out(OHE_COLS))
        return np.hstack([num, ordv, ohe]).astype(np.float32)

    def transform(self, df):
        df = engineer_features(df) if "spend_per_stay" not in df else df
        num = self.scaler.transform(self._numeric(df))
        ordv = self.ord_enc.transform(df[ORD_COLS]) / 4.0
        ohe = self.encoder.transform(df[OHE_COLS])
        return np.hstack([num, ordv, ohe]).astype(np.float32)

    def save(self, folder):
        joblib.dump(self.scaler, folder / "scaler.joblib")
        joblib.dump({"ordinal": self.ord_enc, "onehot": self.encoder}, folder / "encoder.joblib")
        joblib.dump(self, folder / "feature_pipeline.joblib")

    @staticmethod
    def load(folder):
        return joblib.load(folder / "feature_pipeline.joblib")


class SequenceNormalizer:
    """log1p + standardise (gap, spend, lead); property id kept as categorical index."""

    def fit(self, seq, mask):
        x = self._pre(seq)
        m = mask.astype(bool)
        self.mean_ = np.array([x[..., c][m].mean() for c in (0, 1, 3)], dtype=np.float32)
        self.std_ = np.array([x[..., c][m].std() + 1e-6 for c in (0, 1, 3)], dtype=np.float32)
        return self

    @staticmethod
    def _pre(seq):
        x = seq.copy()
        for c in (0, 1, 3):
            x[..., c] = np.log1p(np.clip(x[..., c], 0, None))
        return x

    def transform(self, seq, mask):
        """Returns (cont (N,L,3), prop_ids (N,L) long, mask)."""
        x = self._pre(seq)
        cont = np.stack([(x[..., c] - self.mean_[i]) / self.std_[i] for i, c in enumerate((0, 1, 3))], -1)
        cont = cont * mask[..., None]
        return cont.astype(np.float32), seq[..., 2].astype(np.int64), mask.astype(np.float32)
