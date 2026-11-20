"""Synthetic guest data simulation and preprocessing."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import config as C

PERSONAS = ["Business Road Warrior", "Luxury Leisure Spender", "Weekend Family Getaway",
            "Budget OTA Shopper", "Loyal Direct Regular"]
_WEIGHTS = [0.20, 0.12, 0.23, 0.25, 0.20]


def _beta(rng, mean, conc, n):
    mean = float(np.clip(mean, 0.01, 0.99))
    return rng.beta(mean * conc, (1 - mean) * conc, n)


def _persona_block(rng, p: int, m: int) -> dict:
    B, LN, G = _beta, rng.lognormal, rng.gamma
    if p == 0:    # business road warrior
        return dict(recency_days=G(2, 25, m), annual_stays=rng.poisson(18, m) + 4,
                    avg_length_of_stay=1 + G(2, 0.7, m), booking_lead_time_days=G(2, 4, m),
                    adr_usd=LN(np.log(200), 0.25, m), fb_spend_ratio=B(rng, .25, 20, m),
                    spa_golf_spend_usd=G(1, 25, m), room_upgrade_rate=B(rng, .2, 15, m),
                    loyalty_tenure_years=G(5, 1.4, m), direct_booking_ratio=B(rng, .6, 15, m),
                    cancellation_rate=B(rng, .12, 25, m), weekend_stay_ratio=B(rng, .08, 25, m),
                    business_trip_ratio=B(rng, .88, 25, m))
    if p == 1:    # luxury leisure
        return dict(recency_days=G(2, 80, m), annual_stays=rng.poisson(3, m) + 1,
                    avg_length_of_stay=2 + G(4, 1.1, m), booking_lead_time_days=G(5, 14, m),
                    adr_usd=LN(np.log(550), 0.3, m), fb_spend_ratio=B(rng, .40, 15, m),
                    spa_golf_spend_usd=LN(np.log(900), 0.5, m), room_upgrade_rate=B(rng, .6, 12, m),
                    loyalty_tenure_years=G(4, 1.5, m), direct_booking_ratio=B(rng, .55, 12, m),
                    cancellation_rate=B(rng, .08, 25, m), weekend_stay_ratio=B(rng, .45, 15, m),
                    business_trip_ratio=B(rng, .05, 25, m))
    if p == 2:    # weekend family
        return dict(recency_days=G(2, 110, m), annual_stays=rng.poisson(2, m) + 1,
                    avg_length_of_stay=2 + G(3, 0.8, m), booking_lead_time_days=G(4, 20, m),
                    adr_usd=LN(np.log(170), 0.25, m), fb_spend_ratio=B(rng, .35, 15, m),
                    spa_golf_spend_usd=G(1.2, 40, m), room_upgrade_rate=B(rng, .12, 15, m),
                    loyalty_tenure_years=G(2, 1.2, m), direct_booking_ratio=B(rng, .35, 12, m),
                    cancellation_rate=B(rng, .18, 15, m), weekend_stay_ratio=B(rng, .88, 25, m),
                    business_trip_ratio=B(rng, .04, 25, m))
    if p == 3:    # budget OTA
        return dict(recency_days=G(2, 150, m), annual_stays=rng.poisson(1.5, m) + 1,
                    avg_length_of_stay=1 + G(2, 0.9, m), booking_lead_time_days=G(3, 15, m),
                    adr_usd=LN(np.log(95), 0.2, m), fb_spend_ratio=B(rng, .10, 20, m),
                    spa_golf_spend_usd=G(0.5, 10, m), room_upgrade_rate=B(rng, .03, 20, m),
                    loyalty_tenure_years=G(1.2, 0.9, m), direct_booking_ratio=B(rng, .08, 25, m),
                    cancellation_rate=B(rng, .38, 12, m), weekend_stay_ratio=B(rng, .4, 10, m),
                    business_trip_ratio=B(rng, .15, 15, m))
    # loyal direct regular
    return dict(recency_days=G(2, 40, m), annual_stays=rng.poisson(7, m) + 2,
                avg_length_of_stay=1.5 + G(3, 0.8, m), booking_lead_time_days=G(3, 12, m),
                adr_usd=LN(np.log(260), 0.22, m), fb_spend_ratio=B(rng, .30, 15, m),
                spa_golf_spend_usd=G(2, 120, m), room_upgrade_rate=B(rng, .35, 12, m),
                loyalty_tenure_years=G(8, 1.4, m), direct_booking_ratio=B(rng, .88, 25, m),
                cancellation_rate=B(rng, .05, 30, m), weekend_stay_ratio=B(rng, .30, 12, m),
                business_trip_ratio=B(rng, .25, 12, m))


def simulate_guests(n: int = C.N_GUESTS, seed: int = C.SEED) -> pd.DataFrame:
    """Simulate guests from 5 latent personas. `true_persona` is for sanity checks only."""
    rng = np.random.default_rng(seed)
    z = rng.choice(len(PERSONAS), size=n, p=_WEIGHTS)
    parts = []
    for p in range(len(PERSONAS)):
        idx = np.where(z == p)[0]
        df = pd.DataFrame(_persona_block(rng, p, len(idx)), index=idx)
        df["true_persona"] = p
        parts.append(df)
    df = pd.concat(parts).sort_index().reset_index(drop=True)
    for col, (lo, hi) in C.RANGES.items():
        df[col] = df[col].clip(lo, hi)
    for col in ("annual_stays", "recency_days", "booking_lead_time_days"):
        df[col] = df[col].round().astype(int)
    return df[C.FEATURES + ["true_persona"]]


def load_or_simulate(csv_path: str | None = None, n: int = C.N_GUESTS) -> pd.DataFrame:
    """Ingest a PMS/CRM CSV with FEATURES columns, or simulate."""
    if csv_path:
        df = pd.read_csv(csv_path)
        missing = set(C.FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        return df
    df = simulate_guests(n)
    df.to_csv(C.DATA_DIR / "guests.csv", index=False)
    return df


def build_preprocessor() -> ColumnTransformer:
    other = [f for f in C.FEATURES if f not in C.LOG_FEATURES]
    log_pipe = Pipeline([("log", FunctionTransformer(np.log1p, validate=False)),
                         ("sc", StandardScaler())])
    return ColumnTransformer([("log", log_pipe, C.LOG_FEATURES), ("std", StandardScaler(), other)])


def feature_names_after_transform() -> list[str]:
    """Column order of the preprocessor output."""
    return C.LOG_FEATURES + [f for f in C.FEATURES if f not in C.LOG_FEATURES]


def prepare(df: pd.DataFrame) -> dict:
    """Split, fit preprocessor on train only, save it, return scaled arrays and raw frames."""
    strat = df["true_persona"] if "true_persona" in df else None
    df_tr, df_te = train_test_split(df, test_size=0.2, random_state=C.SEED, stratify=strat)
    df_tr, df_te = df_tr.reset_index(drop=True), df_te.reset_index(drop=True)
    pre = build_preprocessor()
    X_tr = pre.fit_transform(df_tr[C.FEATURES]).astype(np.float32)
    X_te = pre.transform(df_te[C.FEATURES]).astype(np.float32)
    joblib.dump(pre, C.PREPROCESSOR_FILE)
    return dict(X_train=X_tr, X_test=X_te, df_train=df_tr, df_test=df_te, preprocessor=pre)
