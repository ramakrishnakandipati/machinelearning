"""Synthetic hospitality guest / rebooking event generator.

Event times follow a Weibull accelerated-failure-time model with:
  * non-linear covariate effects (U-shaped ADR, saturating app engagement, log prior stays),
  * a lead-time x trip-purpose interaction,
  * covariate-dependent Weibull *shape* (business vs leisure, high tiers) which produces
    crossing hazards -> a deliberate proportional-hazards violation that favours
    tree / deep models over CoxPH,
  * independent right-censoring tuned by bisection to the configured censoring rate.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .utils import TIERS, get_logger, load_config, resolve

log = get_logger("data_generator")


def _simulate_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    tier_idx = rng.choice(4, size=n, p=[0.45, 0.28, 0.19, 0.08])
    business = rng.binomial(1, 0.38 + 0.08 * (tier_idx >= 2))
    direct = rng.binomial(1, np.clip(0.30 + 0.12 * tier_idx, 0, 0.9))

    adr = np.clip(rng.lognormal(np.log(140), 0.35, n) * (1 + 0.12 * tier_idx), 55, 900)
    los = np.clip(rng.gamma(2.2, 1.2, n) * np.where(business == 1, 0.75, 1.25), 1, 21)
    lead = np.clip(rng.gamma(2.0, 22, n) * np.where(business == 1, 0.55, 1.2), 0, 365)
    prior = rng.poisson(1.5 + 2.2 * tier_idx + 1.2 * business) + 1
    since = np.clip(rng.gamma(3.0, 40, n) / (1 + 0.15 * tier_idx), 5, 720)
    ancillary = np.clip(rng.beta(1.8, 6, n) + 0.05 * tier_idx, 0, 1)
    open_rate = np.clip(rng.beta(2.2, 4.5, n) + 0.04 * tier_idx, 0, 1)
    sessions = rng.poisson(np.clip(3 + 3.5 * tier_idx + 4 * open_rate, 0.5, None))
    seasonality = np.clip(rng.normal(1.0, 0.18, n), 0.5, 1.5)

    return pd.DataFrame({
        "loyalty_tier": np.array(TIERS)[tier_idx],
        "adr_avg": adr.round(2),
        "length_of_stay_avg": los.round(2),
        "lead_time_days": lead.round(1),
        "days_since_last_stay": since.round(1),
        "total_prior_stays": prior,
        "ancillary_spend_ratio": ancillary.round(4),
        "promo_email_open_rate": open_rate.round(4),
        "app_sessions_30d": sessions,
        "channel_direct": direct,
        "trip_purpose_business": business,
        "seasonality_index": seasonality.round(4),
    })


def _event_times(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    tier = df["loyalty_tier"].map({t: i for i, t in enumerate(TIERS)}).to_numpy()
    biz = df["trip_purpose_business"].to_numpy()
    adr_z = (np.log(df["adr_avg"]) - np.log(df["adr_avg"]).mean()) / np.log(df["adr_avg"]).std()
    lead_z = (df["lead_time_days"] - df["lead_time_days"].mean()) / df["lead_time_days"].std()

    mu = (
        6.15
        + np.array([0.0, -0.15, -0.30, -0.45])[tier]
        + 0.12 * adr_z ** 2                                   # U-shape: extreme ADR -> slower rebook
        - 0.18 * np.log1p(df["total_prior_stays"])
        - 0.35 * np.tanh(df["app_sessions_30d"] / 8.0 - 1.0)  # saturating engagement
        - 0.45 * df["promo_email_open_rate"]
        + 0.22 * np.log(df["days_since_last_stay"] / 120.0)
        - 0.22 * df["channel_direct"]
        - 0.20 * biz
        + 0.15 * lead_z * (1 - 2 * biz)                       # lead-time x purpose interaction
        - 0.45 * (df["seasonality_index"] - 1.0)
        - 0.30 * df["ancillary_spend_ratio"]
        + rng.normal(0, 0.15, len(df))                        # unobserved heterogeneity
    ).to_numpy()

    shape = np.where(biz == 1, 0.85, 1.65) * np.where(tier >= 2, 0.90, 1.0)   # PH violation
    u = rng.uniform(1e-12, 1 - 1e-12, len(df))
    return np.exp(mu) * (-np.log(u)) ** (1.0 / shape)


def _tune_censoring(t_event: np.ndarray, target: float, c_max: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Bisection on the upper bound of Uniform(15, c_hi) censoring to hit `target` censoring."""
    c_unit = rng.uniform(0, 1, len(t_event))
    lo, hi = 30.0, c_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        c = 15 + c_unit * (mid - 15)
        frac = np.mean(c < t_event)
        if frac > target:
            lo = mid           # too much censoring -> widen window
        else:
            hi = mid
    c_hi = 0.5 * (lo + hi)
    return 15 + c_unit * (c_hi - 15), c_hi


def generate(n: int, target_censoring: float, c_max: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _simulate_features(n, rng)
    t_event = _event_times(df, rng)
    c, c_hi = _tune_censoring(t_event, target_censoring, c_max, rng)
    duration = np.minimum(t_event, c)
    df.insert(0, "guest_id", [f"G{i:06d}" for i in range(n)])
    df["duration_days"] = np.maximum(duration, 1.0).round(1)
    df["event_observed"] = (t_event <= c).astype(int)
    log.info("Generated %d guests | censored=%.1f%% | median duration=%.0fd | c_hi=%.0f",
             n, 100 * (1 - df.event_observed.mean()), df.duration_days.median(), c_hi)
    return df


def main(config_path: str | None = None, n_override: int | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    n = n_override or cfg["data"]["n_samples"]
    df = generate(n, cfg["data"]["target_censoring"], cfg["data"]["max_follow_up_days"], cfg["seed"])
    out = resolve(cfg, "data")
    df.to_csv(out, index=False)
    log.info("Saved dataset -> %s", out)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    main(a.config, a.n)
