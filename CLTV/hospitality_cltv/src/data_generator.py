"""Synthetic hospitality guest + booking generator (BG/NBD-style latent process with hospitality effects)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TIERS = ["Member", "Silver", "Gold", "Platinum", "Titanium"]
CHANNELS = ["Direct Web", "Mobile App", "OTA", "Corporate GDS"]
REGIONS = ["North America", "Europe", "APAC", "Middle East", "LATAM"]
ROOMS = ["Standard", "Deluxe", "Suite", "Club"]
PROP_TIERS = ["Luxury", "Premium", "Select Service"]


def _guests(n: int, rng: np.random.Generator) -> pd.DataFrame:
    tier_p = np.array([0.52, 0.22, 0.14, 0.08, 0.04])
    tier = rng.choice(len(TIERS), n, p=tier_p)
    channel = rng.choice(len(CHANNELS), n, p=[0.30, 0.25, 0.30, 0.15])
    region = rng.choice(len(REGIONS), n, p=[0.35, 0.25, 0.22, 0.08, 0.10])
    card = (rng.random(n) < (0.05 + 0.08 * tier)).astype(int)
    p_biz = np.where(channel == 3, 0.85, 0.25)
    biz = (rng.random(n) < p_biz).astype(int)
    # latent behaviour: purchase rate (stays / month), dropout prob per stay, value multiplier
    lam = rng.gamma(1.6, 0.11, n) * (1 + 0.35 * tier) * (1 + 0.25 * card) * np.where(biz == 1, 1.3, 1.0)
    p_drop = np.clip(rng.beta(1.1, 3.0, n) * (1 - 0.10 * tier), 0.02, 0.9)
    value = rng.lognormal(0.0, 0.35, n) * (1 + 0.18 * tier)
    anc_aff = rng.beta(1.5, 3.5, n) * (1 + 0.10 * tier)
    acq_day = rng.uniform(0, 22 * 30.4, n)
    return pd.DataFrame(dict(
        guest_id=np.arange(n), loyalty_tier=np.array(TIERS)[tier], acquisition_channel=np.array(CHANNELS)[channel],
        home_region=np.array(REGIONS)[region], cobrand_card=card, is_business_guest=biz, acq_day=acq_day,
        _lam=lam, _p_drop=p_drop, _value=value, _anc=anc_aff, _tier_i=tier, _chan_i=channel))


def _simulate_times(g: pd.DataFrame, scale: float, total_days: int, rng: np.random.Generator, kmax: int = 80):
    n = len(g)
    lam_day = (g["_lam"].values * scale / 30.4)[:, None]
    gaps = rng.exponential(1.0, (n, kmax)) / lam_day
    gaps[:, 0] = 0.0
    t = g["acq_day"].values[:, None] + np.cumsum(gaps, axis=1)
    # dropout: after each stay the guest churns w.p. p_drop
    churn_at = rng.geometric(g["_p_drop"].values)[:, None]  # index (1-based) of last alive stay
    alive = np.arange(1, kmax + 1)[None, :] <= churn_at
    return t, alive & (t < total_days)


def generate(cfg: dict, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (guests, bookings). Rate scale is calibrated so total bookings ~ target_bookings."""
    d = cfg["data"]
    rng = np.random.default_rng(seed)
    n = d["n_guests"]
    total_days = int(d["months_total"] * 30.4)
    g = _guests(n, rng)
    target = d["target_bookings"] * n / 50000 * 1.045  # compensates for off-peak thinning
    lo, hi = 0.2, 6.0
    for _ in range(14):  # bisection on the rate scale
        mid = (lo + hi) / 2
        _, m = _simulate_times(g, mid, total_days, np.random.default_rng(seed + 1))
        if m.sum() < target:
            lo = mid
        else:
            hi = mid
    scale = (lo + hi) / 2
    t, mask = _simulate_times(g, scale, total_days, np.random.default_rng(seed + 2))
    gi, ki = np.nonzero(mask)
    b = pd.DataFrame({"guest_id": gi, "checkin_day": t[gi, ki]})
    gg = g.iloc[gi].reset_index(drop=True)
    m = len(b)
    start = pd.Timestamp(d["start_date"])
    b["checkin_date"] = start + pd.to_timedelta(np.floor(b["checkin_day"]).astype(int), unit="D")
    month = b["checkin_date"].dt.month.values
    peak = np.isin(month, [6, 7, 8, 12])
    is_biz = (rng.random(m) < np.where(gg["is_business_guest"] == 1, 0.8, 0.12)).astype(int)
    # leisure stays skew to peak months: thin off-peak leisure stays slightly
    keep = np.where((is_biz == 0) & (~peak) & (ki > 0), rng.random(m) < 0.85, True)
    b, gg, is_biz, peak, month = b[keep].reset_index(drop=True), gg[keep].reset_index(drop=True), is_biz[keep], peak[keep], month[keep]
    m = len(b)
    los = 1 + rng.poisson(np.where(is_biz == 1, 1.2, 2.2), m)
    b["length_of_stay"] = los
    b["checkout_date"] = b["checkin_date"] + pd.to_timedelta(los, unit="D")
    b["is_business"] = is_biz
    b["booking_lead_time"] = np.clip(rng.gamma(np.where(is_biz == 1, 1.5, 2.2), np.where(is_biz == 1, 6, 22)), 0, 365).round().astype(int)
    ti = gg["_tier_i"].values
    prop_p = np.stack([0.12 + 0.07 * ti, 0.38 + 0.0 * ti, 0.50 - 0.07 * ti], 1)
    prop_p = np.clip(prop_p, 0.02, None); prop_p /= prop_p.sum(1, keepdims=True)
    pr = (rng.random(m)[:, None] > prop_p.cumsum(1)).sum(1).clip(0, 2)
    b["property_tier"] = np.array(PROP_TIERS)[pr]
    room = np.clip(rng.choice(4, m, p=[0.45, 0.30, 0.15, 0.10]) + (ti >= 3) * rng.integers(0, 2, m), 0, 3)
    b["room_type"] = np.array(ROOMS)[room]
    base = np.array([320.0, 190.0, 110.0])[pr] * (1 + 0.22 * room) * (1 + 0.20 * peak) * gg["_value"].values
    b["adr"] = (base * rng.lognormal(0, 0.12, m)).round(2)
    # lead-time effect: early bookers get a small discount
    b["adr"] = (b["adr"] * (1 - 0.0006 * np.minimum(b["booking_lead_time"], 120))).round(2)
    room_rev = b["adr"].values * los
    aff = gg["_anc"].values
    lux = (pr == 0) * 1.0 + (pr == 1) * 0.6 + (pr == 2) * 0.25
    b["fb_spend"] = (rng.random(m) < 0.4 + 0.5 * aff) * rng.gamma(2, 22 * los * lux * (1 + aff))
    b["spa_spend"] = (rng.random(m) < 0.05 + 0.5 * aff * lux) * rng.gamma(2, 60 * lux)
    b["golf_spend"] = (rng.random(m) < 0.03 + 0.2 * aff * lux * (is_biz == 0)) * rng.gamma(2, 90 * lux)
    b["parking_spend"] = (rng.random(m) < 0.3) * 18 * los
    for c in ["fb_spend", "spa_spend", "golf_spend", "parking_spend"]:
        b[c] = b[c].round(2)
    p_cancel = np.clip(0.07 + 0.001 * np.minimum(b["booking_lead_time"], 150) - 0.012 * ti + 0.05 * (gg["acquisition_channel"] == "OTA"), 0.01, 0.5)
    b["is_cancelled"] = (rng.random(m) < p_cancel).astype(int)
    b["room_revenue"] = (room_rev * (1 - b["is_cancelled"])).round(2)
    b["ancillary_revenue"] = ((b[["fb_spend", "spa_spend", "golf_spend", "parking_spend"]].sum(1)) * (1 - b["is_cancelled"])).round(2)
    b["total_revenue"] = b["room_revenue"] + b["ancillary_revenue"]
    b["booking_channel"] = np.where(rng.random(m) < 0.65, gg["acquisition_channel"].values, rng.choice(CHANNELS, m))
    b = b.sort_values(["guest_id", "checkin_day"]).reset_index(drop=True)
    b.insert(0, "booking_id", np.arange(len(b)))
    # guest attrs: first stay defines the tenure origin
    guests = g[["guest_id", "loyalty_tier", "acquisition_channel", "home_region", "cobrand_card", "is_business_guest", "acq_day"]].copy()
    guests["first_stay_date"] = start + pd.to_timedelta(np.floor(guests["acq_day"]).astype(int), unit="D")
    guests = guests[guests["guest_id"].isin(b["guest_id"])].reset_index(drop=True)
    return guests, b
