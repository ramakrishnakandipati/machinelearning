"""Feature engineering: leakage-safe calibration-window features, targets, and sequence tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_generator import CHANNELS, PROP_TIERS, REGIONS, ROOMS, TIERS

DAYS_PER_MONTH = 30.4
CAT_COLS = ["loyalty_tier", "acquisition_channel", "home_region"]
CAT_LEVELS = {"loyalty_tier": TIERS, "acquisition_channel": CHANNELS, "home_region": REGIONS}
NUM_COLS = [
    "recency_days", "T_days", "frequency", "repeat_x", "t_x", "monetary_mean", "monetary_total", "ia_mean", "ia_std",
    "spend_trend", "adr_expansion", "anc_ratio", "direct_share", "cancel_rate", "weekend_ratio", "peak_affinity",
    "mean_los", "mean_lead", "biz_share", "luxury_share", "stays_6m", "stays_12m", "rev_12m", "mean_adr",
    "tier_ord", "cobrand_card", "is_business_guest", "n_bookings",
]
SEQ_FEATS = ["log_rev", "log_adr", "los", "log_lead", "is_business", "is_cancelled", "anc_ratio", "weekend",
             "peak", "prop_ord", "room_ord", "direct", "log_dt", "log_since"]
BOOKING_COLS = ["guest_id", "checkin_day", "checkin_date", "total_revenue", "room_revenue", "ancillary_revenue", "adr",
                "length_of_stay", "booking_lead_time", "is_business", "is_cancelled", "booking_channel",
                "property_tier", "room_type"]


@dataclass
class Dataset:
    """Container for modelling tables (feature frame, targets, sequences, splits)."""
    df: pd.DataFrame            # index=guest_id; NUM_COLS + CAT_COLS + targets
    seq: np.ndarray             # (N, L, F) float32
    seq_mask: np.ndarray        # (N, L) bool, True=real stay
    seq_dt: np.ndarray          # (N, L) days since previous stay
    seq_abs: np.ndarray         # (N, L) days before cutoff
    split: np.ndarray           # array of 'train'/'val'/'test'
    cutoff_day: float
    horizon_days: float

    def idx(self, name: str) -> np.ndarray:
        return np.where(self.split == name)[0]


def encode_cats(df: pd.DataFrame) -> pd.DataFrame:
    """Integer-code categoricals with fixed level lists (no fitted state)."""
    out = df.copy()
    for c, lv in CAT_LEVELS.items():
        out[c + "_code"] = pd.Categorical(out[c], categories=lv).codes.astype(int)
    return out


def tree_columns() -> list[str]:
    return NUM_COLS + [c + "_code" for c in CAT_COLS]


def _slope(x: np.ndarray, y: np.ndarray, g: np.ndarray, n: int) -> np.ndarray:
    """Per-group OLS slope of y on x (0 where undefined)."""
    cnt = np.bincount(g, minlength=n).astype(float)
    sx, sy = np.bincount(g, x, n), np.bincount(g, y, n)
    sxy, sxx = np.bincount(g, x * y, n), np.bincount(g, x * x, n)
    num = cnt * sxy - sx * sy
    den = cnt * sxx - sx ** 2
    ok = den > 1e-2
    return np.clip(np.where(ok, num / np.where(ok, den, 1), 0.0), -5000, 5000)


def build_features(guests: pd.DataFrame, bookings: pd.DataFrame, cutoff_day: float) -> pd.DataFrame:
    """Build per-guest features using only bookings with checkin_day < cutoff_day."""
    b = bookings[bookings["checkin_day"] < cutoff_day].copy()
    b = b.sort_values(["guest_id", "checkin_day"])
    ids = np.sort(guests["guest_id"].values)
    pos = {g: i for i, g in enumerate(ids)}
    b = b[b["guest_id"].isin(pos)]
    gi = b["guest_id"].map(pos).values
    n = len(ids)
    ok = b["is_cancelled"].values == 0
    day = b["checkin_day"].values

    def safe(a: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.divide(a, c, out=np.zeros(n), where=c > 0)

    cnt_all = np.bincount(gi, minlength=n).astype(float)
    cnt = np.bincount(gi[ok], minlength=n).astype(float)
    first_any = np.full(n, cutoff_day)
    np.minimum.at(first_any, gi, day)
    first_ok = np.full(n, np.inf)
    np.minimum.at(first_ok, gi[ok], day[ok])
    last_ok = np.full(n, -np.inf)
    np.maximum.at(last_ok, gi[ok], day[ok])
    has = cnt > 0
    first_ok = np.where(has, first_ok, first_any)
    last_ok = np.where(has, last_ok, first_any)
    tot = np.bincount(gi[ok], b["total_revenue"].values[ok], n)

    bo = b[ok]
    gio = gi[ok]
    d_o = bo["checkin_day"].values
    dd = np.diff(d_o, prepend=np.nan)
    first_flag = np.r_[True, gio[1:] != gio[:-1]] if len(gio) else np.array([], bool)
    dd = np.where(first_flag, np.nan, dd)
    v = ~np.isnan(dd)
    ia_n = np.bincount(gio[v], minlength=n).astype(float)
    ia_s = np.bincount(gio[v], dd[v], n)
    ia_ss = np.bincount(gio[v], dd[v] ** 2, n)
    ia_mean = safe(ia_s, ia_n)
    ia_std = np.sqrt(np.maximum(safe(ia_ss, ia_n) - ia_mean ** 2, 0))
    trend = _slope(d_o / DAYS_PER_MONTH, bo["total_revenue"].values, gio, n)
    adr_o = bo["adr"].values
    rank = bo.groupby("guest_id").cumcount().values
    nstay = np.bincount(gio, minlength=n)
    first_adr = np.bincount(gio[rank == 0], adr_o[rank == 0], n)
    last_mask = rank == (nstay[gio] - 1)
    last_adr = np.bincount(gio[last_mask], adr_o[last_mask], n)
    adr_exp = np.where(first_adr > 0, last_adr / np.maximum(first_adr, 1e-9), 1.0)
    room_rev = np.bincount(gio, bo["room_revenue"].values, n)
    anc_rev = np.bincount(gio, bo["ancillary_revenue"].values, n)
    ci = pd.to_datetime(bo["checkin_date"])
    weekend = ci.dt.dayofweek.isin([4, 5]).values.astype(float)
    peak = ci.dt.month.isin([6, 7, 8, 12]).values.astype(float)
    lux = (bo["property_tier"] == "Luxury").values.astype(float)
    ball_direct = b["booking_channel"].isin(["Direct Web", "Mobile App"]).values.astype(float)
    r12 = d_o >= cutoff_day - 12 * DAYS_PER_MONTH

    f = pd.DataFrame(index=ids)
    f["recency_days"] = cutoff_day - last_ok
    f["T_days"] = cutoff_day - first_any
    f["frequency"] = cnt
    f["repeat_x"] = np.maximum(cnt - 1, 0)
    f["t_x"] = np.where(cnt > 1, last_ok - first_ok, 0.0)
    f["monetary_mean"] = safe(tot, cnt)
    f["monetary_total"] = tot
    f["ia_mean"], f["ia_std"], f["spend_trend"], f["adr_expansion"] = ia_mean, ia_std, trend, adr_exp
    f["anc_ratio"] = safe(anc_rev, room_rev)
    f["direct_share"] = safe(np.bincount(gi, ball_direct, n), cnt_all)
    f["cancel_rate"] = safe(cnt_all - cnt, cnt_all)
    f["weekend_ratio"] = safe(np.bincount(gio, weekend, n), cnt)
    f["peak_affinity"] = safe(np.bincount(gio, peak, n), cnt)
    f["mean_los"] = safe(np.bincount(gio, bo["length_of_stay"].values, n), cnt)
    f["mean_lead"] = safe(np.bincount(gio, bo["booking_lead_time"].values, n), cnt)
    f["biz_share"] = safe(np.bincount(gio, bo["is_business"].values, n), cnt)
    f["luxury_share"] = safe(np.bincount(gio, lux, n), cnt)
    f["stays_6m"] = np.bincount(gio[d_o >= cutoff_day - 6 * DAYS_PER_MONTH], minlength=n)
    f["stays_12m"] = np.bincount(gio[r12], minlength=n)
    f["rev_12m"] = np.bincount(gio[r12], bo["total_revenue"].values[r12], n)
    f["mean_adr"] = safe(np.bincount(gio, adr_o, n), cnt)
    f["n_bookings"] = cnt_all
    g = guests.set_index("guest_id").loc[ids]
    f["tier_ord"] = pd.Categorical(g["loyalty_tier"], categories=TIERS).codes
    f["cobrand_card"] = g["cobrand_card"].values
    f["is_business_guest"] = g["is_business_guest"].values
    for c in CAT_COLS:
        f[c] = g[c].values
    f.index.name = "guest_id"
    return f


def build_sequences(bookings: pd.DataFrame, ids: np.ndarray, cutoff_day: float, L: int):
    """Right-aligned (most recent last), left-padded chronological stay tensors."""
    b = bookings[bookings["checkin_day"] < cutoff_day].sort_values(["guest_id", "checkin_day"])
    pos = {g: i for i, g in enumerate(ids)}
    b = b[b["guest_id"].isin(pos)].copy()
    gi = b["guest_id"].map(pos).values
    n = len(ids)
    cnt = np.bincount(gi, minlength=n)
    rank = b.groupby("guest_id").cumcount().values
    day_all = b["checkin_day"].values
    newg_all = np.r_[True, gi[1:] != gi[:-1]]
    dt_all = np.where(newg_all, 0.0, day_all - np.r_[np.nan, day_all[:-1]])
    rfe = cnt[gi] - 1 - rank
    keep = rfe < L
    b, gi, rfe, dt = b[keep], gi[keep], rfe[keep], dt_all[keep]
    slot = L - 1 - rfe
    day = b["checkin_day"].values
    ci = pd.to_datetime(b["checkin_date"])
    room_rev, anc = b["room_revenue"].values, b["ancillary_revenue"].values
    feats = np.stack([
        np.log1p(b["total_revenue"].values), np.log(b["adr"].values + 1), b["length_of_stay"].values / 5.0,
        np.log1p(b["booking_lead_time"].values), b["is_business"].values.astype(float),
        b["is_cancelled"].values.astype(float),
        np.minimum(np.divide(anc, np.maximum(room_rev, 1), out=np.zeros_like(anc), where=room_rev > 0), 3.0),
        ci.dt.dayofweek.isin([4, 5]).values.astype(float), ci.dt.month.isin([6, 7, 8, 12]).values.astype(float),
        pd.Categorical(b["property_tier"], categories=PROP_TIERS).codes / 2.0,
        pd.Categorical(b["room_type"], categories=ROOMS).codes / 3.0,
        b["booking_channel"].isin(["Direct Web", "Mobile App"]).values.astype(float),
        np.log1p(dt), np.log1p(cutoff_day - day),
    ], 1).astype(np.float32)
    seq = np.zeros((n, L, len(SEQ_FEATS)), np.float32)
    mask = np.zeros((n, L), bool)
    sdt = np.zeros((n, L), np.float32)
    sabs = np.zeros((n, L), np.float32)
    seq[gi, slot] = feats
    mask[gi, slot] = True
    sdt[gi, slot] = dt
    sabs[gi, slot] = cutoff_day - day
    return seq, mask, sdt, sabs


def build_targets(guests: pd.DataFrame, bookings: pd.DataFrame, cutoff_day: float, horizon_days: float) -> pd.DataFrame:
    """Future-window targets: revenue, #stays, time-to-first-return (censored at horizon), event flag."""
    ids = np.sort(guests["guest_id"].values)
    pos = {g: i for i, g in enumerate(ids)}
    n = len(ids)
    b = bookings[(bookings["checkin_day"] >= cutoff_day) & (bookings["checkin_day"] < cutoff_day + horizon_days)
                 & (bookings["is_cancelled"] == 0) & bookings["guest_id"].isin(pos)]
    gi = b["guest_id"].map(pos).values
    y = np.bincount(gi, b["total_revenue"].values, n)
    ny = np.bincount(gi, minlength=n)
    first = np.full(n, horizon_days)
    np.minimum.at(first, gi, b["checkin_day"].values - cutoff_day)
    return pd.DataFrame({"y": y, "y_n": ny, "y_ret": (ny > 0).astype(int), "y_time": np.maximum(first, 0.5),
                         "y_event": (ny > 0).astype(int)}, index=pd.Index(ids, name="guest_id"))


def build_dataset(guests: pd.DataFrame, bookings: pd.DataFrame, cfg: dict[str, Any], seed: int = 42) -> Dataset:
    """Assemble features, targets, sequences and a stratified guest-level train/val/test split."""
    d = cfg["data"]
    cutoff = d["calibration_months"] * DAYS_PER_MONTH
    horizon = d["horizon_months"] * DAYS_PER_MONTH
    df = build_features(guests, bookings, cutoff).join(build_targets(guests, bookings, cutoff, horizon))
    df = encode_cats(df)
    seq, mask, sdt, sabs = build_sequences(bookings, df.index.values, cutoff, d["seq_len"])
    s = cfg["split"]
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=1 - s["train"], random_state=seed, stratify=df["y_ret"])
    va, te = train_test_split(rest, test_size=s["test"] / (s["val"] + s["test"]), random_state=seed,
                              stratify=df["y_ret"].values[rest])
    split = np.empty(len(df), dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    return Dataset(df, seq, mask, sdt, sabs, split, cutoff, horizon)
