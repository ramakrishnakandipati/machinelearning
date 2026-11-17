"""Module 2c - Synthetic omnichannel retail world with known ground truth.

The generator creates (1) raw transaction / session history, (2) a logged
campaign produced by a stochastic *legacy CRM* policy mixed with 5% epsilon
exploration (so propensities are known and positivity holds), and (3) delayed,
right-censored outcomes obeying the 7-day conversion + 30-day return windows.
True CATEs are kept in ``truth_*`` columns purely for diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    ACTION_COSTS, CONVERSION_WINDOW_DAYS, EXPLORATION_EPSILON, N_CATEGORIES,
    REWARD_MATURITY_DAYS, RETURN_WINDOW_DAYS, SEED,
)
from .exploration import mix_with_uniform, sample_from_probs
from .features import FeatureTransformer, build_features

CUTOFF = pd.Timestamp("2026-06-30")  # "now": data is observed up to here

# latent segment ids: 0 full-price loyalist, 1 discount hunter, 2 at-risk, 3 mainstream
_SEG_P = np.array([0.25, 0.30, 0.20, 0.25])
_MD_BETA = [(1, 9), (8, 2), (3, 4), (3, 5)]
_RATE = np.array([9.0, 6.0, 3.0, 4.5])      # orders / year
_AOV = np.array([110.0, 70.0, 85.0, 90.0])
_P0 = np.array([0.30, 0.12, 0.035, 0.07])   # baseline 7-day purchase prob
_RET = np.array([0.15, 0.25, 0.20, 0.20])   # baseline return rate
_LIFT = {  # additive purchase-probability lift per action x segment
    1: np.array([0.012, 0.020, 0.010, 0.045]),
    2: np.array([0.030, 0.100, 0.020, 0.200]),
    3: np.array([0.060, 0.200, 0.060, 0.130]),
    4: np.array([0.050, 0.030, 0.010, 0.090]),
}
_MARGIN_RATE = {0: 0.50, 1: 0.50, 2: 0.50, 3: 0.35, 4: 0.50}  # 15% off erodes margin
_UNSUB_BASE = 0.003
_UNSUB_EFFECT = {
    1: np.array([0.0005, 0.0005, 0.0010, 0.0005]),
    2: np.array([0.0010, 0.0015, 0.0030, 0.0010]),
    3: np.array([0.0100, 0.0120, 0.0250, 0.0090]),
    4: np.array([0.0005, 0.0005, 0.0010, 0.0005]),
}


@dataclass
class World:
    """Raw history + latent customer traits."""

    transactions: pd.DataFrame
    sessions: pd.DataFrame
    profile: pd.DataFrame
    category_stock: pd.Series
    as_of: pd.Series          # decision timestamp per customer
    latent: pd.DataFrame      # hidden traits (never exposed to the model)


def generate_world(n_customers: int = 30000, seed: int = SEED) -> World:
    """Generate customer history up to each customer's decision timestamp."""
    rng = np.random.default_rng(seed)
    n = n_customers
    ids = np.arange(n)
    seg = rng.choice(4, size=n, p=_SEG_P)
    guest = rng.random(n) < 0.08
    md_true = np.empty(n)
    for s in range(4):
        m = seg == s
        md_true[m] = rng.beta(*_MD_BETA[s], size=m.sum())
    rate = _RATE[seg] * rng.lognormal(0, 0.25, n)
    rate[guest] = 0.4
    price = _AOV[seg] * rng.lognormal(0, 0.2, n)
    online_share = rng.beta(2, 2, n)
    app_active = (rng.random(n) < 0.6).astype(float)
    top_cat = rng.choice(N_CATEGORIES, size=n)
    size_run = rng.beta(8, 2, n)
    cab_rate = rng.beta(2, 6, n)
    category_stock = pd.Series(rng.uniform(0.1, 0.95, N_CATEGORIES), index=range(N_CATEGORIES))
    decision_days_ago = rng.uniform(10, 150, n)
    as_of = CUTOFF - pd.to_timedelta(decision_days_ago, unit="D")

    # ---- transactions (history window 400d before each decision) ---------
    n_orders = rng.poisson(rate * 400 / 365)
    rep = np.repeat(ids, n_orders)
    lo = np.where(seg[rep] == 2, 100.0, 1.0)  # at-risk customers stopped buying recently
    offset = lo + rng.random(len(rep)) * (400.0 - lo)
    tx = pd.DataFrame({
        "customer_id": rep,
        "ts": as_of[rep] - pd.to_timedelta(offset, unit="D"),
        "amount": price[rep] * rng.lognormal(-0.06, 0.35, len(rep)),
        "is_markdown": (rng.random(len(rep)) < md_true[rep]).astype(float),
        "is_online": (rng.random(len(rep)) < online_share[rep]).astype(float),
        "category": np.where(rng.random(len(rep)) < 0.55, top_cat[rep], rng.integers(0, N_CATEGORIES, len(rep))),
    })

    # ---- sessions -----------------------------------------------------------
    sess_rate = np.array([12.0, 10.0, 5.0, 9.0])[seg] * rng.lognormal(0, 0.3, n)
    n_sess = np.where(guest, rng.integers(0, 2, n), 2 + rng.poisson(sess_rate))
    srep = np.repeat(ids, n_sess)
    ss = pd.DataFrame({
        "customer_id": srep,
        "ts": as_of[srep] - pd.to_timedelta(rng.uniform(0.01, 365, len(srep)), unit="D"),
        "cart_abandon": (rng.random(len(srep)) < cab_rate[srep]).astype(float),
    })
    profile = pd.DataFrame({"customer_id": ids, "size_run_factor": size_run, "app_active_30d": app_active})
    latent = pd.DataFrame({
        "seg": seg, "guest": guest, "md_true": md_true, "rate": rate, "price": price,
        "online_share": online_share, "app_active": app_active, "top_cat": top_cat,
    }, index=pd.Index(ids, name="customer_id"))
    return World(tx, ss, profile, category_stock, pd.Series(as_of, index=pd.Index(ids, name="customer_id")), latent)


def legacy_crm_distribution(x: pd.DataFrame) -> np.ndarray:
    """Stochastic rule-based CRM: the historical logging policy (n, 5).

    Rules key off observable features only. Note the legacy flaw this project
    fixes: it blasts discounts at at-risk and full-price customers.
    """
    n = len(x)
    out = np.tile(np.array([0.15, 0.30, 0.20, 0.20, 0.15]), (n, 1))
    md, rec, f90 = x["markdown_share"].to_numpy(), x["recency_days"].to_numpy(), x["freq_90d"].to_numpy()
    loyal = (md < 0.25) & (f90 >= 2)
    hunter = md > 0.60
    atrisk = rec > 120
    out[loyal] = [0.15, 0.45, 0.10, 0.20, 0.10]
    out[hunter] = [0.10, 0.15, 0.10, 0.55, 0.10]
    out[atrisk] = [0.10, 0.10, 0.10, 0.60, 0.10]
    return out / out.sum(axis=1, keepdims=True)


def simulate_campaign(world: World, raw_features: pd.DataFrame, x_imputed: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Sample logged actions and delayed outcomes; returns the campaign log."""
    rng = np.random.default_rng(seed + 1)
    n = len(raw_features)
    lat = world.latent.loc[raw_features.index]
    seg = lat["seg"].to_numpy()
    ts = world.as_of.loc[raw_features.index]

    # ---- logging policy: legacy CRM + 5% epsilon exploration ---------------
    probs = mix_with_uniform(legacy_crm_distribution(x_imputed), EXPLORATION_EPSILON)
    action = sample_from_probs(probs, rng)
    propensity = probs[np.arange(n), action]

    # ---- ground-truth response surfaces -----------------------------------
    trait = rng.lognormal(0, 0.25, n)  # unobserved persuadability
    p0 = _P0[seg] * (lat["rate"].to_numpy() / _RATE[seg]) ** 0.5 * rng.lognormal(0, 0.2, n)
    p0 = np.clip(p0 * np.where(lat["guest"], 0.5, 1.0), 0.005, 0.6)
    avail = x_imputed["size_availability_score"].to_numpy()
    app = lat["app_active"].to_numpy()
    onl = lat["online_share"].to_numpy()
    scale = {1: 0.25 + 0.75 * avail, 2: (0.3 + 0.7 * onl) * (0.4 + 0.6 * app), 3: 1.0, 4: 0.1 + 0.9 * app}
    p = np.zeros((n, 5)); p[:, 0] = p0
    u = np.zeros((n, 5)); u[:, 0] = np.where(seg == 2, 1.8, 1.0) * _UNSUB_BASE
    ret = np.zeros((n, 5)); ret[:, 0] = _RET[seg]
    for a in (1, 2, 3, 4):
        p[:, a] = np.clip(p0 + _LIFT[a][seg] * scale[a] * trait, 0, 0.9)
        u[:, a] = u[:, 0] + _UNSUB_EFFECT[a][seg] * trait
        ret[:, a] = ret[:, 0] + (0.07 if a == 3 else 0.0)
    for a in (1, 2, 4):
        ret[:, a] = ret[:, 0]
    price = lat["price"].to_numpy()
    m_rate = np.array([_MARGIN_RATE[a] for a in range(5)])
    exp_margin = p * price[:, None] * m_rate[None, :] * (1 - ret)  # E[gross margin | action]
    truth_tau_margin = exp_margin[:, 1:] - exp_margin[:, [0]]
    truth_tau_unsub = u[:, 1:] - u[:, [0]]

    # ---- realize outcomes of the *logged* action ---------------------------
    idx = np.arange(n)
    bought = rng.random(n) < p[idx, action]
    value = price * rng.lognormal(-0.06, 0.35, n)
    returned = bought & (rng.random(n) < ret[idx, action])
    buy_lag = rng.uniform(0, CONVERSION_WINDOW_DAYS, n)           # attributed only if <= 7d
    ret_lag = buy_lag + rng.uniform(1, RETURN_WINDOW_DAYS, n)     # return window closes at +30d after purchase
    elapsed = (CUTOFF - ts).dt.total_seconds().to_numpy() / 86400.0
    mature = elapsed >= REWARD_MATURITY_DAYS
    # Right-censoring for immature rows: purchases / returns not yet observed.
    bought_obs = bought & (buy_lag <= elapsed)
    returned_obs = returned & (ret_lag <= elapsed)
    kept = bought_obs & ~returned_obs
    margin = np.where(kept, value * m_rate[action], 0.0)
    revenue = np.where(kept, value * np.where(action == 3, 0.85, 1.0), 0.0)
    unsub = rng.random(n) < u[idx, action]
    promos7 = np.clip(rng.poisson(0.7, n), 0, 4)

    log = pd.DataFrame({
        "customer_id": raw_features.index.to_numpy(),
        "timestamp": ts.to_numpy(),
        "action_id": action,
        "propensity_score": propensity,
        "incremental_revenue": revenue,
        "gross_margin_dollars": margin,
        "returned_flag": returned_obs,
        "purchased_flag": bought_obs,
        "unsubscribed_flag": unsub,
        "reward_mature": mature,
        "promos_last_7d": promos7,
    })
    log["net_margin_30d"] = log["gross_margin_dollars"] - log["action_id"].map(ACTION_COSTS)
    for a in range(5):
        log[f"logp_a{a}"] = probs[:, a]
    for j, a in enumerate((1, 2, 3, 4)):
        log[f"truth_tau_margin_a{a}"] = truth_tau_margin[:, j]
        log[f"truth_tau_unsub_a{a}"] = truth_tau_unsub[:, j]
    log["truth_p0"] = p0
    return log.reset_index(drop=True)


@dataclass
class Dataset:
    """Everything downstream modules need."""

    log: pd.DataFrame            # campaign log (one row per decision)
    raw_features: pd.DataFrame   # point-in-time features before imputation
    x: pd.DataFrame              # imputed features (train+test), aligned to ``log``
    transformer: FeatureTransformer
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_dataset(n_customers: int = 30000, seed: int = SEED, test_frac: float = 0.3) -> Dataset:
    """End-to-end synthetic dataset with a time-based, maturity-aware split."""
    world = generate_world(n_customers, seed)
    raw = build_features(world.transactions, world.sessions, world.as_of, world.profile, world.category_stock)
    transformer = FeatureTransformer().fit(raw)
    x = transformer.transform(raw)
    log = simulate_campaign(world, raw, x, seed)
    x = x.reset_index(drop=True)
    raw = raw.reset_index(drop=True)
    # Only rewards whose 7d + 30d windows have closed may be used; split by time.
    mature_idx = log.index[log["reward_mature"]].to_numpy()
    order = mature_idx[np.argsort(log.loc[mature_idx, "timestamp"].to_numpy())]
    cut = int(len(order) * (1 - test_frac))
    return Dataset(log, raw, x, transformer, np.sort(order[:cut]), np.sort(order[cut:]))


if __name__ == "__main__":
    ds = make_dataset(5000)
    print(ds.log.head())
    print("train/test:", len(ds.train_idx), len(ds.test_idx), "| mature share:", ds.log["reward_mature"].mean().round(3))
    print("cold-start share:", ds.x["is_cold_start"].mean().round(3))
