"""Synthetic hospitality guest-session generator with logged bandit feedback.

Produces, per session: context X, the logged action A drawn from a *confounded*
logging policy (propensity depends on context), the logged propensity, the realised
net contribution margin R, and (for simulation-only diagnostics) the ground-truth
expected margin of every action (`true_r_a*`) plus the full logging distribution
(`log_p_a*`).  The ground-truth columns must never be used as model inputs.
"""
import numpy as np
import pandas as pd

from .config import (
    ANC_MARGIN, ARCHETYPES, CHANNELS, DAYS, N_ACTIONS, ROOM_MARGIN, SEED, TIERS,
)

DISCOUNT = np.array([0.0, 0.10, 0.0, 0.0, 0.0, 0.0])
ANC_MULT = np.array([1.00, 1.00, 1.12, 1.40, 1.05, 1.05])
LOG_EPS = 0.12  # uniform mixing in the logging policy -> min propensity >= 0.02 (overlap)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _sample_context(n, rng):
    arch = rng.choice(4, size=n, p=[0.22, 0.28, 0.28, 0.22])  # Business, Family, Couple, Solo
    tier_p = np.array([[.20, .30, .30, .20], [.40, .30, .20, .10], [.35, .30, .20, .15], [.45, .30, .17, .08]])
    tier = (rng.random(n)[:, None] > np.cumsum(tier_p[arch], 1)).sum(1).clip(0, 3)

    p_ota = 0.45 - 0.10 * tier
    p_app = 0.15 + 0.10 * tier
    u = rng.random(n)
    channel = np.where(u < p_ota, 2, np.where(u < p_ota + p_app, 1, 0))  # 0 Web, 1 App, 2 OTA

    med_lead = np.array([8, 50, 35, 20])[arch]
    lead = np.clip(rng.lognormal(np.log(med_lead), 0.6), 0, 365).astype(int)

    los = np.where(arch == 0, 1 + rng.poisson(1.5, n),
          np.where(arch == 1, 3 + rng.poisson(2.5, n),
          np.where(arch == 2, 2 + rng.poisson(2.0, n), 1 + rng.poisson(2.0, n)))).clip(1, 14)
    party = np.where(arch == 0, 1 + (rng.random(n) < 0.1),
            np.where(arch == 1, 3 + rng.poisson(0.8, n),
            np.where(arch == 2, 2, 1))).clip(1, 8)

    peak = np.array([3, 3, 3, 1.2, 1, 1.2, 3, 3, 1, 1, 1.2, 3.5], dtype=float)  # month weights
    month = rng.choice(np.arange(1, 13), size=n, p=peak / peak.sum())
    dow_w = np.array([[1.4, 1.4, 1.4, 1.3, .6, .4, .4],      # Business: weekdays
                      [.7, .7, .8, 1, 1.4, 1.5, 1],          # Family
                      [.6, .6, .7, 1, 1.6, 1.6, 1],          # Couple
                      [1, 1, 1, 1, 1, 1, 1]])                # Solo
    dow_p = dow_w[arch] / dow_w[arch].sum(1, keepdims=True)
    dow = (rng.random(n)[:, None] > np.cumsum(dow_p, 1)).sum(1).clip(0, 6)

    hist_spend = np.exp(rng.normal(5.5 + 0.5 * tier + 0.2 * (arch == 1), 0.6)).round(2)
    anc_rate = np.clip(rng.beta(2 + 0.5 * tier, 5 - 0.5 * tier) + 0.08 * (arch == 1) + 0.06 * (arch == 2), 0, 1).round(3)
    return dict(arch=arch, tier=tier, channel=channel, lead=lead, los=los, party=party,
                month=month, dow=dow, hist_spend=hist_spend, anc_rate=anc_rate)


def _economics(c):
    """Return expected margin matrix (n,K), conversion prob matrix, room revenue, ancillary means."""
    n = len(c["tier"])
    tier, arch, ch, lead, los, party = c["tier"], c["arch"], c["channel"], c["lead"], c["los"], c["party"]
    month, dow, anc_rate = c["month"], c["dow"], c["anc_rate"]
    ota, app = (ch == 2).astype(float), (ch == 1).astype(float)
    business, family, couple, solo = [(arch == i).astype(float) for i in range(4)]

    season = np.where(np.isin(month, [12, 1, 2, 3, 7, 8]), 1.30, np.where(np.isin(month, [5, 6, 9, 10]), 0.85, 1.0))
    adr = 260 * season * np.array([1, 1.05, 1.12, 1.25])[tier] * np.array([.95, 1.05, 1.15, .85])[arch] \
        * np.where(dow >= 4, 1.10, 1.0)
    room_rev = adr * los
    anc_base = 30 + 110 * anc_rate * np.sqrt(party) * np.sqrt(los)

    z0 = (-0.55 + 0.30 * tier + 0.5 * app - 0.35 * ota
          + 0.7 * business * (lead < 14) + 0.5 * np.tanh((anc_rate - 0.3) * 4)
          - 0.45 * np.minimum(room_rev / 1500.0, 2.5) + 0.25 * family * (lead > 30))

    peak = np.isin(month, [12, 1, 2, 3, 7, 8]).astype(float)
    price_sens = 0.8 * ota + 0.5 * (tier == 0) + 0.3 * solo + 0.2 * (tier == 1)
    uplift = np.zeros((n, N_ACTIONS))
    uplift[:, 1] = 0.15 + 1.0 * price_sens - 0.5 * (tier == 3) - 0.35 * peak
    uplift[:, 2] = 0.10 + 0.7 * couple + 0.5 * (tier >= 2) + 0.35 * (los >= 4) + 0.2 * peak
    uplift[:, 3] = 0.15 + 0.7 * family + 0.5 * couple * (anc_rate > 0.3) + 0.3 * (party >= 3)
    uplift[:, 4] = 0.05 + 1.0 * business + 0.4 * (lead < 7) + 0.3 * app
    uplift[:, 5] = 0.00 + 0.8 * (tier == 3) + 0.4 * (tier == 2) + 0.3 * app + 0.25 * business
    p_conv = _sigmoid(z0[:, None] + uplift)

    cost = np.zeros((n, N_ACTIONS))
    cost[:, 2] = 60 + 15 * los
    cost[:, 3] = 50 * (0.5 + 0.5 * anc_rate)
    cost[:, 4] = 40 + 3 * party
    cost[:, 5] = 0.02 * room_rev
    anc_mean = anc_base[:, None] * ANC_MULT[None, :]
    room_margin = ROOM_MARGIN * room_rev[:, None] * (1 - DISCOUNT[None, :])
    cond_margin = room_margin + ANC_MARGIN * anc_mean - cost  # margin given conversion
    return p_conv * cond_margin, p_conv, room_margin, anc_mean, cost


def _logging_policy(c):
    n = len(c["tier"])
    tier, arch, ch, lead, anc = c["tier"], c["arch"], c["channel"], c["lead"], c["anc_rate"]
    lg = np.zeros((n, N_ACTIONS))
    lg[:, 0] = 0.6
    lg[:, 1] = 0.9 * (ch == 2) + 0.7 * (tier <= 1) + 0.3 * (lead > 60)
    lg[:, 2] = 0.9 * (tier >= 2) + 0.4 * (arch == 2)
    lg[:, 3] = 0.7 * (arch == 1) + 0.3 * (anc > 0.3)
    lg[:, 4] = 0.8 * (arch == 0) + 0.2 * (lead < 7)
    lg[:, 5] = 0.6 * (tier == 3) + 0.3 * (ch == 1)
    e = np.exp(lg - lg.max(1, keepdims=True))
    sm = e / e.sum(1, keepdims=True)
    return (1 - LOG_EPS) * sm + LOG_EPS / N_ACTIONS


def generate_dataset(n=30000, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = _sample_context(n, rng)
    exp_margin, p_conv, room_margin, anc_mean, cost = _economics(c)
    logp = _logging_policy(c)
    a = (rng.random(n)[:, None] > np.cumsum(logp, 1)).sum(1).clip(0, N_ACTIONS - 1)
    idx = np.arange(n)

    conv = rng.random(n) < p_conv[idx, a]
    room_noise = np.exp(rng.normal(-0.5 * 0.15 ** 2, 0.15, n))
    anc_noise = np.exp(rng.normal(-0.5 * 0.5 ** 2, 0.5, n))
    realised = room_margin[idx, a] * room_noise + ANC_MARGIN * anc_mean[idx, a] * anc_noise - cost[idx, a]
    reward = np.where(conv, realised, 0.0)

    df = pd.DataFrame({
        "loyalty_tier": np.array(TIERS)[c["tier"]],
        "lead_time": c["lead"],
        "los": c["los"],
        "booking_channel": np.array(CHANNELS)[c["channel"]],
        "hist_spend": c["hist_spend"],
        "ancillary_rate": c["anc_rate"],
        "month": c["month"],
        "day_of_week": np.array(DAYS)[c["dow"]],
        "party_size": c["party"],
        "guest_archetype": np.array(ARCHETYPES)[c["arch"]],
        "action": a,
        "propensity": logp[idx, a],
        "converted": conv.astype(int),
        "reward": reward,
    })
    for k in range(N_ACTIONS):
        df[f"true_r_a{k}"] = exp_margin[:, k]
    for k in range(N_ACTIONS):
        df[f"log_p_a{k}"] = logp[:, k]
    return df


def true_reward_matrix(df):
    return df[[f"true_r_a{k}" for k in range(N_ACTIONS)]].to_numpy()


def logging_prob_matrix(df):
    return df[[f"log_p_a{k}" for k in range(N_ACTIONS)]].to_numpy()


if __name__ == "__main__":
    d = generate_dataset()
    print(d.describe().T)
