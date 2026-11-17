"""Module 5a - Off-policy (counterfactual) estimators and policy builders.

All estimates use the *logged* propensities, so a policy can be evaluated on
holdout logs before it is ever deployed:

    SNIPS(pi) = sum_i w_i * Y_i / sum_i w_i,   w_i = pi(a_i | x_i) / p_log(a_i | x_i)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier

from .config import ACTION_COSTS, DEFAULT_LAMBDA, SEED, UNSUB_RATE_CAP
from .policy import COST_VEC, choose_actions, decide
from .synthetic_data import legacy_crm_distribution
from .uplift_model import UpliftBundle

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]
MAX_WEIGHT = 50.0  # weight cap to bound SNIPS variance (reported via ESS)


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def importance_weights(a_logged: IntArr, p_logged: FloatArr, pi_prob_of_logged: FloatArr) -> FloatArr:
    """w = pi(a_logged | x) / p_logged, capped at ``MAX_WEIGHT``."""
    return np.minimum(pi_prob_of_logged / np.clip(p_logged, 1e-6, 1.0), MAX_WEIGHT)


def snips(y: FloatArr, w: FloatArr) -> float:
    """Self-normalised importance-sampling estimate of policy value."""
    s = w.sum()
    return float((w * y).sum() / s) if s > 0 else float("nan")


def effective_sample_size(w: FloatArr) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0


def bootstrap_snips_ci(y: FloatArr, w: FloatArr, n_boot: int = 1000, seed: int = SEED) -> tuple[float, float]:
    """95% percentile bootstrap CI (one decision per customer => row bootstrap)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        sw = w[i].sum()
        stats[b] = (w[i] * y[i]).sum() / sw if sw > 0 else np.nan
    return float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


@dataclass
class PolicyReport:
    name: str
    value: float
    ci_low: float
    ci_high: float
    ess: float
    match_rate: float
    oracle_incremental: float | None = None   # synthetic-only sanity check vs ground truth


def evaluate_policy(name: str, log: pd.DataFrame, pi_prob_of_logged: FloatArr, chosen: IntArr | None = None,
                    outcome: str = "net_margin_30d") -> PolicyReport:
    """SNIPS value + bootstrap CI of a (possibly stochastic) target policy."""
    y = log[outcome].to_numpy(dtype=float)
    w = importance_weights(log["action_id"].to_numpy(), log["propensity_score"].to_numpy(), pi_prob_of_logged)
    lo, hi = bootstrap_snips_ci(y, w)
    oracle = None
    if chosen is not None and "truth_tau_margin_a1" in log:
        tm = log[[f"truth_tau_margin_a{a}" for a in (1, 2, 3, 4)]].to_numpy()
        tm = np.hstack([np.zeros((len(tm), 1)), tm]) - COST_VEC[None, :]
        oracle = float(tm[np.arange(len(chosen)), chosen].mean())
    return PolicyReport(name, snips(y, w), lo, hi, effective_sample_size(w), float((pi_prob_of_logged > 0).mean()), oracle)


# --------------------------------------------------------------------------- #
# Candidate policies
# --------------------------------------------------------------------------- #
def legacy_policy_probs(x: pd.DataFrame, a_logged: IntArr) -> FloatArr:
    """pi_legacy(a_logged | x): the (stochastic) historical CRM rules."""
    return legacy_crm_distribution(x)[np.arange(len(x)), a_logged]


def fit_naive_propensity_model(x_train: pd.DataFrame, purchased: np.ndarray, seed: int = SEED) -> LGBMClassifier:
    """Naive baseline: P(purchase | x), ignoring treatment (targets Sure Things)."""
    return LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=50, verbose=-1,
                          n_jobs=4, random_state=seed).fit(x_train, purchased.astype(int))


def naive_propensity_actions(model: LGBMClassifier, x: pd.DataFrame, top_frac: float = 0.4) -> IntArr:
    """Send the 15%-off promo (A3) to the top ``top_frac`` most-likely buyers."""
    s = model.predict_proba(x)[:, 1]
    return np.where(s >= np.quantile(s, 1 - top_frac), 3, 0).astype(np.int64)


def unconstrained_uplift_actions(bundle: UpliftBundle, x: pd.DataFrame) -> IntArr:
    """argmax of raw margin uplift: ignores cost, unsubscribe risk and guardrails."""
    tm, _ = bundle.predict_tau(x.to_numpy(dtype=np.float32))
    full = np.hstack([np.zeros((len(tm), 1)), tm])
    return full.argmax(axis=1).astype(np.int64)


def deterministic_probs(chosen: IntArr, a_logged: IntArr) -> FloatArr:
    return (chosen == a_logged).astype(np.float64)


# --------------------------------------------------------------------------- #
# Lambda sweep (margin vs unsubscribe Pareto frontier)
# --------------------------------------------------------------------------- #
LAMBDA_GRID: tuple[float, ...] = (0, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200)


def baseline_unsub_rate(log: pd.DataFrame) -> float:
    """IPW (Hajek) unsubscribe rate under no action, from control-arm rows."""
    c = log[log["action_id"] == 0]
    w = 1.0 / c["propensity_score"].to_numpy()
    return snips(c["unsubscribed_flag"].to_numpy(dtype=float), w)


def lambda_sweep(bundle: UpliftBundle, x: pd.DataFrame, log: pd.DataFrame, grid=LAMBDA_GRID) -> pd.DataFrame:
    """Evaluate the guardrailed policy for each risk penalty lambda.

    Margin: SNIPS on logged net margin. Unsubscribe rate: control-arm baseline
    plus the mean predicted unsubscribe uplift of the chosen action (direct
    method - unsubscribe events are too rare for a stable SNIPS estimate).
    """
    base = baseline_unsub_rate(log)
    promos = log["promos_last_7d"].to_numpy()
    rows = []
    for lam in grid:
        d = decide(bundle, x, promos, lam=lam)
        tu_full = np.hstack([np.zeros((len(x), 1)), np.clip(d.tau_unsub, 0.0, None)])
        unsub = base + float(tu_full[np.arange(len(x)), d.actions].mean())
        rep = evaluate_policy(f"lambda={lam}", log, deterministic_probs(d.actions, log["action_id"].to_numpy()), d.actions)
        rows.append({"lambda": lam, "net_margin": rep.value, "ci_low": rep.ci_low, "ci_high": rep.ci_high,
                     "unsub_rate": unsub, "treat_rate": float((d.actions > 0).mean()), "oracle": rep.oracle_incremental})
    return pd.DataFrame(rows)


def select_lambda(sweep: pd.DataFrame, cap: float = UNSUB_RATE_CAP) -> float:
    """Smallest lambda whose estimated unsubscribe rate respects the business cap."""
    ok = sweep[sweep["unsub_rate"] <= cap]
    return float(ok["lambda"].min()) if len(ok) else float(sweep["lambda"].max())
