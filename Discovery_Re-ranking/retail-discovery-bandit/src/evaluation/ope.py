"""Off-Policy Evaluation: IPS, SNIPS, Doubly Robust, Switch-DR (+ DM baseline), bootstrap CIs, ESS and weight diagnostics.

Slot-level formulation (matches the logged schema `(user, item, slot, propensity, context, reward)`):
every policy is evaluated as its *stochastic Boltzmann variant*, i.e. sequential sampling without replacement
(Plackett-Luce) from softmax(score/tau) over the guard-railed candidate pool (epsilon-greedy mixes in a uniform
component), honouring the max-2-per-subcategory cap. A logged record is one (request, slot) decision with exact
conditional propensities mu(i | u, earlier slots) and pi(i | u, earlier slots); slot 0 reduces to the plain
item-level Boltzmann propensity P(i|u) = softmax(score/tau) logged by the serving pipeline.

Reward model for DM/DR: q_hat(u,i) = r_click * p_click + r_buy * p_buy from the calibrated primary ranker,
which was trained on *historical* logs independent of the OPE log -- and is biased for cold-start SKUs.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


# ----------------------------------------------------------------------------------------
# OPE log collection (behaviour = Boltzmann sampling of the pure-ranker utility)
# ----------------------------------------------------------------------------------------
def collect_ope_log(pipe, env, frozen_state, bandits: Dict, cfg: Dict, n_requests: int, seed: int = 123,
                    logger=None) -> pd.DataFrame:
    ocfg, sim = cfg["ope"], cfg["simulation"]
    pipe.state = frozen_state.clone()
    pipe.bandits = bandits
    pipe.rng = np.random.default_rng(seed)
    env.rng = np.random.default_rng(seed + 1)
    rng = np.random.default_rng(seed + 2)
    users = rng.integers(0, pipe.world.n_users, size=n_requests)
    rw = cfg["reward"]
    targets = list(pipe.policies.keys())
    rows = []
    for t in range(n_requests):
        u = int(users[t])
        day = sim["start_day"] + t // sim["requests_per_day"]
        out = pipe.recommend(u, "pure_ranker", day=day, sample=True, tau=ocfg["behavior_tau"], request_id=t)
        pack, picks, slate = out["_pack"], out["_picks"], out["slate"]
        items = np.array([s["item_id"] for s in slate])
        res = env.respond(u, items, np.array([s["unseen_subcat"] for s in slate]))
        # sequential (Plackett-Luce) slot-level propensities on the identical candidate pool; each slot is one
        # contextual-bandit decision conditioned on the earlier slots and on the subcategory cap
        true_r = env.expected_reward(np.full(len(pack.items), u), pack.items, pack.unseen_sub)
        q_all = rw["click"] * pack.pc + rw["buy"] * pack.pb
        cap = cfg["guardrails"]["max_per_subcategory"]
        base_pol = pipe.policies["pure_ranker"]
        dist = {"__behavior__": pipe.policy_probs(pipe.score(pack, base_pol)["score"], base_pol, tau=ocfg["behavior_tau"]).cpu().numpy().astype(np.float64)}
        for name in targets:
            pol = pipe.policies[name]
            dist[name] = pipe.policy_probs(pipe.score(pack, pol)["score"], pol, tau=ocfg["target_taus"].get(name, pol.tau)).cpu().numpy().astype(np.float64)
        avail = np.ones(len(pack.items), dtype=bool)
        counts: Dict[int, int] = {}
        for s, rec in enumerate(slate):
            j = picks[s]
            mask = avail.copy()
            for sc_, c_ in counts.items():
                if c_ >= cap:
                    mask &= pack.subcat != sc_
            if not mask[j]:
                mask = avail.copy()
            row = {"request": t, "user": u, "slot": s, "item": rec["item_id"], "reward": float(res["reward"][s]),
                   "click": int(res["click"][s]), "buy": int(res["buy"][s]), "q_hat": float(q_all[j]),
                   "is_cold": int(rec["is_cold_start"]), "tier": rec["novelty_tier_id"]}
            for name, p_ in dist.items():
                w = p_ * mask
                z = w.sum()
                key = "mu" if name == "__behavior__" else name
                cond = w / z
                if name == "__behavior__":
                    row["mu"] = float(cond[j])
                else:
                    row[f"pi_{key}"] = float(cond[j])
                    row[f"dm_{key}"] = float((cond * q_all).sum())
                    row[f"true_{key}"] = float((cond * true_r).sum())
            rows.append(row)
            avail[j] = False
            counts[pack.subcat[j]] = counts.get(pack.subcat[j], 0) + 1
        if logger and (t + 1) % 1000 == 0:
            logger.info("OPE log collection %d/%d", t + 1, n_requests)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------------------
ESTIMATORS = ["DM", "IPS", "SNIPS", "DR", "Switch-DR"]


def _per_record_terms(df: pd.DataFrame, policy: str, clip: float, switch_tau: float) -> Dict[str, np.ndarray]:
    pi, mu = df[f"pi_{policy}"].values, df["mu"].values
    r, q, dm = df["reward"].values, df["q_hat"].values, df[f"dm_{policy}"].values
    w = pi / np.maximum(mu, 1e-12)
    wc = np.minimum(w, clip)
    return {
        "w": w, "wc": wc,
        "DM": dm,
        "IPS": wc * r,
        "SNIPS_num": wc * r, "SNIPS_den": wc,
        "DR": dm + wc * (r - q),
        "Switch-DR": np.where(w <= switch_tau, dm + w * (r - q), dm),
    }


def _combine(sums: Dict[str, np.ndarray], cnt: np.ndarray) -> Dict[str, float]:
    n = cnt.sum()
    return {"DM": sums["DM"].sum() / n, "IPS": sums["IPS"].sum() / n,
            "SNIPS": sums["SNIPS_num"].sum() / max(sums["SNIPS_den"].sum(), 1e-12),
            "DR": sums["DR"].sum() / n, "Switch-DR": sums["Switch-DR"].sum() / n}


def evaluate_policy(df: pd.DataFrame, policy: str, clip: float, switch_tau: float, n_boot: int = 1000,
                    seed: int = 0) -> Dict[str, Dict[str, float]]:
    terms = _per_record_terms(df, policy, clip, switch_tau)
    req = df["request"].values
    uniq, inv = np.unique(req, return_inverse=True)
    n_req = len(uniq)
    cnt = np.bincount(inv, minlength=n_req).astype(np.float64)
    per_req = {k: np.bincount(inv, weights=terms[k], minlength=n_req) for k in
               ("DM", "IPS", "SNIPS_num", "SNIPS_den", "DR", "Switch-DR")}
    point = _combine(per_req, cnt)
    rng = np.random.default_rng(seed)
    boots = {k: np.empty(n_boot) for k in ESTIMATORS}
    for b in range(n_boot):
        idx = rng.integers(0, n_req, size=n_req)
        est = _combine({k: v[idx] for k, v in per_req.items()}, cnt[idx])
        for k in ESTIMATORS:
            boots[k][b] = est[k]
    truth = float(df[f"true_{policy}"].mean())      # record-level truth (matches estimator weighting)
    w, wc = terms["w"], terms["wc"]
    diag = {
        "ess": float(w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-12)),
        "ess_fraction": float(w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-12) / len(w)),
        "ess_clipped": float(wc.sum() ** 2 / np.maximum((wc ** 2).sum(), 1e-12)),
        "mean_weight": float(w.mean()), "max_weight": float(w.max()), "p99_weight": float(np.percentile(w, 99)),
        "clip_threshold": float(clip), "frac_clipped": float((w > clip).mean()),
        "frac_switched_to_dm": float((w > switch_tau).mean()), "n_records": int(len(w)), "n_requests": int(n_req),
    }
    out = {}
    for k in ESTIMATORS:
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        out[k] = {"estimate": float(point[k]), "ci_low": float(lo), "ci_high": float(hi), "true_value": truth,
                  "abs_error": float(abs(point[k] - truth)), "rel_error_pct": float(100 * (point[k] - truth) / truth),
                  "ci_covers_truth": bool(lo <= truth <= hi)}
    out["diagnostics"] = diag
    return out


def run_ope(df: pd.DataFrame, policies: List[str], cfg: Dict, slot0_only: bool = False) -> Dict[str, Dict]:
    o = cfg["ope"]
    d = df[df["slot"] == 0] if slot0_only else df
    return {p: evaluate_policy(d, p, o["clip"], o["switch_threshold"], o["n_bootstrap"], seed=7 + i)
            for i, p in enumerate(policies)}


def ope_table(res: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for p, r in res.items():
        for k in ESTIMATORS:
            rows.append({"policy": p, "estimator": k, **r[k], **{f"diag_{a}": b for a, b in r["diagnostics"].items()}})
    return pd.DataFrame(rows)


def clipping_sweep(df: pd.DataFrame, policy: str, clips=(2, 5, 10, 20, 50, 100), switch_tau: float = 10.0) -> pd.DataFrame:
    rows = []
    truth = float(df[f"true_{policy}"].mean())      # record-level truth (matches estimator weighting)
    for c in clips:
        t = _per_record_terms(df, policy, c, min(switch_tau, c))
        n = len(df)
        rows.append({"policy": policy, "clip": c, "IPS": t["IPS"].sum() / n, "SNIPS": t["SNIPS_num"].sum() / t["SNIPS_den"].sum(),
                     "DR": t["DR"].sum() / n, "Switch-DR": t["Switch-DR"].sum() / n, "true": truth,
                     "frac_clipped": float((t["w"] > c).mean())})
    return pd.DataFrame(rows)
