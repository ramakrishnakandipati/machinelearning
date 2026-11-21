"""Off-policy evaluation (DM / IPS / SNIPS / DR) and online progressive replay."""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import N_ACTIONS


class DMRewardModel:
    """Policy-independent reward model for DM/DR: one gradient-boosted regressor per action (CPU, sklearn)
    -- deliberately a different model family/implementation from the bandits being evaluated."""

    def __init__(self, K=N_ACTIONS, seed=0):
        self.K = K
        self.models = [HistGradientBoostingRegressor(max_depth=5, learning_rate=0.06, max_iter=250,
                                                      l2_regularization=1.0, random_state=seed) for _ in range(K)]

    def fit(self, X, a, r):
        for k, m in enumerate(self.models):
            msk = a == k
            m.fit(X[msk], r[msk])
        return self

    def predict(self, X):
        return np.stack([m.predict(X) for m in self.models], 1)


def _estimates(pi, a, r, p, q, w_clip):
    idx = np.arange(len(a))
    w = np.minimum(pi[idx, a] / p, w_clip)
    dm_terms = (pi * q).sum(1)
    ips_terms = w * r
    dr_terms = dm_terms + w * (r - q[idx, a])
    return dm_terms, ips_terms, dr_terms, w


def evaluate_policy(pi, a, r, p, q, true_r=None, w_clip=20.0, n_boot=200, seed=0):
    """pi: (n,K) target-policy action probabilities on the logged contexts."""
    dm_t, ips_t, dr_t, w = _estimates(pi, a, r, p, q, w_clip)
    est = {"DM": dm_t.mean(), "IPS": ips_t.mean(), "SNIPS": (w * r).sum() / w.sum(), "DR": dr_t.mean()}
    rng = np.random.default_rng(seed)
    n = len(a)
    boots = {k: [] for k in est}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boots["DM"].append(dm_t[i].mean())
        boots["IPS"].append(ips_t[i].mean())
        boots["SNIPS"].append((w[i] * r[i]).sum() / w[i].sum())
        boots["DR"].append(dr_t[i].mean())
    out = {k: float(v) for k, v in est.items()}
    out["ci95"] = {k: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] for k, b in boots.items()}
    out["ess"] = float(w.sum() ** 2 / (w ** 2).sum())
    out["mean_weight"] = float(w.mean())
    out["max_weight"] = float(w.max())
    if true_r is not None:
        out["TRUE"] = float((pi * true_r).sum(1).mean())  # ground truth (simulator only)
    return out


def baseline_policies(true_r, log_p):
    n, K = true_r.shape
    ctrl = np.zeros((n, K))
    ctrl[:, 0] = 1
    orc = np.zeros((n, K))
    orc[np.arange(n), true_r.argmax(1)] = 1
    return {"Logging policy": log_p, "Uniform random": np.full((n, K), 1.0 / K), "Always control": ctrl,
            "Oracle (true best)": orc}


def run_ope(models, X_test, a, r, p, true_r, log_p, dm_model, w_clip=20.0, n_boot=200, audit_path=None, meta=None):
    q = dm_model.predict(X_test)
    results = {}
    for name, pi in baseline_policies(true_r, log_p).items():
        results[name] = {**evaluate_policy(pi, a, r, p, q, true_r, w_clip, n_boot), "type": "baseline"}
    for name, m in models.items():
        pi = m.action_probs(X_test)
        res = evaluate_policy(pi, a, r, p, q, true_r, w_clip, n_boot)
        res["type"] = "bandit"
        res["action_share"] = pi.mean(0).tolist()
        results[name] = res
    if audit_path:
        audit = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_test": int(len(a)), "w_clip": w_clip,
                 "n_bootstrap": n_boot, "dm_reward_model": "HistGradientBoostingRegressor per action",
                 "logged_mean_reward": float(r.mean()), "min_logged_propensity": float(p.min()),
                 "meta": meta or {}, "results": results}
        Path(audit_path).write_text(json.dumps(audit, indent=2))
    return results


def replay(factory, X, a, r, p, true_r, n_rounds=20, seed=0, logger=None):
    """Progressive (prequential) online replay on a shuffled logged stream.

    Each round the *current* policy chooses actions for the new contexts (scored against the simulator's
    ground-truth expected margins -> regret), then learns from the round's logged feedback (a, r, p)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    order = rng.permutation(n)
    chunks = np.array_split(order, n_rounds)
    model = factory(seed)
    margin, regret, picks = [], [], []
    for t, idx in enumerate(chunks):
        acts = model.select_action(X[idx])
        tr = true_r[idx]
        margin.append(tr[np.arange(len(idx)), acts])
        regret.append(tr.max(1) - tr[np.arange(len(idx)), acts])
        picks.append(acts)
        t0 = time.time()
        model.update(X[idx], a[idx], r[idx], p[idx])
        if logger:
            logger({"event": "replay_round", "model": model.name, "round": t + 1,
                    "mean_regret": float(regret[-1].mean()), "mean_margin": float(margin[-1].mean()),
                    "update_seconds": round(time.time() - t0, 3)})
    margin, regret = np.concatenate(margin), np.concatenate(regret)
    ref = {"Logging policy": true_r[order][np.arange(n), a[order]], "Uniform random": true_r[order].mean(1),
           "Always control": true_r[order][:, 0], "Oracle (true best)": true_r[order].max(1)}
    return {"cum_margin": np.cumsum(margin), "cum_regret": np.cumsum(regret),
            "ref_cum_margin": {k: np.cumsum(v) for k, v in ref.items()},
            "ref_cum_regret": {k: np.cumsum(true_r[order].max(1) - v) for k, v in ref.items()},
            "final_model": model}
