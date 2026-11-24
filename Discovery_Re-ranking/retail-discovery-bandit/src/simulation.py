"""Online bandit simulation: each policy serves requests against the hidden-ground-truth environment."""
from __future__ import annotations

import copy
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data.synthetic_retail import SimEnvironment
from src.reranker.pipeline import DiscoveryPipeline
from src.utils import StageTimer


def run_policy_simulation(pipe: DiscoveryPipeline, env: SimEnvironment, base_state, bandit_template: Dict,
                          policy: str, n_requests: int, cfg: Dict, seed: int = 0, logger=None
                          ) -> Tuple[pd.DataFrame, Dict]:
    sim = cfg["simulation"]
    world = pipe.world
    pipe.state = base_state.clone()
    pipe.timer = StageTimer()
    pipe.bandits = {k: _clone_bandit(b) for k, b in bandit_template.items()}
    rng = np.random.default_rng(seed)                     # common random numbers across policies
    users = rng.integers(0, world.n_users, size=n_requests)
    env.rng = np.random.default_rng(seed + 1)
    pipe.rng = np.random.default_rng(seed + 2)
    persona = world.users["persona"].values
    rows = []
    for t in range(n_requests):
        u = int(users[t])
        day = sim["start_day"] + t // sim["requests_per_day"]
        out = pipe.recommend(u, policy, day=day, request_id=t)
        slate, picks, ctx = out["slate"], out["_picks"], out["_ctx"]
        items = np.array([s["item_id"] for s in slate])
        unseen = np.array([s["unseen_subcat"] for s in slate])
        res = env.respond(u, items, unseen)
        pipe.log_propensity(out, res)
        for s, rec in enumerate(slate):
            pipe.bandit_update(policy, ctx[picks[s]], rec["category"], float(res["reward"][s]), rec["ranker_q"])
            rows.append({
                "policy": policy, "request": t, "day": day, "user": u, "persona": int(persona[u]), "item": rec["item_id"],
                "slot": rec["slot"], "click": int(res["click"][s]), "buy": int(res["buy"][s]), "repeat": int(res["repeat"][s]),
                "reward": float(res["reward"][s]), "tier": rec["novelty_tier_id"], "is_cold": int(rec["is_cold_start"]),
                "margin": rec["margin"], "price": rec["price"], "category": rec["category"], "subcat": rec["subcat"],
                "ranker_topk": int(rec["ranker_topk"]), "ild": out["ild"], "p_click_true": float(res["p_click"][s]),
                "p_buy_true": float(res["p_buy"][s]), "propensity": rec["propensity_score"],
                "base_utility": rec["base_utility"], "exploration_bonus": rec["exploration_bonus"],
                "learned_residual": rec["learned_residual"], "mmr_penalty": rec["mmr_penalty"],
            })
        pipe.state.record(u, items, res["click"], res["buy"], day)
        if logger and (t + 1) % 1000 == 0:
            r = pd.DataFrame(rows[-5000:])
            logger.info("[%s] %d/%d requests | recent reward/impr=%.4f | cold share=%.3f", policy, t + 1, n_requests,
                        r["reward"].mean(), r["is_cold"].mean())
    if pipe.prop_log:
        pipe.prop_log.flush()
        pipe.inference_log.flush()
        pipe.guard_log.flush()
    return pd.DataFrame(rows), {"latency": pipe.timer.summary(), "bandits": pipe.bandits,
                                "guardrail_totals": dict(pipe.guard.totals)}


def _clone_bandit(b):
    import torch
    nb = copy.copy(b)
    for k, v in list(b.__dict__.items()):
        if isinstance(v, torch.Tensor):
            setattr(nb, k, v.clone())
    return nb
