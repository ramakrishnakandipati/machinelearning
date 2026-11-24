"""Glue: train all models, (re)load a full serving system, warm-start bandits."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data.state import UserState, novelty_tier_pairs
from src.data.synthetic_retail import RetailWorld
from src.evaluation.metrics import ranker_metrics
from src.models.primary_ranker import FeatureBuilder, PrimaryRanker, cold_start_gap
from src.models.two_tower import (TwoTowerRetriever, benchmark_retrieval_latency, recall_at_k, train_two_tower)
from src.reranker.bandit import load_bandit_states, save_bandits
from src.reranker.pipeline import DiscoveryPipeline
from src.utils import get_device, output_dir, save_json


def day_rows(world: RetailWorld, a: int, b: int):
    df = world.interactions
    return df[(df.day >= a) & (df.day <= b)]


def pos_pairs(df) -> np.ndarray:
    return df[df.click == 1][["user", "item"]].values


def train_all(cfg: Dict, world: RetailWorld, device: torch.device, logger=None, jsonl=None) -> Dict[str, Any]:
    d = cfg["data"]
    fe = d["feature_window"][1]
    # ---------------- Stage 1: two-tower ------------------------------------------------
    tr_pairs = pos_pairs(day_rows(world, d["feature_window"][0], fe))
    va_pairs = pos_pairs(day_rows(world, d["ranker_train_window"][0], d["ranker_train_window"][1]))
    te_pairs = pos_pairs(day_rows(world, d["ranker_calib_window"][0], d["ranker_test_window"][1]))
    cold_pairs = world.cold_eval[world.cold_eval.click == 1][["user", "item"]].values
    retriever = TwoTowerRetriever(world, cfg, device)
    retriever.set_user_history(world, fe)
    hist = train_two_tower(retriever, world, tr_pairs, va_pairs, logger, jsonl)
    ks = tuple(cfg["retrieval"]["seed_eval_k"])
    ret_metrics = {"warm_test": recall_at_k(retriever, te_pairs, ks), "cold_start_test": recall_at_k(retriever, cold_pairs, ks),
                   "latency": benchmark_retrieval_latency(retriever, k=cfg["retrieval"]["top_n"]),
                   "n_train_pairs": int(len(tr_pairs)), "n_test_pairs": int(len(te_pairs)), "n_cold_test_pairs": int(len(cold_pairs))}
    ret_metrics = {"stage1_retrieval": {
        "Recall@50": ret_metrics["warm_test"]["recall@50"], "Recall@200": ret_metrics["warm_test"]["recall@200"],
        "HitRate@200": ret_metrics["warm_test"]["hitrate@200"],
        "ColdStart_Recall@200": ret_metrics["cold_start_test"]["recall@200"],
        "ColdStart_HitRate@200": ret_metrics["cold_start_test"]["hitrate@200"],
        "ColdStart_Recall@50": ret_metrics["cold_start_test"]["recall@50"],
        "latency_ms": ret_metrics["latency"], "n_train_pairs": ret_metrics["n_train_pairs"],
        "n_test_pairs": ret_metrics["n_test_pairs"], "n_cold_test_pairs": ret_metrics["n_cold_test_pairs"]}}
    if logger:
        logger.info("retrieval: %s", ret_metrics["stage1_retrieval"])
    # ---------------- Stage 3: XGBoost ranker --------------------------------------------
    builder = FeatureBuilder(world, retriever, fe)
    tr, ca, te = (day_rows(world, *d["ranker_train_window"]), day_rows(world, *d["ranker_calib_window"]),
                  day_rows(world, *d["ranker_test_window"]))
    Xtr, Xca, Xte = (builder.build(r.user.values, r.item.values) for r in (tr, ca, te))
    ranker = PrimaryRanker(cfg)
    rinfo = ranker.fit(Xtr, tr.click.values, tr.buy.values, Xca, ca.click.values, ca.buy.values, logger)
    pc, pb = ranker.predict(Xte)
    rm = ranker_metrics(te.user.values, te.click.values, te.buy.values, pc, pb)
    gap = cold_start_gap(world, builder, ranker, te)
    rm_out = {"stage3_ranker": rm, "fit": rinfo, "feature_importance": ranker.feature_importance()}
    if logger:
        logger.info("ranker test: click AUC=%.4f PR-AUC=%.4f | buy AUC=%.4f | NDCG@5=%.4f NDCG@10=%.4f",
                    rm["click"]["roc_auc"], rm["click"]["pr_auc"], rm["buy"]["roc_auc"],
                    rm["ranking"]["ndcg@5"], rm["ranking"]["ndcg@10"])
        logger.info("COLD-START BOTTLENECK: mean pred click cold=%.4f vs warm=%.4f | matched-relevance relative gap=%.1f%% | confirmed=%s",
                    gap["mean_pred_click_cold"], gap["mean_pred_click_warm"], 100 * gap["mean_relative_gap_matched"],
                    gap["bottleneck_confirmed"])
    return {"retriever": retriever, "builder": builder, "ranker": ranker, "retrieval_metrics": ret_metrics,
            "ranker_metrics": rm_out, "cold_gap": gap, "retriever_history": hist}


def save_models(cfg: Dict, retriever: TwoTowerRetriever, ranker: PrimaryRanker) -> None:
    mdir = output_dir(cfg, "models")
    torch.save({"state_dict": retriever.state_dict(), "item_embedding_matrix": retriever.item_matrix.cpu(),
                "user_embedding_matrix": retriever.user_matrix.cpu(), "tower_dim": cfg["retrieval"]["tower_dim"]},
               mdir / "two_tower_retriever.pt")
    ranker.save(mdir)


def load_system(cfg: Dict, device: Optional[torch.device] = None, bandit_file: Optional[str] = "linucb_bandit_state.pt",
                log_dir: Optional[Path] = None, seed: int = 0) -> Dict[str, Any]:
    device = device or get_device()
    mdir = output_dir(cfg, "models")
    world = RetailWorld.load(mdir / "world.pkl")
    retriever = TwoTowerRetriever(world, cfg, device)
    ck = torch.load(mdir / "two_tower_retriever.pt", map_location=device, weights_only=False)
    retriever.load_state_dict(ck["state_dict"])
    retriever.eval()
    ranker = PrimaryRanker.load(cfg, mdir)
    builder = FeatureBuilder(world, retriever, cfg["data"]["feature_window"][1])
    state = UserState(world, cfg["simulation"]["start_day"])
    pipe = DiscoveryPipeline(world, retriever, builder, ranker, state, cfg, device, log_dir=log_dir, seed=seed)
    bandits = pipe.new_bandits()
    if bandit_file and (mdir / bandit_file).exists():
        states = load_bandit_states(mdir / bandit_file)
        for k, b in bandits.items():
            if k in states:
                b.load_state_dict(states[k])
    pipe.bandits = bandits
    return {"world": world, "retriever": retriever, "builder": builder, "ranker": ranker, "state": state,
            "pipeline": pipe, "device": device}


@torch.no_grad()
def warm_start_bandits(pipe: DiscoveryPipeline, n_samples: int = 20000, seed: int = 0, logger=None) -> int:
    """Offline warm start: replay logged history impressions (days 80-89) through every bandit.
    Cold-start SKUs have no such rows, so their contexts keep a high exploration bonus."""
    world, cfg = pipe.world, pipe.cfg
    df = world.interactions
    df = df[df.day >= cfg["data"]["ranker_calib_window"][0]]
    df = df.sample(n=min(n_samples, len(df)), random_state=seed)
    u, i = df["user"].values, df["item"].values
    X = pipe.builder.build(u, i)
    pc, pb = pipe.ranker.predict(X)
    tier = novelty_tier_pairs(pipe.state, u, i)
    nov = pipe_novelty = None
    from src.reranker.utility import novelty_from_tier, utility_weights
    nov = novelty_from_tier(tier, cfg)
    w1, w2, w3 = utility_weights(cfg, True)
    u_full = w1 * pc + w2 * pb * pipe.i_margin[i] + w3 * nov
    unseen = ~pipe.state.seen_sub[u, pipe.state.items_subcat[i]]
    ctx = pipe.make_context(u, i, X, pc, pb, tier, unseen, u_full)
    reward = df["click"].values * cfg["reward"]["click"] + df["buy"].values * cfg["reward"]["buy"]
    groups = pipe.i_cat[i]
    target = reward - (cfg['reward']['click'] * pc + cfg['reward']['buy'] * pb) if cfg["bandit"]["residual_target"] else reward
    ctx_t = torch.as_tensor(ctx, device=pipe.dev)
    n_upd = 0
    first = None
    for name, b in pipe.bandits.items():
        if first is not None:
            b.load_state_dict(copy.deepcopy(first.state_dict()))
            continue
        for r in range(len(df)):
            b.update(ctx_t[r], int(groups[r]), float(target[r]))
        n_upd = len(df)
        first = b
    if logger:
        logger.info("bandits warm-started with %d historical impressions", n_upd)
    return n_upd
