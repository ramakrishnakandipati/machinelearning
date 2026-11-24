"""Train the retriever + ranker on GPU, warm-start the bandits, run the online simulation for all policies,
and persist models, structured logs, and metric reports under output/.

    python scripts/train_and_simulate.py [--quick] [--requests N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.data.state import UserState
from src.data.synthetic_retail import SimEnvironment, generate_world
from src.evaluation.metrics import slate_metrics
from src.reranker.bandit import save_bandits
from src.reranker.pipeline import DiscoveryPipeline
from src.simulation import run_policy_simulation
from src.system import save_models, train_all, warm_start_bandits
from src.utils import (JsonlLogger, get_device, get_logger, gpu_info, load_config, output_dir, save_json, set_seed)


def data_profile(world, cfg) -> None:
    out = output_dir(cfg, "data_profiles")
    it, us, df = world.items, world.users, world.interactions
    prof = {
        "n_items": int(len(it)), "n_users": int(len(us)), "n_categories": int(it.category.nunique()),
        "n_subcategories": int(it.subcat.nunique()), "n_brands": int(it.brand.nunique()),
        "cold_start_items": int(it.is_cold.sum()), "cold_start_share": float(it.is_cold.mean()),
        "cold_start_items_with_history": int(df["item"].isin(it.item_id[it.is_cold]).sum()),
        "out_of_stock_share": float(1 - it.in_stock.mean()),
        "n_impressions": int(len(df)), "ctr": float(df.click.mean()), "buy_rate": float(df.buy.mean()),
        "cold_eval_impressions": int(len(world.cold_eval)), "cold_eval_ctr": float(world.cold_eval.click.mean()),
        "price": it.price.describe().to_dict(), "margin": it.margin.describe().to_dict(),
        "days_since_launch": it.days_since_launch.describe().to_dict(),
    }
    save_json(prof, out / "dataset_profile.json")
    it.groupby("category_name").agg(n_items=("item_id", "size"), mean_price=("price", "mean"), mean_margin=("margin", "mean"),
                                    cold_share=("is_cold", "mean")).to_csv(out / "category_stats.csv")
    us.groupby("persona_name").agg(n_users=("user_id", "size"), avg_item_price=("avg_item_price", "mean"),
                                   avg_basket_value=("avg_basket_value", "mean"), buys_90d=("n_buys_90d", "mean")).to_csv(out / "persona_stats.csv")
    d = df.assign(persona=us.persona_name.values[df["user"].values])
    d.groupby("persona").agg(impressions=("click", "size"), ctr=("click", "mean"), buy_rate=("buy", "mean")).to_csv(out / "persona_interaction_stats.csv")
    df.groupby("day").agg(impressions=("click", "size"), ctr=("click", "mean"), buy_rate=("buy", "mean")).to_csv(out / "daily_stats.csv")
    df.groupby("item").size().rename("impressions").to_csv(out / "item_impression_counts.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small simulation for smoke tests")
    ap.add_argument("--requests", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg["simulation"]["n_requests_per_policy"] = 300
        cfg["simulation"]["warm_start_samples"] = 3000
    if args.requests:
        cfg["simulation"]["n_requests_per_policy"] = args.requests
    log_dir = output_dir(cfg, "logs")
    mdir = output_dir(cfg, "models")
    metrics_dir = output_dir(cfg, "metrics")
    logger = get_logger("train", log_dir / "training.log")
    train_jsonl = JsonlLogger(log_dir / "training_log.jsonl", truncate=True)
    for f in ("propensity_log.jsonl", "inference_log.jsonl", "guardrail_log.jsonl"):
        (log_dir / f).unlink(missing_ok=True)
    set_seed(cfg["seed"])
    device = get_device()
    logger.info("device=%s | %s", device, gpu_info())
    t0 = time.time()

    # ------------------------------------------------------------------ data
    world = generate_world(cfg, device, logger)
    world.save(mdir / "world.pkl")
    data_profile(world, cfg)

    # ------------------------------------------------------------------ models
    art = train_all(cfg, world, device, logger, train_jsonl)
    retriever, builder, ranker = art["retriever"], art["builder"], art["ranker"]
    save_models(cfg, retriever, ranker)
    save_json(art["retrieval_metrics"], metrics_dir / "stage1_retrieval_metrics.json")
    save_json(art["ranker_metrics"], metrics_dir / "stage3_ranker_metrics.json")
    save_json(art["cold_gap"], metrics_dir / "cold_start_bottleneck.json")
    pd.DataFrame(art["retriever_history"]).to_csv(metrics_dir / "two_tower_training_curve.csv", index=False)
    train_jsonl.log({"stage": "cold_start_bottleneck", **{k: v for k, v in art["cold_gap"].items() if k != "matched_relevance_bins"}})
    train_jsonl.flush()

    # ------------------------------------------------------------------ pipeline + bandits
    state0 = UserState(world, cfg["simulation"]["start_day"])
    pipe = DiscoveryPipeline(world, retriever, builder, ranker, state0, cfg, device, log_dir=log_dir, seed=cfg["seed"])
    pipe.bandits = pipe.new_bandits()
    warm_start_bandits(pipe, cfg["simulation"]["warm_start_samples"], cfg["seed"], logger)
    template = {k: b for k, b in pipe.bandits.items()}
    env = SimEnvironment(world, cfg["seed"])

    # ------------------------------------------------------------------ online simulation
    n_req = cfg["simulation"]["n_requests_per_policy"]
    all_rec, summaries, latencies, final_bandits, guard_totals = [], {}, [], {}, {}
    for name in pipe.policies:
        t1 = time.time()
        logger.info("=== simulating policy %s (%d requests) ===", name, n_req)
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        rec, extra = run_policy_simulation(pipe, env, state0, template, name, n_req, cfg, seed=cfg["seed"], logger=logger)
        rec.to_parquet(metrics_dir / f"sim_records_{name}.parquet", index=False)
        m = slate_metrics(rec, world.n_items, int(world.items.in_stock.sum()))
        m["wall_seconds"] = time.time() - t1
        summaries[name] = m
        for stage, v in extra["latency"].items():
            latencies.append({"policy": name, "stage": stage, **v})
        guard_totals[name] = extra["guardrail_totals"]
        for k, b in extra["bandits"].items():
            if k == name:
                final_bandits[k] = b
        all_rec.append(rec)
        logger.info("[%s] CTR=%.4f buy/impr=%.4f reward/impr=%.4f cold share=%.3f cold conv=%.4f discovery conv=%.4f coverage=%.2f%% ILD=%.3f",
                    name, m["ctr"], m["buy_rate_per_impression"], m["avg_reward_per_impression"], m["cold_start_impression_share"],
                    m["cold_start_conversion_rate"], m["discovery_conversion_rate"], m["catalog_coverage_pct"], m["ild"])
        train_jsonl.log({"stage": "simulation", "policy": name, **m})
    train_jsonl.flush()

    # ------------------------------------------------------------------ persist
    for k in template:                                  # keep warm-started template for policies without final state
        final_bandits.setdefault(k, template[k])
    save_bandits(final_bandits, mdir / "linucb_bandit_state.pt")
    pd.concat(all_rec).to_parquet(metrics_dir / "sim_records_all.parquet", index=False)
    save_json(summaries, metrics_dir / "stage4_policy_summary.json")
    pd.DataFrame(summaries).T.to_csv(metrics_dir / "stage4_policy_summary.csv")
    pd.DataFrame(latencies).to_csv(metrics_dir / "latency_by_stage_policy.csv", index=False)
    save_json({"filter_drop_counts_all_requests": pipe.drop_totals, "guardrail_totals_by_policy": guard_totals},
              metrics_dir / "filter_and_guardrail_drop_counts.json")
    save_json({"bandit_inverse_max_error": {k: b.max_inverse_error() for k, b in final_bandits.items()},
               "bandit_updates": {k: b.n_updates for k, b in final_bandits.items()}}, metrics_dir / "bandit_diagnostics.json")
    gi = gpu_info()
    save_json(gi, metrics_dir / "gpu_utilization.json")
    logger.info("GPU: %s", gi)
    logger.info("done in %.1fs. Models in %s", time.time() - t0, mdir)


if __name__ == "__main__":
    main()
