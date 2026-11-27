"""End-to-end orchestrator: data gen -> encode/index -> tune hybrid -> LTR lists -> train -> evaluate -> artifacts."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.common.utils import load_config, percentiles, resolve, save_json, setup_logging
from src.common.world import World
from src.data_gen import generate_retail_data
from src.engine import SearchEngine
from src.evaluation.offline_metrics import eval_ranked, eval_scores, retrieval_recall
from src.ranking.dataset import build_lists
from src.ranking.feature_engineering import FEATURE_NAMES
from src.ranking.ltr_models import save_models, train_all
from src.ranking.multi_task_scorer import MultiTaskScorer
from src.reranking.business_rules import Constraints
from src.search_service import SearchService, rerank


def main(args):
    cfg = load_config()
    if args.small:
        cfg["ranking"].update(n_train_lists=500, n_val_lists=120, n_test_lists=150)
        cfg["ranking"]["gbdt"]["n_estimators"] = 120
        cfg["ranking"]["dcn"]["epochs"] = 6
    setup_logging(cfg, "pipeline")
    data_dir, mdir, met = resolve(cfg, "data"), resolve(cfg, "models"), resolve(cfg, "metrics")
    T0 = time.time()
    # 1 ------------------------------------------------------------------------------------ data
    if args.regen or not (data_dir / "catalog.parquet").exists() or args.small:
        generate_retail_data.main(cfg if not args.small else None, small=args.small)
    world = World(cfg, load_latent=True)
    sessions = pd.read_parquet(data_dir / "sessions.parquet")
    logger.info("World: {} SKUs ({} cold), {} users, {} queries, {} impressions", world.n, int(world.cold.sum()),
                len(world.users), len(world.queries), len(sessions))
    funnel = {"impressions": int(len(sessions)), "clicks": int(sessions.click.sum()), "add_to_cart": int(sessions.add_to_cart.sum()),
              "purchases": int(sessions.purchase.sum()), "returns": int(sessions.returned.sum())}
    save_json(funnel, met / "clickstream_funnel.json")
    # 2 ------------------------------------------------------------------------------ engine/index
    engine = SearchEngine.build(cfg, world, sessions)
    engine.warmup()
    del sessions
    # 3 ---------------------------------------------------------------- retrieval tuning + compare
    val_q = world.queries[world.queries.split == "val"].head(250)
    test_q = world.queries[world.queries.split == "test"].head(250)
    grid = {}
    for a in [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]:
        grid[a] = retrieval_recall(engine, val_q, "alpha", alpha=a, n=50)["recall@50"]
    best_alpha = max(grid, key=grid.get)
    cfg["retrieval"]["alpha"] = best_alpha
    engine.cfg = cfg
    engine.rc = cfg["retrieval"]
    logger.info("alpha grid (val recall@50): {}  -> alpha={}", {k: round(v, 4) for k, v in grid.items()}, best_alpha)
    save_json({"alpha": best_alpha, "grid_val_recall@50": grid}, mdir / "tuned.json")
    ret_res = {}
    for name, mode in [("sparse_only (BM25+char+learned)", "sparse"), ("dense_only (bge+multimodal)", "dense"),
                       ("hybrid_rrf", "rrf"), (f"hybrid_alpha={best_alpha}", "alpha")]:
        r = retrieval_recall(engine, test_q, mode, alpha=best_alpha, n=200)
        r50 = retrieval_recall(engine, test_q, mode, alpha=best_alpha, n=50)
        ret_res[name] = {**r, "recall@50": r50["recall@50"], "cold_start_recall@50": r50["cold_start_recall"]}
        logger.info("  {:<34} R@200={:.4f} R@50={:.4f} cold R@200={:.4f} hit={:.3f} p95={:.1f}ms", name, r["recall@200"],
                    r50["recall@50"], r["cold_start_recall"], r["hit_rate"], r["latency_ms"]["p95"])
    save_json(ret_res, met / "retrieval_metrics.json")
    pd.DataFrame({k: {m: v for m, v in d.items() if not isinstance(v, dict)} for k, d in ret_res.items()}).T.to_csv(met / "retrieval_metrics.csv")
    # 4 ------------------------------------------------------------------------------- LTR lists
    rk = cfg["ranking"]
    L = rk["labeling_impressions"]
    logger.info("Building LTR lists (train/val/test)")
    tr = build_lists(engine, "train", rk["n_train_lists"], L, cfg["seed"] + 1)
    va = build_lists(engine, "val", rk["n_val_lists"], L, cfg["seed"] + 2)
    te = build_lists(engine, "test", rk["n_test_lists"], L, cfg["seed"] + 3)
    logger.info("lists: train {} rows, val {}, test {}", len(tr["X"]), len(va["X"]), len(te["X"]))
    # 5 ------------------------------------------------------------------------------- train + eval
    logger.info("Training LTR models")
    models, times = train_all(tr, va, cfg)
    save_models(models, mdir / "ltr_models.joblib")
    mtl = MultiTaskScorer().fit(tr, cfg)
    joblib.dump(mtl, mdir / "multitask.joblib")
    rows, per_model = [], {}
    fused_col = FEATURE_NAMES.index("x_fused")
    base = eval_scores(te, world, te["X"][:, fused_col])
    per_model["retrieval_order (no LTR)"] = {"paradigm": "baseline", **base, "train_s": 0.0}
    for name, m in models.items():
        t0 = time.perf_counter()
        sc = m.predict(te["X"])
        inf_ms = (time.perf_counter() - t0) * 1e3 / len(te["sizes"])
        res = eval_scores(te, world, sc)
        per_model[name] = {"paradigm": getattr(m, "paradigm", ""), **res, "train_s": times.get(name, 0.0), "infer_ms_per_list": inf_ms}
        logger.info("  {:<18} NDCG@10(rel)={:.4f} NDCG@10(eng)={:.4f} Rev@10={:.2f} Margin@10={:.2f}", name,
                    res["ndcg_rel@10"], res["ndcg_eng@10"], res["expected_revenue@10"], res["expected_margin@10"])
    save_json(per_model, met / "ranking_metrics.json")
    pd.DataFrame(per_model).T.to_csv(met / "ranking_metrics.csv")
    # multi-task head quality (AUC vs latent-sampled labels)
    from sklearn.metrics import roc_auc_score

    heads = mtl.predict(te["X"])
    mt = {"pCTR_auc": float(roc_auc_score(te["n_click"] > 0, heads["pctr"])),
          "pATC_auc(|click)": float(roc_auc_score(te["n_atc"][te["n_click"] > 0] > 0, heads["patc"][te["n_click"] > 0])),
          "pCVR_auc(|cart)": float(roc_auc_score(te["n_pur"][te["n_atc"] > 0] > 0, heads["pcvr"][te["n_atc"] > 0])) if (te["n_atc"] > 0).sum() > 30 else None,
          "pReturn_corr_with_latent": float(np.corrcoef(heads["pret"], te["p_ret"])[0, 1])}
    save_json(mt, met / "multitask_metrics.json")
    logger.info("multi-task: {}", mt)
    # 6 -------------------------------------------------------------------- re-ranking ablation
    best_name = "ensemble"
    sc = models[best_name].predict(te["X"])
    util = mtl.utility(heads, world.price[te["idx"]], world.margin_usd[te["idx"]])
    off = te["offsets"]
    variants = {
        "1. ranker only (no rules)": dict(apply_rules=False, margin_opt=False, diversify=False),
        "2. + hard filters & soft penalties": dict(apply_rules=True, margin_opt=False, diversify=False),
        "3. + margin-aware re-rank": dict(apply_rules=True, margin_opt=True, diversify=False),
        "4. + MMR diversity (full funnel)": dict(apply_rules=True, margin_opt=True, diversify=True),
    }
    abl = {}
    for vname, kw in variants.items():
        ranked = []
        for i in range(len(off) - 1):
            s, e = off[i], off[i + 1]
            u = int(te["uidx"][i])
            cons = Constraints(store_id=world.user_store[u])
            r = rerank(world, cfg, engine.doc_emb, te["idx"][s:e], sc[s:e], util["expected_profit"][s:e], u, cons,
                       float(te["pmax"][i]), 20, brand_intent=bool(te["pbrand"][i]), **kw)
            ranked.append(r.order)
        abl[vname] = eval_ranked(te, world, ranked, 10)
        logger.info("  {:<40} NDCG={:.4f} Rev={:.2f} Margin={:.2f} OOS={:.3f} ret={:.3f}", vname, abl[vname]["ndcg_rel@10"],
                    abl[vname]["expected_revenue@10"], abl[vname]["expected_margin@10"], abl[vname]["oos_rate@10"],
                    abl[vname]["avg_return_prob@10"])
    save_json(abl, met / "rerank_ablation.json")
    pd.DataFrame(abl).T.to_csv(met / "rerank_ablation.csv")
    # 7 ------------------------------------------------------------------------ latency benchmark
    svc = SearchService(cfg, world, engine, models, mtl)
    qs = world.queries[world.queries.split == "test"].head(200)
    for q in qs.head(10).itertuples():  # warm-up
        svc.search(q.typo_text, user_id=world.users.user_id.values[0])
    lat = {k: [] for k in ["query_understanding_ms", "retrieval_ms", "feature_ms", "ranking_ms", "rerank_ms", "total_ms"]}
    for i, q in enumerate(qs.itertuples()):
        out = svc.search(q.typo_text, user_id=world.users.user_id.values[i % len(world.users)])
        for k in lat:
            lat[k].append(out["latency_ms"][k])
    lat_p = {k: percentiles(v) for k, v in lat.items()}
    lat_p["budget_ms"] = cfg["latency_budget_ms"]
    save_json(lat_p, met / "latency.json")
    logger.info("latency p50/p95/p99 total: {:.1f}/{:.1f}/{:.1f} ms", lat_p["total_ms"]["p50"], lat_p["total_ms"]["p95"], lat_p["total_ms"]["p99"])
    # summary + feature importance
    try:
        imp = models["lgbm_lambdamart"].m.booster_.feature_importance("gain")
        fi = sorted(zip(FEATURE_NAMES, imp / imp.sum()), key=lambda x: -x[1])[:20]
        save_json({k: float(v) for k, v in fi}, met / "feature_importance_lambdamart.json")
    except Exception as e:  # pragma: no cover
        logger.warning("feature importance failed: {}", e)
    save_json({"elapsed_s": time.time() - T0, "small": bool(args.small), "alpha": best_alpha,
               "n_skus": world.n, "n_cold_start": int(world.cold.sum())}, met / "run_summary.json")
    logger.info("Pipeline finished in {:.1f}s", time.time() - T0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="fast smoke run (small data)")
    ap.add_argument("--regen", action="store_true", help="regenerate data even if present")
    main(ap.parse_args())
