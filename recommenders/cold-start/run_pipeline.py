"""End-to-end orchestrator:  python run_pipeline.py [--fast] [--skip-visual] [--config PATH]

Stages: hardware -> generate -> EDA -> preprocess -> classical models -> two-tower -> router -> evaluate ->
coverage sweep -> save artifacts -> latency benchmark -> charts -> Playwright visual verification.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.generator import CUTOFF_DAY, generate_dataset, summarize  # noqa: E402
from src.data.preprocessor import EvalCase, Preprocessor  # noqa: E402
from src.evaluation import visualizer as viz  # noqa: E402
from src.evaluation.metrics import add_cold_start_lift, evaluate_strategy, recall_at_k, catalog_coverage, top_k_from_scores  # noqa: E402
from src.models.content_based import ContentBasedRecommender, ItemIndex  # noqa: E402
from src.models.contextual_bandit import ContextualBandit  # noqa: E402
from src.models.hybrid_two_tower import TwoTowerRecommender  # noqa: E402
from src.models.non_personalized import ContextualPopularity  # noqa: E402
from src.models.router import MetaRouter  # noqa: E402
from src.utils.hardware import get_device, seed_everything, write_diagnostics  # noqa: E402
from src.utils.logger import StageLogger  # noqa: E402


def load_config(path: Path, fast: bool) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if fast:
        cfg["data"].update(n_users=2500, n_items=800, n_interactions=35000)
        cfg["two_tower"].update(epochs=4, batch_size=512)
        cfg["eval"].update(n_eval_users=300, latency_requests=120)
        cfg["bandit"]["alpha"] = cfg["bandit"]["alpha"]
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str((ROOT / v).resolve())
    return cfg


def percentiles(ms: List[float]) -> Dict[str, float]:
    a = np.asarray(ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean()), "n": int(len(a))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--fast", action="store_true", help="small data / few epochs smoke run")
    ap.add_argument("--skip-visual", action="store_true", help="skip Playwright screenshots")
    args = ap.parse_args()
    cfg = load_config(Path(args.config), args.fast)
    P = {k: Path(v) for k, v in cfg["paths"].items()}
    for p in P.values():
        p.mkdir(parents=True, exist_ok=True)
    (P["logs"] / "pipeline_stages.jsonl").unlink(missing_ok=True)
    seed_everything(cfg["seed"])
    sl = StageLogger(P["logs"])
    sl.info(f"Cold-start pipeline (fast={args.fast}) -> {P['outputs']}")
    K = int(cfg["eval"]["k"])
    device = get_device(cfg["hardware"]["use_gpu"])
    extra: Dict[str, Any] = {"fast": args.fast}

    with sl.stage("hardware_diagnostics") as rec:
        hw = write_diagnostics(P["logs"] / "hardware_diagnostics.json")
        rec.update(device=str(device), gpu=[g["name"] for g in hw["gpus"]], faiss=hw["faiss_mode"])
        sl.info(f"  torch {hw['torch_version']} cuda={hw['cuda_available']} faiss={hw['faiss_mode']}")

    with sl.stage("generate_data") as rec:
        ds = generate_dataset(cfg, cfg["seed"])
        rec.update(summarize(ds))
        (P["metrics"] / "dataset_summary.json").write_text(json.dumps(summarize(ds), indent=2), encoding="utf-8")
        sl.info(f"  {summarize(ds)}")

    with sl.stage("eda_charts"):
        viz.plot_cold_start_distribution(ds.interactions, ds.users, ds.items, P["charts"] / "cold_start_distribution.png")

    with sl.stage("preprocess") as rec:
        pre = Preprocessor(cfg).fit(ds)
        train, val = pre.build_rows(ds, ["train"]), pre.build_rows(ds, ["val"])
        cases = pre.build_eval_cases(ds, int(cfg["eval"]["n_eval_users"]), cfg["seed"])
        rec.update(embedding_backend=pre.embedder.backend, content_dim=pre.content_dim, train_rows=len(train["item_pos"]),
                   val_rows=len(val["item_pos"]), cases={k: len(v) for k, v in cases.items()})
        sl.info(f"  embeddings={pre.embedder.backend} content_dim={pre.content_dim} cases={rec['cases']}")

    with sl.stage("fit_popularity_content_bandit") as rec:
        tv = ds.interactions[ds.interactions.split.isin(["train", "val"])]
        pop = ContextualPopularity(cfg).fit(tv, ds.users, pre.item_pos, now_day=CUTOFF_DAY)
        index = ItemIndex(pre.content_dim, cfg["hardware"]["faiss_gpu"])
        index.build(pre.item_content)
        content = ContentBasedRecommender(index)
        bandit = ContextualBandit(pre, cfg, cfg["seed"])
        subidx = np.array([bandit.arm_idx.get(s, 0) for s in pre.item_subcat])
        n_rep = bandit.fit_replay(tv, ds.users, subidx)
        rec.update(faiss_backend=index.backend, bandit_replay_events=n_rep, popularity_segments=len(pop.seg_share))
        sl.info(f"  faiss={index.backend} segments={len(pop.seg_share)} bandit replay={n_rep}")

    with sl.stage("train_two_tower") as rec:
        tower = TwoTowerRecommender(pre, cfg, device)
        hist = tower.fit(train, val, log=sl.info)
        tower.save(P["models"])
        pd.DataFrame(hist).to_csv(P["metrics"] / "two_tower_training_history.csv", index=False)
        rec.update(epochs=len(hist), best_val_recall=max(h["val_recall_at_10"] for h in hist), amp=tower.amp_used)
        viz.plot_training_curve(hist, P["charts"] / "two_tower_training_curve.png")

    pos_pairs = tv[tv.positive == 1].item_id.map(pre.item_pos)
    sales = np.bincount(pos_pairs.values, minlength=len(pre.item_ids)).astype(float)
    router = MetaRouter(cfg, pre, ds.items, pop, index, tower, bandit, sales)

    with sl.stage("evaluate_strategies") as rec:
        n_items = len(pre.item_ids)
        rng = np.random.default_rng(cfg["seed"])
        pop_counts = sales.copy()
        pop_top = set(np.argsort(-pop_counts)[:50].tolist())
        n_cands = {"warm": int((~pre.is_cold_item).sum()), "cold_user": n_items,
                   "cold_item": int(pre.is_cold_item.sum()), "session": n_items}

        def pctx(c: EvalCase) -> Dict[str, Any]:
            return dict(region=c.user_row.get("region"), device=c.user_row.get("device"),
                        acq_channel=c.user_row.get("acq_channel"), season=c.ctx["season"], hour=c.ctx["hour"],
                        channel=c.ctx["channel"])

        def rank(scores: np.ndarray, c: EvalCase) -> List[int]:
            return top_k_from_scores(scores + rng.random(len(scores)) * 1e-9, c.candidates, c.exclude, K)

        strategies = {
            "Random": lambda c: rank(rng.random(n_items), c),
            "Popularity": lambda c: rank(pop.score_all(pctx(c))[:n_items], c),
            "Content-Based": lambda c: rank(content.score_all(pre.profile_vec(c.history, c.user_row.get("onboarding_categories")),
                                                              n_items), c),
            "Two-Tower (ID Dropout)": lambda c: rank(tower.score_all(c.user_row, c.ctx, c.history), c),
            "Hybrid Meta-Router": lambda c: [i.position for i in router.recommend(
                c.user_row, c.ctx, c.history, k=K, exclude=c.exclude, candidates=c.candidates, explore=False).items],
        }
        rows: List[Dict[str, Any]] = []
        for name, fn in strategies.items():
            t = time.perf_counter()
            rows += evaluate_strategy(name, fn, cases, K, pop_counts, pre.is_cold_item, n_cands, pop_top)
            sl.info(f"  {name:<24} {time.perf_counter() - t:5.1f}s")
        df = add_cold_start_lift(pd.DataFrame(rows), f"recall@{K}")
        df.to_csv(P["metrics"] / "cold_start_eval_metrics.csv", index=False)
        (P["metrics"] / "cold_start_eval_metrics.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
        viz.plot_strategy_comparison(df, P["charts"] / "strategy_comparison_benchmark.png", f"recall@{K}")
        piv = df.pivot(index="strategy", columns="tier", values=f"recall@{K}").round(4)
        sl.info("\n" + piv.to_string())
        rec["recall_table"] = piv.to_dict()

    with sl.stage("coverage_novelty_sweep"):
        pos = ds.interactions[(ds.interactions.positive == 1) & (ds.interactions.split == "test_cold_item")]
        cold_t = pos.groupby("user_id")["item_id"].apply(lambda s: {pre.item_pos[i] for i in s}).to_dict()
        mixed = []
        for c in cases["warm"]:
            c2 = EvalCase(c.split, c.user_id, c.user_row, c.ctx, c.history, sorted(set(c.targets) | cold_t.get(c.user_id, set())),
                          None, c.exclude)
            mixed.append(c2)
        pts = []
        for slots in range(0, 6):
            router.new_slots = slots
            recs = [[i.position for i in router.recommend(c.user_row, c.ctx, c.history, k=K, exclude=c.exclude,
                                                          include_new_items_boost=slots > 0, explore=False).items] for c in mixed]
            cov = catalog_coverage([[i for i in r if pre.is_cold_item[i]] for r in recs], int(pre.is_cold_item.sum()))
            pts.append(dict(label=f"{slots} slots", slots=slots, recall=float(np.mean([recall_at_k(r, set(c.targets), K)
                                                                                    for r, c in zip(recs, mixed)])),
                            cold_item_coverage=cov))
        router.new_slots = int(cfg["router"]["new_item_slots"])
        pdf = pd.DataFrame(pts)
        pdf.to_csv(P["metrics"] / "coverage_vs_recall.csv", index=False)
        viz.plot_coverage_novelty(pdf, P["charts"] / "catalog_coverage_novelty.png")
        sl.info("\n" + pdf.round(4).to_string(index=False))

    with sl.stage("save_artifacts"):
        # a fresh index/preprocessor snapshot (router may have mutated nothing yet, but be explicit)
        pre.save(P["models"])
        index.save(P["models"] / "item_index.faiss")
        bandit.save(P["models"])
        hist_by_user = (tv[tv.positive == 1].sort_values("ts").groupby("user_id")["item_id"]
                        .apply(lambda s: [pre.item_pos[i] for i in s]).to_dict())
        users = {r["user_id"]: r for r in ds.users.drop(columns=["user_type"]).to_dict("records")}
        with open(P["models"] / "serving_state.pkl", "wb") as f:
            pickle.dump(dict(popularity=pop, users=users, history=hist_by_user, items_df=ds.items, sales=sales), f)
        (P["models"] / "config_used.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        (P["models"] / "scalers.json").write_text(json.dumps(dict(price_mu=pre.price_mu, price_sd=pre.price_sd)), encoding="utf-8")
        ds.items.to_csv(P["models"] / "items_catalog.csv", index=False)
        sl.info(f"  saved: {sorted(p.name for p in P['models'].iterdir())}")

    with sl.stage("latency_benchmark") as rec:
        from fastapi.testclient import TestClient

        os.environ["COLDSTART_CONFIG"] = str(P["models"] / "config_used.yaml")
        os.environ["COLDSTART_MODELS"] = str(P["models"])
        from src.serving.app import app

        n_req = int(cfg["eval"]["latency_requests"])
        r = np.random.default_rng(cfg["seed"] + 1)
        warm_u = [u for u in ds.users[ds.users.user_type == "warm"].user_id if u in hist_by_user]
        cold_u = ds.users[ds.users.user_type == "cold"].to_dict("records")
        payloads = []
        for i in range(n_req):
            m = i % 4
            if m == 0:
                payloads.append(dict(user_id=str(r.choice(warm_u)), k=K))
            elif m == 1:
                u = cold_u[int(r.integers(len(cold_u)))]
                payloads.append(dict(user_context=dict(region=u["region"], device=u["device"], acquisition_channel=u["acq_channel"],
                                                       onboarding_categories=u["onboarding_categories"], hour=int(r.integers(24))), k=K,
                                     include_new_items_boost=True))
            elif m == 2:
                payloads.append(dict(session_item_ids=[str(x) for x in r.choice(pre.item_ids[: len(ds.items)], 3)],
                                     user_context=dict(hour=int(r.integers(24))), k=K))
            else:
                payloads.append(dict(user_context=dict(region=str(r.choice(["north", "south", "east", "west"])),
                                                       hour=int(r.integers(24))), k=K))
        by_scenario = {s: [] for s in ("warm_user", "cold_user_onboarding", "session_clickstream", "anonymous")}
        names = list(by_scenario)
        with TestClient(app) as client:
            for p in payloads[:10]:
                client.post("/recommend", json=p)  # warm-up
            t_all = time.perf_counter()
            for i, p in enumerate(payloads):
                t = time.perf_counter()
                resp = client.post("/recommend", json=p)
                by_scenario[names[i % 4]].append((time.perf_counter() - t) * 1000)
                resp.raise_for_status()
            total = time.perf_counter() - t_all
            cached_ms = []
            for p in payloads[:30]:
                t = time.perf_counter(); client.post("/recommend", json=p); cached_ms.append((time.perf_counter() - t) * 1000)
        # raw FAISS retrieval latency
        q = pre.item_content[r.integers(0, len(pre.item_ids), 500)]
        faiss_ms = []
        for v in q:
            t = time.perf_counter(); index.search(v, 10); faiss_ms.append((time.perf_counter() - t) * 1000)
        allms = [x for v in by_scenario.values() for x in v]
        out = dict(note="In-process ASGI (TestClient) latency; includes routing, models, serialization",
                   device=str(device), overall=percentiles(allms), throughput_rps=round(len(payloads) / total, 2),
                   by_scenario={k: percentiles(v) for k, v in by_scenario.items()}, cache_hit=percentiles(cached_ms),
                   faiss_search_top10=percentiles(faiss_ms), faiss_backend=index.backend, faiss_p99_under_10ms=bool(np.percentile(faiss_ms, 99) < 10))
        (P["metrics"] / "inference_latency_benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        rec.update(p50=out["overall"]["p50_ms"], p99=out["overall"]["p99_ms"], rps=out["throughput_rps"])
        sl.info(f"  /recommend p50={out['overall']['p50_ms']:.1f}ms p95={out['overall']['p95_ms']:.1f}ms "
                f"p99={out['overall']['p99_ms']:.1f}ms rps={out['throughput_rps']} | faiss p99={out['faiss_search_top10']['p99_ms']:.3f}ms")

    if not args.skip_visual:
        try:
            with sl.stage("playwright_visual_verification") as rec:
                r = subprocess.run([sys.executable, str(ROOT / "tests" / "visual_e2e_test.py")], cwd=ROOT,
                                   capture_output=True, text=True, timeout=300)
                sl.info(r.stdout[-1500:])
                if r.returncode != 0:
                    raise RuntimeError((r.stderr or r.stdout)[-800:])
                rec["screenshots"] = sorted(p.name for p in P["screenshots"].glob("*.png"))
        except Exception as exc:
            extra["visual_error"] = repr(exc)
            sl.info(f"  visual verification failed (non-fatal): {exc}")
    summ = sl.write_summary(extra)
    sl.info(f"Pipeline finished in {summ['total_seconds']}s  all_ok={summ['all_ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
