"""Collect every number shown in the deck into deck_data.json (offline metrics + live flow runs with all rankers).

Run from the project root:  python presentation/prepare_deck_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.common.utils import load_config, resolve
from src.flows import get_flows, run_flow
from src.search_service import SearchService

STAGES = ["A retrieval order", "B + LTR ensemble", "C + hard rules & penalties", "D + margin re-rank", "E + MMR (final page)"]
KEYS = ["ndcg@10", "precision@10", "exp_revenue@10", "exp_margin@10", "unfulfillable@10", "avg_return_prob@10"]


def mean(rs, f):
    return float(np.mean([f(r) for r in rs]))


def main():
    cfg = load_config()
    met = resolve(cfg, "metrics")
    J = lambda n: json.loads((met / n).read_text())
    svc = SearchService.load(cfg)
    svc.world.load_latent()
    flows = get_flows(svc.world)
    for _ in range(2):
        svc.search("running shoes")
    out = {"retrieval": J("retrieval_metrics.json"), "ranking": J("ranking_metrics.json"), "ablation": J("rerank_ablation.json"),
           "latency": J("latency.json"), "multitask": J("multitask_metrics.json"), "funnel": J("clickstream_funnel.json"),
           "importance": J("feature_importance_lambdamart.json"), "summary": J("run_summary.json"),
           "world": {"n_skus": svc.world.n, "n_cold": int(svc.world.cold.sum()), "n_users": len(svc.world.users),
                     "n_queries": len(svc.world.queries)}, "flows": []}
    models = list(svc.models)
    allres = []
    for fl in flows:
        res = run_flow(svc, fl, all_models=True)
        allres += res
        d = {"id": fl["id"], "title": fl["title"], "blurb": fl["blurb"], "n_examples": len(res),
             "stages": {s: {k: mean(res, lambda r: r["variants"][s][k]) for k in KEYS} for s in STAGES},
             "stage1_ms": mean(res, lambda r: r["stage1"]["ms"]), "facet_accuracy": mean(res, lambda r: r["stage1"]["facet_accuracy"]),
             "recall200": {k: mean(res, lambda r: r["stage2"]["retrievers"][k]["recall@200"]) for k in ["sparse", "dense", "hybrid"]},
             "recall50": {k: mean(res, lambda r: r["stage2"]["retrievers"][k]["recall@50"]) for k in ["sparse", "dense", "hybrid"]},
             "latency_total": mean(res, lambda r: r["latency_ms"]["total_ms"]),
             "models": {m: {"B": {k: mean(res, lambda r: r["per_model"][m]["B"][k]) for k in ["ndcg@10", "exp_revenue@10", "exp_margin@10"]},
                            "E": {k: mean(res, lambda r: r["per_model"][m]["E"][k]) for k in ["ndcg@10", "exp_revenue@10", "exp_margin@10"]}} for m in models},
             "examples": [{"label": r["example"]["label"], "query": r["example"]["query"], "user": r["example"]["user_id"],
                           "store": r["example"].get("store_id"), "fulfillment": r["example"].get("fulfillment", "any"),
                           "stage1": {k: r["stage1"][k] for k in ["corrected", "edits", "intent", "intent_conf", "facet_accuracy", "ms"]},
                           "stage2": {"n_rel": r["stage2"]["n_relevant_in_catalog"], "n_found": r["stage2"]["n_relevant_in_candidates"],
                                      "recall200": {k: v["recall@200"] for k, v in r["stage2"]["retrievers"].items()}, "ms": r["stage2"]["ms"]},
                           "rules": r["rules"], "variants": r["variants"], "latency": r["latency_ms"], "final": r["final"][:5],
                           "cold_target": r.get("cold_target")} for r in res]}
        out["flows"].append(d)
    out["all_examples"] = len(allres)
    out["stage_means_all"] = {s: {k: mean(allres, lambda r: r["variants"][s][k]) for k in KEYS} for s in STAGES}
    out["model_means_all"] = {m: {"B": {k: mean(allres, lambda r: r["per_model"][m]["B"][k]) for k in ["ndcg@10", "exp_revenue@10", "exp_margin@10"]},
                                  "E": {k: mean(allres, lambda r: r["per_model"][m]["E"][k]) for k in ["ndcg@10", "exp_revenue@10", "exp_margin@10"]}}
                              for m in models}
    out["models"] = models
    (Path(__file__).parent / "deck_data.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("flows:", [f["id"] for f in out["flows"]], "examples:", len(allres))


if __name__ == "__main__":
    main()
