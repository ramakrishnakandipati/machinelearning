"""Run every end-to-end flow example, save metrics (json + markdown) and refresh the README flow section."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.common.utils import ROOT, load_config, resolve, save_json
from src.flows import get_flows, run_flow
from src.search_service import SearchService

START, END = "<!-- FLOWS:START -->", "<!-- FLOWS:END -->"


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def to_markdown(flows, results):
    L = ["### End-to-end flows (per-stage metrics against simulator ground truth)", "",
         "Each example runs the real funnel. Stage variants share the same 200 hybrid candidates: "
         "**A** retrieval order, **B** + LTR ensemble, **C** + hard rules & penalties, **D** + margin re-rank, **E** + MMR (final page). "
         "Metrics are @10; revenue/margin are expected values per page from the latent model.", ""]
    for fl in flows:
        L += [f"#### {fl['title']}", "", fl["blurb"], "",
              "| Example | Stage 1: corrected / intent / facet acc. | Stage 2: relevant in 200 (R@200 sparse/dense/hybrid) | NDCG A→B→E | Margin A→E | Unfulfillable A→E | ms total |",
              "|---|---|---|---|---|---|---|"]
        for r in results[fl["id"]]:
            s1, s2, v = r["stage1"], r["stage2"], r["variants"]
            A, B, E = v["A retrieval order"], v["B + LTR ensemble"], v["E + MMR (final page)"]
            rt = s2["retrievers"]
            L.append(f"| {r['example']['label']} | `{s1['corrected']}` / {s1['intent']} ({s1['intent_conf']:.0%}) / {s1['facet_accuracy']:.0%} | "
                     f"{s2['n_relevant_in_candidates']}/{min(s2['n_relevant_in_catalog'], 200)} ({f(rt['sparse']['recall@200'],2)}/{f(rt['dense']['recall@200'],2)}/{f(rt['hybrid']['recall@200'],2)}) | "
                     f"{A['ndcg@10']:.3f}→{B['ndcg@10']:.3f}→{E['ndcg@10']:.3f} | {A['exp_margin@10']:.1f}→{E['exp_margin@10']:.1f} | "
                     f"{A['unfulfillable@10']}→{E['unfulfillable@10']} | {r['latency_ms']['total_ms']:.0f} |")
        L.append("")
        if fl["id"] == "cold_start":
            L += ["| Cold SKU | impressions | rank sparse | rank dense | rank hybrid | rank in final page |", "|---|---|---|---|---|---|"]
            for r in results[fl["id"]]:
                c = r["cold_target"]
                L.append(f"| {c['title'][:60]} | {c['impressions']} | {c['rank_sparse']} | {c['rank_dense']} | {c['rank_hybrid']} | {c['rank_final']} |")
            L.append("")
        if fl["id"] == "personalization":
            L += ["| User | preferred-brand share@10 (A→E) | exp. revenue A→E |", "|---|---|---|"]
            for r in results[fl["id"]]:
                v = r["variants"]
                A, E = v["A retrieval order"], v["E + MMR (final page)"]
                L.append(f"| {r['example']['user_id']} | {A['pref_brand_share@10']:.2f}→{E['pref_brand_share@10']:.2f} | {A['exp_revenue@10']:.1f}→{E['exp_revenue@10']:.1f} |")
            L.append("")
        # flow-level mean per stage
        L += ["Mean over this flow's examples:", "", "| Stage | NDCG@10 | Precision@10 | Exp. revenue | Exp. margin | Unfulfillable | Avg return prob |", "|---|---|---|---|---|---|---|"]
        for k in results[fl["id"]][0]["variants"]:
            m = lambda key: np.mean([r["variants"][k][key] for r in results[fl["id"]]])
            L.append(f"| {k} | {m('ndcg@10'):.3f} | {m('precision@10'):.3f} | {m('exp_revenue@10'):.1f} | {m('exp_margin@10'):.1f} | {m('unfulfillable@10'):.1f} | {m('avg_return_prob@10'):.3f} |")
        L.append("")
    return "\n".join(L)


def main():
    cfg = load_config()
    svc = SearchService.load(cfg)
    svc.world.load_latent()
    flows = get_flows(svc.world)
    for r in range(2):  # warm-up
        svc.search("running shoes")
    results = {fl["id"]: run_flow(svc, fl) for fl in flows}
    met = resolve(cfg, "metrics")
    save_json({"flows": [{k: v for k, v in fl.items()} for fl in flows], "results": results}, met / "flows.json")
    md = to_markdown(flows, results)
    (met / "flows.md").write_text(md, encoding="utf-8")
    readme = ROOT / "README.md"
    txt = readme.read_text(encoding="utf-8")
    block = f"{START}\n{md}\n{END}"
    if START in txt:
        txt = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda m: block, txt, flags=re.S)
    else:
        txt = txt.replace("## Screenshots", block + "\n\n## Screenshots", 1)
    readme.write_text(txt, encoding="utf-8")
    print("flows written:", sum(len(v) for v in results.values()), "examples")


if __name__ == "__main__":
    main()
