"""Master orchestrator: data -> features -> train 4 learners -> fusion/tuning -> FAISS -> offline eval -> plots -> API benchmark.

  python run_pipeline.py                 # full run
  python run_pipeline.py --quick         # smoke run (fewer epochs)
  python run_pipeline.py --stages eval plots
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from data.generate_retail_data import generate  # noqa: E402
from src.evaluation import eda_and_plots as P  # noqa: E402
from src.evaluation.offline_metrics import build_eval_sets, evaluate_method, save_metrics  # noqa: E402
from src.indexing.faiss_manager import FaissManager  # noqa: E402
from src.models.common import build_bundle, scaled_epochs  # noqa: E402
from src.models.contrastive_ssl import train_ssl  # noqa: E402
from src.models.inductive_graphsage import train_graphsage  # noqa: E402
from src.models.maml_user_adapter import train_maml  # noqa: E402
from src.models.multimodal_two_tower import train_two_tower  # noqa: E402
from src.models.representation_fusion import RepresentationHub, load_all, save_all  # noqa: E402
from src.utils.hardware import get_device, hardware_info, set_seed  # noqa: E402
from src.utils.logger import StageLogger  # noqa: E402

ALL_STAGES = ["data", "train", "fusion", "index", "eval", "plots", "api"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stages", nargs="*", default=ALL_STAGES, choices=ALL_STAGES)
    ap.add_argument("--device", default=None)
    ap.add_argument("--regen-data", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.device:
        cfg["hardware"]["device"] = a.device
    set_seed(cfg["seed"])
    dev = get_device(cfg["hardware"]["device"])
    out = ROOT / cfg["paths"]["outputs"]
    for sub in ("charts", "logs", "metrics", "models", "screenshots"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    log = StageLogger(out / "logs" / "pipeline_execution.jsonl", reset=ALL_STAGES[0] in a.stages and a.stages == ALL_STAGES)
    log.log("pipeline", "hardware", **hardware_info(dev), amp=cfg["hardware"]["amp"], quick=a.quick, stages=a.stages)
    data_dir = ROOT / cfg["paths"]["data_dir"]
    scale = cfg["train"]["quick_epochs_scale"]
    ep = lambda n: scaled_epochs(n, a.quick, scale)

    if "data" in a.stages and (a.regen_data or not (data_dir / "interactions.parquet").exists() or a.stages == ALL_STAGES):
        with log.stage("data"):
            manifest = generate(cfg, data_dir, cfg["seed"], verbose=False)
            log.log("data", "manifest", **{k: v for k, v in manifest.items() if k != "slices"}, slices=manifest["slices"])
            json.dump(manifest, open(out / "metrics" / "dataset_summary.json", "w"), indent=2)

    with log.stage("features"):
        bundle = build_bundle(cfg, ROOT, dev)
        bundle.enc.save(out / "models" / "preprocessors.pkl")
        log.log("features", "built", n_users=bundle.n_users, n_items=bundle.n_items, train_rows=len(bundle.tr),
                val_rows=len(bundle.va), text_dim=bundle.text_dim, vis_dim=bundle.vis_dim)

    hist_path = out / "metrics" / "training_history.json"
    histories = {}
    models = None
    if "train" in a.stages:
        with log.stage("train"):
            tcfg = cfg["train"]
            with log.stage("train_two_tower"):
                tt, histories["two_tower"] = train_two_tower(bundle, ep(tcfg["two_tower"]["epochs"]), log)
            with log.stage("train_graphsage"):
                gm, graph, histories["graphsage"] = train_graphsage(bundle, ep(tcfg["graphsage"]["epochs"]), log)
            with log.stage("train_maml"):
                adapter, histories["maml"] = train_maml(bundle, tt, ep(tcfg["maml"]["epochs"]), log)
            with log.stage("train_ssl"):
                ssl, histories["ssl"] = train_ssl(bundle, ep(tcfg["ssl"]["epochs"]), log)
            json.dump(histories, open(hist_path, "w"), indent=2)
            rows = [{"model": m, **r} for m, h in histories.items() for r in h]
            pd.DataFrame(rows).to_csv(out / "metrics" / "training_history.csv", index=False)
            models = (tt, gm, graph, adapter, ssl)
            save_all(out / "models", bundle, tt, gm, adapter, ssl, {"item": {}, "r0": {}, "r1": {}, "r2": {}})
    elif hist_path.exists():
        histories = json.load(open(hist_path))

    need_hub = any(s in a.stages for s in ("fusion", "index", "eval", "plots", "api"))
    hub = None
    cold_mask = torch.as_tensor(bundle.feats.item_cold, device=dev)
    if need_hub:
        if models is None:
            tt, gm, graph, adapter, ssl, w = load_all(bundle, out / "models")
        else:
            tt, gm, graph, adapter, ssl = models
            w = None
        hub = RepresentationHub(bundle, tt, gm, graph, adapter, ssl, weights=w if (w and w.get("item")) else None)
        val_sets = build_eval_sets(bundle, bundle.val_df, cfg["eval"]["max_users_per_slice"], seed=1)
        if "fusion" in a.stages:
            with log.stage("fusion"):
                hub.tune(val_sets, cold_mask, log)
                save_all(out / "models", bundle, tt, gm, adapter, ssl, hub.w)
                log.log("fusion", "weights", **{k: v for k, v in hub.w.items()})
                np.save(out / "models" / "unified_item_embeddings.npy", hub.E_uni.cpu().numpy())

    test_sets = build_eval_sets(bundle, bundle.test_df, cfg["eval"]["max_users_per_slice"], seed=2) if hub else None

    if "index" in a.stages:
        with log.stage("index"):
            fc = cfg["faiss"]
            meta = {"l1": bundle.items.l1.values}
            fm = FaissManager(cfg["embedding_dim"]).build(hub.E_uni.cpu().numpy(), meta, fc["hnsw_m"], fc["ef_construction"], fc["ef_search"],
                                                          (fc["ivfpq"]["nlist"], fc["ivfpq"]["m"], fc["ivfpq"]["nbits"]))
            fm.save(out / "models")
            qs = []
            for sd in test_sets.values():
                raw = hub.raw_for_slice(sd)
                qs.append(hub.unified_query(raw, sd.nhist)[0])
            q = torch.cat(qs)
            q = q[torch.randperm(len(q), generator=torch.Generator().manual_seed(0))[: fc["n_bench_queries"]].to(q.device)].cpu().numpy()
            res = fm.benchmark(q, ef_sweep=fc["ef_search_sweep"], out_path=out / "metrics" / "ann_benchmark.json")
            h = [r for r in res["results"] if r["index"] == "IndexHNSWFlat" and r.get("efSearch") == fc["ef_search"]]
            log.log("index", "benchmark", n_queries=res["n_queries"], hnsw_recall=h[0]["recall@10_vs_exact"] if h else None,
                    hnsw_p95_ms=h[0]["p95_ms"] if h else None)
            P.plot_ann_tradeoff(res, out / "charts" / "05_ann_recall_latency.png")

    eval_df = None
    if "eval" in a.stages:
        with log.stage("eval"):
            rows = []
            pop = torch.bincount(bundle.tr.i, minlength=bundle.n_items).float()
            rows += evaluate_method("popularity", None, None, test_sets, cold_mask, pop_scores=pop, ks=cfg["eval"]["ks"])
            for name, (qf, im) in hub.method_table().items():
                rows += evaluate_method(name, qf, im, test_sets, cold_mask, ks=cfg["eval"]["ks"])
            eval_df = save_metrics(rows, out / "metrics")
            piv = eval_df.pivot(index="method", columns="slice", values="recall@20").round(3)
            log.log("eval", "recall@20", table=piv.to_dict())
            print(piv.to_string())

    if "plots" in a.stages:
        with log.stage("plots"):
            items, users, inter = bundle.items, bundle.users, bundle.inter
            manifest = json.load(open(out / "metrics" / "dataset_summary.json")) if (out / "metrics" / "dataset_summary.json").exists() \
                else json.load(open(data_dir / "split_manifest.json"))
            P.plot_eda(items, users, inter, manifest, out / "charts" / "01_eda_coldstart_distributions.png")
            if histories:
                P.plot_loss_curves(histories, out / "charts" / "02_training_validation_loss_curves.png")
            name = P.plot_embedding_space(hub.E_uni.cpu().numpy(), items, out / "charts" / "03_representation_space_umap_tsne.png")
            log.log("plots", "projection", method=name)
            if eval_df is None and (out / "metrics" / "offline_evaluation.csv").exists():
                eval_df = pd.read_csv(out / "metrics" / "offline_evaluation.csv")
            if eval_df is not None:
                P.plot_slice_comparison(eval_df, out / "charts" / "04_coldstart_slice_comparison.png")

    if "api" in a.stages:
        with log.stage("api_benchmark"):
            from src.serving.app import benchmark_api
            res = benchmark_api(ROOT, cfg)
            json.dump(res, open(out / "metrics" / "api_benchmark.json", "w"), indent=2)
            log.log("api_benchmark", "done", **{k: v for k, v in res.items() if not isinstance(v, (dict, list))})
    log.log("pipeline", "finished")


if __name__ == "__main__":
    main()
