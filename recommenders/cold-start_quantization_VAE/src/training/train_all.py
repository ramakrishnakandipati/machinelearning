"""Training stage: tokenizers (VQ-VAE, RQ-VAE) -> catalogs -> generative model + bridge towers. Saves artifacts + logs."""
from __future__ import annotations

import json
import time

import numpy as np

from src.artifacts import (assemble_recommenders, build_catalog, export_semantic_artifacts, make_model,
                           save_tokenizer)
from src.config import LOGS_DIR, MODELS_DIR, Config, DEVICE, seed_everything
from src.recommenders.codebook_bridge_rec import CodebookBridgeRecommender
from src.recommenders.generative_seq_rec import GenerativeSeqRecommender
from src.training.evaluator import codebook_stats
from src.training.trainer import train_quantizer, write_logs


def train_all(ds: dict, cfg: Config, verbose: bool = True):
    seed_everything()
    X = ds["item_feats"]
    cold = ds["items"]["is_cold"].values
    warm = np.where(~cold)[0]
    rng = np.random.default_rng(0)
    rng.shuffle(warm)
    n_val = int(0.1 * len(warm))
    va, tr = warm[:n_val], warm[n_val:]
    q = cfg.quant
    logs, models = {}, {}
    for kind, name in (("vq", "vq_vae"), ("rq", "rq_vae")):
        seed_everything()
        m = make_model(kind, X.shape[1], q)
        logs[name] = train_quantizer(m, X[tr], X[va], X[cold], q, name, verbose)
        save_tokenizer(m, kind, X.shape[1], q)
        models[kind] = m

    rq = build_catalog("rq_vae", ds, models["rq"], q.codebook_size)
    vq = build_catalog("vq_vae", ds, models["vq"], q.codebook_size)
    summary = {}
    for name, cat in (("vq_vae", vq), ("rq_vae", rq)):
        summary[name] = {
            "all_items": codebook_stats(cat.codes, q.codebook_size),
            "warm_items": codebook_stats(cat.codes[~cold], q.codebook_size),
            "cold_items": codebook_stats(cat.codes[cold], q.codebook_size),
        }
        summary[name]["cold_items_new_id_rate"] = float(np.mean(
            [tuple(c) not in set(map(tuple, cat.codes[~cold])) for c in cat.codes[cold]]))
    write_logs(logs, extra=summary)
    export_semantic_artifacts(rq, vq, models["rq"], ds)

    # ---- representation-conditioned recommenders
    t0 = time.time()
    gen = GenerativeSeqRecommender(rq, cfg.gen)
    gen_log = gen.fit(ds, verbose=verbose)
    gen.save(MODELS_DIR / "generative_seq.pt")
    b_rq = CodebookBridgeRecommender(rq, cfg.bridge)
    b_rq_log = b_rq.fit(ds, verbose=verbose)
    b_rq.save(MODELS_DIR / "codebook_bridge_rq.pt")
    b_vq = CodebookBridgeRecommender(vq, cfg.bridge)
    b_vq_log = b_vq.fit(ds, verbose=False)
    b_vq.save(MODELS_DIR / "codebook_bridge_vq.pt")
    (LOGS_DIR / "recommender_training_logs.json").write_text(json.dumps(
        {"generative_seq": gen_log, "codebook_bridge_rq": b_rq_log, "codebook_bridge_vq": b_vq_log,
         "train_seconds": time.time() - t0}, indent=2))
    bundle = assemble_recommenders(ds, rq, vq, gen, b_rq, b_vq)
    bundle.meta = {"device": str(DEVICE), "codebook_size": rq.K, "n_levels": rq.M, "tokenizer_logs": logs,
                   "tokenizer_summary": summary, "recommender_logs": {"generative_seq": gen_log, "bridge": b_rq_log}}
    return bundle
