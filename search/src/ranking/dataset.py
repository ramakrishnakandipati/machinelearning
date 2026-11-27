"""Builds (query,user) candidate lists with features, simulated engagement labels and latent ground truth."""
from __future__ import annotations

import numpy as np
from loguru import logger

from src.engine import SearchEngine
from src.ranking.feature_engineering import FEATURE_NAMES, FeatureBuilder


def build_lists(engine: SearchEngine, split: str, n_lists: int, labeling_impressions: int, seed: int,
                fusion: str | None = None, alpha: float | None = None, use_typo: bool = True) -> dict:
    w = engine.world
    rng = np.random.default_rng(seed)
    fb = FeatureBuilder(w)
    qs = w.queries[w.queries.split == split].reset_index(drop=True)
    picks = rng.choice(len(qs), size=n_lists, replace=len(qs) < n_lists)
    out = {k: [] for k in ["X", "sizes", "y_eng", "rel", "y_click", "y_atc", "y_pur", "y_ret", "n_click", "n_atc",
                           "n_pur", "idx", "qid", "uidx", "pmax", "pbrand", "p_click", "p_atc", "p_cvr", "p_ret"]}
    L = labeling_impressions
    for j, qi in enumerate(picks):
        q = qs.iloc[qi]
        uidx = int(rng.integers(0, len(w.users)))
        raw = q.typo_text if use_typo else q.text
        ctx = engine.understand(raw)
        cand = engine.retrieve(ctx, mode=fusion, alpha=alpha)
        X = fb.build(ctx, cand, uidx)
        idx = cand.idx
        pc, pa, pv, pr = w.true_probs(q.query_id, uidx, idx)
        n = len(idx)
        clicks = rng.binomial(L, pc)
        atcs = rng.binomial(clicks, pa)
        purs = rng.binomial(atcs, pv)
        rets = rng.binomial(purs, pr)
        stage = np.where(purs > 0, 3, np.where(atcs > 0, 2, np.where(clicks > 0, 1, 0)))
        out["X"].append(X); out["sizes"].append(n); out["y_eng"].append(stage)
        out["rel"].append(w.rel_true(q.query_id)[idx])
        out["y_click"].append(clicks / L); out["n_click"].append(clicks)
        out["n_atc"].append(atcs); out["n_pur"].append(purs)
        out["y_atc"].append(atcs); out["y_pur"].append(purs); out["y_ret"].append(rets)
        out["idx"].append(idx); out["qid"].append(q.query_id); out["uidx"].append(uidx); out["pmax"].append(ctx.parsed.max_price); out["pbrand"].append(bool(ctx.parsed.brand))
        out["p_click"].append(pc); out["p_atc"].append(pa); out["p_cvr"].append(pv); out["p_ret"].append(pr)
        if (j + 1) % 500 == 0:
            logger.info("  {} lists {}/{}", split, j + 1, n_lists)
    data = {k: (np.concatenate(v) if k in ("X", "y_eng", "rel", "y_click", "y_atc", "y_pur", "y_ret", "n_click",
                                            "n_atc", "n_pur", "idx", "p_click", "p_atc", "p_cvr", "p_ret") else np.array(v))
            for k, v in out.items()}
    data["X"] = data["X"].astype(np.float32)
    data["feature_names"] = FEATURE_NAMES
    data["L"] = L
    data["offsets"] = np.concatenate([[0], np.cumsum(data["sizes"])])
    return data
