"""Saving / loading of trained artifacts and construction of the serving bundle (catalogs + recommenders)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch

from src.config import DEVICE, MODELS_DIR, BridgeConfig, GenConfig, QuantConfig
from src.data.generate_data import load_dataset
from src.models.rq_vae import RQVAE
from src.models.vq_vae import VQVAE
from src.recommenders.base import Catalog
from src.recommenders.baselines import PopularityRecommender, RandomRecommender, RawContentKNNRecommender
from src.recommenders.codebook_bridge_rec import CodebookBridgeRecommender
from src.recommenders.generative_seq_rec import GenerativeSeqRecommender
from src.recommenders.hybrid_cold_start_rec import HybridColdStartRecommender
from src.recommenders.semantic_trie_rec import SemanticTrieRecommender


def make_model(kind: str, in_dim: int, q: QuantConfig):
    kw = dict(latent_dim=q.latent_dim, hidden=q.hidden_dim, n_blocks=q.n_res_blocks, num_codes=q.codebook_size,
              beta=q.beta, decay=q.ema_decay, dead_threshold=q.dead_code_threshold, dropout=q.dropout)
    return VQVAE(in_dim, **kw) if kind == "vq" else RQVAE(in_dim, n_levels=q.n_levels, **kw)


def save_tokenizer(model, kind: str, in_dim: int, q: QuantConfig) -> None:
    torch.save({"state": model.state_dict(), "kind": kind, "in_dim": in_dim, "cfg": q.__dict__},
               MODELS_DIR / f"{kind}_vae.pt")


def load_tokenizer(kind: str):
    ck = torch.load(MODELS_DIR / f"{kind}_vae.pt", map_location=DEVICE, weights_only=False)
    q = QuantConfig(**ck["cfg"])
    model = make_model(kind, ck["in_dim"], q).to(DEVICE)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, q


def build_catalog(name: str, dataset: dict, model, num_codes: int) -> Catalog:
    X = torch.tensor(dataset["item_feats"], device=DEVICE)
    model.eval()
    with torch.no_grad():
        codes = torch.cat([model.encode_to_codes(X[i:i + 2048]) for i in range(0, len(X), 2048)])
        zq = model.codes_to_embedding(codes)
        ze = model.encode(X)
    inter = dataset["interactions"]
    tr = inter[inter["split"] == "train"]
    pop = np.bincount(tr["item_id"].values, minlength=len(dataset["items"]))
    return Catalog(name, dataset, model, codes.cpu().numpy(), zq.cpu().numpy(), ze.cpu().numpy(), pop, num_codes)


def export_semantic_artifacts(rq_cat: Catalog, vq_cat: Catalog, rq_model, dataset: dict) -> None:
    (MODELS_DIR / "semantic_id_map.json").write_text(json.dumps(
        {"rq_vae": rq_cat.id_map(), "vq_vae": vq_cat.id_map(),
         "n_levels": rq_cat.M, "codebook_size": rq_cat.K}, indent=1))
    np.savez(MODELS_DIR / "rq_codebooks.npz", **{f"level_{l + 1}": q.embedding.cpu().numpy()
                                                  for l, q in enumerate(rq_model.quantizers)})
    f = dataset["featurizer"]
    (MODELS_DIR / "scalers.json").write_text(json.dumps({
        "numeric_mean": f.num_mean.tolist(), "numeric_std": f.num_std.tolist(),
        "text_mean": f.txt_mean.tolist(), "text_std": f.txt_std.tolist(),
        "image_mean": f.img_mean.tolist(), "image_std": f.img_std.tolist(),
        "price_edges": f.price_edges.tolist()}, indent=1))


@dataclass
class Bundle:
    dataset: dict
    rq: Catalog
    vq: Catalog
    recs: Dict[str, object] = field(default_factory=dict)       # name -> recommender
    main: Dict[str, object] = field(default_factory=dict)       # the 4 headline approaches
    meta: dict = field(default_factory=dict)


def assemble_recommenders(ds: dict, rq: Catalog, vq: Catalog, gen: GenerativeSeqRecommender,
                          bridge_rq: CodebookBridgeRecommender, bridge_vq: CodebookBridgeRecommender) -> Bundle:
    hybrid = HybridColdStartRecommender(rq, ds)
    trie = SemanticTrieRecommender(rq)
    main = {"semantic_trie": trie, "generative_seq": gen, "codebook_bridge": bridge_rq, "hybrid": hybrid}
    recs = dict(main)
    recs.update({
        "bridge_dot": bridge_rq.with_mode("dot"), "bridge_overlap": bridge_rq.with_mode("overlap"),
        "trie_vq": SemanticTrieRecommender(vq), "bridge_vq": bridge_vq, "hybrid_vq": HybridColdStartRecommender(vq, ds),
        "popularity": PopularityRecommender(rq), "random": RandomRecommender(rq),
        "raw_content_knn": RawContentKNNRecommender(rq, ds["item_feats"]),
    })
    return Bundle(ds, rq, vq, recs, main)


def load_bundle() -> Bundle:
    ds = load_dataset()
    rq_model, rq_q = load_tokenizer("rq")
    vq_model, vq_q = load_tokenizer("vq")
    rq = build_catalog("rq_vae", ds, rq_model, rq_q.codebook_size)
    vq = build_catalog("vq_vae", ds, vq_model, vq_q.codebook_size)
    gen = GenerativeSeqRecommender(rq)
    gen.load(MODELS_DIR / "generative_seq.pt")
    b_rq = CodebookBridgeRecommender(rq); b_rq.load(MODELS_DIR / "codebook_bridge_rq.pt")
    b_vq = CodebookBridgeRecommender(vq); b_vq.load(MODELS_DIR / "codebook_bridge_vq.pt")
    bundle = assemble_recommenders(ds, rq, vq, gen, b_rq, b_vq)
    bundle.meta = {"device": str(DEVICE), "codebook_size": rq.K, "n_levels": rq.M}
    return bundle
