"""Stage 1 + Stage 2 of the funnel: query understanding and hybrid multimodal candidate retrieval."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.common.text import tokens
from src.common.utils import resolve, get_device
from src.common.world import World
from src.query_understanding.ner_intent import ParsedQuery, QueryNER
from src.query_understanding.query_expander import QueryExpander
from src.query_understanding.spell_corrector import SpellCorrector
from src.retrieval.dense_retriever import DenseIndex, MultimodalFuser, TextEncoder, doc_text
from src.retrieval.hybrid_fusion import fuse
from src.retrieval.sparse_retriever import SparseRetriever


@dataclass
class QueryContext:
    raw: str
    corrected: str
    edits: list
    parsed: ParsedQuery
    expansion: dict
    learned: dict
    weighted_terms: dict
    qvec: np.ndarray
    timings: dict = field(default_factory=dict)


@dataclass
class Candidates:
    idx: np.ndarray
    fused: np.ndarray
    dense_cos: np.ndarray
    bm25: np.ndarray
    char: np.ndarray
    dense_rank: np.ndarray
    sparse_rank: np.ndarray
    dense_ids: np.ndarray
    sparse_ids: np.ndarray
    timings: dict = field(default_factory=dict)


class SearchEngine:
    def __init__(self, cfg: dict, world: World):
        self.cfg, self.world = cfg, world
        self.rc = cfg["retrieval"]
        self.spell = self.ner = self.expander = self.sparse = self.dense = self.encoder = None
        self.doc_emb = self.text_emb = None
        self._qcache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------------------------------
    def _load_encoder(self):
        names = [self.rc["dense_model"], self.rc["dense_fallback"]]
        self.encoder = TextEncoder(names, self.rc["query_prefix"], get_device())
        logger.info("Encoder {} on {}", self.encoder.name, self.encoder.device)

    def _fit_light(self):
        w = self.world
        extra = [t for terms in w.synonyms.values() for t in terms] + list(w.synonyms)
        self.spell = SpellCorrector().fit(w.catalog, extra)
        train_q = w.queries[w.queries.split == "train"]
        self.ner = QueryNER(w.synonyms).fit(w.catalog, train_q)
        self.expander = QueryExpander(w.synonyms, self.rc["expansion"]["n_neighbors"])

    @classmethod
    def build(cls, cfg: dict, world: World, sessions: pd.DataFrame) -> "SearchEngine":
        self = cls(cfg, world)
        mdir = resolve(cfg, "models")
        self._load_encoder()
        self._fit_light()
        logger.info("Encoding {} documents", world.n)
        t0 = time.time()
        self.text_emb = self.encoder.encode_docs(doc_text(world.catalog))
        logger.info("  doc encoding {:.1f}s", time.time() - t0)
        self.fuser = MultimodalFuser(self.rc["fusion_weights"]).fit(world.catalog, world.img, self.text_emb)
        self.doc_emb = self.fuser.fuse(world.catalog, world.img, self.text_emb)
        self.dense = DenseIndex(self.rc["hnsw"]).build(self.doc_emb)
        self.sparse = SparseRetriever(self.rc["field_weights"], **self.rc["bm25"]).fit(world.catalog)
        train_q = world.queries[world.queries.split == "train"]
        self.sparse.fit_learned_expansion(world.catalog, sessions, train_q)
        logger.info("Learned sparse expansion for {} query terms", len(self.sparse.learned))
        self.expander.build_vocab(world.catalog, self.encoder.encode)
        np.save(mdir / "doc_emb.npy", self.doc_emb)
        np.save(mdir / "text_emb.npy", self.text_emb)
        self.dense.save(mdir / "faiss_hnsw.index")
        joblib.dump({"sparse": self.sparse, "fuser": self.fuser}, mdir / "retrieval.joblib")
        return self

    @classmethod
    def load(cls, cfg: dict, world: World) -> "SearchEngine":
        self = cls(cfg, world)
        mdir = resolve(cfg, "models")
        self._load_encoder()
        self._fit_light()
        art = joblib.load(mdir / "retrieval.joblib")
        self.sparse, self.fuser = art["sparse"], art["fuser"]
        self.doc_emb = np.load(mdir / "doc_emb.npy")
        self.text_emb = np.load(mdir / "text_emb.npy")
        self.dense = DenseIndex.load(mdir / "faiss_hnsw.index", self.doc_emb, self.rc["hnsw"])
        self.expander.build_vocab(world.catalog, self.encoder.encode)
        self.warmup()
        return self

    def warmup(self, n: int = 3):
        for q in ["waterproof trail kicks", "warm winter puffer", "office chair"][:n]:
            self.retrieve(self.understand(q))

    # ------------------------------------------------------------------------------------------
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Batch encode (used for offline training/eval)."""
        pref = self.encoder.prefix_used
        v = self.encoder.model.encode([pref + t for t in texts], normalize_embeddings=True, batch_size=256,
                                      convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        return v

    def understand(self, raw: str, qvec: np.ndarray | None = None) -> QueryContext:
        tm: dict = {}
        t0 = time.perf_counter()
        corrected, edits = self.spell.correct(raw)
        tm["spell_ms"] = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        syn_probe = self.expander.expand(corrected)  # synonyms only (no vector yet)
        syn_text = " ".join(syn_probe["synonyms"])
        if qvec is None:
            texts = [corrected] + ([f"{corrected} {syn_text}"] if syn_text else [])
            vs = self.encoder.model.encode([self.encoder.prefix_used + t for t in texts], normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            qvec = vs.mean(0)
            qvec /= np.linalg.norm(qvec) + 1e-9
            base_vec = vs[0]
        else:
            base_vec = qvec
        tm["encode_ms"] = (time.perf_counter() - t1) * 1e3
        t2 = time.perf_counter()
        exp = self.expander.expand(corrected, base_vec)
        qt = list(tokens(corrected))
        learned = self.sparse.learned_terms(qt)
        weighted = dict(exp["weighted_terms"])
        for t, w in learned.items():
            weighted[t] = max(weighted.get(t, 0.0), w)
        ext_tokens = [t for t in weighted if t not in qt]
        parsed = self.ner.parse(raw, corrected, ext_tokens)
        tm["expand_ner_ms"] = (time.perf_counter() - t2) * 1e3
        tm["understand_ms"] = (time.perf_counter() - t0) * 1e3
        return QueryContext(raw, corrected, edits, parsed, exp, learned, weighted, qvec, tm)

    def retrieve(self, ctx: QueryContext, mode: str | None = None, alpha: float | None = None,
                 n: int | None = None) -> Candidates:
        n = n or self.rc["n_candidates"]
        mode = mode or self.rc["fusion"]
        alpha = self.rc["alpha"] if alpha is None else alpha
        tm: dict = {}
        t0 = time.perf_counter()
        s_ids, s_sc, bm_all, ch_all = self.sparse.search(ctx.weighted_terms, ctx.corrected, self.rc["n_sparse"])
        tm["sparse_ms"] = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        d_ids, d_sc = self.dense.search(ctx.qvec, self.rc["n_dense"])
        tm["dense_ms"] = (time.perf_counter() - t1) * 1e3
        t2 = time.perf_counter()
        ids, fused = fuse(mode, d_ids, d_sc, s_ids, s_sc, alpha, self.rc["rrf_k"], n)
        cos = self.doc_emb[ids] @ ctx.qvec
        d_rank = np.full(len(ids), self.rc["n_dense"] + 1, dtype=np.float32)
        s_rank = np.full(len(ids), self.rc["n_sparse"] + 1, dtype=np.float32)
        dpos = {int(d): r for r, d in enumerate(d_ids)}
        spos = {int(d): r for r, d in enumerate(s_ids)}
        for j, d in enumerate(ids):
            d_rank[j] = dpos.get(int(d), d_rank[j])
            s_rank[j] = spos.get(int(d), s_rank[j])
        tm["fuse_ms"] = (time.perf_counter() - t2) * 1e3
        tm["retrieval_ms"] = (time.perf_counter() - t0) * 1e3
        return Candidates(ids.astype(np.int64), fused, cos.astype(np.float32), bm_all[ids].astype(np.float32),
                          ch_all[ids].astype(np.float32), d_rank, s_rank, d_ids, s_ids, tm)
