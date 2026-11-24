# cold-start_quantization_VAE / src

## 1. Purpose & scope

This package holds all source code of the Semantic-ID cold-start project. It owns the following:

- synthetic data generation
- VQ-VAE and RQ-VAE tokenizers
- four Semantic-ID recommenders plus baselines
- training and evaluation harnesses
- plots
- the FastAPI service and its screenshot capture

It delegates orchestration to `../run_pipeline.py` and holds no tests; there is no `tests/` folder in this project. Project-level results and install steps are in [`../README.md`](../README.md).

## 2. Subfolders at This Level

| Subfolder | Purpose | High-level metrics |
|---|---|---|
| `data/` | Synthetic dataset and `ItemFeaturizer` | 1 module (`generate_data.py`) |
| `models/` | `VQVAE` (EMA codebook) and `RQVAE` (residual, K-Means++ init) | 2 modules |
| `recommenders/` | Catalog/context base, baselines, semantic trie, generative sequence model, codebook bridge, hybrid | 6 modules |
| `training/` | Tokenizer trainer, training orchestration, evaluator and metrics | 3 modules |
| `visualization/` | Six-figure plot suite | 1 module |
| `api/` | FastAPI app, test-harness HTML, Playwright screenshot capture | 3 modules |

All are small packages inside this one tree, so they have no READMEs of their own; this file covers them.

## 3. Metrics & health

| Metric | Value |
|---|---|
| Python files | 25 (7 `__init__.py`, 2 top-level modules `config.py` and `artifacts.py`, 16 package modules; counted via Glob) |
| Test files | 0 |
| Entry points | None directly; used by `../run_pipeline.py` and `uvicorn src.api.app:app` |

**Verified (observed)**. Source: `../outputs/logs/*`.

| Metric | Definition | Observed | Where defined / output |
|---|---|---|---|
| Recall@10, hybrid, item_cold / user_cold / system_cold / warm | Fraction of ground-truth items in the top 10 | 0.5872 / 0.0835 / 0.0111 / 0.1204 | `training/evaluator.py` -> `evaluation_metrics.csv` |
| Recall@10, codebook_bridge, item_cold | Same | 0.5334 | same |
| Recall@10, raw_content_knn, item_cold | Strong baseline | 0.5410 | same |
| Recall@10, popularity / random, item_cold | Baselines | 0.0592 / 0.0592 | same |
| RQ-VAE all-items collision_rate / unique_ids | Share of items sharing a full ID | 0.002 / 1998 of 2000 | `codebook_stats` -> `validation_logs.json` `tokenizer_summary` |
| VQ-VAE all-items collision_rate / unique_ids | Same | 0.997 / 256 of 2000 | same |
| RQ-VAE mean perplexity / utilization | Codebook health | 208.65 / 99.61% | same |
| VQ-VAE perplexity / utilization | Codebook health (single level) | 219.23 / 100% | same |
| `cold_items_new_id_rate` (RQ / VQ) | Cold items given an ID unseen among warm items | 1.0 / 0.0 | same |
| Best val recon (RQ / VQ) | Reconstruction MSE on warm validation split | 0.0330 (186 epochs) / 0.1126 (66 epochs) | `validation_logs.json` `final` |
| Pipeline timing (s) | Per-stage runtime | data 7.7, train 73.0, eval 97.1, viz 8.7, screenshots 20.8 | `../outputs/logs/pipeline_timing.json` |
| Mean latency, generative_seq vs hybrid vs trie | Per-request ms | about 56 / about 0.2 to 3.1 / about 0.05 | `evaluation_metrics.csv` (`latency_ms`) |

**Tracked metrics (definitions in code)**

| Metric | Definition | Where |
|---|---|---|
| recall@K, ndcg@K, hit@K for K in (10, 20), MRR, coverage@10 | Standard ranking metrics; MRR cutoff 100 | `training/evaluator.py`; `Config.eval_k = (10, 20)`, `eval_max_users = 800` |
| Codebook perplexity, utilization %, collision rate, `level_*` per level | Computed per epoch and at the end | `training/evaluator.py` `codebook_stats`; logged to `training_logs.json`, `validation_logs.json`, `tokenizer_epoch_metrics.csv` |
| latency_ms, latency_p95_ms | Per-recommend latency per regime | `evaluator.py` -> `evaluation_metrics.csv` |

## 4. Key files & components

| File | Main classes / functions | Responsibility |
|---|---|---|
| `config.py` | `DataConfig`, `QuantConfig`, `GenConfig`, `BridgeConfig`, `Config`, `get_config`, `seed_everything` | Dataclass hyper-parameters, paths, device/AMP; `quick()` preset |
| `artifacts.py` | `make_model`, `save_tokenizer`, `load_tokenizer`, `build_catalog`, `export_semantic_artifacts`, `Bundle`, `assemble_recommenders`, `load_bundle` | Persist and reload tokenizers/catalogs/recommenders; exports `semantic_id_map.json` and `rq_codebooks.npz` |
| `data/generate_data.py` | `ItemFeaturizer`, `generate_dataset`, `summarize`, `save_dataset`, `load_dataset` | Synthetic users, items and interactions; item-cold and user-cold holdouts |
| `models/vq_vae.py` | `EMAVectorQuantizer`, `VQVAE`, `ResBlock`, `make_mlp` | EMA codebook, straight-through estimator, commitment loss, dead-code reset |
| `models/rq_vae.py` | `RQVAE`, `kmeans_pp_init` | Multi-level residual quantization (default 4 x 256) |
| `training/trainer.py` | `train_quantizer`, `encode_all`, `write_logs` | Tokenizer training with early stopping; writes training/validation logs and the epoch CSV |
| `training/train_all.py` | `train_all` | Trains VQ and RQ, builds catalogs, trains the generative and bridge recommenders, saves artifacts |
| `training/evaluator.py` | `codebook_stats`, `recall_at_k`, `ndcg_at_k`, `mrr`, `build_eval_requests`, `evaluate_recommender`, `run_benchmark` | Metrics and the four-regime benchmark (warm, item_cold, user_cold, system_cold) |
| `recommenders/base.py` | `UserContext`, `Catalog`, `ColdStartRecommender` | Shared catalog, semantic-ID helpers and the recommender interface |
| `recommenders/baselines.py` | `RandomRecommender`, `PopularityRecommender`, `RawContentKNNRecommender` | Baselines |
| `recommenders/semantic_trie_rec.py` | `SemanticTrie`, `SemanticTrieRecommender` | Prefix-fallback retrieval over Semantic IDs (approach 1) |
| `recommenders/generative_seq_rec.py` | `TokenSpace`, `SeqTransformer`, `GenerativeSeqRecommender` | Causal Transformer with trie-constrained beam search (approach 2) |
| `recommenders/codebook_bridge_rec.py` | `UserTower`, `ItemTower`, `CodebookBridgeRecommender` | Metadata -> codebook-space two-tower; dot, overlap or fused scoring (approach 3) |
| `recommenders/hybrid_cold_start_rec.py` | `HybridColdStartRecommender` | Content kNN + collaborative smoothing + segment/popularity/margin re-ranking, MMR (approach 4) |
| `visualization/plot_suite.py` | `make_all` and 6 `plot_*` functions | Writes the six figures to `outputs/visualizations` |
| `api/app.py` | `Service`, request models, routes | `/health`, `/recommend/item-cold-start`, `/recommend/user-cold-start`, `/recommend/system-cold-start`, `/recommend/compare-approaches`, `/semantic-id/inspect/{item_id}`, `/ui/test-harness`, `/` |
| `api/harness_html.py` | `render_harness` | HTML test harness page |
| `api/capture_screenshots.py` | `start_server`, `wait_for_health`, `capture_all`, `run_with_server` | Starts the API on port 8765 and captures screenshots with Playwright |

## 5. Inputs, outputs & dependencies

| Item | Detail |
|---|---|
| Inputs | Generated dataset (`outputs/models/dataset.pkl`); env `PIPELINE_QUICK=1` selects the quick config; no `.env` file or other env vars |
| Outputs | `outputs/models` (`vq_vae.pt`, `rq_vae.pt`, `generative_seq.pt`, `codebook_bridge_{rq,vq}.pt`, `semantic_id_map.json`, `rq_codebooks.npz`, `scalers.json`, `dataset.pkl`); `outputs/logs`; `outputs/visualizations`; `outputs/api_screenshots` |
| Dependencies | torch, numpy, pandas, scikit-learn, faiss-cpu, matplotlib, seaborn, fastapi, uvicorn, pydantic, requests, playwright, tqdm; optional umap-learn |
| Cross-module | `api/app.py` -> `artifacts.load_bundle` -> `recommenders/*`; `training/*` -> `models/*` and `recommenders/*` |

## 6. Usage & verification

- `python ../run_pipeline.py` runs the full pipeline. Add `--quick`, `--skip-train` or `--skip-screenshots` as needed.
- `uvicorn src.api.app:app --port 8000`, run from the project root after a pipeline run, serves the API. Open `/docs` and `/ui/test-harness`.
- Verify by checking `outputs/logs/evaluation_metrics.csv`, the `tokenizer_summary` in `validation_logs.json`, and the six PNGs in `outputs/visualizations`.
