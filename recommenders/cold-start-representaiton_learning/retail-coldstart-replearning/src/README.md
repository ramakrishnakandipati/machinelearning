# src - Library code

Owns all model, feature, indexing, evaluation and serving code for the retail cold-start project. Does not own configuration (`../config/config.yaml`), data generation (`../data/`) or orchestration (`../run_pipeline.py`), which call into this package.

## Subfolders at This Level

| Subfolder | Purpose | High-level metrics |
|---|---|---|
| `features/` | `RepresentationEncoders` (fit/transform users, items, context; save/load preprocessors) and `Features`; `load_data` | 1 module |
| `models/` | Four learners plus fusion: `MultimodalTwoTower`, `HeteroGraphSAGE` (+`HeteroGraph`), `MAMLUserAdapter`, `SSLModel`, `RepresentationHub`, shared `common.py` (`info_nce`, `nt_xent`, `alignment`, `uniformity`, `Bundle`, `build_bundle`) | 6 modules |
| `indexing/` | `FaissManager`: builds Flat, HNSW, IVFPQ, search with filters, benchmark, save | 1 module |
| `evaluation/` | `offline_metrics.py` (slice sets, recall/ndcg/mrr/coverage/alignment/uniformity) and `eda_and_plots.py` (5 plot functions) | 2 modules |
| `serving/` | FastAPI app (`Cache`, `Engine`, `create_app`, `benchmark_api`) and Pydantic `schemas.py` | 2 modules |
| `utils/` | `hardware.py` (`set_seed`, `get_device`, `hardware_info`, autocast/scaler) and `logger.py` (`StageLogger`, JSONL) | 2 modules |

## High-level metrics & health

- 21 Python files (7 `__init__.py`, 14 modules). Tests live in `../tests/` (19 test functions; none import-test individual modules directly except through the API).
- Stale caches: `__pycache__` folders are present in every package.

| Metric | Definition | Value | Defined / output |
|---|---|---|---|
| recall@k, ndcg@10, mrr | Per-user ranking metrics over all 2500 items, seen history masked | See root README (e.g. unified_ensemble warm recall@20 0.1662) | `evaluation/offline_metrics.py` -> `../outputs/metrics/offline_evaluation.json` |
| catalog_coverage_pct, cold_items_recommended_pct, cold_item_coverage_pct | Top-10 coverage measures | Logged at runtime (committed values in root README) | `offline_metrics.aggregate` |
| alignment, uniformity | Embedding quality (`models/common.py`) | Logged at runtime | `offline_metrics.evaluate_method` |
| ANN recall@10 vs exact, p50/p95/p99 ms | HNSW/IVFPQ benchmark | HNSW efSearch=128 recall 1.0 | `indexing/faiss_manager.py` -> `../outputs/metrics/ann_benchmark.json` |
| Request telemetry, `/metrics` | Latency percentiles, cache hits, per-strategy and per-cold-start counts | Logged at runtime | `serving/app.py` |

## Key files & components

| File | Main classes/functions | Responsibility |
|---|---|---|
| `features/representation_encoders.py` | `RepresentationEncoders`, `Features`, `load_data` | Text (TF-IDF+SVD), categorical, context encoding |
| `models/common.py` | `Bundle`, `build_bundle`, `build_interactions`, `build_history`, `info_nce`, `nt_xent`, `GRN` | Shared tensors, losses, metrics helpers |
| `models/multimodal_two_tower.py` | `ItemTower`, `UserTower`, `MultimodalTwoTower`, `train_two_tower` | Content+history two-tower |
| `models/inductive_graphsage.py` | `HeteroGraph`, `SAGELayer`, `HeteroGraphSAGE`, `train_graphsage` | Inductive graph embeddings for new users/items |
| `models/maml_user_adapter.py` | `MAMLUserAdapter`, `EpisodeSampler`, `train_maml` | Few-shot user adaptation |
| `models/contrastive_ssl.py` | `SSLModel`, `L3Sampler`, `train_ssl`, `ssl_user_query` | Item/sequence contrastive pretraining |
| `models/representation_fusion.py` | `RepresentationHub`, `route_of`, `STRATEGIES`, `save_all`, `load_all` | Procrustes alignment, fusion, validation tuning, router |
| `indexing/faiss_manager.py` | `FaissManager` | ANN indexes and benchmark |
| `evaluation/offline_metrics.py` | `build_eval_sets`, `rank_metrics`, `evaluate_method`, `save_metrics` | Stratified offline evaluation |
| `evaluation/eda_and_plots.py` | `plot_eda`, `plot_loss_curves`, `plot_embedding_space`, `plot_slice_comparison`, `plot_ann_tradeoff` | Charts |
| `serving/app.py`, `serving/schemas.py` | `Cache`, `Engine`, `create_app`, `benchmark_api`; request/response models | REST API |
| `utils/hardware.py`, `utils/logger.py` | `get_device`, `set_seed`, `StageLogger` | Device and JSONL stage logging |

## Inputs, outputs & dependencies

| Kind | Detail |
|---|---|
| Inputs | `../config/config.yaml`; `../data/generated/*.parquet` and `split_manifest.json` |
| Outputs | `../outputs/models/*` (weights, indexes, preprocessors, embeddings, `fusion_weights.json`), `../outputs/metrics/*`, `../outputs/charts/*`, `../outputs/logs/pipeline_execution.jsonl` |
| Dependencies | torch, faiss-cpu, scipy (Procrustes), scikit-learn, pandas, FastAPI/Pydantic, redis (optional), matplotlib/seaborn |

## Usage & verification

```
python ../run_pipeline.py --stages train fusion index eval
uvicorn src.serving.app:app --port 8000      # run from the project root
pytest ../tests
```
