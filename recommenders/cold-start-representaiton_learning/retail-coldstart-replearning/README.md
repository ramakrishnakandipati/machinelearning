# Retail Cold-Start Representation Learning

A retail recommender that serves warm users, new users, new items and new stores from **one unified 128-d embedding space**. Four learners (multimodal Two-Tower, inductive GraphSAGE, MAML user adapter, contrastive SSL) are aligned with orthogonal Procrustes, fused, indexed in FAISS and served through a FastAPI microservice. A router picks the strategy from the amount of user history.

- **Problem:** cold-start for users (0 / 1-3 interactions), items (zero-shot onboarding) and stores (system cold-start) on synthetic retail data.
- **Capabilities:** synthetic data generation, 4 learners, validation-tuned fusion, FAISS Flat/HNSW/IVFPQ with recall/latency benchmark, stratified offline evaluation (4 slices), REST API with Redis-or-memory cache, zero-shot item embedding, Playwright visual tests.
- **Users:** ML engineers evaluating cold-start strategies; platform engineers integrating the `/recommend` API.

## 1. High-level metrics

### Codebase summary

| Item | Value |
|---|---|
| Python files | 26 (21 under `src/` incl. 7 `__init__.py`, `run_pipeline.py`, `data/generate_retail_data.py`, 3 under `tests/`) |
| Notebook | `retail_coldstart_replearning.ipynb` (its log stage is named `notebook`) |
| Frameworks | PyTorch, FAISS (cpu), FastAPI/Pydantic v2, scikit-learn, pandas/pyarrow, Redis (optional), Playwright |
| Test files | 2 test modules plus `conftest.py` |
| Test cases (static count of `def test_`) | 19 (17 in `test_api_endpoints.py`, 2 in `test_visual_playwright.py`; `test_validation_errors` is parametrized, so more are collected at runtime) |
| Entry points | `run_pipeline.py` (CLI), `uvicorn src.serving.app:app` (API), `data/generate_retail_data.py` (via pipeline) |

### KPIs

**Verified (observed)** - test split, 4 slices, from `outputs/metrics/offline_evaluation.json`:

| Metric | Purpose | Observed | Source |
|---|---|---|---|
| recall@20, warm | Ranking quality, warm users | unified_ensemble 0.1662; MAML 0.1588; two_tower 0.1517; popularity 0.0265 | `outputs/metrics/offline_evaluation.json` |
| recall@20, user_cold | New-user quality | unified_ensemble 0.1208; two_tower 0.1202; MAML 0.1136; popularity 0.0242 | same |
| recall@20, item_cold | New-item quality | contrastive_ssl 0.1229; unified_ensemble 0.1121; two_tower 0.1012; GraphSAGE 0.0162; popularity 0.0 | same |
| recall@20, system_cold | New-store quality | contrastive_ssl 0.0858; unified_ensemble 0.0496; two_tower 0.0453; popularity 0.0 | same |
| cold_items_recommended_pct (warm) | Cold-item exposure | unified_ensemble 12.93; contrastive_ssl 18.11; popularity 0.0 | same |
| catalog_coverage_pct (warm) | Diversity | unified_ensemble 47.08; contrastive_ssl 87.76; popularity 0.52 | same |
| HNSW recall@10 vs exact (efSearch=128) | ANN fidelity | 1.0 (efSearch=32: 0.9836; efSearch=8: 0.8534); p95 0.0644 ms | `outputs/metrics/ann_benchmark.json` |
| IVFPQ recall@10 vs exact | Compression trade-off | 0.5912 (nprobe=1) to 0.6692 (nprobe>=8); code bytes 30000 vs flat 1280000 | same |
| API uncached p95 (max over scenarios) | Serving latency | 11.109 ms (300 requests, in-process TestClient, memory cache) | `outputs/metrics/api_benchmark.json` |
| API cache hits (server side) | Cache effectiveness | 310 of 900 requests | same |
| Dataset | Scale and leakage | 5000 users, 2500 items, 25 stores, 50172 interactions; train/val/test 24000/11593/10938; `leakage_checks: passed` | `outputs/metrics/dataset_summary.json` |
| Hardware of committed run | Reproducibility | CUDA, NVIDIA GeForce RTX 3090, torch 2.15.0+cu132, amp on | `outputs/logs/pipeline_execution.jsonl` |

**Tracked / target KPIs** (defined in code, value produced per run):

| Metric | Definition | Where logged |
|---|---|---|
| recall@10/20, ndcg@10, mrr | Per-user, averaged per slice; seen history excluded; all 2500 items ranked | `src/evaluation/offline_metrics.py` -> `outputs/metrics/offline_evaluation.{csv,json}` |
| alignment / uniformity (users, items) | Representation-quality diagnostics (`src/models/common.py`) | same |
| cold_item_coverage_pct | Share of cold items appearing in any top-10 | same |
| Training train/val loss per epoch | InfoNCE / NT-Xent and episodic losses | `outputs/metrics/training_history.{json,csv}`, `outputs/logs/pipeline_execution.jsonl` |
| Telemetry per request | `total_ms`, `embed_ms`, `ann_ms`, `cache_hit`, `cache_backend` | `/recommend` response, `/metrics` |

## 2. Subfolders at This Level

| Subfolder | Purpose | High-level metrics |
|---|---|---|
| [`src/`](src/README.md) | Models, features, FAISS indexing, evaluation, FastAPI serving, utilities | 21 Python files in 6 packages |
| `config/` | `config.yaml` single source of hyperparameters and paths | 1 file, 8 top-level sections plus `seed` and `embedding_dim` |
| `data/` | `generate_retail_data.py` plus `generated/` (users, items, stores, interactions parquet and `split_manifest.json`) | 5000 users, 2500 items, 50172 interactions |
| `outputs/` | Generated artifacts: `charts/` (5 PNGs), `logs/` (JSONL), `metrics/` (7 files), `models/` (4 `.pt`, 3 FAISS indexes, embeddings, preprocessors, fusion weights), `screenshots/` (4 PNGs) | See KPIs above |
| `tests/` | API tests (FastAPI TestClient, fakeredis) and Playwright visual tests | 19 test functions |

## 3. Architecture & data flow

```
generate_retail_data -> parquet + split_manifest (time split, 4 slices)
        |
build_bundle (encoders: TF-IDF+SVD text, visual 64-d, categorical, context)
        |
 +------+-----------+-----------+------------+
 | TwoTower | GraphSAGE | MAML adapter | Contrastive SSL |   (each -> L2-normalised 128-d)
 +------+-----------+-----------+------------+
        |  orthogonal Procrustes on WARM items -> one space
RepresentationHub (weights tuned on validation only)
        |-> E_uni -> FAISS Flat / HNSW / IVFPQ
        |-> router by history count n: 0 content+graph | 1-3 meta_adapter+graph | >3 multimodal+contrastive_ensemble
FastAPI: /recommend /items/embed /health /metrics /strategies  (cache: Redis, fallback in-memory)
```

## 4. Prerequisites & setup

- Python 3.12 (compiled caches are `cpython-312`). Pinned in `requirements.txt`: torch 2.15.0, faiss-cpu 1.16.0, numpy 2.5.4, pandas 3.1.1, fastapi 0.163.2, uvicorn 0.54.0, pydantic 2.14.3, redis 8.1.0, fakeredis 2.40.0, playwright 1.65.2, pytest 9.2.0 (plus scipy, scikit-learn, pyarrow, matplotlib, seaborn, PyYAML, httpx). `umap-learn` is optional (falls back to t-SNE).
- GPU optional: `hardware.device: auto` picks cuda/mps/cpu; the committed run used CUDA.
- Playwright tests need browsers installed (`playwright install`, not documented in the repo).
- Redis is optional; the cache silently falls back to memory.

```
pip install -r requirements.txt
```

## 5. Configuration (`config/config.yaml`; no environment variables are read)

| Key | Default | Required | Description |
|---|---|---|---|
| `seed` | 42 | yes | Global seed |
| `embedding_dim` | 128 | yes | Unified embedding size |
| `paths.data_dir` / `paths.outputs` | `data/generated` / `outputs` | yes | I/O locations |
| `hardware.device` / `hardware.amp` | `auto` / true | yes | Device choice, mixed precision |
| `data.*` | 5000 users, 2500 items, 25 stores, 50000 interactions, cold fractions 0.15, `zero_shot_user_share` 0.5, `system_cold_stores` 2, `visual_dim` 64 | yes | Synthetic data generator |
| `features.*` | `text_svd_dim` 64, `tfidf_max_features` 4000, `use_transformer_text` false | yes | Text features |
| `train.<model>.*` | two_tower 25 epochs, graphsage 40, maml 30, ssl 40; `quick_epochs_scale` 0.25 | yes | Per-learner hyperparameters |
| `fusion.*` | `weights_content` 0.6, `weights_graph` 0.4, `history_alpha` 0.5 | yes | Initial weights (tuned later) |
| `faiss.*` | `hnsw_m` 32, `ef_construction` 200, `ef_search` 128, sweep [8..256], IVFPQ nlist 32/m 16/nbits 6, `n_bench_queries` 500 | yes | Index build and benchmark |
| `eval.*` | `ks` [10, 20], `max_users_per_slice` 1500 | yes | Evaluation |
| `api.*` | host 127.0.0.1, port 8000, `default_k` 10, `max_k` 100 | no | Serving defaults |
| `redis.*` | `url` redis://localhost:6379/0, `ttl_seconds` 300, `enabled` true | no | Cache |

## 6. Usage (text only)

```
python run_pipeline.py                      # all stages: data train fusion index eval plots api
python run_pipeline.py --quick              # fewer epochs (quick_epochs_scale)
python run_pipeline.py --stages eval plots  # subset (choices: data train fusion index eval plots api)
python run_pipeline.py --device cpu --regen-data --config config/config.yaml
uvicorn src.serving.app:app --port 8000     # serve (needs outputs/models from training)
pytest tests                                # tests
```

Artifacts: models, FAISS indexes, preprocessors, embeddings in `outputs/models/`; metrics in `outputs/metrics/`; charts in `outputs/charts/`; stage log in `outputs/logs/pipeline_execution.jsonl`.

API endpoints: `GET /health`, `POST /recommend`, `POST /items/embed`, `GET /metrics`, `GET /strategies`.

## 7. Known gaps

- `src/serving/app.py` builds `app = create_app()` at import time, and the Engine loads on startup; serving requires trained artifacts to exist.
- `api.host/port/default_k/max_k` and `use_transformer_text` are in the config; `api.*` is not read by `app.py` (`max_k`/`default_k` may be enforced in `schemas.py`; not verified).
- Committed `__pycache__`, `.pytest_cache` and model binaries exist in the folder (check `.gitignore`).
