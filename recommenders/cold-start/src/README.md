# cold-start / src

## 1. Purpose & scope

This is the library package of the cold-start recommender. It owns the following:

- synthetic data generation and feature preprocessing
- the four base models (popularity, content/FAISS, two-tower, LinUCB bandit) and the `MetaRouter` that fuses them
- evaluation metrics and plots
- the FastAPI serving layer
- hardware and logging utilities

It does not own orchestration. `../run_pipeline.py` drives the stages, `../config/config.yaml` holds configuration, and `../tests/` holds the tests.

## 2. Subfolders at This Level

| Subfolder | Purpose | High-level metrics |
|---|---|---|
| `data/` | Synthetic retail dataset and `Preprocessor`/`TextEmbedder` | 2 modules + `__init__` |
| `models/` | Popularity, content-based (FAISS), two-tower, bandit, router | 5 modules + `__init__` |
| `evaluation/` | Ranking metrics and chart functions | 2 modules + `__init__` |
| `serving/` | FastAPI app, response cache, Pydantic schemas | 2 modules + `__init__` |
| `utils/` | Seeding/device/diagnostics and stage logging | 2 modules + `__init__` |

These are small packages inside one source tree, so they have no README of their own; this file covers them.

## 3. Metrics & health

| Metric | Value |
|---|---|
| Python files | 19 (6 `__init__.py`, 13 modules) |
| Test coverage by file | No unit tests for modules; `../tests/test_api.py` (12 cases) exercises the stack via the API |
| `__pycache__` present on disk | yes (gitignored by `../.gitignore`) |

Domain metrics (observed values are in the project root README):

| Metric | Definition | Value | Defined / output |
|---|---|---|---|
| recall@k, ndcg@k, hit_rate@k | Top-k ranking metrics | Logged at runtime | `evaluation/metrics.py` -> `outputs/metrics/cold_start_eval_metrics.*` |
| catalog_coverage, novelty, serendipity | Coverage, popularity-based surprise, and surprise relative to the top-popular set | Logged at runtime | `evaluation/metrics.py` |
| add_cold_start_lift | Relative lift vs the Popularity baseline | Logged at runtime | `evaluation/metrics.py` |
| Bandit reward map | view 0.0, skip 0.0, click 0.4, cart 0.7, purchase 1.0 | Threshold (constant) | `models/contextual_bandit.py` `REWARD` |
| Router weights | See the table in the project root README | Threshold (constant) | `models/router.py` `WEIGHTS` |
| Stage timings / status | Per-stage seconds and `ok/failed` | Logged at runtime | `utils/logger.py` -> `pipeline_summary.json` |

## 4. Key files & components

| File | Main classes / functions | Responsibility |
|---|---|---|
| `data/generator.py` | `RetailDataset`, `generate_dataset`, `summarize`, `hour_bucket`, `season_of_day` | Synthetic users/items/interactions with warm, cold, session user types and cold items; splits |
| `data/preprocessor.py` | `TextEmbedder` (transformer or TF-IDF+SVD), `Vocab`, `EvalCase`, `Preprocessor` | Feature encoding, `encode_new_item`, `build_rows`, `build_eval_cases`, save/load |
| `models/non_personalized.py` | `ContextualPopularity`, `wilson_lower_bound` | Segment popularity with time decay, trending, and Bayesian/Wilson smoothing; live `observe` and `grow` |
| `models/content_based.py` | `ItemIndex` (FAISS, CPU/GPU), `ContentBasedRecommender` | Cosine kNN over item content; `add` supports live item injection |
| `models/hybrid_two_tower.py` | `TwoTower`, `TwoTowerRecommender` | Two-tower with ID dropout, AMP, early stopping; `add_item`, `score_all` |
| `models/contextual_bandit.py` | `ContextualBandit`, `REWARD` | LinUCB per-subcategory arms; replay fit, online `update` |
| `models/router.py` | `MetaRouter`, `RankedItem`, `RouteResult`, `WEIGHTS` | Strategy selection by interaction count, score fusion, new-item slots, `onboard_item` |
| `evaluation/metrics.py` | `recall_at_k`, `ndcg_at_k`, `hit_rate_at_k`, `catalog_coverage`, `novelty`, `serendipity`, `evaluate_strategy`, `add_cold_start_lift` | Metrics and the per-tier benchmark harness |
| `evaluation/visualizer.py` | `plot_cold_start_distribution`, `plot_training_curve`, `plot_strategy_comparison`, `plot_coverage_novelty` | PNG charts |
| `serving/app.py` | `ResponseCache`, `ServiceState`, routes `/health`, `/recommend`, `/items/onboard`, `/feedback` | FastAPI service; lifespan loads artifacts; `/health` reports degraded if loading fails |
| `serving/schemas.py` | `RecommendRequest/Response`, `OnboardItemRequest/Response`, `FeedbackRequest/Response`, `HealthResponse`, `UserContext` | Request/response validation |
| `utils/hardware.py` | `seed_everything`, `get_device`, `faiss_gpu_available`, `autocast_ctx`, `write_diagnostics` | Device, AMP and diagnostics JSON |
| `utils/logger.py` | `get_logger`, `StageLogger` | Timed stage logging to console, `.log`, `.jsonl` and summary JSON |

## 5. Inputs, outputs & dependencies

| Item | Detail |
|---|---|
| Inputs | `../config/config.yaml`; the API reads `outputs/models` (`item_index.faiss`, two-tower weights, `bandit.pkl`, `serving_state.pkl`, `preprocessor.pkl`) via `COLDSTART_CONFIG`, `COLDSTART_MODELS` and `COLDSTART_REDIS_URL` |
| Outputs | Metrics CSV/JSON, charts and logs (written by `../run_pipeline.py` through these modules) |
| Dependencies | torch, faiss, numpy/pandas/scikit-learn, FastAPI/Pydantic, pyyaml, optional redis and sentence-transformers |
| Cross-module | `router.py` depends on all of `models/` plus `data/preprocessor.py`; `serving/app.py` depends on `router.py` and `utils/hardware.py` |

## 6. Usage & verification

- `python ../run_pipeline.py --fast --skip-visual` builds the artifacts.
- `uvicorn src.serving.app:app --port 8000` serves them. Run it from the project root.
- `pytest ../tests/test_api.py` verifies the API. It is skipped automatically if `outputs/models/serving_state.pkl` is missing.
- Check `GET /health` for the `models_loaded`, `faiss_backend`, `redis` and `cache` fields.
