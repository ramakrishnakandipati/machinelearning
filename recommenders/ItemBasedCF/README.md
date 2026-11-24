# Item-Based Collaborative Filtering (IBCF) Retail Recommender

Offline batch pipeline plus a FastAPI microservice that serves item-to-item ("similar items") and user-personalised recommendations for an omnichannel retailer. It is built on synthetic data, so the metrics show relative model health only.

- **Problem:** recommend relevant, in-stock, diverse products without letting very popular "staple" items dominate the neighbour lists. Handle cold-start users, durable-vs-consumable repurchase rules and sub-category diversity.
- **Capabilities:** time-decayed implicit-feedback matrix, BM25-weighted item-item cosine, co-occurrence shrinkage (beta) and top-K pruning, temporal-split grid search, a self-verification report, and an online engine with guardrails (stock, repurchase, diversity cap) and a popularity fallback.
- **Users:** data scientists (train and verify) and personalisation or platform engineers (serve through the API).

## 1. High-level metrics

### Codebase summary

| Item | Value |
|---|---|
| Python modules | 4 (`train_pipeline.py`, `recommender_engine.py`, `api.py`, `verify_and_screenshot_api.py`) plus notebook `ibcf_demo.ipynb` |
| Frameworks | numpy, scipy, pandas, pyarrow, scikit-learn, FastAPI, uvicorn, pydantic v2, matplotlib, seaborn, Pillow, httpx (`requirements.txt`) |
| Test files / cases | 0 pytest files, 0 `def test_`. Verification is script-based: `verify_and_screenshot_api.py` (5 scenarios, 35 assertions in `api_test_summary.json`) |
| Entry points | `python train_pipeline.py`, `uvicorn api:app --port 8000`, `python verify_and_screenshot_api.py` |
| Container | `Dockerfile` (python:3.12-slim, runs `train_pipeline.py` at build, serves on 8000) |

### KPIs

**Verified (observed)** values come from `output/artifacts/verification_report.json` unless noted.

| Metric | Purpose | Observed value | Source |
|---|---|---|---|
| NDCG@10 (validation, best beta=5, K=10) | Ranking quality | 0.2731 | verification_report.json |
| Recall@10 | Hit rate | 0.3746 | verification_report.json |
| Coverage (items appearing in top-10 across eval users) | Catalogue breadth | 0.6824 | verification_report.json |
| Popularity baseline NDCG@10 / Recall@10 / coverage | Lift reference | 0.1039 / 0.1613 / 0.0205 | verification_report.json |
| Raw co-occurrence top-K NDCG@10 / Recall@10 / coverage | Lift reference | 0.1594 / 0.2749 / 0.2172 | verification_report.json |
| Staple slot share, final vs raw co-occurrence | Staple-bias removal | 0.0 vs 0.345 | verification_report.json |
| Intra-category score share, final vs raw | Neighbour purity | 1.0 vs 0.7073 | verification_report.json |
| Eval users / split day | Evaluation size | 2777 users, split at day 144 of 180 | verification_report.json |
| Self-checks | Regression gate | 6 of 6 true (`all_checks_passed: true`) | verification_report.json |
| Model version, n_items, n_users | Identity | `ibcf-ca75a092`, 488, 3000 | model_meta.json |
| API checks | Behavioural QA | 0 failures across 5 scenarios | output/api_tests/api_test_summary.json |
| Recommend p99 latency | Serving SLO | 0.686 ms over warm-up (assertion threshold below 30 ms) | api_test_summary.json |
| Serving `catalog_coverage` (share of items that appear as a neighbour) | Graph coverage | 0.9098 | api_test_summary.json |

**Tracked / target KPIs (runtime)**

| Metric | Definition | Where logged |
|---|---|---|
| Per-route latency p50/p95/p99 and `latency_ms_recommend_overall` | Rolling 5000-sample window per route (`/v1/` routes aggregated) | `GET /metrics` (`api.py: Metrics`) |
| Requests and errors by route | Counters, errors are status >= 400 | `GET /metrics` |
| Strategy mix | Counts of `item_similarity`, `personalized`, `session`, `fallback` | `GET /metrics` -> `strategies` |
| `X-Latency-Ms` | Per-request wall time | Response header (middleware) |
| Uptime | Seconds since start | `GET /metrics` -> `uptime_s` |

## 2. Subfolders at This Level

| Subfolder | Purpose | High-level metrics |
|---|---|---|
| `output/artifacts/` | Serving artifacts written by `train_pipeline.py`: `item_sim_map.parquet`, `catalog.parquet`, `user_history.parquet`, `fallback_popular.json`, `model_meta.json`, `verification_report.json` | 6 files, 488 items, version `ibcf-ca75a092` |
| `output/charts/` | 4 diagnostic PNGs (sparsity and long tail, time decay and funnel, tuning curves, similarity heatmap) | 4 images |
| `output/api_tests/` | Screenshots and `api_test_summary.json` from `verify_and_screenshot_api.py` | 5 PNG cards, 0 failures |
| `output/notebook/` | Copy of artifacts and charts produced by running `ibcf_demo.ipynb` | Mirrors `artifacts/` and `charts/` |
| `__pycache__/` | Python bytecode (not source) | Ignore |

## 3. Architecture and data flow

```
train_pipeline.py (offline)
 synthetic data (3000 users, 180 days, 10 categories)
   -> time-decayed user x item matrix (half-life 30d; view=1, add_to_cart=3, purchase=5)
   -> BM25 item-item cosine + shrinkage S*n/(n+beta)
   -> temporal split (80/20) + grid search beta x K -> best by NDCG@10 -> refit on all data
   -> output/artifacts/* + verification_report.json + output/charts/*
                 |
                 v
recommender_engine.RecommenderEngine.load()  <- api.py lifespan (env IBCF_ARTIFACT_DIR)
   /v1/recommend/item/{id}   -> neighbours (similar_item)      -> fallback if empty or unknown
   /v1/recommend/user        -> history seeds + session boost  -> top-up/fallback for cold start
   guardrails: in-stock, durable repurchase block, consumable window 14d,
               max 3 per sub-category, up to 500 candidates
```

## 4. Prerequisites and setup

- Python 3.12 (Dockerfile base image); the repo contains `cpython-312` bytecode. No GPU required.
- Install: `pip install -r requirements.txt` (pyarrow is mandatory; the pipeline exits without it).

## 5. Configuration

| Key | Default | Required | Description |
|---|---|---|---|
| `IBCF_ARTIFACT_DIR` (env) | `./output/artifacts` | No | Artifact directory loaded by `api.py` |
| `SEED`, `N_USERS`, `N_DAYS` | 42, 3000, 180 | No | Constants in `train_pipeline.py` |
| `HALF_LIFE_DAYS`, `EVENT_W0` | 30.0, view=1 / add_to_cart=3 / purchase=5 | No | Time decay and event weights |
| `BM25_K1`, `BM25_B` | 1.2, 0.75 | No | BM25 parameters |
| `BETAS`, `KS`, `TOPN`, `VAL_FRACTION` | [0,5,10,25,50,100], [10,20,50,100], 10, 0.2 | No | Grid and evaluation settings |
| `SESSION_BOOST`, `MAX_PER_SUBCATEGORY`, `MAX_CANDIDATES` | 2.0, 3, 500 | No | Engine constants in `recommender_engine.py` |
| `consumable_repurchase_days` | 14 | No | Written to `model_meta.json` by the pipeline |

## 6. Usage

```
python train_pipeline.py                      # build artifacts, report, charts under ./output
uvicorn api:app --port 8000                   # serve (Dockerfile uses --host 0.0.0.0 --port 8000)
python verify_and_screenshot_api.py           # starts its own uvicorn on a free port, runs 5 scenarios, writes output/api_tests/
docker build -t ibcf . && docker run -p 8000:8000 ibcf
```

API endpoints (from `api.py`):

| Method and path | Notes |
|---|---|
| `POST /v1/recommend/user` | Body: `user_id` (1-64 chars), `recent_item_ids` (max 100), `exclude_item_ids` (max 500), `limit` (1-50, default 10). Extra fields are forbidden. |
| `GET /v1/recommend/item/{item_id}` | Query: `limit` (1-50), `in_stock_only` (default true) |
| `GET /health` | 200 when artifacts are loaded, otherwise 503 `degraded` |
| `GET /metrics` | Model info, coverage, latency percentiles, counters |

Validation errors return 422 `{"error": "validation_error", "details": [...]}`. A missing model returns 503 `model_unavailable`.

## 7. Audit notes

- No pytest suite, and no unit tests for `recommender_engine.py` or the metric functions. Verification relies on the screenshot script.
- `output/artifacts/` and `output/notebook/artifacts/` duplicate the same artifact set.
- `__pycache__/*.pyc` is present, and no `.gitignore` exists in this folder.
- `evaluate()` coverage (0.68) and the serving `catalog_coverage` (0.91) are different definitions; the names are close but the values are not comparable.
- `staple_slot_share = 0.0` and `intra_category_score_share = 1.0` are the exact observed values. With a synthetic block-structured dataset these are likely over-optimistic and not a real-world expectation.
- `Dockerfile` regenerates synthetic data at build time, and `verify_and_screenshot_api.py` and the screenshot dependencies are shipped inside the serving image.
