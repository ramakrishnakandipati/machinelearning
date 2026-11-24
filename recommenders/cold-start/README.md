# Cold-Start Personalization (Omnichannel Retail)

## 1. Purpose

A retail recommender that handles three cold-start problems in one service: new users (zero history), new SKUs (zero interactions) and anonymous session-only users. A **Meta-Router** picks a strategy per request from the number of known interactions. It fuses four models:

- contextual popularity
- FAISS content-based retrieval
- a two-tower network trained with ID dropout
- a LinUCB contextual bandit

New SKUs can be embedded and injected into the live FAISS index without downtime, and a FastAPI service serves everything.

| Item | Detail |
|---|---|
| Problem | Personalize when history is missing (new user, new item, anonymous session) |
| Capabilities | Synthetic retail data, 5-strategy benchmark across 4 tiers, coverage/novelty sweep, latency benchmark, `/recommend` API with Redis/LRU cache, live item onboarding, bandit feedback loop |
| Users | ML engineers, recommender-platform and merchandising teams |

## 2. High-level metrics

### Codebase summary

| Metric | Value (counted statically) |
|---|---|
| Python files under `src/` | 19 (6 are `__init__.py`; 13 are modules) |
| Other Python | `run_pipeline.py` (orchestrator), `tests/test_api.py`, `tests/visual_e2e_test.py` |
| Notebook | `cold_start_recommender.ipynb` |
| Test files / test cases | 2 files / 12 `def test_` cases (all in `tests/test_api.py`); `visual_e2e_test.py` is a Playwright script, not pytest cases |
| Language / frameworks | Python 3.12 (committed `.pyc` are `cpython-312`); PyTorch, FAISS, scikit-learn, FastAPI, Pydantic, Redis (optional), Playwright |
| Entry points | `python run_pipeline.py [--fast] [--skip-visual] [--config PATH]`; `uvicorn src.serving.app:app --port 8000`; `pytest tests/test_api.py` |

### KPIs

**Verified (observed)**. Source: `outputs/metrics/*.csv|json` and `outputs/logs/pipeline_summary.json`, from a non-fast run (`fast: false`) on an RTX 3090.

| Metric | Purpose | Observed value | Source |
|---|---|---|---|
| Recall@10, Hybrid Meta-Router, cold_user | Zero-history quality | 0.1786 (Popularity 0.1452) | `outputs/metrics/cold_start_eval_metrics.csv` |
| Recall@10, Hybrid Meta-Router, cold_item | New-SKU quality | 0.2127 (Content-Based 0.3177, Two-Tower 0.1913) | same |
| Recall@10, Hybrid Meta-Router, warm | Warm quality | 0.1193 (Two-Tower 0.1202, Popularity 0.1152) | same |
| Recall@10, Hybrid Meta-Router, session | Clickstream quality | 0.1099 (Popularity 0.1353, Two-Tower 0.1248) | same |
| Lift vs Popularity, Hybrid, cold_user | Relative gain | +0.2295 (cold_item +3.31) | same |
| Best val Recall@10, two-tower | Training health | 0.1987 (epoch 3 of 6 run) | `outputs/metrics/two_tower_training_history.csv` |
| Cold-item coverage vs recall, 0 to 5 new-item slots | Coverage/accuracy trade-off | coverage 0.29 to 0.985; recall 0.1135 to 0.0850 | `outputs/metrics/coverage_vs_recall.csv` |
| `/recommend` latency, in-process | Serving SLO | p50 3.92 ms, p95 6.08 ms, p99 7.64 ms, 251.95 rps (n=400) | `outputs/metrics/inference_latency_benchmark.json` |
| FAISS top-10 search p99 | Retrieval SLO | 0.0617 ms (CPU backend; `faiss_p99_under_10ms: true`) | same |
| Cache-hit latency p50 | Cache benefit | 1.47 ms | same |
| Pipeline wall time | Reproducibility | 61.694 s, `all_ok: true` | `outputs/logs/pipeline_summary.json` |
| Dataset | Scale | 10,000 users, 2,500 items, 150,199 interactions, 200 cold items | `outputs/metrics/dataset_summary.json` |

**Tracked / target KPIs (definitions in code, values logged each run)**

| Metric | Definition | Where logged |
|---|---|---|
| recall@k, ndcg@k, hit_rate@k | Standard top-k ranking metrics, k=10 | `src/evaluation/metrics.py` -> `cold_start_eval_metrics.csv/json` |
| coverage, cold_item_coverage | Fraction of the catalog (or of cold items) recommended | same |
| novelty, serendipity | Popularity-based surprise; hits outside the top-50 popular set | same |
| lift_vs_popularity | `add_cold_start_lift` relative to the Popularity baseline | same |
| Latency p50/p95/p99, throughput | TestClient `/recommend` timings by scenario | `inference_latency_benchmark.json` |

## 3. Subfolders at This Level

| Subfolder | Purpose | High-level metrics |
|---|---|---|
| [`src/`](src/README.md) | Library code: data generation, preprocessing, models, router, metrics, serving, utils | 19 py files; 6 packages |
| `config/` | `config.yaml` (single YAML config: seed, paths, hardware, data, models, router, serving, eval); no README (config only) | 1 file |
| `tests/` | `test_api.py` (pytest API tests) and `visual_e2e_test.py` (Playwright screenshot script run by the pipeline) | 2 files; 12 test cases |
| `outputs/` | Pipeline artifacts: `charts/`, `logs/`, `metrics/`, `models/`, `screenshots/` (PNG and HTML) | Verified metrics above come from here |
| `outputs_nb/` | Artifacts produced by the notebook run (`charts/`, `logs/`, `metrics/`, `models/`); same layout as `outputs/` minus screenshots and some model files | Notebook counterpart of `outputs/` |

## 4. Architecture & data flow

```
config.yaml --> generator (synthetic retail, 4 user tiers, cold items)
   --> Preprocessor (TF-IDF+SVD or MiniLM text embeddings, vocabs, eval cases)
   --> ContextualPopularity | ItemIndex (FAISS) + ContentBased | ContextualBandit (LinUCB replay)
   --> TwoTowerRecommender (ID dropout, AMP)
   --> MetaRouter  --(n_interactions)-->  cold_zero_history | semi_cold_session | warm_personalized
         + new-item injection slots (include_new_items_boost)
   --> evaluate (Random, Popularity, Content-Based, Two-Tower, Hybrid) x (warm, cold_user, cold_item, session)
   --> save artifacts --> latency benchmark (TestClient) --> charts --> Playwright screenshots
Serving: FastAPI /health /recommend /items/onboard /feedback; ResponseCache = Redis or in-memory LRU
```

| Route | Condition (from `router.py`, `config.yaml`) | Weights (pop / tt / content / bandit) |
|---|---|---|
| `cold_zero_history` | 0 interactions | 0.15 / 0.40 / 0.30 / 0.15 |
| `semi_cold_session` | 1 to 4 interactions (`warm_threshold: 5`) | 0.10 / 0.50 / 0.30 / 0.10 |
| `warm_personalized` | 5 or more interactions | 0.00 / 0.85 / 0.00 / 0.15 |

## 5. Prerequisites & setup

- Python 3.12 (inferred from committed `cpython-312` bytecode; `requirements.txt` states no Python version).
- Packages: numpy, pandas, scipy, scikit-learn, pyyaml, matplotlib, seaborn, torch>=2.2, faiss-cpu>=1.8, fastapi, uvicorn, pydantic>=2.6, httpx, pytest, playwright. `sentence-transformers` and `redis` are optional (fallbacks: TF-IDF+SVD, in-memory LRU).
- GPU is optional. `config.yaml` sets `use_gpu`, `use_amp` and `faiss_gpu` to true; the requirements pin `faiss-cpu` ("replace with faiss-gpu on Linux/CUDA hosts"). The committed run used CUDA with FAISS on CPU.
- Install (documented, not run): `pip install -r requirements.txt` then `playwright install chromium`.

## 6. Configuration

All keys are in `config/config.yaml`. The 2 env vars below are read by `src/serving/app.py`.

| Key | Default | Required | Description |
|---|---|---|---|
| `seed` | 42 | no | Global seed |
| `paths.*` | `outputs`, `outputs/{charts,logs,metrics,models,screenshots}` | no | Artifact directories (resolved relative to the project root) |
| `hardware.use_gpu` / `use_amp` / `faiss_gpu` | true / true / true | no | CUDA, mixed precision, faiss-gpu when available |
| `data.n_users` / `n_items` / `n_interactions` | 10000 / 2500 / 150000 | no | Synthetic data size (`--fast` overrides to 2500 / 800 / 35000) |
| `data.cold_user_frac` / `cold_item_frac` / `session_user_frac` | 0.10 / 0.08 / 0.04 | no | Cold-start population shares |
| `embeddings.*` | `prefer_transformer: true`, `all-MiniLM-L6-v2`, `text_dim: 64` | no | Text embedding backend and dimension |
| `popularity.*` | half_life 14, prior 20, min_support 30, trending 7 d / 0.3 | no | Contextual popularity |
| `two_tower.*` | embed 64, hidden 128, id_dim 32, id_dropout user/item 0.5/0.4, epochs 12, batch 1024, lr 0.002, temperature 0.1, patience 3 | no | Two-tower training |
| `bandit.*` | alpha 0.6, ridge 1.0, policy `linucb` | no | Contextual bandit (`linucb` or `thompson`) |
| `router.*` | warm_threshold 5, cold_threshold 1, candidate_pool 200, new_item_slots 2 | no | Routing and new-item injection |
| `serving.*` | host 127.0.0.1, port 8000, redis_url `redis://localhost:6379/0`, timeout 0.2 s, ttl 60 s, lru 2048 | no | API and cache |
| `eval.*` | k 10, n_eval_users 1500, latency_requests 400 | no | Evaluation size |
| env `COLDSTART_CONFIG` | `config/config.yaml` | no | Config path used by the API |
| env `COLDSTART_MODELS` | `<paths.models>` | no | Model directory used by the API |
| env `COLDSTART_REDIS_URL` | `serving.redis_url` | no | Overrides the Redis URL (not mentioned in the app docstring) |

## 7. Usage (documented commands, text only)

- Full pipeline: `python run_pipeline.py`
- Smoke run: `python run_pipeline.py --fast --skip-visual`
- Serve (after the pipeline has produced `outputs/models`): `uvicorn src.serving.app:app --port 8000`, then open `/docs` and `/health`.
- Tests: `pytest tests/test_api.py`. These need trained artifacts in `outputs/models`.
- Artifact locations:
  - `outputs/metrics`: evaluation CSV/JSON, training history, coverage sweep, latency benchmark.
  - `outputs/logs`: `pipeline_execution.log`, `pipeline_stages.jsonl`, `pipeline_summary.json`, `hardware_diagnostics.json`.
  - `outputs/models`: model files and serving state.
  - `outputs/charts` and `outputs/screenshots`: charts and API screenshots.
