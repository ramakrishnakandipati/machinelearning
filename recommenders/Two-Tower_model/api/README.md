# api - online serving layer

## 1. Purpose and scope

FastAPI service that turns the offline artifacts into low-latency recommendations: user tower inference (TorchScript, GPU if available), FAISS HNSW nearest-neighbour retrieval, post-retrieval filtering, Redis (or fakeredis) feature lookups, and hot-reload.

- **Owns:** HTTP contract and validation, the model bundle lifecycle, the feature store adapter, filtering rules and telemetry.
- **Delegates:** training, index building and artifact creation go to `../src/`. Behaviour checks go to `../tests/`.
- **Start command (from project root):** `uvicorn api.main:app --port 8000`

## 2. Subfolders at This Level

None apart from `__pycache__/`, which holds bytecode only.

## 3. High-level metrics and health

| Item | Value |
|---|---|
| Python files | 5 (`__init__`, `main`, `serving`, `feature_store`, `schemas`) |
| Endpoints | 4: `GET /health`, `POST /recommend/user`, `POST /recommend/similar-items`, `POST /index/reload` |
| Tests | 7 cases in `../tests/test_api_e2e.py`, all passed in `../outputs/logs/06_e2e_test_execution.log` |

Domain metrics. Verified values come from `../outputs/metrics/api_benchmark.json` (100 requests, concurrency 16, `/recommend/user`, top_k=10).

| Metric | Definition | Observed value or threshold | Where defined / output |
|---|---|---|---|
| Throughput | Requests per second under 16 concurrent clients | 408.4 rps (wall 0.245 s) | `api_benchmark.json` |
| Server total latency p50 / p95 / p99 / mean | `telemetry.total_ms` | 12.36 / 21.23 / 23.17 / 12.59 ms. Test threshold: p99 < 100 ms | `api_benchmark.json`, `test_07` |
| Client latency p50 / p95 / p99 | Round trip including HTTP | 30.27 / 46.05 / 55.31 ms. Test threshold: p99 < 1000 ms | `api_benchmark.json` |
| Hot reload time | Build a new bundle and swap | 264.37 ms (`reload_ms`, one observed run) | `06_e2e_test_execution.log` |
| Health snapshot | device, FAISS and store state | `status: ok`, device cuda (RTX 3090), `faiss_ntotal` 5000, store `fakeredis`, 19,997 users loaded | `06_e2e_test_execution.log` |
| Per-request telemetry | `inference_ms`, `ann_search_ms`, `total_ms` | Logged at runtime (response body) | `serving.py`, `schemas.Telemetry` |
| Access log | method, path, status, latency | Logged at runtime | `outputs/logs/05_api_server.log` |

Note: there is no `/metrics` endpoint, no request counters and no percentile aggregation inside the service. Only per-request telemetry and the access log exist.

## 4. Key files and components

| File | Main classes / functions | Responsibility |
|---|---|---|
| `main.py` | `lifespan`, `health`, `recommend_user`, `similar_items`, `reload_index`, `access_log`, `validation_handler` | App wiring; a startup failure leaves the app degraded (503 for model routes) instead of crashing; atomic bundle swap under a lock |
| `serving.py` | `ModelBundle.load`, `embed_user`, `retrieve`, `recommend_user`, `similar_items` | Loads the TorchScript tower, FAISS index, embeddings, segment vectors and metadata. Over-fetches 3x top_k and doubles k until filters are satisfied. Cold start uses channel segment embedding plus a 0.05 popularity re-rank. Warms up with 5 dummy inferences. Version string is `two-tower-<mtime of two_tower_best.pt>` |
| `feature_store.py` | `FeatureStore.connect`, `load_snapshot`, `get_user`, `purchased` | Redis via `REDIS_URL`, falling back to fakeredis. Keys: `user:{id}` hash, `purchased:{id}` set |
| `schemas.py` | `UserRecRequest`, `SimilarItemsRequest`, `RecItem`, `RecResponse`, `Telemetry`, `ReloadResponse` | Pydantic v2 contracts with `extra="forbid"` |

Request contracts:

| Endpoint | Fields (default) |
|---|---|
| `POST /recommend/user` | `user_id` (str, 1-64), `top_k` (10, 1-100), `channel` (Web / Mobile / In-Store, optional), `exclude_purchased` (true), `category_filter` (optional, matches `category_l1` case-insensitively), `in_stock_only` (true) |
| `POST /recommend/similar-items` | `item_id` (int >= 0; 404 if >= catalogue size), `top_k` (10, 1-100), `category_filter`, `in_stock_only` (true) |
| `POST /index/reload` | No body. 500 and the previous model keeps serving if the load fails |

Response `strategy` is one of `personalized`, `cold_start_segment`, `item_similarity`. Validation errors return 422 with `{"error": "validation_error", "details": [...]}`.

## 5. Inputs, outputs and dependencies

| Direction | Item | Detail |
|---|---|---|
| Input | `outputs/models/*` | `preprocessors.joblib`, `user_tower_serving.pt`, `retail_items.faiss`, `item_embeddings.npy`, `segment_embeddings.npz`, `item_metadata.parquet`, `two_tower_best.pt` (mtime only) |
| Input | `outputs/data/user_features_snapshot.parquet`, `purchased.parquet` | Loaded into the feature store at startup |
| Env | `REDIS_URL` (default `redis://localhost:6379/0`) | Falls back to in-process fakeredis when unreachable (0.3 s connect timeout) |
| Output | `outputs/logs/05_api_server.log` | Rotating access and lifecycle log |
| Dependencies | fastapi, uvicorn, pydantic, torch, faiss, redis, fakeredis, joblib, pandas, numpy | Imports `src.config`, `src.logger`, `src.train.U_KEYS`, `src.data_generator.CHANNEL_DEVICE` |

## 6. Usage and verification

```
uvicorn api.main:app --port 8000        # run from project root, after python -m src.run_pipeline
curl http://localhost:8000/health
curl -X POST http://localhost:8000/recommend/user -H "Content-Type: application/json" -d "{\"user_id\":\"42\",\"top_k\":10,\"channel\":\"Mobile\"}"
curl -X POST http://localhost:8000/recommend/similar-items -H "Content-Type: application/json" -d "{\"item_id\":10,\"top_k\":8}"
curl -X POST http://localhost:8000/index/reload
pytest -v                               # E2E suite (see ../tests/README.md)
```

Interactive docs: `/docs` (Swagger).

## 7. Audit notes

- `/health` hard-codes `faiss_backend` to `"cpu-index (HNSW)"`; it is not detected at runtime.
- `/health` calls `store.ping()` twice, and the endpoint returns 503 if the model failed to load.
- `recommend_user` accepts any digit string; a numeric id absent from the feature store is treated as cold start (this is intended and tested).
- `embed_user` always encodes the user's stored channel and the device from the requested channel, and day/hour from the server's local clock (`datetime.now()`), which makes responses time-dependent.
- No authentication, rate limiting or request-count metrics. The `_swap_lock` guards the swap but readers do not take it.
- `__pycache__/` is present in the folder.
