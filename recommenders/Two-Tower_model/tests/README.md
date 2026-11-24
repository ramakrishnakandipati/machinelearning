# tests - end-to-end API verification

## 1. Purpose and scope

Black-box tests that boot a real `uvicorn api.main:app` process on a free port and exercise every endpoint. They also record a 100-request concurrency benchmark and generate screenshot evidence. They require the offline pipeline outputs to exist (`../outputs/models`, `../outputs/data`); they do not train anything.

- **Owns:** API contract checks, validation errors, hot-reload, the latency benchmark and screenshot artifacts.
- **Delegates:** model quality (see `../src/evaluate.py` and `offline_test_metrics.json`) and serving logic (see `../api/`). There are no unit tests for the `src/` modules.

## 2. Subfolders at This Level

None apart from `__pycache__/`, which holds bytecode only.

## 3. High-level metrics and health

| Item | Value |
|---|---|
| Python files | 4 (`__init__.py`, `conftest.py`, `shots.py`, `test_api_e2e.py`) |
| Test files | 1 (`test_api_e2e.py`) |
| Test cases | 7 (`def test_` count), plus 2 fixtures in the test module and 1 session fixture in `conftest.py` |
| Last recorded result | 7 of 7 PASSED in `../outputs/logs/06_e2e_test_execution.log` (Swagger screenshot test took 10.76 s, the others under 1 s each) |

Domain metrics

| Metric | Definition | Observed value or threshold | Where defined / output |
|---|---|---|---|
| Throughput | 100 requests, 16 threads | 408.4 rps | `test_07`, `../outputs/metrics/api_benchmark.json` |
| Server p99 | `telemetry.total_ms` | 23.17 ms observed; assert < 100 ms | `test_07` |
| Client p99 | Round trip | 55.31 ms observed; assert < 1000 ms | `test_07` |
| Pass/fail | pytest outcome per test | Logged at runtime | `06_e2e_test_execution.log` (via `log_file` in `pytest.ini`) |

## 4. Key files and components

| File | Main classes / functions | Responsibility |
|---|---|---|
| `conftest.py` | `server` (session fixture), `pytest_runtest_logreport` | Picks a free port, starts uvicorn from the project root, waits up to about 90 s for `/health`, appends to `outputs/logs/05_api_server.log`, and terminates the process afterwards |
| `shots.py` | `render_cards`, `capture_swagger`, `abbreviate` | Renders request/response cards to PNG (HTML via Playwright, Pillow fallback), and the Swagger UI screenshot |
| `test_api_e2e.py` | `call`, `client`, `warm_user` and the 7 tests below | The test suite |

Test cases

| Test | Covers |
|---|---|
| `test_01_swagger_docs_screenshot` | `/openapi.json` lists `/health`, `/recommend/user`, `/recommend/similar-items`, `/index/reload`; Swagger PNG is larger than 5000 bytes |
| `test_02_health` | HTTP 200, `status == ok`, tower loaded, FAISS ready with `ntotal == 5000`, Redis or fakeredis connected, device is cuda or cpu |
| `test_03_user_recommendations_warm_and_cold` | Warm user returns 10 `personalized` items with descending scores, no purchased items, all in stock, telemetry keys present. `category_filter=Apparel` is honoured. Unknown numeric and non-numeric user ids return `cold_start_segment` |
| `test_04_similar_items` | 8 items, `item_similarity`, seed excluded, descending scores; unknown `item_id` 999999 returns 404 |
| `test_05_validation_errors` | 3 bad bodies (empty id with top_k 0, extra field, invalid channel) each return 422 `validation_error` with details |
| `test_06_hot_reload_keeps_serving` | `/index/reload` returns `reloaded` with `n_items == 5000`; a recommendation after the reload still works |
| `test_07_concurrency_latency_benchmark` | 100 requests at concurrency 16 all return 200, p99 thresholds hold, writes `api_benchmark.json` |

## 5. Inputs, outputs and dependencies

| Direction | Item | Detail |
|---|---|---|
| Input | `outputs/models/*`, `outputs/data/{user_features_snapshot,purchased}.parquet` | Produced by `python -m src.run_pipeline` |
| Config | `pytest.ini` | `testpaths = tests`, `log_file = outputs/logs/06_e2e_test_execution.log`, INFO level |
| Output | `outputs/screenshots/01_swagger_ui_docs.png` ... `04_similar_items_response.png` | 4 PNGs written by tests 01-04 |
| Output | `outputs/metrics/api_benchmark.json`, `outputs/logs/05_api_server.log`, `06_e2e_test_execution.log` | Benchmark and logs |
| Dependencies | pytest, httpx, pandas, playwright (Chromium), Pillow | Listed in `../requirements.txt`; `setup_gpu.sh` installs Chromium |

## 6. Usage and verification

```
pytest -v                                    # from the project root
pytest tests/test_api_e2e.py::test_05_validation_errors -v
```

The tests must run from the project root because `conftest.py` inserts the root onto `sys.path` and starts uvicorn there.

## 7. Audit notes

- Only one test module and only API-level coverage; no unit tests for `Preprocessors`, `info_nce`, `metrics_from_ranks` or the feature-store logic.
- Tests share one server process and depend on test order (`test_06` reload before `test_07`) and on pre-built artifacts; they cannot run on a clean checkout.
- `test_03` and `test_06` use `warm_user`, the user with the most purchases, so the result depends on committed data.
- The benchmark file is overwritten on every run, so the committed numbers are a single run on an RTX 3090 with fakeredis, not a baseline.
- `.pytest_cache/` and `__pycache__/` (including `.pyc` for pytest 9.2.0) exist in the project folder.
