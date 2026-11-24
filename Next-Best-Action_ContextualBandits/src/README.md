# `src/` - Next-Best-Action contextual-bandit engine (library + API)

## 1. Purpose & scope
`src/` is the importable package behind `run_pipeline.py` (training/evaluation entry point) and the FastAPI service (`src.api:app`). It owns: configuration and the discrete offer space, the synthetic confounded hospitality data generator, feature engineering shared by training and serving, off-policy evaluation (OPE) plus online replay, JSONL telemetry, plotting, the serving API/dashboard, and Playwright screenshots. The bandit algorithms themselves live in [`models/`](models/README.md) and are only consumed here.

It does **not** own: the CLI orchestration (`../run_pipeline.py`), tests (`../tests/`), generated artifacts (`../output/`, `../data/`) or the notebook/presentation. Problem framing and the narrative are in the [project README](../README.md).

## 2. Subfolders at This Level
| Subfolder | Purpose | High-level metrics |
|---|---|---|
| [`models/`](models/README.md) | The 9 bandit classes (10 configured policies), common `BaseBandit` interface, `model_zoo`, `load_model` | 6 source files; 10 policies; 8 algorithm families per project README |
| `__pycache__/` | Compiled bytecode (ignored by `.gitignore`) | n/a |

## 3. High-level metrics & health
**Module/code metrics**

| Item | Value |
|---|---|
| Python modules directly in `src/` | 9 (`__init__`, `config`, `data_generator`, `preprocessing`, `evaluation`, `telemetry`, `visualization`, `api`, `screenshots`) |
| Test file covering `src/` | `../tests/test_smoke.py` - 5 test functions, 14 cases (10 are parametrized over the model zoo) |
| Language / frameworks | Python 3.12 bytecode present; numpy, pandas, scikit-learn, PyTorch, XGBoost, FastAPI, pydantic v2, matplotlib/seaborn, Playwright |
| Modules with no direct test | `api.py`, `visualization.py`, `screenshots.py`, `telemetry.py` |

**Domain metrics** (values from `../output/logs/ope_evaluation_audit.json`, a `mode: full`, `device: cuda` run; `entry: notebook`, created 2026-11-21)

| Metric | Definition | Observed value / status | Where defined / output |
|---|---|---|---|
| Feature count | One-hot (tier, channel, archetype, day) + 7 numeric | 25 (Verified, `model_registry.json`) | `preprocessing.py`, `output/models/model_registry.json` |
| Train / test size | 70/30 split of 30000 sessions | 21000 train, 9000 test (Verified) | `../run_pipeline.py`, audit `meta.n_train`, `n_test` |
| Min logged propensity (test) | Overlap diagnostic | 0.0819 (Verified); test asserts >= 0.02 | audit JSON; `LOG_EPS = 0.12` in `data_generator.py` |
| Logged mean reward | Mean realised margin ($) | 488.58 (Verified) | audit JSON |
| TRUE value: Logging / Uniform / Always control / Oracle | Simulator ground-truth expected margin of policy ($) | 491.34 / 466.94 / 401.11 / 629.08 (Verified) | audit JSON `results.*.TRUE` |
| Best model by DR | `argmax DR` among bandits | LinUCB-Disjoint, DR 609.92, TRUE 617.33 (Verified) | `model_registry.json` |
| ESS of best model | `(sum w)^2 / sum w^2`, weights clipped at 20 | 1757.6 of 9000 (Verified) | `evaluation.evaluate_policy` |
| Per-model fit seconds | Wall time of `fit` | 0.07 s (LinearTS) to 9.72 s (NeuralTS) (Verified) | audit `meta.fit_seconds` |
| Online replay regret per session | `cum_regret[-1] / n` | Printed to stdout by the pipeline; `Logged at runtime` | `evaluation.replay`, `run_pipeline.py` |
| API latency | `latency_ms` per `/predict_nba` | Logged at runtime to `output/logs/api_requests.jsonl` | `api.py` |

## 4. Key files & components
| File | Main classes / functions | Responsibility |
|---|---|---|
| `config.py` | `ROOT`, `*_DIR`, `SEED=42`, `DEVICE`, `XGB_DEVICE`, `N_ACTIONS=6`, `ACTIONS`, `ACTION_SHORT`, `ACTION_ATTR`, `TIERS/CHANNELS/ARCHETYPES/DAYS`, `ROOM_MARGIN=0.65`, `ANC_MARGIN=0.45` | Paths (auto-created on import), seed, CUDA/CPU selection, action space and vocabularies |
| `data_generator.py` | `generate_dataset(n=30000, seed)`, `_sample_context`, `_economics`, `_logging_policy`, `true_reward_matrix`, `logging_prob_matrix` | Synthetic sessions with confounded logging policy (`LOG_EPS=0.12`), realised margin, and ground-truth `true_r_a*` / `log_p_a*` columns (must not be used as features) |
| `preprocessing.py` | `add_derived`, `build_preprocessor`, `fit_preprocessor`, `transform`, `save_/load_preprocessor`, `RAW_COLS`, `NUM_COLS`, `CAT_COLS` | log1p lead time/spend, month sin/cos, StandardScaler + fixed-vocabulary OneHotEncoder; shared by training and API |
| `evaluation.py` | `DMRewardModel`, `evaluate_policy`, `baseline_policies`, `run_ope`, `replay` | DM/IPS/SNIPS/DR with weight clip 20, 200-sample bootstrap 95% CI, ESS, ground-truth TRUE; writes `ope_evaluation_audit.json`; prequential replay with regret |
| `telemetry.py` | `log_jsonl`, `training_logger` | Thread-safe timestamped JSONL appender (`training_metrics.jsonl`) |
| `visualization.py` | `plot_data_distributions`, `plot_correlations_importance`, `plot_training_curves`, `plot_model_comparison`, `plot_replay`, `plot_segment_heatmap` | Six 300-DPI figures into `output/plots/` |
| `api.py` | `app`, `GuestContext`, `PredictResponse`, `RewardFeedback`, `/predict_nba`, `/update_reward`, `/health`, `/model_info`, `/dashboard`, `EXPLORATION_EPS=0.05`, `UPDATE_EVERY=25` | Loads best model from `model_registry.json`; serves action with logged propensity; pending-request map capped at 20000; online update every 25 rewards; inline HTML dashboard |
| `screenshots.py` | `capture(port=8765)` | Starts uvicorn subprocess, captures `/docs` and `/dashboard` with headless Chromium |

## 5. Inputs, outputs & dependencies
| Kind | Item | Notes |
|---|---|---|
| Input | `output/models/model_registry.json`, `preprocessor.joblib`, per-model weight files | Required by `api.py` at startup (lifespan); the API fails to start if absent |
| Input | Request body `GuestContext` (10 fields, validated ranges) | `lead_time` 0-365, `los` 1-21, `ancillary_rate` 0-1, `month` 1-12, `party_size` 1-10 |
| Output | `data/guest_sessions.csv`, `output/models/*`, `output/logs/*.json[l]`, `output/plots/01..06*.png`, `output/screenshots/*.png` | Directories are created by `config.py` on import |
| Dependencies | See `../requirements.txt`: numpy, pandas, scikit-learn, torch (CUDA recommended), xgboost>=2.0, joblib, matplotlib, seaborn, fastapi, uvicorn, pydantic>=2, playwright, requests, pytest | No environment variables are read anywhere in `src/`; all settings are module constants |

## 6. Usage & verification commands (text only)
```
python run_pipeline.py                                   # full train + OPE + replay + plots + screenshots
python run_pipeline.py --quick --n 8000 --rounds 8 --skip-screenshots
python -m pytest tests                                   # from the project root
uvicorn src.api:app --port 8000                          # /docs, /dashboard, /health, /model_info
python -m src.data_generator                              # prints describe() of a generated dataset
python -m src.screenshots                                 # needs playwright chromium installed
```
Verify: `output/models/model_registry.json` lists `best_model`; `output/logs/ope_evaluation_audit.json` holds the OPE table.

**Notes / findings:** `api.py` imports `ACTION_SHORT` and `DEVICE` and returns `device` in each response; `STATE["rng"]` is seeded with 0, so exploration draws are reproducible per process. `output/logs/api_requests.jsonl` holds only 2 records (one predict, one reward). The committed `training_metrics.jsonl` and audit were produced with `entry: notebook`, a key that `run_pipeline.py` does not write.
