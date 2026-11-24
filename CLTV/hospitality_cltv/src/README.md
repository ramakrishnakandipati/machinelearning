# src - Hospitality CLTV library

## 1. Purpose & scope

Owns all pipeline logic: synthetic data, features, models, training, evaluation, plotting, logging and the FastAPI service. It delegates orchestration to `../run_pipeline.py`, settings to `../config/config.yaml`, and screenshot capture to `../scripts/`.

## 2. Subfolders at This Level

| Subfolder | Purpose | High-level metrics |
|---|---|---|
| `models/` | One class per CLTV paradigm plus shared utilities (no README) | 6 modules + `__init__.py`, 7 model classes |
| `__pycache__/` | Tracked bytecode (cpython-312); should be ignored | n/a |

## 3. High-level metrics & health

- Files: 9 modules in `src/` (incl. `__init__.py`), 7 in `src/models/`; tests: 0.

| Metric | Definition | Value | Where |
|---|---|---|---|
| Normalized_Gini | Lorenz Gini / perfect Gini | ft_transformer_ziln 0.7490; ensemble_top3 0.7570 (test) | `evaluate.normalized_gini`; `output/logs/metrics_summary.json` |
| Top10_Revenue_Capture_pct | % revenue in top 10% scored | 49.64 (best) | `evaluate.decile_capture` |
| Calibration_Ratio | sum(pred)/sum(actual) | 0.977 (best) | `evaluate.compute_metrics` |
| RMSE, MAE, R2, MAPE_nonzero, Spearman | Standard | Logged at runtime | `evaluation_report.csv` |
| Best-model score | `Val_Normalized_Gini - 0.05*abs(log(Calibration_Ratio))`, validation only | `ft_transformer_ziln` | `evaluate_all` |
| Tier cut-offs | Validation quantiles 0.95/0.80/0.50 of best-model preds | VIP 3748.85, High 1344.98, Medium 416.95 | `output/models/tier_cutoffs.json` |
| train_time_s, latency_ms_per_1k | Wall time; median of 3 reps | Logged at runtime | `train._latency_ms_per_1k`, `timing.json` |
| Epoch loss / lr / gpu_mem_mb | `log_epoch` | Logged at runtime | `training_history.jsonl` |

## 4. Key files & components

| File | Main classes/functions | Responsibility |
|---|---|---|
| `logger.py` | `load_config`, `ensure_dirs`, `get_logger`, `append_jsonl`, `log_epoch`, `get_device`, `gpu_memory_mb` | Config, console + `execution.log`, JSONL, CUDA selection |
| `data_generator.py` | `generate`, `_guests`, `_simulate_times`; constants `CHANNELS, PROP_TIERS, REGIONS, ROOMS, TIERS` | Synthetic guests and bookings |
| `features.py` | `Dataset`, `build_features`, `build_sequences`, `build_targets`, `build_dataset`, `encode_cats`, `tree_columns` | Leakage-safe features, targets, sequence tensors |
| `models/common.py` | `Inputs`, `NumPrep`, ZILN loss, torch trainer, `subset` | Shared model utilities |
| `models/probabilistic.py` | `BGNBDGammaGamma` | BG/NBD + Gamma-Gamma baseline |
| `models/classical_ml.py` | `XGBTweedie`, `CatBoostTweedie`, `LGBHurdle` | GBDT models (XGBoost has SHAP `contribs`) |
| `models/deep_tabular.py` | `FTNet`, `FTZILN` | FT-Transformer with ZILN head |
| `models/deep_sequential.py` | `SeqNet`, `SeqZILN` | GRU + attention over stays |
| `models/deep_survival.py` | `SurvNet`, `WeibullSurvival` | Weibull cure model x log-normal value |
| `train.py` | `train_all`, `MODEL_ORDER` | Train, time, save, predict val/test |
| `evaluate.py` | `compute_metrics`, `evaluate_all`, `decile_table` | Metrics, ensemble, best model, tiers |
| `visualize.py` | `plot_eda`, `plot_features`, `plot_training`, `plot_comparisons` | 300 DPI plots |
| `api.py` | `app`, `health`, `predict_guest`, `predict_batch`, `_score`, `telemetry` middleware | FastAPI inference (VERSION 1.0.0) |

## 5. Inputs, outputs & dependencies

| Kind | Detail |
|---|---|
| Inputs | `config/config.yaml`; for API: `output/models/*`, `output/logs/metrics_summary.json`, `output/data/bookings.parquet` |
| Outputs | `output/data/*.parquet|npy`, `output/models/*`, `output/logs/*`, `output/plots/**` |
| Dependencies | numpy, pandas, scipy, torch, xgboost, lightgbm, catboost, shap, matplotlib, seaborn, pyyaml, pyarrow, joblib, fastapi, pydantic |
| API behavior | `/predict/*` returns HTTP 422 if any stay is on/after `as_of_date`, 400 for unknown model |

## 6. Usage & verification

```
python ../run_pipeline.py [--n-guests N] [--skip-screenshots]
uvicorn src.api:app --port 8000     # run from the project root
```

Verification: no unit tests exist; check `output/logs/execution.log` for "missing artifacts: none".
