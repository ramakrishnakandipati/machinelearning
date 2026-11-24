# Class_Inbalance - Extreme class-imbalance pipeline (GPU tree models)

## 1. Purpose & scope
Reference pipeline for binary classification with about 0.5% positives (fraud or defect style). It uses only GPU-accelerated tree models. A model whose library or GPU build is missing is excluded from the run and never silently downgraded to CPU. Stages in `imbalance_gpu_pipeline.py`:
1. GPU probe.
2. Synthetic data (or your own CSV/Parquet) and a stratified 70/15/15 split.
3. Leak-free `FeatureEngineer`, fitted on train only.
4. Optuna tuning with Stratified K-Fold, objective PR-AUC.
5. Validation-tuned F-beta threshold, test metrics and plots.

It is a standalone script and has no API and no tests. `imbalance_gpu_pipeline.ipynb` is a notebook companion to the script.

## 2. Subfolders at This Level
| Subfolder | Purpose | High-level metrics |
|---|---|---|
| `outputs/` | Run artifacts: `metrics_summary.csv` plus the three folders below | 3 models evaluated |
| `outputs/data_plots/` | EDA on train only: `class_distribution.png`, `correlation_heatmap.png`, `top4_feature_distributions.png` | 3 PNGs |
| `outputs/eval_plots/` | `pr_curves.png`, `roc_curves.png`, `confusion_matrices.png`, `threshold_curve_best_model.png`, `feature_importance.png` | 5 PNGs |
| `outputs/saved_models/` | `xgboost_gpu.json`, `catboost_gpu.cbm`, `resample_xgb_gpu_pipeline.joblib`, `feature_engineer.joblib`, `best_params.json` | 4 model/preprocessor files plus tuned params |
| `__pycache__/` | Tracked bytecode (see findings) | not an asset |

## 3. High-level metrics & health
| Item | Value |
|---|---|
| Code | 1 script (about 811 lines) and 1 notebook |
| Tests | 0 test files |
| Language / libs | Python 3.12, xgboost>=2.0 (3.4.1 in log), catboost, lightgbm (optional), imbalanced-learn, optuna, scikit-learn, pandas, numpy, scipy, matplotlib, seaborn, joblib |
| Hardware | NVIDIA GeForce RTX 3090 (24576 MiB) per `full_run.log` |

**Verified (observed)**, from `full_run.log` and `outputs/metrics_summary.csv` (test set, threshold tuned on validation):

| Model | Threshold | Val F2 | Val PR-AUC | Test PR-AUC | ROC-AUC | F1 | F2 | Precision | Recall | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost_GPU | 0.1666 | 0.625 | 0.639 | 0.586 | 0.918 | 0.593 | 0.554 | 0.672 | 0.531 | 0.595 |
| CatBoost_GPU (best) | 0.4255 | 0.667 | 0.686 | 0.625 | 0.937 | 0.643 | 0.587 | 0.763 | 0.556 | 0.649 |
| Resample+XGB_GPU | 0.0875 | 0.576 | 0.577 | 0.516 | 0.908 | 0.449 | 0.513 | 0.371 | 0.568 | 0.455 |

| Observed fact | Value | Source |
|---|---|---|
| Rows / positives | 100,000 / 542 (0.542%) | `full_run.log` |
| Split train / val / test | 69,999 / 15,001 / 15,000 rows, 380 / 81 / 81 positives | `full_run.log` |
| Train neg/pos ratio | 183.2 | `full_run.log` |
| Features | 30 raw to 74 engineered | `full_run.log` |
| Best CV PR-AUC (Optuna) | XGB 0.4820, CatBoost 0.5804, Resample+XGB 0.4424 | `full_run.log` |
| Runtime | Total 505.7 s (tuning 459.5 s; CatBoost 322.9 s) | `full_run.log` |
| LightGBM_GPU | Excluded: GPU Tree Learner not enabled in this build | `full_run.log` |

**Tracked/target KPIs.** All are printed or saved at runtime in `metrics_summary.csv`, with the threshold chosen by `tune_threshold(beta=2.0)`.

| Metric | Purpose | Defined in |
|---|---|---|
| PR-AUC (`average_precision_score`) | Optuna objective and model selection ("best model by validation PR-AUC") | `cv_pr_auc`, `evaluate` |
| F2 / F-beta (`--beta`, default 2.0) | Threshold tuning, weights recall 2x over precision | `tune_threshold`, `metrics_at` |
| ROC-AUC, F1, precision, recall, MCC | Reporting | `metrics_at` |

## 4. Key files & components
| File | Main classes / functions | Responsibility |
|---|---|---|
| `imbalance_gpu_pipeline.py` | `Config`, `ModelSpec`, `probe_gpu_models`, `load_data`, `split_data`, `eda_plots`, `FeatureEngineer`, `build_features`, `make_cv_folds`, `tune_and_train`, `tune_threshold`, `metrics_at`, `evaluate`, `eval_plots`, `main` | Whole pipeline. Specs: `_make_xgb_spec`, `_make_lgb_spec`, `_make_cat_spec`, `_make_resample_spec` (SMOTETomek or RandomUnderSampler plus XGB). |
| `imbalance_gpu_pipeline.ipynb` | n/a | Notebook version. |
| `full_run.log` | n/a | Captured full-run console output. |
| `requirements.txt` | n/a | Package list. Targets conda env `causal-gpu`. |

## 5. Inputs, outputs & dependencies
| Kind | Detail |
|---|---|
| Inputs | Synthetic `make_classification` data by default (`n_rows=100000`, `n_features=30`, `pos_weight=0.005`, `seed=42`). Optional `--data-path` (CSV or Parquet; needs a binary `target` column and numeric features). |
| Config defaults (`Config`) | `val_size=0.15`, `test_size=0.15`, `cv_folds=3`, `n_trials=15`, `n_trials_resample=6`, `max_estimators=600`, `beta=2.0`, `dpi=300`, `out_dir=outputs` |
| CLI flags | `--quick` (30k rows, 1% positives, 2 trials, 150 estimators), `--trials`, `--n-rows`, `--beta`, `--data-path`, `--out-dir` |
| Outputs | `outputs/` as described above |
| Dependencies | CUDA GPU. Required: xgboost, optuna. Optional (model is dropped if missing): catboost, lightgbm GPU build, imbalanced-learn. |

## 6. Usage & verification commands
```text
conda activate causal-gpu
pip install imbalanced-learn optuna catboost
python imbalance_gpu_pipeline.py                 # full run (about 506 s observed)
python imbalance_gpu_pipeline.py --quick         # smoke test
python imbalance_gpu_pipeline.py --data-path my.parquet --beta 2.0 --trials 30
```
Verify: `[ OK ] ... GPU probe passed` lines in the console, and `outputs/metrics_summary.csv` matches the table above.
