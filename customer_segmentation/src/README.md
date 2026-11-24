# src - Segmentation pipeline package

## 1. Purpose & scope
Python package for the guest-segmentation system. It generates synthetic hospitality data, engineers features, trains and compares six models, selects a winner, and serves it via FastAPI. It owns all executable logic. Config lives in `../config/config.yaml`. Tests live in `../tests/`. Project-level setup and quickstart are in `../README.md`.

## 2. Subfolders at This Level
| Subfolder | Purpose | High-level metrics |
|---|---|---|
| [models/](models/README.md) | Six clustering/representation model implementations | 4 modules, 7 model classes; winner DEC Autoencoder (composite 1.75) |

## 3. High-level metrics & health
| Item | Value |
|---|---|
| Python modules | 9 here (`api`, `data_generator`, `eda_visualizer`, `evaluate_and_compare`, `feature_engineering`, `train_pipeline`, `utils`, plus `__init__`) plus 5 in `models/` |
| Test files / cases | 1 file (`../tests/test_smoke.py`), 4 test functions, 5 cases (one is parametrised x2) |
| Entry points | `python -m src.train_pipeline [--quick] [--cpu] [--config PATH]`, `uvicorn src.api:app` |
| API endpoints | `GET /health`, `POST /predict/segment`, `POST /predict/batch`, `GET /segments/personas` |

Domain metrics (committed run, `../output/`):

| Metric | Definition | Observed value | Where defined / output |
|---|---|---|---|
| Silhouette (common space) | Silhouette on the standardised features, sampled (`evaluation.silhouette_sample=8000`) | Winner DEC 0.3428 | `evaluate_and_compare.cluster_metrics`; `model_comparisons/model_benchmark_summary.csv` |
| Davies-Bouldin, Calinski-Harabasz | Standard cluster indices | DEC 1.3407 and 10838.6 | same |
| Stability ARI | Bootstrap refit, 5 bootstraps at 0.8 fraction | DEC 0.9999, K-Means 0.758 | `evaluate_and_compare.stability` |
| Composite rank | Mean rank of silhouette, CH, DB and stability | DEC 1.75 (HDBSCAN 1.50 but `can_serve=False`) | `select_best` |
| noise/VIP ratio | HDBSCAN noise share | 0.36% (90 points, 3 clusters) | `logs/evaluation_audit.log` |
| Segment sizes (winner) | Guests per persona | Spa Couples 4285, Corporate 6162, OTA 6203, Families 4487, Dormant VIPs 3863 | `model_comparisons/segment_business_kpis.csv` |
| Churn risk and LTV | Proxies per README: `mean(1-exp(-recency/365))`; LTV = spend/stay x 12m frequency x 3 yrs x (1-churn) | Churn 0.063 to 0.658; avg LTV 1626 to 45984 | `segment_kpis` |
| Per-epoch loss, grad-norm, LR, throughput | Training telemetry | `Logged at runtime` | `logs/training_metrics.jsonl` |
| API request audit | `audit_middleware` in `api.py` | `Logged at runtime` | `logs/api_server.log` |

## 4. Key files & components
| File | Main classes / functions | Responsibility |
|---|---|---|
| `train_pipeline.py` | `main` | Orchestrator: data, features, EDA, K-sweep (3-8), 6 models, evaluation, registry, `guests_segmented.csv`. |
| `data_generator.py` | `generate_dataset(n_guests, seq_len, seed, overlap=0.12)`, `build_and_save` | Synthetic guests, stay events and sequences. Writes `data/guests.csv`, `stay_events.csv`, `sequences.npz`. |
| `feature_engineering.py` | `engineer_features`, `FeaturePipeline`, `SequenceNormalizer` | Derived features, scaling and encoding (42 features in the registry), sequence normalisation. |
| `evaluate_and_compare.py` | `cluster_metrics`, `stability`, `measure_latency`, `name_clusters`, `segment_kpis`, `compare_models`, `select_best`, `plot_metric_comparison`, `plot_radar`, `save_benchmark`, `audit_logger` | Metrics, persona naming by Hungarian matching, selection and benchmark outputs. |
| `eda_visualizer.py` | `plot_raw_distributions`, `plot_channel_breakdown`, `plot_correlation_heatmap`, `plot_feature_importance`, `plot_latent_umap`, `plot_*_loss` | 300 DPI PNGs. |
| `api.py` | `Guest`, `StayEvent`, `BatchRequest`, `Serving`, `lifespan`, `audit_middleware` | FastAPI serving of the registry winner, with HDBSCAN VIP/outlier flag. |
| `utils.py` | `load_config`, `out_dir`, `get_logger`, `set_seed`, `get_device`, `log_gpu_memory`, `timed`, `JsonlMetricsLogger`, `grad_norm` | Config, logging, device and seeding helpers. |

## 5. Inputs, outputs & dependencies
| Kind | Detail |
|---|---|
| Inputs | `../config/config.yaml` (`seed=42`, `use_gpu=true`, `n_guests=25000`, `n_clusters=5`, `latent_dim=8`, `batch_size=512`, `lr=0.001`, `quick.*` overrides) |
| Outputs | `../data/*.csv`, `../data/sequences.npz`. `../output/{logs, data_visualizations, feature_visualizations, training_losses, model_comparisons, saved_models}`. `saved_models/model_registry.json` (version `20261120-111339`, 42 features). |
| Dependencies | numpy, pandas, scipy, scikit-learn>=1.3, torch, matplotlib, seaborn, umap-learn, joblib, pyyaml, fastapi, uvicorn, pydantic, playwright, requests, pytest, httpx (`../requirements.txt`). Optional `cuml-cu12` and `hdbscan` on Linux/WSL2. |

## 6. Usage & verification commands
```text
python -m src.train_pipeline            # full run, from customer_segmentation/
python -m src.train_pipeline --quick    # smoke run
python -m src.train_pipeline --cpu      # CPU fallback
uvicorn src.api:app --port 8000         # docs at /docs
pytest tests/test_smoke.py
```
