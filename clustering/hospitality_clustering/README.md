# hospitality_clustering - Guest persona clustering (6 models) with API

## 1. Purpose & scope
Flat script project that simulates hotel-guest behaviour (or loads your PMS/CRM CSV), preprocesses it, fits six clustering models (CPU or GPU), evaluates and compares them, names personas, produces plots, and serves predictions through FastAPI. Users are CRM and marketing analysts. It does not contain model-selection logic beyond rank aggregation, and it ships no tests.

Models:
- K-Means
- GMM
- Agglomerative (centroid-based predictor)
- DBSCAN/density
- DEC (denoising autoencoder + KL(P||Q) fine-tuning)
- VAE+GMM (beta-VAE latent, GMM)

## 2. Subfolders at This Level
| Subfolder | Purpose | High-level metrics |
|---|---|---|
| `output/data/` | `guests.csv` (simulated data) | 1 file |
| `output/models/` | Fitted models: `preprocessor.joblib`, `kmeans`, `gmm`, `agglomerative`, `dbscan` (`.joblib`), `autoencoder_dec.pt`, `vae_cluster.pt`, `vae_gmm.joblib`, `persona_map.json` | 10 files |
| `output/logs/` | `pipeline.log`, `training_losses.jsonl`, `metrics_summary.json` | 3 files |
| `output/plots/` | `01_eda_distributions` to `06_cluster_persona_heatmap` PNGs | 6 PNGs |
| `output/api_screenshots/` | Swagger, health and predict screenshots | 3 PNGs |

## 3. High-level metrics & health
| Item | Value |
|---|---|
| Python files | 8 (`api`, `config`, `data_pipeline`, `models_dl`, `models_ml`, `run_pipeline`, `visualize_and_eval`, `capture_api_screenshots`) |
| Tests | 0 test files, 0 cases |
| Entry points | `python run_pipeline.py [--quick] [--skip-api-shots] [--n-guests N] [--csv PATH] [--k K] [--epochs E]`, `uvicorn api:app --port 8000` |
| API endpoints | `GET /health`, `GET /models/metrics`, `POST /predict?model=<name>` |
| Hardware observed | `cuda`, NVIDIA GeForce RTX 3090; pipeline "Done in 70.3s" and "71.2s" on two logged runs (`output/logs/pipeline.log`) |

**Verified (observed)**, from `output/logs/metrics_summary.json` (train features, `_device=cuda`, best model `kmeans`, best K=5):

| Model | K | Noise | Silhouette | Calinski-Harabasz | Davies-Bouldin | ARI vs simulated truth | Latency ms/1k |
|---|---|---|---|---|---|---|---|
| kmeans (best) | 5 | 0 | 0.3758 | 4947.5 | 1.124 | 0.9944 | 0.130 |
| gmm | 5 | 0 | 0.3726 | 4941.7 | 1.126 | 0.9994 | 0.771 |
| agglomerative | 5 | 0 | 0.3741 | 4947.5 | 1.124 | 0.9944 | 0.230 |
| dbscan | 2 | 5.17% | 0.3551 | 4516.4 | 1.255 | 0.4248 | 0.113 |
| dec | 5 | 0 | 0.3737 | 4942.6 | 1.125 | 0.9985 | 0.664 |
| vae_gmm | 5 | 0 | 0.3712 | 4943.5 | 1.125 | 0.9934 | 1.333 |

- Latent-space silhouette: DEC 0.7806 and VAE+GMM 0.6129 (not comparable to the table above).
- K selection silhouette: K=3 0.3723, K=5 0.3713, K=2 0.3463. `best_k=5` equals `DEFAULT_K`. The silhouette for K=3 is marginally higher (see findings).
- `ari_vs_simulated_truth` is a synthetic-data sanity check only.

**Tracked/target KPIs**
| Metric | Definition | Where |
|---|---|---|
| Silhouette | Sampled to 5000 points, non-noise only | `visualize_and_eval._metrics_for` |
| Calinski-Harabasz, Davies-Bouldin | Standard | same |
| Best-model rank | `rank(sil desc)+rank(CH desc)+rank(DB asc)` among servable models (kmeans, gmm, agglomerative, dec, vae_gmm) | `pick_best` |
| Latency | ms per 1k rows of `predict_proba`, 5 repeats | `models_ml.timed_predict_ms_per_1k` |
| Training losses | Per-epoch, `Logged at runtime` | `output/logs/training_losses.jsonl` |

Persona names (kmeans): Weekend Family Getaway, Business Road Warrior, Luxury Leisure Spender, Budget OTA Shopper, Loyal Direct Regular (`output/models/persona_map.json`).

## 4. Key files & components
| File | Main classes / functions | Responsibility |
|---|---|---|
| `config.py` | `set_seed`, `setup_logging`, `log_hardware`; constants `FEATURES` (13), `LOG_FEATURES`, `RANGES`, `DL`, `HDBSCAN` | Paths, hyperparameters, device and cuML detection. |
| `data_pipeline.py` | `simulate_guests`, `load_or_simulate`, `build_preprocessor`, `prepare` | Persona-based simulation or CSV load, log1p plus scaling, train/test split. |
| `models_ml.py` | `CentroidModel`, `select_k`, `fit_kmeans`, `fit_gmm`, `fit_agglomerative(max_n=6000)`, `fit_density`, `fit_all` | Classical models. Optional cuML. |
| `models_dl.py` | `DenoisingAE`, `DEC`, `DECPredictor`, `BetaVAE`, `VAEPredictor`, `train_dec`, `train_vae`, `load_dec`, `load_vae`, `fit_all` | Deep models. |
| `visualize_and_eval.py` | `evaluate`, `pick_best`, `build_personas`, `make_all_plots` | Metrics, Hungarian persona naming (5 archetypes), 6 plots. |
| `run_pipeline.py` | `main` | 5-step orchestrator. Exits 1 if expected outputs are missing. |
| `api.py` | `GuestFeatures` (range-validated), `lifespan`, `health`, `models_metrics`, `predict` | FastAPI serving. Returns 503 until artifacts exist. |
| `capture_api_screenshots.py` | `capture` | Playwright screenshots of the API. |

## 5. Inputs, outputs & dependencies
| Kind | Detail |
|---|---|
| Inputs | Simulated data (`N_GUESTS=10000`, `SEED=42`) or `--csv` with the 13 feature columns |
| Config (`config.py`) | `K_RANGE=2..10`, `DEFAULT_K=5`, `DL`: `latent_dim=8`, `hidden=(64,32)`, `noise_std=0.1`, `batch_size=256`, `lr=1e-3`, `ae_epochs=60`, `dec_epochs=40`, `dec_update_interval=5`, `dec_tol=0.001`, `vae_epochs=80`, `beta=1.0`, `kl_warmup=10`. HDBSCAN: `min_cluster_size=150`, `min_samples=15`. `--quick`: `ae_epochs=8`, `dec_epochs=10`, `vae_epochs=10`, `kl_warmup=3`, 3000 guests. No environment variables are used. |
| Outputs | `output/{data,models,logs,plots,api_screenshots}` |
| Dependencies | torch, scikit-learn>=1.3, pandas, numpy, scipy, umap-learn, matplotlib, seaborn, fastapi, uvicorn, pydantic>=2, joblib, httpx, playwright. Optional `cuml-cu12`, `cudf-cu12`. |

## 6. Usage & verification commands
```text
pip install -r requirements.txt
playwright install chromium
python run_pipeline.py                  # full run; add --quick or --skip-api-shots
uvicorn api:app --port 8000             # http://127.0.0.1:8000/docs
```
Verify: the last log line reads `Done in ...s. Missing outputs: none`. `GET /health` returns `status: ok`.
