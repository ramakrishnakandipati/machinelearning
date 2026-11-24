# src - offline pipeline (data, training, evaluation, ANN index)

## 1. Purpose and scope

Everything that runs offline to produce the artifacts the API serves: synthetic data, feature engineering with a temporal split, Two-Tower training (GPU or CPU), held-out evaluation, and FAISS index benchmarking and persistence.

- **Owns:** data generation, preprocessors, model definition, training and export, test-set metrics, ANN index build, logging, and GPU diagnostics.
- **Delegates:** online serving, the Redis feature store and HTTP validation go to `../api/`. End-to-end verification goes to `../tests/`. Project-level setup is in `../README.md`.
- **Output root:** `../outputs/` (`logs/ metrics/ charts/ models/ data/ screenshots/`, created on import of `config.py`).

## 2. Subfolders at This Level

None apart from `__pycache__/`, which holds bytecode only and is ignored.

## 3. High-level metrics and health

| Item | Value |
|---|---|
| Python files | 11 (`__init__`, `config`, `logger`, `check_gpu`, `data_generator`, `dataset`, `model`, `train`, `evaluate`, `ann_index`, `run_pipeline`) |
| Test files | None here. Covered by `../tests/test_api_e2e.py` (7 cases, API level only) |
| Key classes | `UserTower`, `ItemTower`, `TwoTower`, `Preprocessors`, `Prepared`, `DataCfg`, `TrainCfg` |

Domain metrics. Observed values were read from committed files in `../outputs/`; these runs used synthetic data.

| Metric | Definition | Observed value / threshold | Where defined / output |
|---|---|---|---|
| Recall@10, Recall@50 | Share of positives whose rank (items scoring above the positive) is below 10 or 50 | Test overall 0.0506 and 0.1269. Popularity baseline 0.0251 and 0.0695. Random Recall@50 is 0.01 | `train.metrics_from_ranks`, `outputs/metrics/offline_test_metrics.json` |
| NDCG@10, MRR@10 | 1/log2(rank+2) and 1/(rank+1), applied only when rank < 10 | Test overall 0.0288 and 0.0222 | same |
| Warm vs cold-start users (test) | Recall@50 on seen vs unseen-in-train users | 0.1301 (n=87,662) vs 0.1078 (n=14,748) | `evaluate.test_metrics` |
| Best epoch / early stopping | Checkpoint on best val Recall@50, patience 4 | Best epoch 15, stopped at epoch 19; best val Recall@50 about 0.1286 | `outputs/metrics/training_history.json`, `03_model_training.log` |
| Model size | Parameter count | 2,163,093 | `03_model_training.log` |
| Temporal leakage check | train max day < val min day | passed (125.0 < 126.0) | `dataset.prepare`, `outputs/metrics/dataset_summary.json` |
| Dataset | users / items / events / sparsity | 20,000 / 5,000 / 1,200,044 / 99.42% | `dataset_summary.json` |
| Split sizes | train / val / test examples | 411,655 / 102,611 / 102,410 | `dataset_summary.json` |
| Hard-negative coverage (train) | Share of train rows with a hard negative | 91.42% | `dataset_summary.json` |
| ANN recall@10 vs exact | Recall of HNSW and IVF-PQ against `IndexFlatIP` | HNSW efSearch=32 0.9515, 64 0.9874. IVFPQ best 0.4076 (nprobe=32) | `ann_index.bench`, `outputs/metrics/ann_benchmark_metrics.csv` |
| Serving index choice | Smallest efSearch with recall >= 0.95, else the largest | HNSW M=32, efSearch=32, 5000 items | `outputs/metrics/serving_index.json` |
| ANN latency (CPU backend) | p50/p99 per single query | HNSW efSearch=32 0.0294 / 0.0374 ms; Flat 0.0538 / 0.0723 ms | `ann_benchmark_metrics.csv` |
| Training logged per epoch | train_loss, val_loss, val R@10/R@50/NDCG/MRR, lr, temperature, AMP scale, duration, peak GPU memory | Logged at runtime | `training_history.csv` and `.json` |

Targets in code: HNSW selection threshold recall >= 0.95; TorchScript vs eager max difference < 1e-4 (assertion in `train.export`).

## 4. Key files and components

| File | Main classes / functions | Responsibility |
|---|---|---|
| `config.py` | `DataCfg`, `TrainCfg`, paths, `EMB_DIM=128`, `HIST_LEN=20`, `SEED=42` | Shared constants; creates output dirs on import |
| `logger.py` | `get_logger`, `stage_timer`, `mem_snapshot` | stdout and rotating file logs (5 MB x 3), RSS and VRAM in stage timings |
| `check_gpu.py` | `probe`, `main` | Writes `01_gpu_diagnostics.log` and `hardware_profile.json` (nvidia-smi, torch, faiss, AMP check) |
| `data_generator.py` | `generate`, `main` | Latent-factor synthetic users, items and events, with cold-start users and items |
| `dataset.py` | `Preprocessors`, `prepare`, `_rfm_and_history`, `eda_chart` | Point-in-time RFM and history features, train-only vocabularies and scalers, temporal split (days 126 and 153), hard negatives, logQ, summary |
| `model.py` | `UserTower`, `ItemTower`, `TwoTower`, `mlp` | L2-normalised 128-d towers, learnable temperature (init 0.07, clamped 0.01-1.0) |
| `train.py` | `info_nce`, `train`, `export`, `ranks_of_positive` | In-batch InfoNCE with logQ correction, hard negatives, AdamW with warmup and cosine schedule, AMP, grad-clip 5.0, early stopping. Exports item embeddings, a TorchScript user tower and segment embeddings |
| `evaluate.py` | `test_metrics`, `charts`, `load_model` | Test metrics overall and by segment, popularity baseline, training and embedding charts (PCA and t-SNE) |
| `ann_index.py` | `build_hnsw`, `build_ivfpq`, `bench`, `run`, `main` | Benchmarks Flat, HNSW and IVF-PQ; persists the serving HNSW index and item metadata |
| `run_pipeline.py` | `main` | Sequences check_gpu, train, evaluate, ann_index |

Training defaults (`TrainCfg`): epochs 30, batch 4096, lr 2e-3, weight decay 1e-5, patience 4, dropout 0.1, `hard_neg` and `logq` both True.

## 5. Inputs, outputs and dependencies

| Direction | Item | Detail |
|---|---|---|
| Input | None external | Data is generated (`DataCfg`: 20k users, 5k items, 1.2M interactions, 180 days) |
| Input (stage re-runs) | `outputs/data/{users,items,events}.parquet` | Read by `evaluate.main` and `ann_index.main` when called standalone |
| Output | `outputs/data/` | `user_features_snapshot.parquet`, `purchased.parquet` (the API loads these) |
| Output | `outputs/models/` | `two_tower_best.pt`, `user_tower_serving.pt`, `item_embeddings.npy`, `segment_embeddings.npz`, `preprocessors.joblib`, `retail_items.faiss`, `item_metadata.parquet` |
| Output | `outputs/metrics/` | `dataset_summary.json`, `training_history.{csv,json}`, `offline_test_metrics.json`, `ann_benchmark_metrics.csv`, `serving_index.json`, `hardware_profile.json` |
| Output | `outputs/logs/`, `outputs/charts/` | `01_`-`04_*.log`; EDA, loss, retrieval, embedding and ANN charts |
| Dependencies | torch, numpy, pandas, scikit-learn, faiss (cpu or gpu), matplotlib, seaborn, joblib, psutil | See `../requirements.txt` |

## 6. Usage and verification

```
python -m src.run_pipeline       # run from the project root; all stages
python -m src.check_gpu          # diagnostics only
python -m src.dataset            # data + features only
python -m src.train              # train + export (generates data if needed)
python -m src.evaluate           # needs outputs/data/*.parquet and a checkpoint
python -m src.ann_index          # needs item_embeddings.npy and a checkpoint
```

Verify by checking that `outputs/metrics/offline_test_metrics.json` has `model.overall.recall_50` above `popularity_baseline.overall.recall_50`, and that `dataset_summary.json` has `leakage_check.passed = true`.

## 7. Audit notes

- `outputs/logs/03_model_training.log` holds two full training runs appended in one file (runs are about 1 hour apart and best Recall@50 differs slightly, 0.1286 vs 0.1287). Treat the last run as the current one.
- `ann_index.maybe_gpu` and `size_mb` contain GPU branches that the committed benchmark did not exercise (all rows show `backend=cpu`).
- `ann_index` writes `serving_index.json` as a hand-built string rather than via `json.dumps`.
- The EDA and embedding chart file names (`embedding_space_umap_pca.png`) mention UMAP, but the code draws PCA and t-SNE.
- `evaluate.main` and `ann_index.main` read the data from a relative `outputs/data/` path, so they only work when run from the project root.
- `__pycache__/` is present in the folder.
