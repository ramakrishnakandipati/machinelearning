# src/models - Clustering model implementations

## 1. Purpose & scope
Holds the six clustering/representation models compared by `src/train_pipeline.py`. It owns model classes plus their train/save/load helpers. It delegates data generation, feature engineering, metrics, model selection and serving to the parent `src/` package (see `../README.md`).

## 2. Subfolders at This Level
None (only `__pycache__/`, which is ignored by `.gitignore`).

## 3. High-level metrics & health
| Item | Value |
|---|---|
| Python modules | 4 (`classical_ml.py`, `deep_autoencoder.py`, `deep_vae.py`, `sequence_transformer.py`) plus `__init__.py` |
| Model classes | 7 (`KMeansGPU`, `GMMGPU`, `HDBSCANModel`, `DenoisingAE`, `DEC`, `BetaVAE`, `SeqTransformerAE`) |
| Dedicated tests | None. `tests/test_smoke.py` covers `KMeansGPU`, `GMMGPU` and `HDBSCANModel` only. The deep models have no direct test. |

Domain metrics (from `../../output/model_comparisons/model_benchmark_summary.csv`; common standardised feature space; Verified, observed on the committed run):

| Model | Silhouette | Davies-Bouldin | Calinski-Harabasz | Stability ARI | Train ms | Infer ms | Composite rank (lower is better) |
|---|---|---|---|---|---|---|---|
| K-Means (GPU) | 0.2694 | 1.989 | 9308.1 | 0.758 | 46.1 | 0.41 | 4.25 |
| GMM (GPU EM) | 0.2037 | 3.505 | 8112.3 | 0.737 | 48.2 | 0.32 | 5.25 |
| HDBSCAN (3 clusters, 0.36% noise) | 0.3687 | 1.117 | 12808.8 | 0.993 | 7790.6 | 1.37 | 1.50 (`can_serve=False`) |
| DEC Autoencoder (selected) | 0.3428 | 1.341 | 10838.6 | 0.9999 | 33800.9 | 2.16 | 1.75 |
| Beta-VAE + K-Means | 0.3416 | 1.341 | 10749.3 | 0.9998 | 18239.2 | 2.49 | 2.75 |
| Sequence Transformer | 0.0998 | 3.879 | 4570.5 | 0.964 | 14973.8 | 32.80 | 5.50 |

Per-epoch loss, grad-norm, LR and throughput are written by `JsonlMetricsLogger` to `output/logs/training_metrics.jsonl`. Individual values are `Logged at runtime`.

## 4. Key files & components
| File | Main classes / functions | Responsibility |
|---|---|---|
| `classical_ml.py` | `KMeansGPU` (k-means++, Lloyd, `predict_proba`), `GMMGPU` (diagonal EM, `bic`), `HDBSCANModel` (PCA to `pca_dim=10`, `min_cluster_size=150` default, `predict`), `_sqdist` | PyTorch GPU/CPU clustering. `HAS_CUML` is imported by the pipeline for the optional cuML path. Save and load via joblib. |
| `deep_autoencoder.py` | `mlp`, `DenoisingAE`, `DEC`, `train_dec`, `dec_predict`, `save_dec`, `load_dec` | Denoising AE pretraining, then DEC fine-tuning (Student-t soft assignment, target distribution P). |
| `deep_vae.py` | `BetaVAE`, `train_vae`, `vae_latents`, `save_vae`, `load_vae` | KL-annealed beta-VAE. Cluster on the latent mean mu with a K-Means cluster file. |
| `sequence_transformer.py` | `SeqTransformerAE`, `train_seq`, `seq_embeddings`, `save_seq`, `load_seq` | Transformer autoencoder over the last-10-stay sequences (continuous and property inputs plus a mask). Embeddings are clustered with K-Means. |

## 5. Inputs, outputs & dependencies
| Kind | Detail |
|---|---|
| Inputs | Standardised feature matrix `X` (42 features in the committed registry), or sequence tensors `(cont, prop, mask)`. Hyperparameters come from `config/config.yaml`: `model.latent_dim=8`, `model.hidden=[128,64]`, `training.*`. |
| Outputs | `output/saved_models/`: `kmeans.joblib`, `gmm.joblib`, `hdbscan.joblib`, `dec_autoencoder.pt`, `beta_vae.pt`, `beta_vae_kmeans.joblib`, `sequence_transformer.pt`, `sequence_kmeans.joblib`. |
| Dependencies | `torch`, `numpy`, `scikit-learn`, `joblib`, optional `hdbscan` or `cuml`. `src.utils.JsonlMetricsLogger` for metrics. |

## 6. Usage & verification commands
```text
python -m src.train_pipeline            # trains all six models (from customer_segmentation/)
python -m src.train_pipeline --quick    # smoke run
pytest tests/test_smoke.py              # covers KMeansGPU, GMMGPU, HDBSCANModel
```
