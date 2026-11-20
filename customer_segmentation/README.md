# Hospitality Customer Segmentation

End-to-end guest segmentation: synthetic hospitality data → feature engineering → 6 models (classical, deep, sequential) trained on GPU → comparison → best model served through FastAPI. All artifacts land in `output/`.

## Quickstart
```bash
pip install -r requirements.txt
python -m playwright install chromium

python -m src.train_pipeline            # full run (~2 min on an RTX 3090)
python -m src.train_pipeline --quick    # smoke run (~50 s, 4k guests)
python -m src.train_pipeline --cpu      # force CPU fallback

uvicorn src.api:app --port 8000         # then open http://127.0.0.1:8000/docs
python tests/capture_api_screenshots.py # Swagger + response screenshots
pytest tests/test_smoke.py
```

## Models
| # | Model | Paradigm | Implementation |
|---|-------|----------|----------------|
| 1 | K-Means | Centroid | PyTorch GPU (k-means++, Lloyd); cuML if installed |
| 2 | GMM | Probabilistic | Diagonal-covariance EM in PyTorch, soft assignments |
| 3 | HDBSCAN | Density | `hdbscan` pkg or `sklearn.cluster.HDBSCAN`; noise = VIP/outlier flag |
| 4 | DEC Autoencoder | Deep | Denoising AE (BN/LeakyReLU/Dropout) + KL(P‖Q) Student-t fine-tune |
| 5 | β-VAE + K-Means | Deep generative | KL-annealed β-VAE, clustering on μ |
| 6 | Sequence Transformer | Sequential | Transformer encoder-decoder on last-10-stay sequences |

**GPU note:** RAPIDS cuML has no native Windows build, so on Windows the PyTorch CUDA implementations are the accelerated path; the code auto-uses cuML on Linux/WSL2 when importable and logs every fallback.

## Model selection
Composite = mean rank of Silhouette, Calinski-Harabasz, Davies-Bouldin and bootstrap stability (ARI), measured on the **common standardised feature space** so latent-space models are not flattered by their own geometry (own-space silhouette is reported separately). Only models that yield the target K segments are eligible for serving, so HDBSCAN (3 dense clusters + noise) acts as the VIP/outlier detector. `ari_vs_planted_archetypes` in the benchmark is a synthetic-data sanity check only.

## Outputs (`output/`)
- `logs/` – `pipeline_execution.log`, `training_metrics.jsonl` (per-epoch loss, grad-norm, LR, throughput), `evaluation_audit.log` (metrics, K-sweep, selection rationale), `api_server.log`
- `data_visualizations/`, `feature_visualizations/`, `training_losses/`, `model_comparisons/` – 300 DPI PNGs, `model_benchmark_summary.csv/.json`, `segment_business_kpis.csv`
- `saved_models/` – `scaler.joblib`, `encoder.joblib`, classical `.joblib`, PyTorch `.pt` (state_dict + architecture config), `model_registry.json`
- `api_screenshots/`

## API
`GET /health`, `POST /predict/segment`, `POST /predict/batch`, `GET /segments/personas`. Example:
```bash
curl -X POST localhost:8000/predict/segment -H 'Content-Type: application/json' \
  -d '{"loyalty_tier":"Platinum","recency_days":18,"frequency_stays_12m":9,"business_center_usage":0.8}'
```
Unspecified fields use defaults from `Guest` in `src/api.py`. `stay_history` (≤10 stays) is used only if the sequence model wins.

## Notes
- Personas are named by Hungarian matching of cluster profiles to signature rules in `evaluate_and_compare.py`; playbooks/next-best-offers come from `config/config.yaml`.
- Churn risk = `mean(1 − exp(−recency/365))`; LTV = spend/stay × 12m frequency × 3 yrs × (1 − churn). Both are simple proxies, not fitted models.
