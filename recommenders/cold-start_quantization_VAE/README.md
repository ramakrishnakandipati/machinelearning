# Omnichannel retail cold-start recommender: VQ-VAE / RQ-VAE Semantic IDs

```bash
pip install -r requirements.txt && playwright install chromium
python run_pipeline.py              # data -> train -> eval -> charts -> FastAPI -> screenshots (~3.5 min on an RTX 3090)
python run_pipeline.py --quick      # smoke test
uvicorn src.api.app:app --port 8000 # serve after a pipeline run; open /docs and /ui/test-harness
```

Everything lands in `outputs/` (`models/`, `logs/`, `visualizations/`, `api_screenshots/`).

## What is implemented
* **Data** (`src/data/generate_data.py`): 5k users, 2k items, ~50k interactions over web / app / in-store; 10% of items and 10% of
  users held out of training; Day-0 evaluated with no logs. Item text is embedded with TF-IDF+SVD, so a new SKU's description can be encoded at serving time.
* **Tokenizers**: `VQVAE` (EMA codebook, STE, commitment loss, dead-code reset) and `RQVAE` (4 x 256, K-Means++ init on the
  first batch, per-level dead-code reset). Both are trained on warm items only; cold items are encoded inductively.
* **Recommenders**: (1) semantic trie with prefix fallback, (2) causal Transformer over Semantic-ID tokens with trie-constrained beam search,
  (3) metadata -> codebook-space two-tower trained with InfoNCE + per-level code prediction (dot / overlap / fused scoring),
  (4) hybrid FAISS content kNN + collaborative smoothing + segment/popularity/margin re-ranking (MMR for Day-0).
  Baselines: random, popularity, raw-feature kNN, plus VQ-VAE variants and bridge ablations.
* **API**: `/health`, `/recommend/{item,user,system}-cold-start`, `/recommend/compare-approaches`, `/semantic-id/inspect/{id}`, `/ui/test-harness`.

## Results (Recall@10, 800 sampled requests per regime; see `outputs/logs/evaluation_metrics.csv`)
| approach | warm | item cold | user cold | system cold |
|---|---|---|---|---|
| 1 semantic trie | 0.075 | 0.348 | 0.047 | 0.015 |
| 2 generative seq | 0.118 | 0.416 | 0.079 | 0.021 |
| 3 codebook bridge | 0.094 | 0.533 | 0.054 | 0.024 |
| 4 hybrid | 0.120 | **0.587** | **0.084** | 0.011 |
| popularity (uses logs) | 0.031 | 0.059 | 0.020 | 0.022 |
| raw-feature kNN | 0.088 | 0.541 | 0.056 | 0.017 |
| random | 0.006 | 0.059 | 0.007 | 0.007 |

Caveats: the data is synthetic, so absolute numbers are not transferable. Raw-feature kNN is a strong item-cold baseline (quantization
mainly buys discrete IDs, hierarchy and generative decoding, not accuracy). System cold-start is bounded by a single landing-category
signal; the Day-0 hybrid trades recall for diversity and margin.
