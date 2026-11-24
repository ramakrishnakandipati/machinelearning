# `src/` — Discovery pipeline library

## Purpose & scope

**Why it exists.** `src/` is the importable library behind the Retail Product Discovery & Contextual-Bandit Re-Ranking system: synthetic retail world and ground truth, Stage 1–3 models (two-tower retrieval, XGBoost ranker), the Stage 4 **discovery engine** (utility, guardrails, LinUCB bonus, MMR, propensities), evaluation (metrics, off-policy evaluation) and the FastAPI service. The project-level overview is in the [project README](../README.md).

**Boundaries.** `src/` owns all logic. It does **not** own entry points that run training or reporting (those are in [`../scripts/`](../scripts/README.md)), hyper-parameters (`../config/config.yaml`), or generated artifacts (`../output/`).

## Subfolders at This Level

| Subfolder | Purpose | High-level metrics (verified in the shipped run unless noted) |
|---|---|---|
| `data/` | `synthetic_retail.py`: 10k-SKU catalog (15 categories x 4 subcategories), 2,000 users, 90-day history, hidden ground truth, `SimEnvironment`; `state.py`: live per-user novelty, purchase and fatigue state | 1,500 cold-start SKUs (15%), 600,000 historical impressions, CTR 6.4%, buy rate 1.8% (`output/logs/training.log`) |
| `models/` | `two_tower.py` (Stage 1 retrieval), `primary_ranker.py` (Stage 3 calibrated P(click)/P(buy), cold-start gap analysis) | Recall@200 0.185, ColdStart_Recall@200 0.022, p99 retrieval 0.17 ms; click ROC-AUC 0.633, NDCG@5 0.345 (`output/metrics/stage1_retrieval_metrics.json`, `stage3_ranker_metrics.json`) |
| `reranker/` | Stage 4 discovery engine: `utility.py`, `guardrails.py`, `bandit.py`, `diversity_mmr.py`, `pipeline.py` | 4 policies simulated; LinUCB + MMR keeps 96.6% of pure-ranker reward with 12x cold-start impressions (`output/metrics/stage4_policy_summary.json`) |
| `evaluation/` | `metrics.py` (ranking, discovery, diversity, exposure, latency) and `ope.py` (IPS, SNIPS, DR, Switch-DR, ESS) | IPS/SNIPS/DR within 0.6–1.6% of ground truth; DM +34–40% (`output/metrics/ope_results.json`) |
| `api/` | `main.py`: `POST /discover`, `POST /feedback`, `GET /health`, `GET /metrics`, plus `/demo` and `/dashboard` HTML views | Per-request latency by stage reported at runtime; example responses in `output/metrics/api_examples.json` |

(Child folders have no README of their own; they are described here.)

## High-level metrics & health indicators (current folder)

| Item | Value | Source |
|---|---|---|
| Python modules in `src/` (excluding `__init__.py`) | 15 (3 at this level, 12 in subpackages) | directory listing |
| Unit tests exercising `src/` | 12 test functions in `../tests/test_core.py` | `../tests/test_core.py` |
| Stage latency (LinUCB + MMR, p50 / p99, ms) | total 18.7 / 26.2; ranking 7.6 / 11.4; MMR 5.5 / 8.9; retrieval 0.55 / 1.75 | `output/metrics/latency_by_stage_policy.csv` |

**Tracked runtime KPIs defined here** (values logged at runtime): per-stage latency (`utils.StageTimer`), GPU memory (`utils.gpu_info`), filter and guardrail drop counts (`pipeline.drop_totals`, `guardrails.Guardrails.totals`), propensity records (`pipeline.log_propensity`).

## Key files & components (this level)

| File | Main classes / functions | Responsibility |
|---|---|---|
| `utils.py` | `load_config`, `get_device`, `set_seed`, `JsonlLogger`, `StageTimer`, `gpu_info`, `save_json` | Config loading, CUDA/CPU selection, structured JSONL logging, CUDA-synchronised stage timers |
| `system.py` | `train_all`, `save_models`, `load_system`, `warm_start_bandits`, `day_rows` | Trains retriever and ranker, persists/reloads the full serving system, offline warm start of the bandits |
| `simulation.py` | `run_policy_simulation` | Online simulation of one policy against the ground-truth environment with common random numbers |

## Inputs, outputs & dependencies

| Direction | Item | Detail |
|---|---|---|
| Input | `../config/config.yaml` | Data sizes, windows, utility weights, bandit, MMR, guardrail, policy and OPE settings |
| Output | `../output/models/` | `two_tower_retriever.pt`, `xgb_click_ranker.json`, `xgb_buy_ranker.json`, `ranker_calibrators.pkl`, `linucb_bandit_state.pt`, `world.pkl` |
| Output | `../output/logs/`, `../output/metrics/` | Training, inference, guardrail and propensity logs; metric JSON/CSV/parquet |
| Dependencies | torch (CUDA), xgboost, scikit-learn, pandas, numpy, pyyaml, fastapi, pydantic | See `../requirements.txt` |

## Usage & verification

```python
# from the project root
from src.utils import load_config
from src.system import load_system          # needs output/models/ produced by scripts/train_and_simulate.py
S = load_system(load_config())
out = S["pipeline"].recommend(7, "linucb_mmr")   # one discovery slate with score decomposition
```

```bash
python -m pytest tests -q                    # unit tests for bandit, MMR, utility, metrics and OPE
uvicorn src.api.main:app --port 8000         # serve the API
```
