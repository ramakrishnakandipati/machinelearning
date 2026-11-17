# Retail Next Best Action (causal uplift + margin-aware decisioning)

```
pip install -r requirements.txt
python run_pipeline.py --n 100000      # synthetic data -> plots -> models -> validation (~30s)
python -m pytest tests -q              # leakage, guardrails, exploration, service
uvicorn nba.service:app --port 8000    # POST /v1/nba/recommend  (set NBA_REDIS_URL for Redis)
python benchmark_latency.py            # handler p50/p95/p99
```

| Module | File(s) |
|---|---|
| 1 Artifact manager | `nba/artifact_manager.py` |
| 2 Schemas, features, synthetic data, data plots | `nba/schemas.py`, `features.py`, `synthetic_data.py`, `data_plots.py` |
| 3 DR-Learner uplift + epsilon-greedy + persistence | `nba/uplift_model.py`, `exploration.py` |
| 4 Guardrails + policy | `nba/guardrails.py`, `policy.py` |
| 5 Off-policy evaluation + plots | `nba/evaluation.py`, `eval_plots.py` |
| 6 FastAPI service | `nba/service.py` |

Key choices: DRLearner (LightGBM nuisance + final stages); rewards used for training only after the
7-day conversion + 30-day return windows close; cold-start shoppers (<2 sessions) get prior-imputed
features, CATEs shrunk to the global ATE, and no discount action; lambda chosen as the smallest value
meeting the 0.5% unsubscribe cap on the holdout Pareto sweep; if Redis fails, promos are suppressed.
