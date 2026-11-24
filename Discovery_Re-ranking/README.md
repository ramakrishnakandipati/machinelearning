# Discovery & Re-Ranking

## Purpose

Product **discovery** for retail: a relevance ranker systematically under-scores new products, so this area adds an exploration layer on top of retrieval and ranking. It re-ranks candidates with a multi-objective utility, hard business guardrails, a LinUCB exploration bonus and MMR diversity, and logs Boltzmann propensities so policies can be evaluated off-policy.

## Subfolders at This Level

| Subfolder | Purpose | High-level metrics (verified) |
|---|---|---|
| [`retail-discovery-bandit/`](retail-discovery-bandit/README.md) | The full system: synthetic retail world, two-tower retrieval, XGBoost ranker, discovery engine, evaluation, FastAPI service, executed notebook | Cold-start score gap -38% at equal relevance; LinUCB + MMR keeps 96.6% of pure-ranker reward with 12x cold-start impressions; IPS/SNIPS/DR within 0.6–1.6% of ground truth; p99 latency 26 ms; 12 unit tests |
| [`presentation/`](presentation/README.md) | 24-slide deck on the discovery flow, engine mechanics and discovery metrics | 24 slides, built from the project's metric files |

## Where to start

| Goal | Open |
|---|---|
| Understand the system and reproduce results | [`retail-discovery-bandit/README.md`](retail-discovery-bandit/README.md) |
| Walk one request through the engine interactively | `retail-discovery-bandit/notebooks/discovery_engine.ipynb` |
| Present it | `presentation/Discovery_Engine_Flow_and_Metrics.pptx` |
