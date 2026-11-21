# Next-Best-Action Audience Optimization — Contextual Bandits (Hospitality & Resorts)

End-to-end NBA engine: synthetic confounded guest logs → 10 contextual-bandit policies (8 algorithm families, PyTorch/XGBoost on CUDA) → off-policy evaluation (DM / IPS / SNIPS / DR) → online replay → artifacts → FastAPI service + dashboard.

## Quick start
```bash
pip install -r requirements.txt && python -m playwright install chromium
python run_pipeline.py                # full run (~3-5 min on an RTX 3090)
python run_pipeline.py --quick --n 8000 --rounds 8 --skip-screenshots   # smoke run
python -m pytest tests                # unit/smoke tests
uvicorn src.api:app --port 8000       # /docs  and  /dashboard
```

## Problem setup
* **Context X**: loyalty tier, lead time, length of stay, booking channel, historical spend, ancillary attach rate, month/day-of-week, party size, guest archetype.
* **Actions A**: 0 control · 1 10% discount · 2 room upgrade · 3 $50 dining/spa credit · 4 late checkout + transfer · 5 double points.
* **Reward R**: net contribution margin = `1[book] · (0.65·room_rev·(1−discount) + 0.45·ancillary − action_cost)` with non-linear, interacting uplifts (e.g. family×dining, business×late-checkout, couple×upgrade, OTA/Bronze×discount).
* **Logging policy**: softmax over context-dependent preferences + 12% uniform mix → confounded but with full overlap (min propensity ≥ 0.02). The simulator also stores the *true expected margin of every action* (`true_r_a*`) so OPE estimates can be validated against ground truth and regret is exact.

## Policies (`src/models/`)
| Family | Class | Notes |
|---|---|---|
| Linear | `LinUCB` (disjoint, hybrid), `LinearThompsonSampling` | exact batch ridge / Bayesian posteriors in float64 on GPU; hybrid uses shared coefficients on context⊗arm-attribute features |
| Tree | `XGBoostEpsilonGreedy`, `XGBoostUCB` | `device="cuda"`; one booster on `[x, onehot(a)]`; UCB from bootstrap ensemble std |
| Neural | `NeuralEpsilonGreedy`, `NeuralLinearBandit`, `NeuralTS`, `NeuralUCB` | MLP + BatchNorm + Dropout; Bayesian last layer; MC-dropout deep-ensemble TS; diagonal-Fisher gradient UCB (last layer + head, `torch.func`) |
| Policy gradient | `DoublyRobustPolicyGradient` | softmax policy maximising a cross-fitted, weight-clipped DR objective |

All share `BaseBandit`: `fit` (offline), `update` (incremental), `select_action`, `action_probs`, `save`/`load_model`.

## Evaluation (`src/evaluation.py`)
DM uses a separate sklearn `HistGradientBoosting` reward model (different family from the bandits). IPS/SNIPS/DR use clipped importance weights (cap 20), 200-sample bootstrap CIs, effective sample size, and the simulator ground truth (`TRUE`). Online replay is *progressive*: each round the current policy acts on fresh contexts (scored against ground-truth margins → regret), then learns from that round's logged feedback `(a, r, propensity)`.

## Artifacts
* `output/models/` — `.pt`, XGBoost `.json`, NumPy `.npz`, `preprocessor.joblib`, `model_registry.json`
* `output/logs/` — `training_metrics.jsonl`, `ope_evaluation_audit.json`, `api_requests.jsonl`
* `output/plots/` — `01…06` PNGs at 300 DPI
* `output/screenshots/` — Playwright captures of `/docs` and `/dashboard`
* `data/guest_sessions.csv` — generated dataset

## API (`src/api.py`)
`POST /predict_nba` → action, serving propensity (5% uniform exploration mixed in), per-action expected margin, request id · `POST /update_reward` → feedback loop (model updated online every 25 rewards) · `GET /dashboard` · `GET /health` · `GET /model_info`. The best model by DR value is served.

## Caveats
Data is synthetic; absolute dollar values are illustrative. NeuralUCB uses a diagonal (not full) gradient covariance. OPE estimates have variance for deterministic target policies — read DR/SNIPS alongside the CI bars and the ground-truth column.
