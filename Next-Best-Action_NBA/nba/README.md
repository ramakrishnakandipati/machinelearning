# `nba/` - Retail Next-Best-Action Python package

## 1. Purpose & scope
`nba/` is the importable package behind the Retail NBA project: causal uplift modelling (EconML DR-Learner with LightGBM), margin-aware decisioning under hard retail guardrails, off-policy evaluation, epsilon-greedy exploration with logged propensities, and a FastAPI inference microservice. `config.py` is the single source of truth shared by offline training/evaluation and online serving, to avoid train/serve skew.

It owns: configuration, schemas, point-in-time features, the synthetic world and logged campaign, uplift models and persistence, guardrails and contact-fatigue stores, policy, OPE, plots, and the service. It delegates orchestration to `../run_pipeline.py`, latency measurement to `../benchmark_latency.py`, tests to `../tests/test_nba.py`, and the narrative to the [project README](../README.md) and `../Retail_NBA_Uplift.ipynb`.

## 2. Subfolders at This Level
None (only `__pycache__/`, compiled bytecode; the project has no `.gitignore`, see findings).

## 3. High-level metrics & health
**Module/code metrics**

| Item | Value |
|---|---|
| Python modules | 15 (`__init__`, `config`, `schemas`, `features`, `synthetic_data`, `uplift_model`, `exploration`, `guardrails`, `policy`, `evaluation`, `service`, `artifact_manager`, `plot_utils`, `data_plots`, `eval_plots`) |
| Tests | `../tests/test_nba.py`, 6 test functions (no-lookahead features, guardrail rules, net-value/choice, epsilon-greedy propensities, contact store, service end-to-end) |
| Language / frameworks | Python 3.12 bytecode present; numpy, pandas, scikit-learn, LightGBM, EconML, pydantic v2, FastAPI, optional Redis, matplotlib/seaborn |
| Entry points using the package | `run_pipeline.py [--n 30000] [--seed 42]`, `uvicorn nba.service:app`, `benchmark_latency.py [--n 2000]`; module self-demos in `uplift_model.py`, `synthetic_data.py`, `artifact_manager.py` |

**Domain metrics.** Verified values come from the committed `../outputs/saved_models/model_manifest.json` (model_version `nba-20261117T234417Z`; econml 0.17.0, lightgbm 4.7.0, sklearn 1.9.1).

| Metric | Definition | Observed value / threshold | Where defined / output |
|---|---|---|---|
| Propensity log-loss (3-fold CV) | Nuisance propensity quality | 1.4223 (Verified) | `uplift_model.cross_validated_nuisance_metrics`, manifest |
| Propensity AUC (OvR) | Treatment predictability | 0.6106 (Verified) | same |
| Margin outcome R2 | Outcome model fit | 0.0550 (Verified) | same |
| Unsubscribe outcome AUC | Unsub model discrimination | 0.7015 (Verified) | same |
| AUUC lift A1 / A2 / A3 / A4 | Holdout area under uplift curve | 1833.8 / 11013.1 / 11602.8 / 2527.6 (Verified) | `eval_plots.auuc`, manifest `holdout_metrics` |
| Decile Spearman A1 / A2 / A3 / A4 | CATE calibration by decile | 0.345 / 0.733 / 0.818 / -0.139 (Verified) | `eval_plots.plot_cate_calibration` |
| SNIPS net margin per customer ($) | (a) legacy CRM / (b) naive propensity / (c) unconstrained uplift / (d) guardrailed NBA | 4.71 / 5.05 / 3.93 / 5.76 (Verified) | `evaluation.evaluate_policy`, manifest |
| Selected lambda | Smallest lambda with estimated unsub rate <= `UNSUB_RATE_CAP` 0.005 | 100.0, unsub 0.00467, treat rate 0.760, margin 5.764 CI [5.224, 6.308] (Verified) | `evaluation.select_lambda`, manifest `lambda_sweep` |
| Thresholds | `SURE_THING_THRESHOLD` 0.22, `FULL_PRICE_LOYALIST_MARKDOWN_MAX` 0.25, `MAX_PROMOS_PER_WINDOW` 2 per 7 days, `MIN_SIZE_AVAILABILITY` 0.20, `EXPLORATION_EPSILON` 0.05, `COLD_START_SHRINKAGE` 0.5 | Config constants | `config.py` |
| Serving latency | `latency_ms` per request; target p99 < 40 ms | `Logged at runtime` (response field, `X-Latency-ms` header, `benchmark_latency.py`); no benchmark result is committed | `service.py` |
| ESS, bootstrap CI | `(sum w)^2 / sum w^2`, 1000-sample 95% CI, weight cap 50 | `Logged at runtime` by `run_pipeline.py` stdout | `evaluation.py` |

## 4. Key files & components
| File | Main classes / functions | Responsibility |
|---|---|---|
| `config.py` | `ACTION_IDS` (0-4), `ACTION_NAMES`, `ACTION_COSTS` (0, 0, 8, 0.5, 2), `TRIGGER_ACTIONS`, `COLD_START_*`, `FEATURE_COLUMNS` (18), `SEGMENT_ORDER`, windows (7d conversion, 30d return), `DEFAULT_LAMBDA=100`, `SEED=42` | Constants and feature-order contract |
| `schemas.py` | `CampaignLogRecord`, `DecisionLogRecord`, `validate_log_frame` | Pydantic validation (sampled 2000 rows plus vectorised range checks) |
| `features.py` | `build_features`, `assign_segment`, `FeatureTransformer` (`fit`, `transform`, `transform_dict`) | Point-in-time features (events strictly before `as_of`), median-prior imputation, hot-path single-row transform |
| `synthetic_data.py` | `generate_world`, `legacy_crm_distribution`, `simulate_campaign`, `make_dataset` | Synthetic history, legacy-CRM logging policy plus 5% exploration, delayed right-censored outcomes, `truth_*` columns, time-based maturity-aware split (test_frac 0.3) |
| `uplift_model.py` | `make_dr_learner`, `UpliftBundle` (`predict_tau`, `baseline_purchase_prob`), `fit_uplift_bundle`, `save_bundle`, `update_manifest`, `load_bundle` | Margin and unsubscribe DR-Learners (LightGBM, cv=3, min_propensity 0.01), baseline purchase model, cold-start shrinkage to global ATE, joblib + manifest persistence |
| `exploration.py` | `mix_with_uniform`, `sample_from_probs`, `EpsilonGreedyPolicy` | Epsilon-greedy over eligible actions with exact logged propensity |
| `guardrails.py` | `apply_guardrails`, `margin_protection`, `contact_fatigue`, `inventory_eligibility`, `cold_start_restriction`, `trigger_restriction`, `InMemoryContactStore`, `RedisContactStore` | Vectorised hard rules with per-rule block matrices for reason codes; fail-safe promo suppression when the store is down |
| `policy.py` | `net_value_matrix`, `choose_actions`, `decide` | `NetValue = tau_margin - cost - lambda * max(tau_unsub, 0)`; guardrail-eligible argmax, action 0 wins ties |
| `evaluation.py` | `importance_weights`, `snips`, `effective_sample_size`, `bootstrap_snips_ci`, `evaluate_policy`, `lambda_sweep`, `select_lambda`, naive and unconstrained baselines | SNIPS OPE with `MAX_WEIGHT=50`, lambda grid (0 to 3200), Pareto sweep |
| `service.py` | `app`, `RecommendRequest`, `RecommendResponse`, `/healthz`, `POST /v1/nba/recommend`, `timing_middleware` | Loads bundle once, store lookup, CATE, guardrails, exploration, background decision log and promo recording |
| `artifact_manager.py` | `ArtifactManager` (`save_figure`, `write_json`, `read_json`), `DEFAULT_ROOT` | Creates `outputs/{data_plots,validation_plots,comparison_plots,saved_models}`, 300-dpi PNG export |
| `plot_utils.py`, `data_plots.py`, `eval_plots.py` | `apply_style`, `plot_propensity_overlap`, `plot_correlations_and_drift`, `plot_margin_and_returns`, `plot_qini`, `plot_cate_calibration`, `plot_nuisance_diagnostics`, `plot_policy_comparison`, `plot_allocation_by_segment`, `plot_pareto` | Diagnostic and validation figures (9 PNGs committed under `../outputs/`) |

## 5. Inputs, outputs & dependencies
| Kind | Item | Notes |
|---|---|---|
| Input | Env `NBA_MODELS_DIR` (default `outputs/saved_models`) | Needs `model_manifest.json`, 4 joblib files, optional `serving_feature_snapshot.csv` (5000 customers) |
| Input | Env `NBA_DECISION_LOG` (default `outputs/decision_log.jsonl`) | Append-only decision log (2126 records committed) |
| Input | Env `NBA_REDIS_URL` (optional) | Falls back to in-memory contact store if unset or unreachable at startup |
| Request | `customer_id`, `trigger` in {app_open, browse_abandon, post_purchase}, optional `context_features` overrides | Unknown customers are treated as cold start |
| Output | `RecommendResponse`: action, net_value, tau_margin, tau_unsub, propensity_score, explored, cold_start, reason_codes, model_version, latency_ms | Reason codes such as `suppressed_a{k}:{rule}`, `exploration`, `cold_start_shrinkage`, `no_positive_net_value` |
| Dependencies | `../requirements.txt`: numpy, pandas, scikit-learn, lightgbm, econml, matplotlib, seaborn, joblib, pydantic>=2, fastapi, uvicorn, httpx, redis, pytest | Versions are unpinned; no GPU needed |

## 6. Usage & verification commands (text only)
```
python run_pipeline.py --n 100000          # README example; script default is 30000
python -m pytest tests -q
uvicorn nba.service:app --port 8000        # POST /v1/nba/recommend, GET /healthz
python benchmark_latency.py --n 2000       # p50/p95/p99 handler latency, in-memory store
```
Verify: `outputs/saved_models/model_manifest.json` contains `cross_validation_metrics`, `holdout_metrics`, `risk_aversion_lambda`.

**Findings:** (1) The committed manifest `holdout_metrics` lacks the `cate_truth_corr` key that `run_pipeline.py` now writes, so artifacts predate the current script; re-run to refresh. (2) `service.py` reads `n_sessions_total` from the feature row, but that column is not in `FEATURE_COLUMNS` or the snapshot, so that branch never triggers; cold start relies on `is_cold_start`. (3) Manifest `lambda_sweep` shows the margin curve is non-monotonic above lambda 400 (5.48, 5.56, 5.47, 5.52), within the CI width. (4) Decile Spearman for A4 is negative (-0.139), indicating weak ranking for the 2x-points action. (5) `__pycache__/` and `.pytest_cache/` exist and the project has no `.gitignore`.
