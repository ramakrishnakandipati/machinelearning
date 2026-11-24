# `scripts/` — Entry points

## Purpose & scope

**Why it exists.** The two executable entry points of the discovery system. They orchestrate the library in [`../src/`](../src) end to end: train and simulate, then evaluate and report.

**Boundaries.** Scripts contain orchestration, reporting and plotting only; model, bandit, guardrail and evaluation logic lives in `src/`. Hyper-parameters come from `../config/config.yaml`.

## High-level metrics & health indicators

| Item | Value | Source |
|---|---|---|
| Scripts | 2 (`train_and_simulate.py`, `generate_artifacts.py`) | directory listing |
| Verified run time (RTX 3090) | training and 4-policy simulation about 5 minutes (316 s in the shipped run); artifacts about 2.4 minutes (143 s) | `output/logs/training.log`, `output/logs/artifacts.log` |

### Metrics each script produces

| Script | Metric group | Definition | Observed value | Output location |
|---|---|---|---|---|
| `train_and_simulate.py` | Stage 1 retrieval | Recall@50/200, HitRate@200, ColdStart_Recall@200, latency percentiles | Recall@200 0.185; ColdStart_Recall@200 0.022; p99 0.17 ms | `output/metrics/stage1_retrieval_metrics.json` |
| `train_and_simulate.py` | Stage 3 ranker | ROC-AUC, PR-AUC, LogLoss, Brier, NDCG@5/10, cold vs warm score gap | click AUC 0.633; NDCG@5 0.345; gap -38% at equal relevance | `stage3_ranker_metrics.json`, `cold_start_bottleneck.json` |
| `train_and_simulate.py` | Stage 4 policy metrics | CTR, reward, discovery and cold-start conversion/share, coverage, Gini, entropy, ILD, serendipity, margin, 7-day repeat | e.g. LinUCB + MMR: reward 0.0855, cold share 1.74% | `stage4_policy_summary.json/.csv` |
| `train_and_simulate.py` | Filter / guardrail drops, bandit diagnostics, GPU | counts per rule; max inverse error; VRAM | max inverse error about 5e-14 | `filter_and_guardrail_drop_counts.json`, `bandit_diagnostics.json`, `gpu_utilization.json` |
| `generate_artifacts.py` | Off-policy evaluation | IPS, SNIPS, DR, Switch-DR, DM with 95% bootstrap CIs, ESS, clipping | IPS/SNIPS/DR within 0.6–1.6% of truth; ESS 74–90% | `ope_results.json`, `ope_results_*.csv` |
| `generate_artifacts.py` | Slices, plots, screenshots | persona / cold / category slices; 9 plots; 7 Playwright screenshots | n/a | `output/metrics/`, `output/plots/`, `output/screenshots/` |

## Key files & components

| File | Main functions | Responsibility |
|---|---|---|
| `train_and_simulate.py` | `main`, `data_profile` | Generate the world, train two-tower and XGBoost, warm-start LinUCB, simulate the 4 policies (3,000 requests each), save models, logs and metrics. Flags: `--quick`, `--requests N`, `--config` |
| `generate_artifacts.py` | `main`, `collect_ope_log` (via `src.evaluation.ope`), `plot_*`, `api_screenshots` | Slice metrics, collect the OPE log (6,000 requests) and score all policies, render plots, start the API and capture Playwright screenshots. Flags: `--skip-screenshots`, `--ope-requests N` |

## Inputs, outputs & dependencies

| Direction | Item |
|---|---|
| Input | `config/config.yaml`; `generate_artifacts.py` also needs the outputs of `train_and_simulate.py` (`output/models/`, `output/metrics/sim_records_*.parquet`) |
| Output | `output/models`, `output/logs`, `output/metrics`, `output/plots`, `output/screenshots`, `output/data_profiles` |
| Dependencies | `src` package, matplotlib, seaborn, playwright (Chromium), httpx, uvicorn |

## Usage & verification

```bash
python scripts/train_and_simulate.py          # full run (use --quick for a smoke test)
python scripts/generate_artifacts.py          # OPE, plots, screenshots (use --skip-screenshots to skip Playwright)
```
