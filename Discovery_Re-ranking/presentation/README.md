# `presentation/` — Discovery Engine deck

## Purpose & scope

**Why it exists.** A 24-slide PowerPoint explaining the discovery engine: the cold-start problem, the end-to-end flow, each of the five engine steps (utility, guardrails, LinUCB bonus, MMR, propensity), discovery-specific metrics, policy comparison, off-policy evaluation and system performance. Speaker notes are included on every slide.

**Boundaries.** The deck only *presents* results; it owns no model or metric logic. All numbers are read at build time from `../retail-discovery-bandit/output/metrics/`. Re-run the training and artifact scripts there first if you want the deck to reflect a new run.

## High-level metrics & health indicators

| Item | Value | Source |
|---|---|---|
| Slides | 24 (title, executive summary, 5 problem/flow slides, 3 section dividers, 12 content slides on engine and metrics, takeaways) | `build_deck.js` |
| Native charts | bar / stacked / percent-stacked column charts built with `addChart` | `build_deck.js` |
| File validation | passes the OOXML schema validator; text-fit estimate found no overflow. Visual rendering was **not** checked (no LibreOffice or PowerPoint available when it was built) | build session |

### Headline figures presented (all verified in `retail-discovery-bandit/output/metrics/`)

| Figure | Value | Source file |
|---|---|---|
| Cold-start score gap at equal relevance | -38% | `cold_start_bottleneck.json` |
| LinUCB + MMR reward vs pure ranker | 96.6% retained (0.0855 vs 0.0885) | `stage4_policy_summary.json` |
| Cold-start impression share, LinUCB + MMR vs pure ranker | 1.74% vs 0.14% | `stage4_policy_summary.json` |
| Discovery share of purchases | 29.0% vs 17.0% | `stage4_policy_summary.json` |
| OPE error vs ground truth (IPS/SNIPS/DR/Switch-DR; DM) | 0.6–1.6%; +34–40% | `ope_results.json` |
| End-to-end latency, LinUCB + MMR | p50 18.7 ms, p99 26.2 ms | `latency_by_stage_policy.csv` |

## Key files & components

| File | Role |
|---|---|
| `Discovery_Engine_Flow_and_Metrics.pptx` | The generated deck |
| `build_deck.js` | pptxgenjs script: theme, three slide layouts (title, section, content), all slides, charts, tables and speaker notes; reads metrics JSON/CSV and `deck_data.json` |
| `deck_data.json` | Pre-aggregated inputs the script cannot compute alone (tier mix and tier purchase rates, per-stage latency, per-request guardrail drops, the worked example slate, cold-start bins, persona cold-start shares) |

## Inputs, outputs & dependencies

| Direction | Item | Detail |
|---|---|---|
| Input | `../retail-discovery-bandit/output/metrics/*.json`, `*.csv` | Policy summary, OPE results, cold-start gap, GPU info, filter/guardrail counts |
| Input | `../retail-discovery-bandit/output/logs/propensity_log.jsonl` | First record shown as the real logging example (slide 13) |
| Output | `Discovery_Engine_Flow_and_Metrics.pptx` | Written next to the script |
| Dependencies | Node.js, `pptxgenjs`; the theme step also uses the `apply_theme.js` helper from the pptx skill (set `APPLY_THEME` to its path) | not vendored in this repo |

## Usage & verification

```bash
# in this folder, with pptxgenjs installed (npm install pptxgenjs) and APPLY_THEME pointing to apply_theme.js
NODE_PATH=<dir containing node_modules> APPLY_THEME=<path>/apply_theme.js node build_deck.js
```
