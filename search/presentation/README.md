# Presentation

`Retail_Search_Funnel.pptx` (46 slides, 16:9) covers the whole search funnel:

- **Introduction:** problem, architecture, data
- **Stage by stage:** design, models and metrics for each of the four stages; all nine rankers compared (quality, revenue/margin, train/inference cost); multi-task heads; re-ranking ablation; latency
- **Overall metrics and uplift:** scoreboard, stage-level uplift versus retrieval order, and a high-level uplift flow (offline replay, shadow, online A/B) with guardrails. Only the offline part is implemented; the rest is a proposed path
- **Flow-level metrics:** 7 flows, 24 examples, per-stage means, and two model-by-flow heat-maps (all rankers on every flow)
- **Scenarios as stories:** one story slide (Streamlit screenshot) and one metrics slide per flow
- **Wrap-up:** limitations, next steps, how to run

All numbers come from `../outputs/metrics` and from live runs of the flow examples (`prepare_deck_data.py`). Nothing is typed in by hand.

## Rebuild
```bash
# from the project root, models and metrics already produced by scripts/run_pipeline.py
python presentation/prepare_deck_data.py     # -> presentation/deck_data.json (runs every flow example with all rankers)
python presentation/capture_story_shots.py   # -> presentation/story_shots/*.png (crops of the Streamlit flows tab)
# needs node with pptxgenjs, react, react-dom, react-icons, sharp (NODE_PATH) and the pptx skill's apply_theme.js (PPTX_SKILL_DIR)
cd presentation && node build_deck.js        # -> Retail_Search_Funnel.pptx
```
