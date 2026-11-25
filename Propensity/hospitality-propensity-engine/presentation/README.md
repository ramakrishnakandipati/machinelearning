# Presentation

`Hospitality_Propensity_Engine.pptx` (32 slides, 16:9) covers the eight propensity flows: overview and method, overall and per-flow metrics, model comparison, calibration, GPU cost, a Streamlit story for every flow, API/dashboard, limits and next steps.

Everything on the slides comes from the real artifacts in `../outputs`; nothing is typed in by hand.

## Rebuild
```bash
# from the project root, with the models/metrics already produced
python -m presentation.prepare_deck_data      # outputs/metrics + examples -> presentation/deck_data.json
python presentation/capture_story_shots.py    # hi-res Streamlit crops -> presentation/story_shots/
# then, with node + pptxgenjs, react, react-dom, react-icons, sharp installed (NODE_PATH points at them):
cd presentation && node build_deck.js         # -> Hospitality_Propensity_Engine.pptx
```
`build_deck.js` also needs the `apply_theme.js` helper from the pptx skill (path set at the top of the script, override with `PPTX_SKILL_DIR`).
