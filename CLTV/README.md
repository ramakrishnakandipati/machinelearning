# CLTV

Customer lifetime value modelling projects (currently hospitality guest CLTV).

## Subfolders at This Level

| Subfolder | Purpose | High-level metrics |
|---|---|---|
| [hospitality_cltv](hospitality_cltv/README.md) | 12-month CLTV scoring for hotel loyalty guests: 7 model paradigms + ensemble, FastAPI service | Best model ft_transformer_ziln: test Normalized Gini 0.749, R2 0.391; ensemble_top3 Gini 0.757, R2 0.412 (n=7,501; `output/logs/metrics_summary.json`); 0 tests |
