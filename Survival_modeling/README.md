# Survival_modeling

Survival-analysis (time-to-rebooking / churn) projects for hospitality loyalty guests.

## Subfolders at This Level
| Subfolder | Purpose | High-level metrics |
|---|---|---|
| [`hospitality_survival_mlops/`](hospitality_survival_mlops/README.md) | Train, compare and serve 10 survival models via FastAPI | Verified: best model `xgboost_aft`, Uno C 0.6712, Harrell C 0.6898, IBS 0.1983 (hold-out n = 3000); 3 API test cases (source: `output/training_logs/model_comparison.csv`) |
