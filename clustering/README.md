# clustering

Unsupervised guest-persona clustering for hospitality: six models compared and served via FastAPI.

## Subfolders at This Level
| Subfolder | Purpose | High-level metrics |
|---|---|---|
| [hospitality_clustering/](hospitality_clustering/README.md) | Simulate or load guests, fit K-Means, GMM, Agglomerative, DBSCAN, DEC and VAE+GMM, evaluate, serve | Best model K-Means, K=5: silhouette 0.3758, CH 4947.5, DB 1.124 (`output/logs/metrics_summary.json`). 8 scripts, 0 tests. |
