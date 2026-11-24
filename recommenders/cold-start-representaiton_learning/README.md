# Cold-Start Representation Learning

Unified-embedding retail recommender that handles new users, new items and new stores (Two-Tower + GraphSAGE + MAML + contrastive SSL, FAISS, FastAPI).

## Subfolders at This Level

| Subfolder | Purpose | High-level metrics |
|---|---|---|
| [`retail-coldstart-replearning/`](retail-coldstart-replearning/README.md) | Full pipeline: data, 4 learners, fusion, FAISS, offline eval, API | Verified: unified_ensemble warm recall@20 0.1662 vs popularity 0.0265; item_cold recall@20 0.1121 (contrastive_ssl 0.1229); HNSW recall@10 1.0 at efSearch=128; API uncached p95 11.1 ms; 19 test functions |
