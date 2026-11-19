# Two-Tower retail retrieval (GPU train -> FAISS ANN -> FastAPI)

```
bash setup_gpu.sh                 # detect GPU, install CUDA torch / faiss-gpu (Linux) or faiss-cpu, playwright chromium
python -m src.run_pipeline        # gpu check -> data -> train/export -> evaluate+charts -> ANN index (~1 min on RTX 3090)
pytest -v                         # boots uvicorn, E2E tests, 100-request concurrency benchmark, screenshots
uvicorn api.main:app --port 8000  # serve manually (needs the pipeline outputs)
```

Everything lands in `outputs/` (`logs/ metrics/ charts/ models/ screenshots/`; intermediate data in `outputs/data/`).

Notes
- Redis: uses `REDIS_URL` (default localhost:6379) if it answers, else in-process `fakeredis`.
- `faiss-gpu` wheels exist for Linux only; elsewhere `faiss-cpu` is used and logged. HNSW is CPU-only in FAISS even with faiss-gpu; Flat/IVF-PQ move to GPU via `index_cpu_to_gpu`.
- Data is synthetic; metrics show relative model health, not real-world performance.
