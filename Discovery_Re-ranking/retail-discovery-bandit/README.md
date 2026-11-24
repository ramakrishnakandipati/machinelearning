# Retail Product Discovery & Contextual-Bandit Re-Ranking

GPU-accelerated multi-stage discovery pipeline for an omnichannel grocery catalog:

```
Stage 1  Two-tower retrieval (PyTorch, CUDA matmul top-200)
Stage 2  Hard filters: out-of-stock, bought in last 14 days (drop counts logged per rule)
Stage 3  XGBoost (device="cuda") calibrated P(click), P(buy)
Stage 4  Utility -> guardrails -> LinUCB bonus -> MMR slate (K=5) -> Boltzmann propensities -> JSONL log
         Off-policy evaluation: IPS / SNIPS / DR / Switch-DR with bootstrap CIs and ESS
```

## Run

```bash
pip install -r requirements.txt            # install the CUDA build of torch for your toolkit
python -m playwright install chromium
python scripts/train_and_simulate.py       # ~5 min on an RTX 3090 (use --quick for a smoke run)
python scripts/generate_artifacts.py       # OPE, metric suites, plots, Playwright screenshots (~2.5 min)
python -m pytest tests -q                  # 12 unit tests
uvicorn src.api.main:app --port 8000       # POST /discover, POST /feedback, GET /health, GET /metrics
```

Extras in the API: `GET /demo?user_id=7&policy=linucb_mmr` (HTML slate with score decomposition) and `GET /dashboard`.
All hyperparameters live in `config/config.yaml`.

## Results from the shipped run (seed 42, RTX 3090)

| Stage | Result |
|---|---|
| Retrieval | Recall@50 0.096, Recall@200 0.185, HitRate@200 0.375, **ColdStart_Recall@200 0.022**; p99 retrieval 0.17 ms (target <10 ms) |
| Ranker (click) | ROC-AUC 0.633, PR-AUC 0.109, LogLoss 0.234, Brier 0.059, NDCG@5 0.345, NDCG@10 0.443 |
| Cold-start bottleneck | mean predicted P(click) cold 0.036 vs warm 0.065; **-38% at equal true relevance** (confirmed) |

Online simulation, 3,000 requests per policy:

| Policy | Reward/impr | Cold impr. share | Cold conv. | Coverage % | ILD | Serendipity |
|---|---|---|---|---|---|---|
| Pure ranker | 0.0885 | 0.1% | 0.0% | 5.3 | 0.942 | 0.0003 |
| Epsilon-greedy | 0.0819 | 1.3% | 1.5% | 11.1 | 0.953 | 0.0014 |
| LinUCB + MMR | 0.0855 | 1.7% | 1.5% | 5.7 | 0.967 | 0.0063 |
| High-explore LinUCB | 0.0848 | 6.3% | 1.8% | 7.5 | 0.992 | 0.0059 |

Exposure remains highly concentrated (Gini ≈ 0.98) in all policies at this request volume.
End-to-end latency p99 is 22-26 ms per request.

Off-policy evaluation (6,000 logged requests, behaviour = Boltzmann over the pure-ranker utility): IPS, SNIPS, DR and
Switch-DR are within about 1-2% of the simulator's ground truth and preserve the policy ordering. The direct method
(ranker as reward model) overestimates by 34-40% because the ranker is over-confident. ESS is 74-90% of n.

## Design notes and honest caveats

- **Bandit mean.** The default score is the spec's `U + alpha*sqrt(x'A^-1 x)`. Adding a learned residual `theta'x`
  (`bandit.use_mean: true`) was tried and *hurt* reward by about 25% in ablations: a linear model on noisy sparse rewards
  ranks worse than the XGBoost scores. LinUCB therefore acts as the exploration bonus. Hybrid LinUCB uses float64
  because the Schur-complement updates were ill-conditioned in float32.
- **OPE propensities.** Each policy is evaluated as its *stochastic Boltzmann variant*: sequential sampling without
  replacement, honouring the subcategory cap, with exact slot-conditional propensities. The deployed slates use
  deterministic greedy MMR; their logged `propensity_score` is the item-level softmax.
- **Add-to-cart** is modelled as the purchase event in the simulator.
- **Novelty** has a fourth tier (0.25 = new SKU in a known subcategory and brand) beyond the three in the spec.
- **Retrieval recall** is bounded by the data: an oracle ranking by true latent affinity reaches only about 0.28
  Recall@200, because impressions are popularity-biased and noisy.
- The ranker is weak (AUC 0.63), since the latent affinity is only partly observable through features.

## Known weak spots: are they worth improving?

| Area | Current | Ceiling | Assessment |
|---|---|---|---|
| Overall retrieval | Recall@200 0.185 | about 0.28 (ranking by the true hidden affinity) | **Not worth it.** The impressions are popularity-biased and noisy, so little headroom is left and tuning would not change any conclusion. |
| Cold-start retrieval | ColdStart_Recall@200 0.022 | **not measured yet** | **Check first.** The true ceiling may also be low. If it is well above 0.022, fix it, because cold items reach the ranker only if retrieval surfaces them. Candidate fixes: a content-only item tower, and a reserved slice of the top-200 for new arrivals. |
| Ranker | click ROC-AUC 0.633 | **not measured yet** | **Check first.** Compare against the AUC of the true click probabilities on the same test rows. If the ceiling is about 0.70 there is little to gain. If it is much higher, the ranker under-uses its features. |

Why the ranker matters: it is the pure-ranker baseline, so a stronger one makes the bandit's gains harder to show, and it is
the reward model for the direct-method baseline in the off-policy evaluation. Suggested next step: compute the two missing
oracle ceilings (a few minutes), then spend effort only where the gap to the ceiling is large.

## Layout

`src/data` synthetic world + ground truth + user state - `src/models` two-tower, ranker - `src/reranker` utility,
bandit, MMR, guardrails, pipeline - `src/evaluation` metrics, OPE - `src/api` FastAPI - `scripts` entry points -
`output/` models, logs (training / inference / guardrail / propensity JSONL), metrics (JSON/CSV), plots, screenshots.
