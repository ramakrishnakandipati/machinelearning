# `src/models/` - contextual-bandit policies

## 1. Purpose & scope
Implements every decision policy compared in the NBA study behind one interface (`BaseBandit`): offline `fit`, incremental `update`, acting `select_action`, off-policy `action_probs`, and `save`/`load_model` persistence. It owns the algorithms, the model registry (`MODEL_CLASSES`) and the `model_zoo` factory that defines the 10 configured policies in `full`, `quick` and `replay` modes. It delegates data, features, OPE, logging and serving to the parent [`src/`](../README.md) package (imports `..config` for `DEVICE`, `XGB_DEVICE`, `ACTION_ATTR`).

## 2. Subfolders at This Level
None (only `__pycache__/`, ignored by `.gitignore`).

## 3. High-level metrics & health
**Module/code metrics**

| Item | Value |
|---|---|
| Source files | 6 (`__init__`, `base`, `linear`, `tree`, `neural`, `policy_gradient`) |
| Concrete policy classes | 9 (`LinUCB`, `LinearThompsonSampling`, `XGBoostEpsilonGreedy`, `XGBoostUCB`, `NeuralEpsilonGreedy`, `NeuralLinearBandit`, `NeuralTS`, `NeuralUCB`, `DoublyRobustPolicyGradient`) |
| Configured policies in `model_zoo` | 10 (LinUCB is instantiated disjoint and hybrid) |
| Tests | `../../tests/test_smoke.py::test_model_fit_probs_and_roundtrip` runs once per zoo entry in `quick` mode (10 cases): fit, `action_probs` shape and row sums, `select_action` shape, save/load round-trip within 0.5 $ on `mean_scores` |
| Frameworks | PyTorch (GPU float64 ridge, MLPs, `torch.func` Jacobians), XGBoost (`device="cuda"` when available) |

**Domain metrics.** Values are Verified from `../../output/models/model_registry.json` and `../../output/logs/ope_evaluation_audit.json` (full mode, 21000 train / 9000 test, cuda). DR is the doubly-robust OPE estimate, TRUE the simulator ground truth (both $ expected margin per session).

| Policy | DR value | TRUE value | Fit seconds | ESS |
|---|---|---|---|---|
| LinUCB-Disjoint (served) | 609.92 | 617.33 | 0.33 | 1757.6 |
| LinUCB-Hybrid | 609.29 | 617.32 | 0.08 | 1757.4 |
| LinearTS | 605.20 | 611.97 | 0.07 | 2327.8 |
| XGBoost-EpsGreedy | 602.86 | 599.93 | 0.89 | 1946.2 |
| XGBoost-UCB | 605.99 | 612.15 | 3.68 | 1744.0 |
| NeuralEpsGreedy | 598.07 | 589.27 | 4.00 | 1806.7 |
| NeuralLinear | 571.54 | 565.45 | 2.62 | 2274.8 |
| NeuralTS | 579.14 | 582.10 | 9.72 | 3649.6 |
| NeuralUCB | 606.63 | 595.91 | 2.66 | 1645.9 |
| DR-PolicyGradient | 592.56 | 601.26 | 4.21 | 1776.3 |

Baseline TRUE values for reference: Logging 491.34, Uniform 466.94, Always control 401.11, Oracle 629.08.

| Metric | Definition | Value | Where |
|---|---|---|---|
| Training loss / validation loss | Masked MSE on logged-action reward (scaled); XGBoost RMSE per boost round (every 5th round) | `Logged at runtime` (700+ lines in `training_metrics.jsonl`) | `neural.train_qnet`, `tree._XGBBandit._fit` via `BaseBandit._log` |
| `fit_done` event | `n` samples and seconds per fit | `Logged at runtime` | `BaseBandit.fit` |
| DR objective | `dr_value` per policy epoch | `Logged at runtime` | `policy_gradient._train_policy` |
| Action share | Mean of `action_probs` per action | In audit JSON, e.g. LinUCB-Disjoint [0.012, 0.186, 0.285, 0.317, 0.085, 0.115] | `evaluation.run_ope` |

## 4. Key files & components
| File | Main classes / functions | Responsibility |
|---|---|---|
| `base.py` | `BaseBandit` (`fit`, `update`, `mean_scores`, `acting_scores`, `select_action`, `action_probs`, `save`, `_restore`) | Reward standardisation (`r_scale = max(std, 1e-3)`), replay buffer, warm refit every `update_every` rounds, epsilon mixing, Monte-Carlo (`n_mc=40`) probabilities for stochastic policies, persistence (`.meta.joblib`, `.buffer.npz` + weights) |
| `linear.py` | `aug`, `arm_stats`, `ArmPosterior`, `LinUCB` (alpha 0.3, lam 1.0, `hybrid`), `LinearThompsonSampling` (v 1.0) | Exact per-arm ridge in float64 on device; hybrid LinUCB with shared coefficients over context x `ACTION_ATTR`; batched Cholesky posterior sampling |
| `tree.py` | `_XGBBandit`, `XGBoostEpsilonGreedy` (eps 0.05), `XGBoostUCB` (8 members, beta 1.0, 200 trees) | One booster on `[x, onehot(a)]`, 10% validation split, bootstrap ensemble std as UCB bonus, `feature_importance()` (gain) |
| `neural.py` | `QNet`, `train_qnet`, `predict_net`, `net_features`, `NeuralEpsilonGreedy`, `NeuralLinearBandit`, `NeuralTS`, `NeuralUCB` | MLP (128,64) + BatchNorm + Dropout, AdamW, grad clip 5.0, best-validation checkpoint; Bayesian last layer; 5-member MC-dropout ensemble; diagonal-Fisher gradient bonus via `vmap(jacrev)` |
| `policy_gradient.py` | `PolicyNet`, `DoublyRobustPolicyGradient` | Softmax policy trained on a 2-fold cross-fitted, weight-clipped (10.0) DR objective with entropy bonus 0.01; `action_probs` is exact (no MC) |
| `__init__.py` | `MODEL_CLASSES`, `model_zoo(d, K, mode, logger)`, `load_model(dir, name)` | Registry and factory; `quick` = 8 epochs, `replay` = 15 epochs with smaller XGBoost/NeuralTS settings |

## 5. Inputs, outputs & dependencies
| Kind | Item | Notes |
|---|---|---|
| Input | `X` float32 (n, 25 in the committed run), `a` int, `r` float, `p` propensity (defaults to 1/K if None) | Rewards are divided by `r_scale` internally; public scores are in $ |
| Output | `output/models/<name>.meta.joblib`, `<name>.buffer.npz`, plus `<name>.npz` (linear), `.m{i}.json` (XGBoost), `.pt` (neural/PG) | Written by `BaseBandit.save`; `load_model` rebuilds from `meta["cls"]` |
| Dependencies | `torch`, `xgboost`, `numpy`, `joblib`, `..config` | Device falls back to CPU when CUDA is unavailable |

## 6. Usage & verification commands (text only)
```
python -m pytest tests/test_smoke.py -k roundtrip          # from the project root; 10 parametrized cases
python run_pipeline.py --quick --n 8000 --rounds 8 --skip-screenshots
```
Python: `from src.models import model_zoo, load_model` then `model_zoo(d=25, K=6, mode="quick")["LinUCB-Disjoint"](seed=0)`.

**Findings:** `torch.load(..., weights_only=False)` is used when restoring neural models (only load trusted artifacts). `NeuralTS._fit` has a redundant tag expression (`member{m}` in both branches). `NeuralLinearBandit` and `NeuralTS` are stochastic, so their DR/TRUE vary with the Monte-Carlo draws. Imports `Path` in `linear.py`, `tree.py` and `numpy` in `policy_gradient.py`/`neural.py` are partly unused (cosmetic).
