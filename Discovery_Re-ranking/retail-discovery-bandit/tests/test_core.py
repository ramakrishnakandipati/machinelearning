import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.metrics import gini, intra_list_diversity, shannon_entropy
from src.evaluation.ope import evaluate_policy
from src.reranker.bandit import LinUCBBandit, ThompsonBandit
from src.reranker.diversity_mmr import mmr_select
from src.reranker.utility import novelty_from_tier, utility
from src.utils import load_config

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("hybrid", [False, True])
def test_sherman_morrison_matches_exact_inverse(hybrid):
    torch.manual_seed(0)
    b = LinUCBBandit(10, 3, DEV, hybrid=hybrid, shared_idx=[0, 1, 2], refresh_every=0)
    for _ in range(300):
        b.update(torch.randn(10, device=DEV), int(np.random.randint(3)), float(np.random.rand()))
    assert b.max_inverse_error() < 1e-3
    drift = b.refresh()
    assert drift < 1e-3 and b.max_inverse_error() < 1e-4


def test_bonus_shrinks_with_data_and_cold_group_stays_uncertain():
    b = LinUCBBandit(6, 2, DEV, hybrid=False, alpha=1.0)
    x = torch.ones(1, 6, device=DEV)
    before = b.scores(x, torch.tensor([0], device=DEV))[1].item()
    for _ in range(50):
        b.update(x[0], 0, 0.1)
    after0 = b.scores(x, torch.tensor([0], device=DEV))[1].item()
    after1 = b.scores(x, torch.tensor([1], device=DEV))[1].item()
    assert after0 < before * 0.3 and after1 == pytest.approx(before, rel=1e-4)


def test_linucb_learns_linear_reward():
    torch.manual_seed(1)
    theta = torch.tensor([0.5, -0.2, 0.1, 0.3], device=DEV)
    b = LinUCBBandit(4, 1, DEV, hybrid=False)
    for _ in range(500):
        x = torch.randn(4, device=DEV)
        b.update(x, 0, float(x @ theta))
    xt = torch.randn(20, 4, device=DEV)
    mean, _ = b.predict(xt, torch.zeros(20, dtype=torch.long, device=DEV))
    assert torch.allclose(mean, xt @ theta, atol=0.05)


def test_hybrid_uses_shared_information_for_unseen_group():
    torch.manual_seed(2)
    b = LinUCBBandit(5, 3, DEV, hybrid=True, shared_idx=[0, 1], alpha=1.0)
    for _ in range(300):
        x = torch.randn(5, device=DEV)
        b.update(x, int(np.random.randint(2)), float(x[0] * 0.4))
    x = torch.randn(1, 5, device=DEV)
    m_unseen, sd_unseen = b.predict(x, torch.tensor([2], device=DEV))
    m_seen, sd_seen = b.predict(x, torch.tensor([0], device=DEV))
    assert sd_unseen > sd_seen            # unseen group is more uncertain...
    assert abs(m_unseen.item() - 0.4 * x[0, 0].item()) < 0.25   # ...but benefits from shared parameters


def test_thompson_sampling_runs():
    b = ThompsonBandit(4, 2, DEV, v=0.1)
    for _ in range(20):
        b.update(torch.randn(4, device=DEV), 0, 1.0)
    mean, noise = b.scores(torch.randn(5, 4, device=DEV), torch.tensor([0, 0, 1, 1, 0], device=DEV))
    assert mean.shape == noise.shape == (5,)


def test_mmr_lambda_one_is_topk_and_cap_enforced():
    torch.manual_seed(0)
    n = 40
    scores = torch.rand(n, device=DEV)
    emb = torch.nn.functional.normalize(torch.randn(n, 8, device=DEV), dim=-1)
    sub = torch.randint(0, 4, (n,), device=DEV)
    picks, _, _ = mmr_select(scores, emb, sub, 5, lam=1.0, max_per_subcat=10)
    assert picks == torch.topk(scores, 5).indices.cpu().tolist()
    picks, _, _ = mmr_select(scores, emb, sub, 5, lam=0.7, max_per_subcat=2)
    counts = np.bincount(sub[picks].cpu().numpy(), minlength=4)
    assert counts.max() <= 2 and len(set(picks)) == 5


def test_mmr_increases_diversity():
    torch.manual_seed(3)
    n = 60
    emb = torch.nn.functional.normalize(torch.randn(n, 16, device=DEV), dim=-1)
    sub = torch.randint(0, 20, (n,), device=DEV)
    scores = torch.rand(n, device=DEV)
    ild = lambda p: intra_list_diversity(emb[p].cpu().numpy())
    p_rel, _, _ = mmr_select(scores, emb, sub, 5, lam=1.0, max_per_subcat=5)
    p_div, _, _ = mmr_select(scores, emb, sub, 5, lam=0.2, max_per_subcat=5)
    assert ild(p_div) >= ild(p_rel) - 1e-6


def test_boltzmann_propensities_sum_to_one():
    s = torch.randn(150, device=DEV)
    for tau in (0.03, 0.1, 1.0):
        p = torch.softmax(s / tau, dim=0)
        assert p.sum().item() == pytest.approx(1.0, abs=1e-5) and (p >= 0).all()


def test_novelty_and_utility():
    cfg = load_config()
    nov = novelty_from_tier(np.array([3, 2, 1, 0]), cfg)
    assert nov.tolist() == [1.0, 0.5, 0.25, 0.0]
    u = utility(torch.tensor([0.2]), torch.tensor([0.1]), torch.tensor([0.3]), torch.tensor([1.0]), 1.0, 2.0, 0.1)
    assert u.item() == pytest.approx(0.2 + 2 * 0.1 * 0.3 + 0.1)


def test_diversity_metric_helpers():
    assert gini(np.ones(100)) == pytest.approx(0.0, abs=1e-9)
    assert gini(np.r_[np.zeros(99), 100.0]) > 0.98
    assert shannon_entropy(np.ones(8)) == pytest.approx(3.0)


def test_ope_estimators_recover_truth_with_known_propensities():
    rng = np.random.default_rng(0)
    n, A = 4000, 20
    rows = []
    for t in range(n):
        q = rng.uniform(0, 1, A)                       # true expected reward per action
        mu = np.exp(rng.normal(0, 0.5, A)); mu /= mu.sum()
        pi = np.exp(2.0 * q); pi /= pi.sum()
        a = rng.choice(A, p=mu)
        r = float(rng.random() < q[a])
        rows.append({"request": t, "slot": 0, "mu": mu[a], "pi_p": pi[a], "reward": r, "q_hat": q[a] * 0.8,
                     "dm_p": float((pi * q * 0.8).sum()), "true_p": float((pi * q).sum())})
    res = evaluate_policy(pd.DataFrame(rows), "p", clip=50.0, switch_tau=20.0, n_boot=200)
    truth = res["IPS"]["true_value"]
    for k in ("IPS", "SNIPS", "DR", "Switch-DR"):
        assert abs(res[k]["estimate"] - truth) < 0.03, k
    assert abs(res["DM"]["estimate"] - truth) > 0.02                       # DM is biased by construction
    assert 0 < res["diagnostics"]["ess"] <= n
