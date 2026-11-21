import tempfile

import numpy as np
import pytest

from src.config import DEVICE
from src.data_generator import generate_dataset, logging_prob_matrix, true_reward_matrix
from src.evaluation import DMRewardModel, baseline_policies, evaluate_policy
from src.models import load_model, model_zoo
from src.preprocessing import fit_preprocessor, transform


@pytest.fixture(scope="module")
def data():
    df = generate_dataset(3000, seed=1)
    pre = fit_preprocessor(df)
    return df, transform(pre, df)


def test_generator_overlap_and_columns(data):
    df, X = data
    assert df.propensity.min() >= 0.02
    assert np.allclose(logging_prob_matrix(df).sum(1), 1)
    assert true_reward_matrix(df).shape == (len(df), 6)
    assert not np.isnan(X).any()


def test_logged_propensity_matches_action(data):
    df, _ = data
    L = logging_prob_matrix(df)
    assert np.allclose(L[np.arange(len(df)), df.action.values], df.propensity.values)


def test_device_is_cuda_when_available():
    import torch
    assert (DEVICE.type == "cuda") == torch.cuda.is_available()


@pytest.mark.parametrize("name", list(model_zoo(25, 6, "quick").keys()))
def test_model_fit_probs_and_roundtrip(data, name):
    df, X = data
    m = model_zoo(X.shape[1], 6, "quick")[name]()
    m.fit(X[:2000], df.action.values[:2000], df.reward.values[:2000], df.propensity.values[:2000])
    P = m.action_probs(X[2000:2200])
    assert P.shape == (200, 6) and np.allclose(P.sum(1), 1, atol=1e-6)
    assert m.select_action(X[2000:2010]).shape == (10,)
    with tempfile.TemporaryDirectory() as d:
        m.save(d)
        m2 = load_model(d, name)
        assert np.allclose(m.mean_scores(X[2000:2050]), m2.mean_scores(X[2000:2050]), atol=0.5)


def test_ope_unbiased_for_logging_policy(data):
    df, X = data
    a, r, p = df.action.values, df.reward.values, df.propensity.values
    T, L = true_reward_matrix(df), logging_prob_matrix(df)
    q = DMRewardModel().fit(X, a, r).predict(X)
    res = evaluate_policy(L, a, r, p, q, T, n_boot=20)
    assert abs(res["IPS"] - r.mean()) < 1e-6          # weights are all 1 for the logging policy
    assert abs(res["DR"] - res["TRUE"]) / res["TRUE"] < 0.1
    oracle = baseline_policies(T, L)["Oracle (true best)"]
    assert evaluate_policy(oracle, a, r, p, q, T, n_boot=5)["TRUE"] > res["TRUE"]
