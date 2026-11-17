"""Sanity tests: leakage, guardrails, exploration propensities, end-to-end service."""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from nba.exploration import EpsilonGreedyPolicy
from nba.features import build_features
from nba.guardrails import InMemoryContactStore, apply_guardrails
from nba.policy import choose_actions, net_value_matrix
from nba.synthetic_data import generate_world


def test_no_lookahead_in_features() -> None:
    w = generate_world(500, seed=1)
    base = build_features(w.transactions, w.sessions, w.as_of, w.profile, w.category_stock)
    # Add a huge future purchase for every customer; features must not change.
    future = pd.DataFrame({"customer_id": w.as_of.index, "ts": w.as_of.to_numpy() + pd.Timedelta(days=1),
                           "amount": 1e6, "is_markdown": 0.0, "is_online": 1.0, "category": 0})
    tx2 = pd.concat([w.transactions, future], ignore_index=True)
    again = build_features(tx2, w.sessions, w.as_of, w.profile, w.category_stock)
    pd.testing.assert_frame_equal(base, again)


def _gr(base_p=0.05, md=0.8, promos=0, avail=0.9, cold=0.0):
    f = lambda v: np.array([v], dtype=float)
    return apply_guardrails(f(base_p), f(md), f(promos), f(avail), f(cold))


def test_guardrails_block_as_specified() -> None:
    assert _gr().eligible[0, 3]                                  # discount hunter: A3 allowed
    assert not _gr(base_p=0.5).eligible[0, 3]                    # sure thing
    assert not _gr(md=0.05).eligible[0, 3]                       # full-price loyalist
    assert not _gr(promos=2).eligible[0, [2, 3, 4]].any()        # fatigue cap
    assert _gr(promos=2).eligible[0, 1]                          # non-promo email still ok
    assert not _gr(avail=0.1).eligible[0, 1]                     # low size availability
    assert not _gr(cold=1.0).eligible[0, 3]                      # cold start: no discount
    assert _gr(base_p=0.9, promos=9, avail=0, cold=1.0).eligible[0, 0]  # no-action always allowed


def test_net_value_and_choice() -> None:
    tm = np.array([[1.0, 20.0, 9.0, 3.0]]); tu = np.array([[0.0, 0.0, 0.05, 0.0]])
    nv = net_value_matrix(tm, tu, lam=100.0)
    assert nv[0, 3] == pytest.approx(9.0 - 0.5 - 5.0)
    assert choose_actions(nv, np.ones((1, 5), bool))[0] == 2
    assert choose_actions(nv, np.array([[True, True, False, True, True]]))[0] == 3
    assert choose_actions(np.array([[0, -1, -2, -3, -4.0]]), np.ones((1, 5), bool))[0] == 0


def test_epsilon_greedy_propensities() -> None:
    pol = EpsilonGreedyPolicy(0.05, seed=0)
    elig = np.array([[True, True, False, True, True]] * 20000)
    greedy = np.full(20000, 3)
    actions, prop, explored = pol.select(greedy, elig)
    assert not (actions == 2).any()                              # ineligible never explored
    assert np.allclose(pol.probabilities(greedy, elig).sum(axis=1), 1.0)
    assert prop[actions == 3].min() == pytest.approx(0.95 + 0.05 / 4)
    assert abs((actions != 3).mean() - 0.05 * 3 / 4) < 0.01


def test_contact_store_counts() -> None:
    async def run() -> int:
        s = InMemoryContactStore()
        await s.record_promo(1); await s.record_promo(1)
        return await s.count_promos(1)
    assert asyncio.run(run()) == 2


def test_service_end_to_end() -> None:
    from fastapi.testclient import TestClient
    from nba.service import app
    with TestClient(app) as c:
        for trig, allowed in {"app_open": {0, 1, 2, 4}, "browse_abandon": {0, 1, 2, 3}, "post_purchase": {0, 1, 4}}.items():
            for cid in (0, 1, 2, 10_000_000):  # last one = unseen guest (cold start)
                r = c.post("/v1/nba/recommend", json={"customer_id": cid, "trigger": trig})
                assert r.status_code == 200, r.text
                body = r.json()
                assert body["action_id"] in allowed  # exploration is restricted to eligible actions
                assert 0 < body["propensity_score"] <= 1
                if cid == 10_000_000:
                    assert body["cold_start"] and body["action_id"] in {0, 1, 4}
        assert c.post("/v1/nba/recommend", json={"customer_id": 1, "trigger": "bad"}).status_code == 422
