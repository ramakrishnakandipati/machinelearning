"""Epsilon-greedy contextual exploration with logged propensities (Module 3b).

Randomized traffic is what makes future off-policy training/evaluation unbiased:
every decision is stored together with the *exact* probability with which the
serving policy chose it.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .config import EXPLORATION_EPSILON

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]


def mix_with_uniform(base_probs: FloatArr, eps: float, eligible: npt.NDArray[np.bool_] | None = None) -> FloatArr:
    """Return ``(1-eps) * base + eps * Uniform(eligible)`` row-wise.

    ``base_probs`` is (n, n_actions); ``eligible`` masks actions that may be
    explored (ineligible actions get probability exactly 0 and the base
    distribution must already respect the mask).
    """
    n, k = base_probs.shape
    if eligible is None:
        eligible = np.ones((n, k), dtype=bool)
    uniform = eligible / eligible.sum(axis=1, keepdims=True)
    return (1.0 - eps) * base_probs + eps * uniform


def sample_from_probs(probs: FloatArr, rng: np.random.Generator) -> IntArr:
    """Vectorized categorical sampling for an (n, k) probability matrix."""
    cum = probs.cumsum(axis=1)
    u = rng.random((probs.shape[0], 1)) * cum[:, -1:]
    return (u > cum).sum(axis=1).astype(np.int64).clip(0, probs.shape[1] - 1)


class EpsilonGreedyPolicy:
    """Wraps a greedy decision with epsilon-random exploration.

    ``select`` returns the chosen action *and* its logged propensity:
    ``P(a) = (1-eps) * 1[a == greedy] + eps / |eligible|``.
    """

    def __init__(self, epsilon: float = EXPLORATION_EPSILON, seed: int | None = None) -> None:
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1)")
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def probabilities(self, greedy: IntArr, eligible: npt.NDArray[np.bool_]) -> FloatArr:
        """Full action distribution induced by the epsilon-greedy rule."""
        base = np.zeros(eligible.shape, dtype=np.float64)
        base[np.arange(len(greedy)), greedy] = 1.0
        return mix_with_uniform(base, self.epsilon, eligible)

    def select(self, greedy: IntArr, eligible: npt.NDArray[np.bool_]) -> tuple[IntArr, FloatArr, npt.NDArray[np.bool_]]:
        """Return ``(actions, logged_propensities, explored_flags)``."""
        probs = self.probabilities(greedy, eligible)
        explore = self.rng.random(len(greedy)) < self.epsilon
        random_pick = sample_from_probs(eligible / eligible.sum(axis=1, keepdims=True), self.rng)
        actions = np.where(explore, random_pick, greedy).astype(np.int64)
        return actions, probs[np.arange(len(actions)), actions], explore
