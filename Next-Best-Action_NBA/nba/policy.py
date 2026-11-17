"""Module 4b - Margin-aware policy optimizer.

    NetValue(x, a) = tau_margin(x, a) - ActionCost(a) - lambda * tau_unsub(x, a)

The decision is the guardrail-eligible argmax; "no action" (value 0) wins
whenever every eligible treatment has non-positive net value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from .config import ACTION_COSTS, DEFAULT_LAMBDA
from .guardrails import GuardrailResult, apply_guardrails
from .uplift_model import UpliftBundle

FloatArr = npt.NDArray[np.float64]
COST_VEC: FloatArr = np.array([ACTION_COSTS[a] for a in range(5)], dtype=np.float64)


def net_value_matrix(tau_margin: FloatArr, tau_unsub: FloatArr, lam: float = DEFAULT_LAMBDA) -> FloatArr:
    """Return (n, 5) net incremental value; column 0 (control) is exactly 0."""
    n = tau_margin.shape[0]
    out = np.zeros((n, 5))
    # Unsubscribe risk is one-sided: estimation noise must never *credit* an action.
    out[:, 1:] = tau_margin - COST_VEC[1:] - lam * np.clip(tau_unsub, 0.0, None)
    return out


def choose_actions(net_value: FloatArr, eligible: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
    """Eligible argmax; ties (and all-negative rows) resolve to action 0."""
    masked = np.where(eligible, net_value, -np.inf)
    return masked.argmax(axis=1).astype(np.int64)


@dataclass
class PolicyDecision:
    actions: npt.NDArray[np.int64]
    net_value: FloatArr
    tau_margin: FloatArr
    tau_unsub: FloatArr
    guardrails: GuardrailResult


def decide(
    bundle: UpliftBundle,
    x: pd.DataFrame,
    promos_last_7d: FloatArr,
    lam: float = DEFAULT_LAMBDA,
    use_guardrails: bool = True,
    allowed_actions: tuple[int, ...] | None = None,
) -> PolicyDecision:
    """Score every action for each row and pick the best allowed one."""
    arr = x.to_numpy(dtype=np.float32)
    tm, tu = bundle.predict_tau(arr)
    nv = net_value_matrix(tm, tu, lam)
    cols = {c: i for i, c in enumerate(bundle.transformer.columns)}
    gr = apply_guardrails(
        bundle.baseline_purchase_prob(arr),
        arr[:, cols["markdown_share"]].astype(np.float64),
        np.asarray(promos_last_7d, dtype=np.float64),
        arr[:, cols["size_availability_score"]].astype(np.float64),
        arr[:, cols["is_cold_start"]].astype(np.float64),
        allowed_actions,
    )
    eligible = gr.eligible if use_guardrails else np.ones_like(gr.eligible)
    return PolicyDecision(choose_actions(nv, eligible), nv, tm, tu, gr)
