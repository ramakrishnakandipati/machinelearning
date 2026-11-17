"""Module 4a - Hard retail guardrails and contact-fatigue state stores.

Guardrails are pure array functions so that the *same* code runs in offline
counterfactual evaluation and in the online service (no train/serve skew).
Each returns a boolean ``blocked`` matrix (n, 5) - True means the action is
suppressed - so the decision log can carry per-action reason codes.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .config import (
    ACTION_IDS, COLD_START_ACTIONS, FATIGUE_WINDOW_DAYS, FULL_PRICE_LOYALIST_MARKDOWN_MAX,
    MAX_PROMOS_PER_WINDOW, MIN_SIZE_AVAILABILITY, PROMOTIONAL_ACTIONS, SURE_THING_THRESHOLD,
)

BoolArr = npt.NDArray[np.bool_]
FloatArr = npt.NDArray[np.float64]
N_ACTIONS = len(ACTION_IDS)


@dataclass
class GuardrailResult:
    """Eligibility mask plus per-rule block matrices for auditing."""

    eligible: BoolArr                 # (n, 5) True = action may be selected
    blocked_by: dict[str, BoolArr]    # rule name -> (n, 5) blocked flags

    def reasons_for(self, row: int, action: int) -> list[str]:
        """Rule names that suppressed ``action`` for one row."""
        return [name for name, m in self.blocked_by.items() if m[row, action]]


def _empty(n: int) -> BoolArr:
    return np.zeros((n, N_ACTIONS), dtype=bool)


def margin_protection(baseline_purchase_prob: FloatArr, markdown_share: FloatArr) -> BoolArr:
    """Suppress A3 (15% off) for "Sure Things" and full-price loyalists."""
    blocked = _empty(len(baseline_purchase_prob))
    sure_thing = baseline_purchase_prob > SURE_THING_THRESHOLD
    loyalist = markdown_share < FULL_PRICE_LOYALIST_MARKDOWN_MAX
    blocked[:, 3] = sure_thing | loyalist
    return blocked


def contact_fatigue(promos_last_7d: npt.NDArray[np.int64] | FloatArr) -> BoolArr:
    """Max ``MAX_PROMOS_PER_WINDOW`` promotional messages per rolling 7 days."""
    blocked = _empty(len(promos_last_7d))
    capped = np.asarray(promos_last_7d) >= MAX_PROMOS_PER_WINDOW
    for a in PROMOTIONAL_ACTIONS:
        blocked[:, a] = capped
    return blocked


def inventory_eligibility(size_availability: FloatArr) -> BoolArr:
    """Suppress A1 when the top-affinity category has <20% size availability."""
    blocked = _empty(len(size_availability))
    blocked[:, 1] = np.asarray(size_availability) < MIN_SIZE_AVAILABILITY
    return blocked


def cold_start_restriction(is_cold_start: FloatArr) -> BoolArr:
    """Guests (<2 sessions) only receive actions without margin risk."""
    blocked = _empty(len(is_cold_start))
    cold = np.asarray(is_cold_start) == 1
    for a in ACTION_IDS:
        if a not in COLD_START_ACTIONS:
            blocked[:, a] = cold
    return blocked


def trigger_restriction(n: int, allowed_actions: tuple[int, ...]) -> BoolArr:
    """Channel / context validity for a live trigger."""
    blocked = _empty(n)
    for a in ACTION_IDS:
        if a not in allowed_actions:
            blocked[:, a] = True
    return blocked


def apply_guardrails(
    baseline_purchase_prob: FloatArr,
    markdown_share: FloatArr,
    promos_last_7d: FloatArr,
    size_availability: FloatArr,
    is_cold_start: FloatArr,
    allowed_actions: tuple[int, ...] | None = None,
    contact_store_unavailable: bool = False,
) -> GuardrailResult:
    """Combine all hard business rules. Action 0 (no action) is always allowed.

    ``contact_store_unavailable`` implements fail-safe behaviour: if Redis can't
    be reached we cannot prove the fatigue cap holds, so promos are suppressed.
    """
    n = len(markdown_share)
    rules = {
        "margin_protection": margin_protection(baseline_purchase_prob, markdown_share),
        "contact_fatigue": contact_fatigue(promos_last_7d),
        "inventory": inventory_eligibility(size_availability),
        "cold_start": cold_start_restriction(is_cold_start),
    }
    if contact_store_unavailable:
        rules["contact_store_unavailable"] = contact_fatigue(np.full(n, MAX_PROMOS_PER_WINDOW))
    if allowed_actions is not None:
        rules["trigger_not_allowed"] = trigger_restriction(n, allowed_actions)
    blocked_any = np.zeros((n, N_ACTIONS), dtype=bool)
    for m in rules.values():
        blocked_any |= m
    blocked_any[:, 0] = False
    return GuardrailResult(eligible=~blocked_any, blocked_by=rules)


# --------------------------------------------------------------------------- #
# Contact-fatigue state (rolling 7-day window)
# --------------------------------------------------------------------------- #
class ContactStore(Protocol):
    """Async interface for the rolling promotional-contact counter."""

    async def count_promos(self, customer_id: int) -> int: ...
    async def record_promo(self, customer_id: int) -> None: ...


class InMemoryContactStore:
    """Process-local fallback (tests / demo without Redis)."""

    def __init__(self, window_days: int = FATIGUE_WINDOW_DAYS) -> None:
        self._window = window_days * 86400
        self._events: dict[int, deque[float]] = defaultdict(deque)

    async def count_promos(self, customer_id: int) -> int:
        q, now = self._events[customer_id], time.time()
        while q and q[0] < now - self._window:
            q.popleft()
        return len(q)

    async def record_promo(self, customer_id: int) -> None:
        self._events[customer_id].append(time.time())


class RedisContactStore:
    """Redis sorted-set per customer: score = epoch seconds of each promo send.

    One pipelined round-trip trims expired entries and returns the count, which
    keeps the hot path to a single network hop. Any Redis error propagates to
    the caller, which then fails safe by suppressing promotions.
    """

    def __init__(self, client, window_days: int = FATIGUE_WINDOW_DAYS) -> None:  # redis.asyncio.Redis
        self._r = client
        self._window = window_days * 86400

    @staticmethod
    def _key(customer_id: int) -> str:
        return f"nba:promo:{customer_id}"

    async def count_promos(self, customer_id: int) -> int:
        now = time.time()
        async with self._r.pipeline(transaction=False) as pipe:
            pipe.zremrangebyscore(self._key(customer_id), 0, now - self._window)
            pipe.zcard(self._key(customer_id))
            _, count = await pipe.execute()
        return int(count)

    async def record_promo(self, customer_id: int) -> None:
        now = time.time()
        async with self._r.pipeline(transaction=False) as pipe:
            pipe.zadd(self._key(customer_id), {f"{now}": now})
            pipe.expire(self._key(customer_id), self._window)
            await pipe.execute()
