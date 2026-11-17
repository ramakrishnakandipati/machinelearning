"""Module 2a - Pydantic schemas for the off-policy campaign log."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from .config import ACTION_IDS


class CampaignLogRecord(BaseModel):
    """One logged decision + its (delayed) outcome.

    ``propensity_score`` is the probability with which the *logging policy*
    chose ``action_id`` given the context; it is what makes unbiased IPW / SNIPS
    evaluation possible. ``incremental_revenue`` is the revenue attributed to the
    decision inside the 7-day conversion window, net of returns inside the 30-day
    return window (true incrementality is what the uplift model estimates).
    """

    customer_id: int
    timestamp: datetime
    context_features: dict[str, float]
    action_id: int
    propensity_score: float = Field(gt=0.0, le=1.0)
    incremental_revenue: float = Field(ge=0.0)
    gross_margin_dollars: float
    returned_flag: bool
    # Extra outcome / bookkeeping fields used by training & evaluation.
    purchased_flag: bool = False
    unsubscribed_flag: bool = False
    reward_mature: bool = True  # False until 7d conversion + 30d return windows close

    @field_validator("action_id")
    @classmethod
    def _valid_action(cls, v: int) -> int:
        if v not in ACTION_IDS:
            raise ValueError(f"action_id must be in {ACTION_IDS}, got {v}")
        return v


class DecisionLogRecord(BaseModel):
    """Record written asynchronously by the online service for every decision."""

    request_id: str
    customer_id: int
    timestamp: datetime
    trigger: str
    action_id: int
    propensity_score: float = Field(gt=0.0, le=1.0)
    explored: bool
    net_value: float
    reason_codes: list[str] = Field(default_factory=list)
    model_version: str


def validate_log_frame(df: pd.DataFrame, feature_cols: list[str], sample: int = 2000, seed: int = 0) -> int:
    """Validate a random sample of rows against the schema (bulk-safe).

    Validating every row of an 8M-row log in pure Python is too slow, so a
    random sample is validated and the remainder is range-checked vectorially.
    Returns the number of rows validated; raises ``ValueError`` on bad data.
    """
    if not df["action_id"].isin(ACTION_IDS).all():
        raise ValueError("Found action_id outside the allowed set")
    if not ((df["propensity_score"] > 0) & (df["propensity_score"] <= 1)).all():
        raise ValueError("propensity_score must be in (0, 1]")
    take = df.sample(min(sample, len(df)), random_state=seed)
    for row in take.itertuples(index=False):
        CampaignLogRecord(
            customer_id=int(row.customer_id),
            timestamp=row.timestamp.to_pydatetime(),
            context_features={c: float(getattr(row, c)) for c in feature_cols},
            action_id=int(row.action_id),
            propensity_score=float(row.propensity_score),
            incremental_revenue=float(row.incremental_revenue),
            gross_margin_dollars=float(row.gross_margin_dollars),
            returned_flag=bool(row.returned_flag),
            purchased_flag=bool(row.purchased_flag),
            unsubscribed_flag=bool(row.unsubscribed_flag),
            reward_mature=bool(row.reward_mature),
        )
    return len(take)
