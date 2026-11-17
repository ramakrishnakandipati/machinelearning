"""Single source of truth for the Retail Next Best Action (NBA) system.

Everything that must stay identical between offline training/evaluation and
online serving (actions, costs, guardrail thresholds, windows, feature order)
lives here so there is no train/serve skew.
"""
from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- #
# Actions (treatments)
# --------------------------------------------------------------------------- #
ACTION_IDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
TREATMENT_IDS: Final[tuple[int, ...]] = (1, 2, 3, 4)  # 0 is the control arm

ACTION_NAMES: Final[dict[int, str]] = {
    0: "Holdout / No Action",
    1: "Email: New Arrivals & Back-in-Stock",
    2: "Push: Free Express Shipping",
    3: "Email+SMS: 15% Off Promo",
    4: "App: 2x Loyalty Points",
}
ACTION_SHORT: Final[dict[int, str]] = {
    0: "A0 Holdout",
    1: "A1 Email Recs",
    2: "A2 Free Ship",
    3: "A3 15% Off",
    4: "A4 2x Points",
}
# Expected per-send cost in dollars that is NOT already visible in gross margin.
#   A1 zero cost, A2 $8 average express-shipping subsidy, A3 message cost (the
#   15% markdown itself already erodes the observed gross margin), A4 deferred
#   points liability.
ACTION_COSTS: Final[dict[int, float]] = {0: 0.0, 1: 0.0, 2: 8.0, 3: 0.5, 4: 2.0}

# Promotional (offer-bearing) actions count towards the contact-fatigue cap.
PROMOTIONAL_ACTIONS: Final[tuple[int, ...]] = (2, 3, 4)

# --------------------------------------------------------------------------- #
# Business objective / risk parameters
# --------------------------------------------------------------------------- #
DEFAULT_LAMBDA: Final[float] = 100.0       # $ penalty per unit of unsubscribe uplift
UNSUB_RATE_CAP: Final[float] = 0.005       # business ceiling: 0.5% unsubscribe rate
EXPLORATION_EPSILON: Final[float] = 0.05   # 5% epsilon-greedy randomized traffic

# --------------------------------------------------------------------------- #
# Guardrail thresholds
# --------------------------------------------------------------------------- #
SURE_THING_THRESHOLD: Final[float] = 0.22      # baseline P(purchase | no action)
FULL_PRICE_LOYALIST_MARKDOWN_MAX: Final[float] = 0.25
MAX_PROMOS_PER_WINDOW: Final[int] = 2
FATIGUE_WINDOW_DAYS: Final[int] = 7
MIN_SIZE_AVAILABILITY: Final[float] = 0.20

# Valid actions per live trigger (online serving). Offline evaluation uses the
# full action set because the historical log has no trigger concept.
TRIGGER_ACTIONS: Final[dict[str, tuple[int, ...]]] = {
    "app_open": (0, 1, 2, 4),
    "browse_abandon": (0, 1, 2, 3),
    "post_purchase": (0, 1, 4),
}
# Cold-start shoppers (<2 sessions) have no reliable history: no discounting.
COLD_START_ACTIONS: Final[tuple[int, ...]] = (0, 1, 4)
COLD_START_MIN_SESSIONS: Final[int] = 2
COLD_START_SHRINKAGE: Final[float] = 0.5  # weight on individual CATE; rest = global ATE

# --------------------------------------------------------------------------- #
# Reward attribution (delayed feedback)
# --------------------------------------------------------------------------- #
CONVERSION_WINDOW_DAYS: Final[int] = 7
RETURN_WINDOW_DAYS: Final[int] = 30
# A reward is "mature" only once decision_ts + 7d + 30d has passed.
REWARD_MATURITY_DAYS: Final[int] = CONVERSION_WINDOW_DAYS + RETURN_WINDOW_DAYS

# --------------------------------------------------------------------------- #
# Features (order is part of the model contract and exported to the manifest)
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS: Final[list[str]] = [
    "recency_days",
    "freq_30d", "freq_90d", "freq_365d",
    "monetary_30d", "monetary_90d", "monetary_365d",
    "aov_365d",
    "markdown_share",
    "discount_tier",
    "top_affinity_cat",
    "affinity_top_share",
    "size_availability_score",
    "online_spend_ratio",
    "cart_abandon_recency_days",
    "n_sessions_90d",
    "app_active_30d",
    "is_cold_start",
]
N_CATEGORIES: Final[int] = 6
CATEGORY_NAMES: Final[list[str]] = ["Running", "Lifestyle", "Training", "Outdoor", "Kids", "Sandals"]

SEGMENT_ORDER: Final[list[str]] = [
    "Full-Price Loyalists",
    "Discount Hunters",
    "At-Risk Churners",
    "Mainstream",
    "Cold-Start Guests",
]

SEED: Final[int] = 42
