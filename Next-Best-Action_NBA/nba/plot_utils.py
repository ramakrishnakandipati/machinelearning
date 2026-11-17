"""Shared plotting style: one colour per action across every figure."""
from __future__ import annotations

import seaborn as sns

# Colour-blind-safe categorical palette (Okabe-Ito based), fixed per action.
ACTION_COLORS: dict[int, str] = {
    0: "#7f7f7f",   # holdout: neutral grey
    1: "#0072B2",   # email recs
    2: "#009E73",   # free shipping
    3: "#D55E00",   # 15% off (margin-risky -> warm)
    4: "#CC79A7",   # loyalty points
}


def apply_style() -> None:
    """Consistent, clean seaborn/matplotlib theme."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
