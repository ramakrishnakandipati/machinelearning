"""Pick 4-5 real held-out test cases per scenario as ready-to-score example guests.

Archetypes (by calibrated score of the scenario's best model on the test snapshot):
  hot_hit   – top decile, outcome happened       hot_miss – top decile, outcome did not happen
  middle    – around the median score            cold     – bottom decile, outcome did not happen
  OTA→Direct additionally: persuadable (highest uplift) and sleeping_dog (most negative uplift)
Output: outputs/examples/examples.json  (payloads are full feature rows, usable with POST /predict/guest).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import get_logger, load_config, path_of
from .feature_engineering import META_COLS

NAMES = {"hot_hit": "High score – event occurred", "hot_miss": "High score – event did not occur", "middle": "Mid-propensity guest",
         "cold": "Low score – event did not occur", "persuadable": "Persuadable (highest uplift)", "sleeping_dog": "Sleeping dog (negative uplift)"}
OUTCOME = {
    "ota_to_direct": ("booked direct in the next 90 days", "did not book direct"),
    "booking_abandonment": ("completed a direct booking within 60 days", "never came back to book"),
    "room_upgrade": ("bought an upgrade", "did not buy an upgrade"),
    "ancillary_spend": ("spent ≥$450 on spa/F&B/experiences", "spent <$450 on ancillaries"),
    "guest_churn": ("did NOT book again within 365 days (churned)", "booked again within 365 days"),
    "property_repeat": ("returned to the same property within 365 days", "did not return to that property"),
    "brand_repeat": ("booked another property in the portfolio", "booked no other property"),
    "cancel_noshow": ("cancelled or was a no-show", "completed the stay"),
}


def story(s: str, r: pd.Series) -> str:
    g = lambda k, d="n/a": (d if k not in r or pd.isna(r[k]) else r[k])  # noqa: E731
    f = lambda k, nd=0: (f"{float(g(k, 0)):.{nd}f}")  # noqa: E731
    tier = g("loyalty_tier")
    who = "Non-member guest" if tier == "Non-Member" else f"{tier} loyalty member"
    base = (f"{who}, {f('n_stays')} completed stays, last stay {f('recency_days')} days ago" if float(g("n_stays", 0)) > 0
            else f"{who} with no completed stays yet")
    if s == "ota_to_direct":
        return f"{base}; {float(g('ota_share_12m', 0)):.0%} of bookings via OTA in the last 12m, email open rate {float(g('email_open_rate_3m', 0)):.0%}, {f('sessions_30d')} web sessions in 30d."
    if s == "booking_abandonment":
        return f"Last session {f('days_since_session')} days ago on {g('device')}: {f('n_searches')} searches, {f('pages_viewed')} pages, checkout {'reached' if g('reached_checkout', 0) == 1 else 'not reached'}; {base}."
    if s in ("room_upgrade", "ancillary_spend", "cancel_noshow"):
        return (f"Upcoming {g('property_type')} stay booked via {g('channel')} ({g('rate_plan')}, {g('room_type')}): {f('los')} nights, "
                f"{f('adults')} adults + {f('children')} children, ADR ${f('adr')}, check-in in {f('days_until_checkin')} days; {base}.")
    return f"{base}; {f('n_distinct_properties')} distinct properties, tenure {f('tenure_days')} days, email open trend {float(g('email_open_trend', 0)):+.2f}."


def clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 6)
    return v


def main():
    cfg = load_config()
    log = get_logger("make_examples", cfg)
    best = json.loads((path_of(cfg, "metrics") / "best_models.json").read_text())
    rng = np.random.default_rng(cfg["seed"])
    out = {}
    for s, info in best.items():
        te = pd.read_parquet(path_of(cfg, "processed") / s / "test.parquet").reset_index(drop=True)
        pr = pd.read_parquet(path_of(cfg, "metrics") / s / f"preds_{info['model']}.parquet").reset_index(drop=True)
        assert len(te) == len(pr) and (te["y"].values == pr["y"].values).all()
        p, y = pr["p_cal"].values, te["y"].values
        hi, lo, med = np.quantile(p, 0.9), np.quantile(p, 0.1), np.median(p)
        picks = {}
        pool = lambda m: np.where(m)[0]  # noqa: E731
        for k, m in (("hot_hit", (p >= hi) & (y == 1)), ("hot_miss", (p >= hi) & (y == 0)),
                     ("middle", np.abs(p - med) < 0.02), ("cold", (p <= lo) & (y == 0))):
            ix = pool(m)
            if not len(ix) and k == "hot_miss":  # e.g. churn: top decile is all positives -> widen to top 40%
                ix = pool((p >= np.quantile(p, 0.6)) & (y == 0))
            picks[k] = int(rng.choice(ix)) if len(ix) else None
        if s == "ota_to_direct":
            up = pd.read_parquet(path_of(cfg, "metrics") / s / "uplift_preds.parquet").reset_index(drop=True)
            picks.pop("middle", None)  # keep 5 examples: the two uplift archetypes replace it
            picks["persuadable"] = int(up["uplift"].values.argmax())
            picks["sleeping_dog"] = int(up["uplift"].values.argmin())
        ex = []
        for k, i in picks.items():
            if i is None:
                continue
            row = te.iloc[i]
            payload = {c: clean(row[c]) for c in te.columns if c not in META_COLS or c == "guest_id"}
            payload["guest_id"] = str(row["guest_id"])
            payload = {c: v for c, v in payload.items() if v is not None}
            pos, neg = OUTCOME[s]
            ex.append({"id": k, "name": NAMES[k], "story": story(s, row), "test_row": i,
                       "observed_y": int(y[i]), "observed_outcome": pos if y[i] == 1 else neg, "model_score_at_selection": round(float(p[i]), 4),
                       "payload": payload})
        out[s] = ex
        log.info("%-20s %d examples: %s", s, len(ex), [(e["id"], e["model_score_at_selection"]) for e in ex])
    d = path_of(cfg, "outputs") if False else (path_of(cfg, "metrics").parent / "examples")
    d.mkdir(parents=True, exist_ok=True)
    (d / "examples.json").write_text(json.dumps(out, indent=1))
    log.info("wrote %s", d / "examples.json")


if __name__ == "__main__":
    main()
