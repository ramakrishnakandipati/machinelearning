"""Realistic synthetic hospitality guest / stay generator driven by latent archetypes."""
import numpy as np
import pandas as pd

from .utils import ROOT, get_logger, load_config, set_seed

REF_DATE = pd.Timestamp("2026-11-01")
PROPERTY_TYPES = ["Urban Business", "Resort/Leisure", "Boutique", "Extended Stay"]
LOYALTY = ["Member", "Silver", "Gold", "Platinum", "Ambassador"]
REGIONS = ["North America", "Europe", "APAC", "Middle East", "Latin America"]
AGES = ["18-29", "30-44", "45-59", "60+"]
ROOM_TIERS = ["Standard", "Deluxe", "Suite", "Villa"]

ARCH_NAMES = ["Corporate Road Warriors", "Luxury Resort Families", "Weekend Spa Couples",
              "OTA Price-Sensitive Travelers", "At-Risk Dormant VIPs"]

# Archetype parameter table.
ARCH = {
    0: dict(p=0.25, stays=7, gap=40, adr=210, nights=2.0, lead=7, wk=0.15, rec=25,
            prop=[.75, .03, .07, .15], loy=[.05, .2, .35, .3, .1], room=[.35, .45, .18, .02],
            spa=.03, dine=.15, golf=.02, biz=.75, app=.55, ota=.12, email=.55, key=.7, redeem=.15,
            cancel=.06, lastmin=.4, trend=0.0, age=[.05, .45, .4, .1], tenure=70),
    1: dict(p=0.18, stays=3, gap=120, adr=460, nights=5.5, lead=75, wk=0.45, rec=110,
            prop=[.02, .8, .13, .05], loy=[.05, .15, .3, .3, .2], room=[.05, .2, .45, .3],
            spa=.14, dine=.32, golf=.18, biz=.03, app=.4, ota=.15, email=.5, key=.4, redeem=.2,
            cancel=.1, lastmin=.05, trend=0.0, age=[.02, .38, .45, .15], tenure=60),
    2: dict(p=0.17, stays=4, gap=90, adr=330, nights=2.3, lead=35, wk=0.85, rec=70,
            prop=[.03, .35, .55, .07], loy=[.1, .3, .3, .2, .1], room=[.1, .45, .38, .07],
            spa=.38, dine=.22, golf=.03, biz=.02, app=.5, ota=.2, email=.6, key=.55, redeem=.12,
            cancel=.08, lastmin=.12, trend=0.0, age=[.2, .5, .25, .05], tenure=45),
    3: dict(p=0.25, stays=2.5, gap=150, adr=115, nights=2.4, lead=22, wk=0.5, rec=150,
            prop=[.3, .2, .15, .35], loy=[.65, .22, .08, .04, .01], room=[.75, .22, .03, 0.0],
            spa=.02, dine=.06, golf=.01, biz=.12, app=.08, ota=.8, email=.2, key=.2, redeem=.03,
            cancel=.28, lastmin=.2, trend=0.0, age=[.35, .4, .2, .05], tenure=20),
    4: dict(p=0.15, stays=6, gap=70, adr=520, nights=3.5, lead=45, wk=0.4, rec=520,
            prop=[.25, .35, .3, .1], loy=[.0, .05, .2, .35, .4], room=[.02, .18, .5, .3],
            spa=.12, dine=.25, golf=.1, biz=.2, app=.3, ota=.1, email=.12, key=.3, redeem=.08,
            cancel=.12, lastmin=.1, trend=1.0, age=[.0, .2, .45, .35], tenure=110),
}


def generate_dataset(n_guests=25000, seq_len=10, seed=42, overlap=0.12):
    """Return (guests_df, events_df, seq_tensor (N,L,4), seq_mask (N,L), archetype_ids)."""
    rng = np.random.default_rng(seed)
    probs = np.array([ARCH[i]["p"] for i in range(5)])
    arch = rng.choice(5, size=n_guests, p=probs / probs.sum())
    # Profile "blur": a fraction of guests behave like another archetype -> realistic overlap.
    behave = arch.copy()
    flip = rng.random(n_guests) < overlap
    behave[flip] = rng.integers(0, 5, flip.sum())

    ev_rows, guests = [], []
    seq = np.zeros((n_guests, seq_len, 4), dtype=np.float32)
    mask = np.zeros((n_guests, seq_len), dtype=np.float32)

    for g in range(n_guests):
        a = ARCH[int(behave[g])]
        n = max(1, int(rng.poisson(a["stays"] * rng.lognormal(0, 0.25))) + 1)
        # newest stay first
        rec = max(1, rng.gamma(2.0, a["rec"] / 2.0))
        gaps = rng.gamma(2.0, a["gap"] / 2.0, n)
        if a["trend"]:  # dormant: gaps widen & spend falls towards the present
            gaps = gaps * np.linspace(2.2, 0.6, n)
        gaps[0] = rec
        days_ago = np.cumsum(gaps)
        nights = np.clip(rng.poisson(a["nights"] * rng.lognormal(0, .2), n), 1, 21)
        adr_g = a["adr"] * rng.lognormal(0, .18)
        adr = adr_g * rng.lognormal(0, .08, n)
        if a["trend"]:
            adr = adr * np.linspace(0.7, 1.15, n)
        lead = np.clip(rng.gamma(2.0, a["lead"] / 2.0 * rng.lognormal(0, .2), n), 0, 365)
        prop = rng.choice(4, size=n, p=a["prop"])
        anc = a["spa"] + a["dine"] + a["golf"]
        spend = nights * adr * (1 + anc * rng.lognormal(0, .3, n) * 3)
        for i in range(n):
            ev_rows.append((g, REF_DATE - pd.Timedelta(days=float(days_ago[i])), prop[i], nights[i],
                            adr[i], spend[i], lead[i]))
        # sequence: chronological order, last seq_len stays, padded at front
        take = np.arange(min(n, seq_len))  # indices 0..k-1 are newest..older
        chron = take[::-1]
        da = days_ago[chron]
        gap = np.concatenate([[0.0], -np.diff(da)])
        k = len(chron)
        seq[g, seq_len - k:, 0] = gap
        seq[g, seq_len - k:, 1] = spend[chron]
        seq[g, seq_len - k:, 2] = prop[chron]
        seq[g, seq_len - k:, 3] = lead[chron]
        mask[g, seq_len - k:] = 1

        # guest-level behavioural ratios
        def r(x, s=0.25):
            return float(np.clip(x * rng.lognormal(0, s), 0, 1))
        app, ota = r(a["app"]), r(a["ota"])
        tot = app + ota
        if tot > 0.98:
            app, ota = app / tot * .98, ota / tot * .98
        loy = rng.choice(5, p=a["loy"])
        guests.append(dict(
            guest_id=g, loyalty_tier=LOYALTY[loy],
            tenure_months=float(np.clip(rng.gamma(3, a["tenure"] / 3), 1, 240)),
            home_region=rng.choice(REGIONS, p=[.35, .25, .2, .1, .1]),
            age_group=AGES[rng.choice(4, p=a["age"])],
            room_tier_preference=ROOM_TIERS[rng.choice(4, p=a["room"])],
            property_type_affinity=PROPERTY_TYPES[int(np.bincount(prop, minlength=4).argmax())],
            weekend_vs_weekday_ratio=float(np.clip(a["wk"] + rng.normal(0, .12), 0, 1)),
            cancellation_rate=r(a["cancel"], .4), last_minute_booking_ratio=r(a["lastmin"], .35),
            spa_spend_ratio=r(a["spa"], .4), dining_fb_spend_ratio=r(a["dine"], .35),
            golf_recreation_spend_ratio=r(a["golf"], .5),
            business_center_usage=r(a["biz"], .35),
            direct_app_booking_ratio=app, ota_booking_ratio=ota,
            email_open_rate=r(a["email"], .3), mobile_key_usage_rate=r(a["key"], .3),
            points_redemption_rate=r(a["redeem"], .5),
            true_archetype=int(arch[g]),
        ))

    events = pd.DataFrame(ev_rows, columns=["guest_id", "stay_date", "property_type_id",
                                            "nights", "adr", "spend", "lead_time_days"])
    events["stay_date"] = pd.to_datetime(events["stay_date"])
    guests = pd.DataFrame(guests)

    # Aggregate RFM-style features from events (consistent with the event log)
    last12 = events[events.stay_date >= REF_DATE - pd.Timedelta(days=365)]
    agg = events.groupby("guest_id").agg(
        recency_days=("stay_date", lambda s: (REF_DATE - s.max()).days),
        monetary_total_spend=("spend", "sum"), avg_daily_rate=("adr", "mean"),
        avg_length_of_stay=("nights", "mean"), booking_lead_time_days=("lead_time_days", "mean"),
        total_stays=("spend", "size"))
    freq = last12.groupby("guest_id").size().rename("frequency_stays_12m")
    guests = guests.merge(agg, left_on="guest_id", right_index=True).merge(
        freq, left_on="guest_id", right_index=True, how="left")
    guests["frequency_stays_12m"] = guests["frequency_stays_12m"].fillna(0)
    return guests, events, seq, mask


def build_and_save(cfg=None, n_guests=None):
    cfg = cfg or load_config()
    log = get_logger(cfg=cfg)
    set_seed(cfg["seed"])
    n = n_guests or cfg["data"]["n_guests"]
    guests, events, seq, mask = generate_dataset(n, cfg["data"]["seq_len"], cfg["seed"])
    d = ROOT / cfg["paths"]["data"]
    d.mkdir(parents=True, exist_ok=True)
    guests.to_csv(d / "guests.csv", index=False)
    events.to_csv(d / "stay_events.csv", index=False)
    np.savez_compressed(d / "sequences.npz", seq=seq, mask=mask)
    log.info("Data generated: guests %s | events %s | sequences %s", guests.shape, events.shape, seq.shape)
    return guests, events, seq, mask


if __name__ == "__main__":
    build_and_save()
