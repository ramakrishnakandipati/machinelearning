"""Realistic synthetic hospitality data generator.

Produces relational tables in data/raw:
  guests, properties, bookings, folios, web_sessions, email_monthly, surveys, complaints
plus _hidden_latents.parquet (generator-only ground truth; NEVER read by feature engineering,
except through `simulate_treatment_outcome`, which models a randomized campaign experiment).

Leakage design
--------------
* Every event carries a timestamp. Outcomes (cancellations, folio spend, later bookings) are
  produced from latent traits + observable attributes, and are timestamped in the future of any
  snapshot that predicts them. Feature engineering only reads rows with ts < feature_cutoff and
  labels only rows in [cutoff, cutoff + window).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .common import get_logger, load_config, path_of

TYPES = ["Luxury Resort", "Urban Business Hotel", "Coastal Vacation"]
CHANNELS = ["OTA", "Direct Web", "Mobile App", "Corporate"]
ROOMS = ["Standard", "Deluxe", "Club", "Suite"]
ROOM_MULT = np.array([1.0, 1.25, 1.5, 1.95])
BASE_ADR = {0: 480.0, 1: 215.0, 2: 310.0}
SEASON = {  # booking check-in seasonality weights by month (Jan..Dec)
    0: np.array([1.1, 1.1, 1.0, 0.9, 0.7, 0.8, 1.1, 1.1, 0.7, 0.8, 0.9, 1.3]),
    1: np.array([0.8, 1.0, 1.1, 1.1, 1.0, 1.0, 0.8, 0.7, 1.2, 1.2, 1.1, 0.7]),
    2: np.array([0.5, 0.5, 0.7, 0.9, 1.0, 1.4, 1.7, 1.7, 1.0, 0.7, 0.5, 0.6]),
}
D = np.timedelta64(1, "D")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _dt(s) -> np.datetime64:
    return np.datetime64(pd.Timestamp(s), "D")


def generate(cfg: dict | None = None, seed: int | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    dc = cfg["data"]
    rng = np.random.default_rng(seed if seed is not None else cfg["seed"])
    N = dc["n_guests"]
    start, end = _dt(dc["history_start"]), _dt(dc["end_date"])

    # ------------------------------------------------------------------ properties
    P = dc["n_properties"]
    ptype = rng.choice(3, P, p=[0.27, 0.40, 0.33])
    n_regions = 10
    reg_xy = rng.uniform(0, 3000, (n_regions, 2))
    p_region = rng.integers(0, n_regions, P)
    p_xy = reg_xy[p_region] + rng.normal(0, 40, (P, 2))
    p_service = rng.normal(0, 1, P)
    p_adr_f = np.exp(rng.normal(0, 0.12, P))
    properties = pd.DataFrame({
        "property_id": np.arange(P),
        "property_name": [f"{TYPES[t].split()[0]} {i:02d}" for i, t in enumerate(ptype)],
        "property_type": [TYPES[t] for t in ptype],
        "region": p_region,
        "star_rating": np.where(ptype == 0, 5, np.where(ptype == 2, 4, rng.choice([3, 4], P))),
        "rooms": rng.integers(80, 650, P),
    })
    pools = [np.where(ptype == t)[0] for t in range(3)]

    # ------------------------------------------------------------------ guests + latents
    business = rng.random(N) < 0.27
    z = lambda: rng.normal(0, 1, N)  # noqa: E731
    sat_z, loy_z, price_z, spend_z, ota_z, e_z, flaky_z = z(), z(), z(), z(), z(), z(), z()
    explore = _sig(rng.normal(-0.4, 1.0, N))            # propensity to try other properties
    prop_loy = np.clip(0.55 * _sig(loy_z * 0.9 - 0.2 - 1.2 * explore) + 0.05, 0.02, 0.8)
    app_user = (rng.random(N) < _sig(-0.6 + 0.7 * e_z - 0.3 * (business * 0))).astype(int)
    member = (rng.random(N) < _sig(-0.3 + 0.9 * loy_z + 0.4 * business)).astype(int)
    cobrand = ((rng.random(N) < _sig(-2.0 + 0.9 * spend_z + 0.5 * member)) & (member == 1)).astype(int)
    corp_acct = (business & (rng.random(N) < 0.6)).astype(int)
    family = (~business) & (rng.random(N) < 0.38)
    age_band = rng.choice(["18-29", "30-44", "45-59", "60+"], N, p=[0.14, 0.38, 0.32, 0.16])
    home_region = rng.integers(0, n_regions, N)
    home_xy = reg_xy[home_region] + rng.normal(0, 60, (N, 2))
    # signup: more weight to earlier dates
    span = int((_dt("2024-10-01") - start) / D)
    signup = start + (rng.beta(1.0, 1.35, N) * span).astype(int) * D
    type_pref = np.where(business, 1, rng.choice([0, 2], N, p=[0.4, 0.6]))
    fav_pool_pick = np.array([rng.choice(pools[t]) for t in type_pref])

    mean_life = 3.8 * np.exp(0.45 * sat_z + 0.30 * (member - 0.3) + 0.20 * loy_z + 0.15 * cobrand)
    life_y = rng.exponential(mean_life)
    active_end = signup + (life_y * 365).astype(int) * D
    eff_end = np.minimum(active_end, end)
    alive_y = np.maximum((eff_end - signup) / D / 365.0, 0.05)
    rate = np.exp(rng.normal(np.log(1.35), 0.45, N)) * (1 + 0.35 * business) * (1 + 0.15 * loy_z.clip(-2, 2))
    n_stays = 1 + rng.poisson(np.clip(rate, 0.15, None) * alive_y)

    guests = pd.DataFrame({
        "guest_id": np.arange(N),
        "signup_date": signup,
        "age_band": age_band,
        "home_region": home_region,
        "is_loyalty_member": member,
        "cobrand_card": cobrand,
        "corporate_account": corp_acct,
        "email_opt_in": (rng.random(N) < _sig(1.4 + 0.5 * e_z)).astype(int),
        "app_installed": app_user,
    })
    latents = pd.DataFrame({
        "guest_id": np.arange(N), "sat_z": sat_z, "loy_z": loy_z, "price_z": price_z,
        "spend_z": spend_z, "ota_z": ota_z, "e_z": e_z, "flaky_z": flaky_z,
        "business": business.astype(int), "active_end": active_end, "prop_loy": prop_loy,
        "explore": explore,
    })

    # ------------------------------------------------------------------ bookings
    gid = np.repeat(np.arange(N), n_stays)
    M = len(gid)
    first = np.r_[True, gid[1:] != gid[:-1]]
    lo = signup[gid] + 7 * D
    hi = np.maximum(eff_end[gid], lo + 14 * D)
    win = ((hi - lo) / D).astype(int)

    # property choice
    u = rng.random(M)
    pick_fav = u < prop_loy[gid]
    pick_type = (~pick_fav) & (u < prop_loy[gid] + (1 - prop_loy[gid]) * (1 - explore[gid] * 0.8))
    prop = rng.integers(0, P, M)
    for t in range(3):
        m = pick_type & (type_pref[gid] == t)
        prop[m] = rng.choice(pools[t], m.sum())
    prop[pick_fav] = fav_pool_pick[gid][pick_fav]
    pt = ptype[prop]

    # check-in date with seasonality (rejection sampling), first stay soon after signup
    checkin = lo + (rng.random(M) * win).astype(int) * D
    checkin[first] = lo[first] + rng.integers(3, 55, first.sum()) * D
    for _ in range(4):
        mon = pd.DatetimeIndex(checkin).month.values - 1
        w = np.array([SEASON[t] for t in range(3)])[pt, mon] / 1.7
        rej = rng.random(M) > w
        if not rej.any():
            break
        checkin[rej] = lo[rej] + (rng.random(rej.sum()) * win[rej]).astype(int) * D
    keep = checkin <= end

    # stay attributes
    biz_b = business[gid]
    los = np.where(biz_b, 1 + rng.geometric(0.55, M) - 1 + 0, 2 + rng.poisson(1.8, M) + (pt == 0) * rng.poisson(1.2, M))
    los = np.clip(los, 1, 21)
    mean_lead = np.where(biz_b, 11.0, np.where(pt == 0, 85.0, 55.0))
    lead = np.minimum(rng.gamma(1.6, mean_lead / 1.6), 330).astype(int)

    # channel
    s_ota = 1.5 * ota_z[gid] * 0.6 + 0.9 * price_z[gid] - 0.9 * member[gid] - 0.5 * cobrand[gid] - 0.5 * e_z[gid] + rng.normal(0, 0.7, M)
    s_web = 0.5 + 0.8 * loy_z[gid] * 0.6 + 0.5 * e_z[gid] + rng.normal(0, 0.7, M)
    s_app = -0.5 + 1.4 * app_user[gid] + 0.4 * loy_z[gid] * 0.6 + rng.normal(0, 0.7, M)
    s_corp = np.where(corp_acct[gid] == 1, 2.2, -9.0)
    S = np.stack([s_ota, s_web, s_app, s_corp], 1)
    Pm = np.exp(S - S.max(1, keepdims=True))
    Pm /= Pm.sum(1, keepdims=True)
    cum = Pm.cumsum(1)
    ch = (rng.random(M)[:, None] > cum).sum(1).clip(0, 3)
    lead = np.where(ch == 0, (lead * 0.85).astype(int), lead)
    booking_date = np.maximum(checkin - lead * D, signup[gid])
    lead = ((checkin - booking_date) / D).astype(int)

    # rate plan, room, party
    nonref = rng.random(M) < _sig(-0.7 + 0.9 * price_z[gid] - 0.8 * biz_b + 0.5 * (ch == 0))
    room_s = 0.8 * spend_z[gid] + 0.5 * (pt == 0) - 0.6 * price_z[gid] + rng.normal(0, 1, M)
    room = np.digitize(room_s, [-0.6, 0.7, 1.7])
    adults = np.where(biz_b, 1, 1 + rng.choice(3, M, p=[0.45, 0.45, 0.10]))
    kids = np.where(family[gid], rng.choice([1, 2, 3], M, p=[0.45, 0.4, 0.15]), (rng.random(M) < 0.04).astype(int))

    # ADR
    m_idx = pd.DatetimeIndex(checkin).month.values - 1
    seas = np.array([SEASON[t] for t in range(3)])[pt, m_idx]
    adr = (np.array([BASE_ADR[t] for t in range(3)])[pt] * p_adr_f[prop] * (0.8 + 0.3 * seas)
           * ROOM_MULT[room] * np.where(nonref, 0.88, 1.0) * np.where(ch == 0, 0.96, 1.0)
           * np.exp(rng.normal(0, 0.10, M) - 0.05 * price_z[gid]))
    dist = np.linalg.norm(home_xy[gid] - p_xy[prop], axis=1)

    # cancellation / no-show
    lead_c = np.minimum(lead, 120) / 30.0
    logit_c = (-3.35 + 1.25 * (~nonref) + 0.55 * (ch == 0) + 0.45 * lead_c - 0.6 * member[gid] - 0.35 * loy_z[gid] * 0.6
               + 0.35 * price_z[gid] * 0.6 + 0.55 * flaky_z[gid] - 0.6 * (ch == 3) + 0.15 * (kids > 0) + rng.normal(0, 0.45, M))
    logit_n = (-4.7 + 0.9 * (~nonref) + 0.5 * (ch == 0) + 0.6 * flaky_z[gid] + 0.4 * biz_b + 0.15 * lead_c + rng.normal(0, 0.4, M))
    p_c, p_n = _sig(logit_c), _sig(logit_n)
    r = rng.random(M)
    status = np.where(r < p_c, "cancelled", np.where(r < p_c + p_n, "no_show", "completed"))
    cancel_date = np.full(M, np.datetime64("NaT"), dtype="datetime64[ns]")
    cm = status == "cancelled"
    frac = rng.beta(2.6, 1.2, cm.sum())
    cancel_date[cm] = (booking_date[cm] + (frac * lead[cm]).astype(int) * D)

    bookings = pd.DataFrame({
        "booking_id": np.arange(M),
        "guest_id": gid,
        "property_id": prop,
        "booking_date": booking_date,
        "checkin_date": checkin,
        "checkout_date": checkin + los * D,
        "los": los,
        "lead_time": lead,
        "channel": np.array(CHANNELS)[ch],
        "rate_plan": np.where(nonref, "Non-Refundable", "Refundable"),
        "room_type": np.array(ROOMS)[room],
        "adults": adults,
        "children": kids,
        "adr": adr.round(2),
        "distance_km": dist.round(0),
        "status": status,
        "cancel_date": cancel_date,
    })
    keepmask = keep & (bookings["booking_date"].values < end)
    bookings = bookings[keepmask].reset_index(drop=True)
    # reindex helper arrays to kept bookings
    idx = np.where(keepmask)[0]
    bid_old2new = {o: n for n, o in enumerate(idx)}
    bookings["booking_id"] = np.arange(len(bookings))
    g_ = bookings["guest_id"].values
    ch_ = ch[idx]; pt_ = pt[idx]; room_ = room[idx]; los_ = los[idx]; kids_ = kids[idx]; adults_ = adults[idx]

    # ------------------------------------------------------------------ folios
    comp = (bookings["status"].values == "completed")
    ci = bookings["checkin_date"].values
    lg = lambda: rng.normal(0, 0.6, len(bookings))  # noqa: E731
    mm = len(bookings)
    rows = []

    def add(cat, p, amt):
        hit = comp & (rng.random(mm) < p)
        sub = np.where(hit)[0]
        rows.append(pd.DataFrame({
            "booking_id": sub, "guest_id": g_[sub], "category": cat,
            "amount": amt[sub].round(2),
            "folio_date": ci[sub] + (rng.random(len(sub)) * los_[sub]).astype(int) * D,
        }))

    sz, lz, pz = spend_z[g_], loy_z[g_], price_z[g_]
    p_spa = _sig(-1.9 + 0.9 * sz + 0.6 * (pt_ == 0) + 0.4 * (pt_ == 2) + 0.1 * np.minimum(los_, 7) + 0.3 * (adults_ >= 2) + lg())
    add("spa", p_spa, np.exp(rng.normal(np.log(150), 0.45, mm)))
    p_fb = _sig(0.3 + 0.6 * sz + 0.2 * np.minimum(los_, 7) - 0.4 * (ch_ == 3) + lg() * 0.5)
    add("fb", p_fb, np.exp(rng.normal(np.log(65), 0.5, mm)) * np.minimum(los_, 7) * (adults_ + 0.5 * kids_) ** 0.7)
    p_exp = _sig(-2.3 + 0.6 * sz + 0.7 * (pt_ == 2) + 0.6 * (pt_ == 0) + 0.5 * (kids_ > 0) + lg())
    add("experience", p_exp, np.exp(rng.normal(np.log(130), 0.5, mm)))
    p_up = _sig(-2.55 + 0.75 * sz + 0.45 * lz * 0.6 - 0.6 * pz + 0.55 * (room_ == 0) - 0.4 * (room_ == 3) * 2
                + 0.3 * np.isin(ch_, [1, 2]) - 0.4 * (kids_ > 0) + 0.12 * np.minimum(los_, 6) + lg())
    add("upgrade", p_up, np.exp(rng.normal(np.log(95), 0.4, mm)))
    add("other", np.full(mm, 0.35), np.exp(rng.normal(np.log(30), 0.5, mm)))
    folios = pd.concat(rows, ignore_index=True)
    folios.insert(0, "folio_id", np.arange(len(folios)))

    # ------------------------------------------------------------------ surveys & complaints
    cs = np.where(comp)[0]
    p_comp = _sig(-3.1 - 0.85 * sat_z[g_[cs]] - 0.4 * p_service[bookings["property_id"].values[cs]] + rng.normal(0, 0.4, len(cs)))
    cflag = rng.random(len(cs)) < p_comp
    csub = cs[cflag]
    complaints = pd.DataFrame({
        "complaint_id": np.arange(len(csub)), "booking_id": csub, "guest_id": g_[csub],
        "complaint_date": ci[csub] + rng.integers(0, 3, len(csub)) * D,
        "severity": rng.choice([1, 2, 3], len(csub), p=[0.5, 0.35, 0.15]),
        "category": rng.choice(["housekeeping", "front_desk", "noise", "facilities", "billing"], len(csub)),
    })
    had_c = np.zeros(mm, bool); had_c[csub] = True
    resp = rng.random(len(cs)) < _sig(-0.7 + 0.5 * e_z[g_[cs]] + 0.3 * member[g_[cs]])
    ss = cs[resp]
    nps = np.clip(np.round(7.2 + 1.9 * sat_z[g_[ss]] + 0.9 * p_service[bookings["property_id"].values[ss]] - 2.6 * had_c[ss] + rng.normal(0, 1.6, len(ss))), 0, 10)
    surveys = pd.DataFrame({
        "survey_id": np.arange(len(ss)), "booking_id": ss, "guest_id": g_[ss],
        "survey_date": bookings["checkout_date"].values[ss] + rng.integers(1, 8, len(ss)) * D,
        "nps": nps.astype(int),
        "csat": np.clip(np.round(nps / 2.5 + rng.normal(0, 0.5, len(ss)) + 0.5), 1, 5).astype(int),
    })

    # ------------------------------------------------------------------ monthly email engagement
    mend_g = np.minimum(active_end + 150 * D, end)
    sm = signup.astype("datetime64[M]")
    n_m = (mend_g.astype("datetime64[M]") - sm).astype(int) + 1
    eg = np.repeat(np.arange(N), n_m)
    off = np.arange(len(eg)) - np.repeat(np.cumsum(n_m) - n_m, n_m)
    month = (sm[eg] + off.astype("timedelta64[M]")).astype("datetime64[ns]")
    after = np.maximum((month - active_end[eg].astype('datetime64[ns]')) / D / 30.0, 0)
    decay = np.exp(-0.35 * after)
    sent = rng.poisson(3 + 4 * member[eg] + 2 * 1, len(eg))
    open_rate = _sig(-1.0 + 0.9 * e_z[eg] + 0.3 * sat_z[eg] * 0.6) * decay
    opens = rng.binomial(sent, np.clip(open_rate, 0, 1))
    clicks = rng.binomial(opens, np.clip(_sig(-1.7 + 0.7 * e_z[eg]), 0, 1))
    email = pd.DataFrame({"guest_id": eg, "month": month, "emails_sent": sent.astype(np.int16),
                          "emails_opened": opens.astype(np.int16), "emails_clicked": clicks.astype(np.int16)})
    email = email[email["month"] <= end].reset_index(drop=True)

    # ------------------------------------------------------------------ web sessions
    direct = np.isin(ch_, [1, 2])
    bd = bookings["booking_date"].values
    sess_parts = []

    def mk(gids, ts, props, intent):
        n = len(gids)
        co = rng.random(n) < (0.38 if intent == "pre" else 0.07)
        pages = 3 + rng.poisson(5 if intent == "pre" else 2.2, n)
        srch = 1 + rng.poisson(2.6 if intent == "pre" else 0.9, n)
        gz = loy_z[gids]
        dev = rng.choice(["mobile", "desktop", "tablet"], n, p=[0.5, 0.4, 0.1])
        dev = np.where(app_user[gids] == 1, np.where(rng.random(n) < 0.6, "app", dev), dev)
        return pd.DataFrame({
            "guest_id": gids, "session_date": ts, "property_id": props,
            "device": dev, "n_searches": srch, "pages_viewed": pages,
            "time_on_site_s": np.exp(rng.normal(5.4 + (0.6 if intent == "pre" else 0), 0.7, n)).astype(int),
            "reached_checkout": co.astype(int),
            "price_quote_adr": np.array([BASE_ADR[t] for t in ptype[props]]) * p_adr_f[props] * np.exp(rng.normal(0, 0.12, n)),
            "logged_in": (rng.random(n) < _sig(-0.2 + 0.9 * member[gids] + 0.4 * gz * 0.6)).astype(int),
        })

    di = np.where(direct)[0]
    k_extra = rng.poisson(1.35, len(di))
    ri = np.repeat(di, k_extra)
    ts = bd[ri] - rng.integers(1, 40, len(ri)) * D
    sess_parts.append(mk(g_[ri], ts, bookings["property_id"].values[ri], "pre"))
    oi = np.where(ch_ == 0)[0]
    oi = oi[rng.random(len(oi)) < 0.42]  # look on brand site, then book OTA
    ts = bd[oi] - rng.integers(1, 30, len(oi)) * D
    sess_parts.append(mk(g_[oi], ts, bookings["property_id"].values[oi], "pre"))
    # browse-only sessions
    br_rate = np.exp(rng.normal(np.log(2.2), 0.7, N)) * (1 + 0.5 * e_z.clip(-2, 2) * 0.5)
    web_end = np.minimum(active_end + 120 * D, end)
    yrs = np.maximum((web_end - signup) / D / 365.0, 0.05)
    nb = rng.poisson(br_rate * yrs)
    bg = np.repeat(np.arange(N), nb)
    bts = signup[bg] + (rng.random(len(bg)) * ((web_end[bg] - signup[bg]) / D)).astype(int) * D
    sess_parts.append(mk(bg, bts, rng.choice(np.arange(P), len(bg)), "browse"))
    web = pd.concat(sess_parts, ignore_index=True)
    web = web[(web["session_date"] <= end)].sort_values("session_date").reset_index(drop=True)
    web.insert(0, "session_id", np.arange(len(web)))

    # restore dtypes
    for df, cols in [(bookings, ["booking_date", "checkin_date", "checkout_date", "cancel_date"]),
                     (folios, ["folio_date"]), (complaints, ["complaint_date"]),
                     (surveys, ["survey_date"]), (web, ["session_date"])]:
        for c in cols:
            df[c] = pd.to_datetime(df[c])
    guests["signup_date"] = pd.to_datetime(guests["signup_date"])
    email["month"] = pd.to_datetime(email["month"])
    latents["active_end"] = pd.to_datetime(latents["active_end"])

    # campaign-effect parameters used by `simulate_treatment_outcome`
    latents["tau"] = 0.30 * _sig(1.5 * ota_z + 0.8 * price_z - 1.2 * loy_z - 0.5)
    latents["neg"] = 0.12 * _sig(2.0 * loy_z - 1.0)

    return {"guests": guests, "properties": properties, "bookings": bookings, "folios": folios,
            "web_sessions": web, "email_monthly": email, "surveys": surveys,
            "complaints": complaints, "_hidden_latents": latents}


def simulate_treatment_outcome(latents: pd.DataFrame, guest_ids: np.ndarray, organic_y: np.ndarray,
                               snapshot, seed: int = 7):
    """Simulate a randomized 'direct-booking offer' campaign sent at `snapshot` (50/50 assignment).

    Returns (treatment, y_observed). Persuadables (high OTA affinity / price sensitivity, low loyalty)
    convert incrementally; very loyal guests are mildly suppressed ("sleeping dogs").
    """
    rng = np.random.default_rng(seed + int(pd.Timestamp(snapshot).strftime("%Y%m%d")))
    lat = latents.set_index("guest_id").loc[guest_ids]
    t = (rng.random(len(guest_ids)) < 0.5).astype(int)
    inc = (rng.random(len(t)) < lat["tau"].values) & (t == 1) & (organic_y == 0)
    sup = (rng.random(len(t)) < lat["neg"].values) & (t == 1) & (organic_y == 1)
    y = np.where(inc, 1, np.where(sup, 0, organic_y))
    return t, y.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    log = get_logger("data_generator", cfg)
    raw = path_of(cfg, "raw")
    tables = generate(cfg)
    for name, df in tables.items():
        df.to_parquet(raw / f"{name}.parquet", index=False)
        log.info("wrote %-16s rows=%9d cols=%d", name, len(df), df.shape[1])
    nb = len(tables["bookings"])
    log.info("guests=%d bookings=%d (min required %d)", len(tables["guests"]), nb, cfg["data"]["min_bookings_total"])
    st = tables["bookings"]["status"].value_counts(normalize=True).round(3).to_dict()
    log.info("status mix: %s | channel mix: %s", st, tables["bookings"]["channel"].value_counts(normalize=True).round(3).to_dict())
    assert nb >= cfg["data"]["min_bookings_total"], "increase n_guests / stay rate"


if __name__ == "__main__":
    main()
