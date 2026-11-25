"""Point-in-time feature engineering and per-scenario dataset construction.

For every snapshot (feature cutoff) `c` the builder only reads events that were *observable* at c:
  * bookings with booking_date < c, with status resolved only if its resolving event is < c
    (cancel_date < c, no-show/checkout < c), otherwise the booking is "pending";
  * folios / surveys / complaints / sessions / email months fully before c.
Labels are read from the raw tables from events in [c, c + window).

Temporal split (per scenario window w):
  train  : snapshots stepping back from (val_cutoff - w) to train_first_cutoff
  val    : snapshot at val_cutoff (labels end <= test_cutoff when w <= 365)
  test   : snapshot at test_cutoff
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .common import get_logger, load_config, path_of
from .data_generator import simulate_treatment_outcome

DAY = pd.Timedelta(days=1)
META_COLS = ["guest_id", "booking_id", "snapshot_date", "y", "treatment", "checkin_date",
             "session_date", "latest_source_ts"]
ZERO_FILL = ["n_stays", "n_nights", "stays_12m", "stays_24m", "nights_24m", "n_ota_12m", "n_booked_12m", "n_surveys", "n_complaints", "n_complaints_12m", "stays_in_upcoming_season", "sessions_30d", "sessions_90d", "searches_30d", "searches_90d", "checkout_reach_30d", "checkout_reach_90d", "web_props_30d", "web_props_90d", "emails_clicked_3m", "last_stay_complaint", "n_upgrades", "n_spa_stays"]
CAT_COLS_ALL = ["channel", "rate_plan", "room_type", "property_type", "last_property_type", "age_band",
                "home_region", "loyalty_tier", "device", "market_type", "property_id", "last_property_id"]


# --------------------------------------------------------------------------- calendar helpers
def calendar_features(d: pd.Series, prefix: str) -> pd.DataFrame:
    m, day = d.dt.month, d.dt.day
    woy = d.dt.isocalendar().week.astype(int)
    out = pd.DataFrame(index=d.index)
    out[f"{prefix}_month_sin"] = np.sin(2 * np.pi * m / 12)
    out[f"{prefix}_month_cos"] = np.cos(2 * np.pi * m / 12)
    out[f"{prefix}_week_sin"] = np.sin(2 * np.pi * woy / 52)
    out[f"{prefix}_week_cos"] = np.cos(2 * np.pi * woy / 52)
    out[f"{prefix}_dow"] = d.dt.dayofweek
    out[f"{prefix}_school_break"] = (m.isin([6, 7, 8]) | ((m == 12) & (day >= 18)) | ((m == 1) & (day <= 3))
                                    | ((m == 3) & (day >= 15)) | ((m == 4) & (day <= 15))).astype(int)
    out[f"{prefix}_holiday_window"] = (((m == 12) & (day >= 22)) | ((m == 1) & (day <= 2)) | ((m == 7) & (day <= 7))
                                       | ((m == 11) & (day >= 22)) | ((m == 5) & (day >= 25)) | ((m == 9) & (day <= 7))).astype(int)
    return out


class Snapshotter:
    """Builds guest-level point-in-time features for any cutoff."""

    def __init__(self, tables: dict[str, pd.DataFrame]):
        b = tables["bookings"].merge(
            tables["properties"][["property_id", "property_type", "region", "star_rating"]], on="property_id")
        b["quarter"] = b["checkin_date"].dt.quarter
        b["rev"] = b["adr"] * b["los"]
        b["is_ota"] = (b["channel"] == "OTA").astype(int)
        b["is_direct"] = b["channel"].isin(["Direct Web", "Mobile App"]).astype(int)
        b["is_app"] = (b["channel"] == "Mobile App").astype(int)
        b["is_corp"] = (b["channel"] == "Corporate").astype(int)
        b["is_nonref"] = (b["rate_plan"] == "Non-Refundable").astype(int)
        b["is_lux"] = (b["property_type"] == "Luxury Resort").astype(int)
        b["is_urb"] = (b["property_type"] == "Urban Business Hotel").astype(int)
        b["is_coast"] = (b["property_type"] == "Coastal Vacation").astype(int)
        self.b = b
        self.guests = tables["guests"].set_index("guest_id")
        self.f = tables["folios"]
        self.s = tables["surveys"]
        self.c = tables["complaints"]
        self.e = tables["email_monthly"]
        self.w = tables["web_sessions"]
        self.props = tables["properties"].set_index("property_id")
        self._cache: dict = {}

    # -- known view ------------------------------------------------------------------
    def known(self, c: pd.Timestamp) -> pd.DataFrame:
        K = self.b[self.b["booking_date"] < c].copy()
        K["kstat"] = np.select(
            [(K.status == "cancelled") & (K.cancel_date < c),
             (K.status == "no_show") & (K.checkin_date < c),
             (K.status == "completed") & (K.checkout_date < c)],
            ["cancelled", "no_show", "completed"], "pending")
        live = K[K.kstat != "cancelled"]
        lk = live.groupby(["property_id", "quarter"])["adr"].mean().rename("prop_season_adr")
        K = K.join(lk, on=["property_id", "quarter"])
        K["adr_vs_prop_season"] = K["adr"] / K["prop_season_adr"]
        K["is_disc"] = (K["adr_vs_prop_season"] < 0.95).astype(int)
        return K

    # -- guest features --------------------------------------------------------------
    def guest_features(self, c) -> tuple[pd.DataFrame, pd.DataFrame]:
        c = pd.Timestamp(c)
        if c in self._cache:
            return self._cache[c]
        K = self.known(c)
        comp = K[K.kstat == "completed"]
        gb = K.groupby("guest_id")
        G = gb.agg(n_booked=("booking_id", "size"), ota_share=("is_ota", "mean"), direct_share=("is_direct", "mean"),
                   app_share=("is_app", "mean"), corp_share=("is_corp", "mean"), nonref_share=("is_nonref", "mean"),
                   avg_lead_time=("lead_time", "mean"), last_booking=("booking_date", "max"),
                   first_booking=("booking_date", "min"), avg_distance_km=("distance_km", "mean"),
                   avg_party_adults=("adults", "mean"), avg_party_children=("children", "mean"))
        G["n_cancelled"] = (K.kstat == "cancelled").groupby(K.guest_id).sum()
        G["n_noshow"] = (K.kstat == "no_show").groupby(K.guest_id).sum()
        up = (K.kstat == "pending") & (K.checkin_date >= c)
        G["n_upcoming"] = up.groupby(K.guest_id).sum()
        K12 = K[K.booking_date >= c - 365 * DAY]
        g12 = K12.groupby("guest_id").agg(n_booked_12m=("booking_id", "size"), ota_share_12m=("is_ota", "mean"),
                                          n_ota_12m=("is_ota", "sum"), direct_share_12m=("is_direct", "mean"))
        G = G.join(g12)
        fin = G["n_cancelled"] + G["n_noshow"]
        # completed-stay aggregates
        gc = comp.groupby("guest_id").agg(
            n_stays=("booking_id", "size"), n_nights=("los", "sum"), room_spend=("rev", "sum"),
            avg_adr=("adr", "mean"), avg_los=("los", "mean"), last_checkout=("checkout_date", "max"),
            first_checkin=("checkin_date", "min"), adr_vs_season_mean=("adr_vs_prop_season", "mean"),
            adr_vs_season_std=("adr_vs_prop_season", "std"), discount_dependency=("is_disc", "mean"),
            n_distinct_properties=("property_id", "nunique"), n_distinct_regions=("region", "nunique"),
            lux_share=("is_lux", "mean"), urban_share=("is_urb", "mean"), coastal_share=("is_coast", "mean"))
        G = G.join(gc)
        c12 = comp[comp.checkin_date >= c - 365 * DAY].groupby("guest_id").agg(stays_12m=("booking_id", "size"))
        c24 = comp[comp.checkin_date >= c - 730 * DAY].groupby("guest_id").agg(stays_24m=("booking_id", "size"), nights_24m=("los", "sum"))
        G = G.join(c12).join(c24)
        cs = comp.sort_values(["guest_id", "checkin_date"])
        gap = cs.groupby("guest_id")["checkin_date"].diff().dt.days
        gg = gap.groupby(cs.guest_id).agg(["mean", "std", "min"])
        gg.columns = ["inter_stay_mean_days", "inter_stay_std_days", "inter_stay_min_days"]
        G = G.join(gg)
        cnt = comp.groupby(["guest_id", "property_id"]).size()
        p = cnt / cnt.groupby(level=0).transform("sum")
        G["property_entropy"] = (-(p * np.log(p))).groupby(level=0).sum()
        G["top_property_share"] = p.groupby(level=0).max()
        last = cs.groupby("guest_id").tail(1).set_index("guest_id")
        G["last_property_id"] = last["property_id"]
        G["last_property_type"] = last["property_type"]
        G["last_stay_adr_ratio"] = last["adr_vs_prop_season"]
        G["last_stay_los"] = last["los"]
        G["last_stay_channel"] = last["channel"]
        G["last_booking_id"] = last["booking_id"]
        G["stays_at_last_property"] = cnt.rename("n").reset_index().merge(
            last[["property_id"]].reset_index(), on=["guest_id", "property_id"]).set_index("guest_id")["n"]
        G["share_at_last_property"] = G["stays_at_last_property"] / G["n_stays"]
        G["days_since_last_stay_at_property"] = (c - cs.groupby(["guest_id", "property_id"])["checkout_date"].max()
                                                 .rename("lc").reset_index()
                                                 .merge(last[["property_id"]].reset_index(), on=["guest_id", "property_id"])
                                                 .set_index("guest_id")["lc"]).dt.days
        # same-season history: completed stays whose check-in month is in the next quarter
        nxt = [((c.month - 1 + k) % 12) + 1 for k in range(3)]
        G["stays_in_upcoming_season"] = comp[comp.checkin_date.dt.month.isin(nxt)].groupby("guest_id").size()
        # RFM + decay
        G["recency_days"] = (c - G["last_checkout"]).dt.days
        G["frequency_per_year"] = G["n_stays"] / np.maximum((c - G["first_booking"]).dt.days / 365.0, 0.25)
        G["monetary_value"] = G["room_spend"]
        G["inter_stay_decay_ratio"] = G["recency_days"] / G["inter_stay_mean_days"].fillna(365.0).clip(lower=14)
        G["tenure_days"] = (c - G["first_booking"]).dt.days
        G["cancel_rate"] = fin / (fin + G["n_stays"].fillna(0)).replace(0, np.nan)

        # folio (completed, known stays only)
        F = self.f[(self.f.folio_date < c) & (self.f.booking_id.isin(comp.booking_id))]
        piv = F.pivot_table(index="guest_id", columns="category", values="amount", aggfunc="sum").add_prefix("folio_").add_suffix("_total")
        G = G.join(piv)
        G["n_upgrades"] = F[F.category == "upgrade"].groupby("guest_id").size()
        G["n_spa_stays"] = F[F.category == "spa"].groupby("guest_id").size()
        for col in ["folio_spa_total", "folio_fb_total", "folio_experience_total", "folio_upgrade_total", "folio_other_total", "n_upgrades", "n_spa_stays"]:
            if col not in G:
                G[col] = 0.0
        G["ancillary_total"] = G[["folio_spa_total", "folio_fb_total", "folio_experience_total"]].sum(axis=1)
        ns = G["n_stays"].replace(0, np.nan)
        G["ancillary_per_stay"] = G["ancillary_total"] / ns
        G["upgrade_rate"] = G["n_upgrades"] / ns
        G["spa_rate"] = G["n_spa_stays"] / ns
        G["fb_per_night"] = G["folio_fb_total"] / G["n_nights"].replace(0, np.nan)
        total_spend = G["room_spend"].fillna(0) + F.groupby("guest_id")["amount"].sum().reindex(G.index).fillna(0)
        G["lifetime_spend"] = total_spend

        # surveys + complaints
        S = self.s[self.s.survey_date < c].sort_values("survey_date")
        sg = S.groupby("guest_id").agg(n_surveys=("nps", "size"), nps_mean=("nps", "mean"), nps_min=("nps", "min"),
                                       csat_mean=("csat", "mean"), nps_last=("nps", "last"), last_survey=("survey_date", "max"))
        sg["detractor_share"] = (S["nps"] <= 6).groupby(S.guest_id).mean()
        G = G.join(sg)
        C = self.c[self.c.complaint_date < c]
        cg = C.groupby("guest_id").agg(n_complaints=("complaint_id", "size"), last_complaint=("complaint_date", "max"),
                                       max_severity=("severity", "max"))
        cg["n_complaints_12m"] = C[C.complaint_date >= c - 365 * DAY].groupby("guest_id").size()
        G = G.join(cg)
        G["days_since_complaint"] = (c - G["last_complaint"]).dt.days
        lastS = S.groupby("guest_id").tail(1).set_index("guest_id")
        G["last_stay_nps"] = lastS.loc[lastS.booking_id == G["last_booking_id"].reindex(lastS.index), "nps"]
        G["last_stay_complaint"] = C[C.booking_id.isin(last["booking_id"])].groupby("guest_id").size()

        # email engagement (complete months before the cutoff month)
        cm = c.to_period("M").to_timestamp()
        E = self.e[self.e.month < cm]
        for wm in (3, 12):
            Ew = E[E.month >= cm - pd.DateOffset(months=wm)].groupby("guest_id")[["emails_sent", "emails_opened", "emails_clicked"]].sum()
            G[f"email_open_rate_{wm}m"] = (Ew.emails_opened / Ew.emails_sent.replace(0, np.nan)).reindex(G.index)
            G[f"email_click_rate_{wm}m"] = (Ew.emails_clicked / Ew.emails_sent.replace(0, np.nan)).reindex(G.index)
            if wm == 3:
                G["emails_clicked_3m"] = Ew.emails_clicked.reindex(G.index)
        G["email_open_trend"] = G["email_open_rate_3m"] - G["email_open_rate_12m"]
        G["email_click_trend"] = G["email_click_rate_3m"] - G["email_click_rate_12m"]

        # web behaviour
        W = self.w[self.w.session_date < c]
        for wd in (30, 90):
            Ww = W[W.session_date >= c - wd * DAY]
            a = Ww.groupby("guest_id").agg(**{f"sessions_{wd}d": ("session_id", "size"), f"searches_{wd}d": ("n_searches", "sum"),
                                                f"checkout_reach_{wd}d": ("reached_checkout", "sum"),
                                                f"avg_pages_{wd}d": ("pages_viewed", "mean"),
                                                f"web_props_{wd}d": ("property_id", "nunique")})
            G = G.join(a)
        G["days_since_last_session"] = (c - W.groupby("guest_id")["session_date"].max()).dt.days.reindex(G.index)
        G["app_session_share"] = (W[W.session_date >= c - 90 * DAY].device == "app").groupby(W.guest_id).mean().reindex(G.index)

        # static guest attributes + derived loyalty
        st = self.guests.reindex(G.index)
        for col in ["is_loyalty_member", "cobrand_card", "corporate_account", "email_opt_in", "app_installed", "age_band", "home_region"]:
            G[col] = st[col]
        G["home_region"] = G["home_region"].astype(str)
        G["account_age_days"] = (c - st["signup_date"]).dt.days
        n24 = G["nights_24m"].fillna(0)
        G["loyalty_tier"] = np.where(G.is_loyalty_member == 0, "Non-Member",
                                     np.where(n24 >= 50, "Platinum", np.where(n24 >= 20, "Gold", "Silver")))
        pts = G["lifetime_spend"].fillna(0) * 10
        G["points_gap_to_free_night"] = np.where(G.is_loyalty_member == 1, 25000 - (pts % 25000), 25000)
        G["market_type"] = np.where(G["avg_distance_km"] > 600, "fly", "drive")
        zero_fill = ZERO_FILL
        for z in zero_fill:
            G[z] = G[z].fillna(0)
        G["days_since_complaint"] = G["days_since_complaint"].fillna(9999)
        G["is_first_stay"] = (G["n_stays"] == 0).astype(int)
        G["month_at_cutoff_sin"] = np.sin(2 * np.pi * c.month / 12)
        G["month_at_cutoff_cos"] = np.cos(2 * np.pi * c.month / 12)

        # leak-audit column: latest source timestamp contributing to any feature
        ts = pd.concat([G["last_booking"], G["last_checkout"], G["last_survey"], G["last_complaint"],
                        W.groupby("guest_id")["session_date"].max().reindex(G.index)], axis=1).max(axis=1)
        G["latest_source_ts"] = ts
        drop = ["last_booking", "first_booking", "last_checkout", "first_checkin", "last_survey", "last_complaint",
                "last_booking_id", "last_stay_channel", "room_spend"]
        G = G.drop(columns=[d for d in drop if d in G.columns])
        self._cache[c] = (G, K)
        if len(self._cache) > 6:
            self._cache.pop(next(iter(self._cache)))
        return G, K


# --------------------------------------------------------------------------- scenario builders
def _recent_cohort(G: pd.DataFrame, c: pd.Timestamp) -> pd.DataFrame:
    """Guests with a completed stay in the last 24 months and no upcoming stay."""
    m = (G["n_stays"] > 0) & (G["recency_days"] <= 730) & (G["n_upcoming"] == 0)
    return G[m]


class ScenarioBuilder:
    def __init__(self, tables, cfg):
        self.t, self.cfg = tables, cfg
        self.S = Snapshotter(tables)
        self.lat = tables["_hidden_latents"] if "_hidden_latents" in tables else None
        self.bk = tables["bookings"]
        self.fol = tables["folios"]

    # ---- guest-level scenarios ------------------------------------------------------
    def guest_snapshot(self, scen, c, w):
        c = pd.Timestamp(c)
        G, K = self.S.guest_features(c)
        coh = _recent_cohort(G, c)
        b = self.bk
        win = b[(b.booking_date >= c) & (b.booking_date < c + w * DAY)]
        out = coh.copy()
        if scen == "guest_churn":
            booked = set(win.guest_id.unique())
            out["y"] = (~out.index.isin(booked)).astype(int)
        elif scen == "ota_to_direct":
            out = out[out["n_ota_12m"] > 0].copy()
            direct = set(win[win.channel.isin(["Direct Web", "Mobile App"])].guest_id.unique())
            organic = out.index.isin(direct).astype(int)
            t, y = simulate_treatment_outcome(self.lat, out.index.values, organic, c)
            out["treatment"], out["y"] = t, y
        elif scen in ("property_repeat", "brand_repeat"):
            out = out[out["last_property_id"].notna()].copy()
            wp = win[["guest_id", "property_id"]]
            same = set(map(tuple, wp.values))
            lp = out["last_property_id"].astype(int)
            if scen == "property_repeat":
                out["y"] = [int((g, p) in same) for g, p in zip(out.index, lp)]
            else:
                other = wp.merge(pd.DataFrame({"guest_id": out.index, "lp": lp.values}), on="guest_id")
                other = other[other.property_id != other.lp]
                out["y"] = out.index.isin(set(other.guest_id)).astype(int)
        out = out.reset_index().rename(columns={"index": "guest_id"})
        out["snapshot_date"] = c
        return out

    # ---- booking-level scenarios ----------------------------------------------------
    def booking_snapshot(self, scen, c, w):
        c = pd.Timestamp(c)
        G, K = self.S.guest_features(c)
        B = K[(K.kstat == "pending") & (K.checkin_date >= c) & (K.checkin_date < c + w * DAY)].copy()
        B = B.rename(columns={"adr_vs_prop_season": "adr_vs_property_season"})
        B["days_until_checkin"] = (B.checkin_date - c).dt.days
        B["days_since_booking"] = (c - B.booking_date).dt.days
        B["party_size"] = B.adults + B.children
        B["has_children"] = (B.children > 0).astype(int)
        B["is_weekend_checkin"] = B.checkin_date.dt.dayofweek.isin([4, 5]).astype(int)
        B["rate_per_night_total"] = B.adr * B.los
        B = pd.concat([B, calendar_features(B.checkin_date, "checkin")], axis=1)
        comp = K[K.kstat == "completed"]
        hist = set(zip(comp.guest_id, comp.checkin_date.dt.month))
        B["same_month_repeat"] = [int((g, m) in hist) for g, m in zip(B.guest_id, B.checkin_date.dt.month)]
        keep = ["booking_id", "guest_id", "property_id", "channel", "rate_plan", "room_type", "property_type", "star_rating",
                "lead_time", "los", "adults", "children", "adr", "distance_km", "adr_vs_property_season", "days_until_checkin",
                "days_since_booking", "party_size", "has_children", "is_weekend_checkin", "rate_per_night_total", "same_month_repeat",
                "checkin_date"] + [c_ for c_ in B.columns if c_.startswith("checkin_") and c_ != "checkin_date"]
        B = B[keep].drop_duplicates("booking_id")
        Gx = G.drop(columns=["last_property_id", "last_property_type", "property_type"], errors="ignore")
        B = B.merge(Gx, left_on="guest_id", right_index=True, how="left", suffixes=("", "_g"))
        raw = self.bk.set_index("booking_id")
        if scen == "cancel_noshow":
            B["y"] = raw.loc[B.booking_id, "status"].isin(["cancelled", "no_show"]).astype(int).values
        else:
            f = self.fol
            if scen == "room_upgrade":
                up = set(f[f.category == "upgrade"].booking_id)
                B["y"] = B.booking_id.isin(up).astype(int)
            else:
                anc = f[f.category.isin(["spa", "fb", "experience"])].groupby("booking_id")["amount"].sum()
                B["y"] = (B.booking_id.map(anc).fillna(0) >= 450).astype(int)
        B["snapshot_date"] = c
        return B

    # ---- session-level scenario -----------------------------------------------------
    def session_snapshot(self, c, w):
        c = pd.Timestamp(c)
        G, K = self.S.guest_features(c)
        W = self.t["web_sessions"]
        Wc = W[(W.session_date >= c - 30 * DAY) & (W.session_date < c)].sort_values("session_date")
        last = Wc.groupby("guest_id").tail(1).copy()
        n30 = Wc.groupby("guest_id").size().rename("open_sessions_30d")
        last = last.join(n30, on="guest_id")
        # exclude guests already resolved: booked (any known booking) on/after the session and before c
        b = self.bk
        known_after = b[(b.booking_date < c)][["guest_id", "booking_date"]].merge(last[["guest_id", "session_date"]], on="guest_id")
        resolved = set(known_after[known_after.booking_date >= known_after.session_date].guest_id)
        last = last[~last.guest_id.isin(resolved)].copy()
        win = b[(b.booking_date >= c) & (b.booking_date < c + w * DAY) & b.channel.isin(["Direct Web", "Mobile App"])]
        last["y"] = last.guest_id.isin(set(win.guest_id)).astype(int)
        last["days_since_session"] = (c - last.session_date).dt.days
        last["searched_property_type"] = last.property_id.map(self.S.props["property_type"])
        last = pd.concat([last, calendar_features(last.session_date, "session")], axis=1)
        Gx = G.drop(columns=["last_property_id", "last_property_type", "property_type"], errors="ignore")
        out = last.merge(Gx, left_on="guest_id", right_index=True, how="left", suffixes=("", "_g"))
        out["price_vs_guest_adr"] = out["price_quote_adr"] / out["avg_adr"]
        out["snapshot_date"] = c
        return out.drop(columns=["session_id"])


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cols = [c for c in df.columns if c not in META_COLS]
    cat = [c for c in cols if c in CAT_COLS_ALL or df[c].dtype == object or str(df[c].dtype) == "category"]
    num = [c for c in cols if c not in cat]
    return num, cat


def build_preprocessor(num_cols: list[str], cat_cols: list[str], mode: str = "gbdt") -> ColumnTransformer:
    """sklearn ColumnTransformer. Output is [categorical ordinal codes..., numeric...] as a DataFrame.

    mode='gbdt': numerics pass through with NaN preserved (trees handle missing natively).
    mode='nn'  : median imputation (+ missing indicators) and standardization.
    """
    cat_pipe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1,
                              dtype=np.float32)
    if mode == "gbdt":
        num_pipe = "passthrough"
    else:
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                             ("sc", StandardScaler())])
    ct = ColumnTransformer([("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)], remainder="drop",
                           verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


# --------------------------------------------------------------------------- dataset assembly
def snapshot_dates(cfg, w: int) -> dict:
    d = cfg["data"]
    val, test = pd.Timestamp(d["val_cutoff"]), pd.Timestamp(d["test_cutoff"])
    first = pd.Timestamp(d["train_first_cutoff"])
    step = max(30, w // 4) if w <= 90 else 91
    tr, cur = [], val - w * DAY
    while cur >= first:
        tr.append(cur)
        cur = cur - step * DAY
    return {"train": sorted(tr), "val": [val], "test": [test]}


def build_scenario(scen: str, builder: ScenarioBuilder, cfg: dict, log) -> dict[str, pd.DataFrame]:
    sc = cfg["scenarios"][scen]
    w, unit = sc["window_days"], sc["unit"]
    snaps = snapshot_dates(cfg, w)
    out = {}
    for split, dates in snaps.items():
        parts = []
        for c in dates:
            if unit == "guest":
                parts.append(builder.guest_snapshot(scen, c, w))
            elif unit == "booking":
                parts.append(builder.booking_snapshot(scen, c, w))
            else:
                parts.append(builder.session_snapshot(c, w))
        df = pd.concat(parts, ignore_index=True)
        if split == "train" and len(df) > cfg["data"]["max_train_rows"]:
            df = df.sample(cfg["data"]["max_train_rows"], random_state=cfg["seed"]).sort_values("snapshot_date").reset_index(drop=True)
        out[split] = df
        log.info("  %-20s %-5s snapshots=%2d rows=%7d pos_rate=%.3f", scen, split, len(dates), len(df), df["y"].mean())
    return out


def save_scenario(scen: str, splits: dict[str, pd.DataFrame], cfg: dict):
    d = path_of(cfg, "processed") / scen
    d.mkdir(parents=True, exist_ok=True)
    num, cat = feature_columns(splits["train"])
    for k, df in splits.items():
        df = df.copy()
        for c in cat:
            df[c] = df[c].astype(str).where(df[c].notna(), np.nan)
        for c in ("last_property_id", "property_id"):
            if c in df:
                df[c] = df[c].map(lambda v: str(int(float(v))) if pd.notna(v) and v != "nan" else np.nan)
        df.to_parquet(d / f"{k}.parquet", index=False)
    schema = {"numeric": num, "categorical": cat, "meta": [c for c in META_COLS if c in splits["train"].columns],
              "window_days": cfg["scenarios"][scen]["window_days"], "unit": cfg["scenarios"][scen]["unit"],
              "rows": {k: len(v) for k, v in splits.items()}, "pos_rate": {k: float(v["y"].mean()) for k, v in splits.items()},
              "snapshots": {k: sorted(map(lambda x: str(x.date()), v["snapshot_date"].unique())) for k, v in splits.items()}}
    (d / "schema.json").write_text(json.dumps(schema, indent=2))


def load_tables(cfg) -> dict[str, pd.DataFrame]:
    raw = path_of(cfg, "raw")
    return {p.stem: pd.read_parquet(p) for p in raw.glob("*.parquet")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    log = get_logger("feature_engineering", cfg)
    tables = load_tables(cfg)
    builder = ScenarioBuilder(tables, cfg)
    for scen in (a.scenarios or list(cfg["scenarios"])):
        log.info("building %s", scen)
        splits = build_scenario(scen, builder, cfg, log)
        save_scenario(scen, splits, cfg)


if __name__ == "__main__":
    main()
