"""Interactive Streamlit dashboard.  Run:  streamlit run dashboard/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_config  # noqa: E402
from src.predictor import PropensityEngine  # noqa: E402

st.set_page_config(page_title="Hospitality Propensity Engine", page_icon=":hotel:", layout="wide")
CFG = load_config()
TITLES = {k: v["title"] for k, v in CFG["scenarios"].items()}


@st.cache_resource(show_spinner="Loading calibrated GPU models…")
def engine() -> PropensityEngine:
    return PropensityEngine()


eng = engine()
h = eng.health()
st.title("Hospitality Propensity Engine")
st.caption(f"8 lifecycle scenarios · calibrated probabilities · GPU: **{h['gpu_device']}** "
           f"(CUDA {'on' if h['cuda_available'] else 'off'}) · {h['gpu_memory_used_gb']} / {h['gpu_memory_total_gb']} GB used")

tab_sim, tab_board, tab_plots = st.tabs(["Guest simulator", "Model leaderboard", "Evaluation gallery"])

# ----------------------------------------------------------------------------- simulator
with tab_sim:
    side, main = st.columns([1, 2.2])
    with side:
        st.subheader("Guest profile")
        preset = st.selectbox("Preset", ["Custom", "Loyal direct booker", "OTA-reliant deal seeker", "Lapsing high-value guest", "First-time family"])
        P = {
            "Custom": {},
            "Loyal direct booker": dict(tier="Platinum", member=1, card=1, n_stays=14, recency=60, gap=120, ota=0.05, nps=9.0, complaints=0, open_rate=0.6, sessions=5),
            "OTA-reliant deal seeker": dict(tier="Non-Member", member=0, card=0, n_stays=4, recency=200, gap=170, ota=0.8, nps=7.0, complaints=0, open_rate=0.25, sessions=4),
            "Lapsing high-value guest": dict(tier="Gold", member=1, card=1, n_stays=9, recency=520, gap=150, ota=0.2, nps=5.0, complaints=2, open_rate=0.08, sessions=0),
            "First-time family": dict(tier="Non-Member", member=0, card=0, n_stays=0, recency=0, gap=0, ota=0.5, nps=8.0, complaints=0, open_rate=0.3, sessions=2),
        }[preset]
        g = lambda k, d: P.get(k, d)  # noqa: E731
        tier = st.selectbox("Loyalty tier", ["Non-Member", "Silver", "Gold", "Platinum"], index=["Non-Member", "Silver", "Gold", "Platinum"].index(g("tier", "Silver")))
        card = st.checkbox("Co-brand credit card", bool(g("card", 0)))
        n_stays = st.slider("Completed stays", 0, 30, int(g("n_stays", 5)))
        recency = st.slider("Days since last stay", 0, 900, int(g("recency", 150)))
        gap = st.slider("Avg days between stays", 0, 500, int(g("gap", 160)))
        ota = st.slider("Historical OTA share", 0.0, 1.0, float(g("ota", 0.3)))
        avg_adr = st.slider("Average ADR ($)", 100, 900, 320)
        nps = st.slider("Avg NPS (0-10)", 0.0, 10.0, float(g("nps", 8.0)))
        complaints = st.slider("Service complaints", 0, 6, int(g("complaints", 0)))
        open_rate = st.slider("Email open rate (3m)", 0.0, 1.0, float(g("open_rate", 0.35)))
        sessions = st.slider("Web sessions (30d)", 0, 15, int(g("sessions", 2)))
        st.markdown("**Upcoming booking / session context**")
        channel = st.selectbox("Booking channel", ["OTA", "Direct Web", "Mobile App", "Corporate"])
        rate_plan = st.selectbox("Rate plan", ["Refundable", "Non-Refundable"])
        room = st.selectbox("Room type", ["Standard", "Deluxe", "Club", "Suite"])
        ptype = st.selectbox("Property type", ["Luxury Resort", "Urban Business Hotel", "Coastal Vacation"])
        lead = st.slider("Lead time (days)", 0, 300, 45)
        los = st.slider("Length of stay", 1, 14, 3)
        adults = st.slider("Adults", 1, 4, 2); kids = st.slider("Children", 0, 4, 0)
        adr = st.slider("Booked ADR ($)", 100, 1200, 340)
        month = st.slider("Check-in month", 1, 12, 7)
        until = st.slider("Days until check-in", 0, 90, 21)
        explain = st.checkbox("Show SHAP drivers (slower)", False)

    guest = dict(
        loyalty_tier=tier, is_loyalty_member=int(tier != "Non-Member"), cobrand_card=int(card), n_stays=n_stays,
        stays_12m=min(n_stays, 3 if n_stays else 0), stays_24m=min(n_stays, 5), recency_days=recency if n_stays else None,
        inter_stay_mean_days=gap or None, avg_adr=avg_adr, ota_share=ota, ota_share_12m=ota, n_ota_12m=int(ota > 0.25) * 2,
        direct_share=1 - ota, nps_mean=nps, nps_last=nps, n_complaints=complaints, email_open_rate_3m=open_rate,
        email_click_rate_3m=open_rate * 0.2, sessions_30d=sessions, sessions_90d=sessions * 2, checkout_reach_90d=int(sessions > 3),
        n_searches=sessions + 1, pages_viewed=4 + sessions, reached_checkout=int(sessions > 3), days_since_session=3,
        channel=channel, rate_plan=rate_plan, room_type=room, property_type=ptype, lead_time=lead, los=los, adults=adults,
        children=kids, adr=adr, checkin_month=month, days_until_checkin=until, days_since_booking=max(lead - until, 0),
        lifetime_spend=n_stays * avg_adr * 3, frequency_per_year=n_stays / 3 if n_stays else None, tenure_days=n_stays * 200,
        cancel_rate=0.25 if rate_plan == "Refundable" else 0.05, n_upcoming=1,
    )
    guest = {k: v for k, v in guest.items() if v is not None}
    res = eng.predict_guests(pd.DataFrame([guest]), explain=explain)[0]

    with main:
        names = [TITLES[s] for s in res]
        vals = [res[s]["propensity"] for s in res]
        base = [res[s]["base_rate"] for s in res]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=base + base[:1], theta=names + names[:1], name="Portfolio base rate", line=dict(color="#94a3b8", dash="dot")))
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=names + names[:1], fill="toself", name="This guest", line=dict(color="#2563eb")))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, max(0.1, max(vals + base) * 1.15)], tickformat=".0%")), height=470,
                          margin=dict(l=40, r=40, t=30, b=30), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Propensity scores & recommended actions")
        rows = [{"Scenario": TITLES[s], "Propensity": r["propensity"] * 100, "Base rate": r["base_rate"] * 100, "Decile (10=top)": r["decile"],
                 "Tier": r["tier"], "Next-best action": r["next_best_action"], "Model": r["model"]} for s, r in res.items()]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, column_config={
            "Propensity": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            "Base rate": st.column_config.NumberColumn(format="%.1f%%")})
        top = max(res, key=lambda s: res[s]["decile"] * 10 + res[s]["propensity"])
        r = res[top]
        st.success(f"**Top action — {TITLES[top]}** (decile {r['decile']}, {r['propensity']:.1%} vs {r['base_rate']:.1%} base): {r['next_best_action']}")
        if res.get("ota_to_direct", {}).get("uplift_segment"):
            r1 = res["ota_to_direct"]
            st.info(f"Uplift segment for the direct-booking offer: **{r1['uplift_segment']}** (estimated incremental effect {r1['uplift']:+.1%}).")
        if explain:
            sc = st.selectbox("Explain scenario", list(res), format_func=lambda s: TITLES[s])
            d = pd.DataFrame(res[sc]["top_drivers"])
            figd = go.Figure(go.Bar(x=d["shap"][::-1], y=(d["feature"] + " = " + d["value"].fillna("NaN"))[::-1], orientation="h",
                                    marker_color=["#16a34a" if v > 0 else "#dc2626" for v in d["shap"][::-1]]))
            figd.update_layout(height=300, title="Top drivers (SHAP, log-odds)", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(figd, use_container_width=True)

# ----------------------------------------------------------------------------- leaderboard
with tab_board:
    lb = eng.leaderboard()
    st.subheader("Master leaderboard (held-out temporal test set, calibrated)")
    scen = st.selectbox("Scenario", ["All"] + list(TITLES), format_func=lambda s: "All scenarios" if s == "All" else TITLES[s])
    show = lb if scen == "All" else lb[lb.scenario == scen]
    cols = ["scenario", "model", "roc_auc", "pr_auc", "logloss", "brier", "top_decile_lift", "ece", "train_time_s", "cv_pr_auc_mean", "device"]
    st.dataframe(show[cols].round(4), use_container_width=True, hide_index=True)
    piv = lb.pivot(index="scenario", columns="model", values="pr_auc")
    st.plotly_chart(go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, text=piv.round(3).values, texttemplate="%{text}", colorscale="Blues")).update_layout(title="Test PR-AUC", height=380), use_container_width=True)

# ----------------------------------------------------------------------------- gallery
with tab_plots:
    plots = ROOT / "outputs" / "plots"
    scen = st.selectbox("Scenario", list(TITLES), format_func=lambda s: TITLES[s], key="gal")
    kinds = {"ROC / PR": f"roc_pr_{scen}.png", "Gains & lift": f"gains_lift_{scen}.png", "Calibration": f"calibration_{scen}.png",
             "SHAP beeswarm": f"shap_beeswarm_{scen}.png", "SHAP bar": f"shap_bar_{scen}.png", "EDA": f"eda_{scen}.png"}
    for label, f in kinds.items():
        if (plots / f).exists():
            st.markdown(f"**{label}**"); st.image(str(plots / f))
    if scen == "ota_to_direct" and (plots / "uplift_ota_to_direct.png").exists():
        st.markdown("**Uplift (T-Learner)**"); st.image(str(plots / "uplift_ota_to_direct.png"))
