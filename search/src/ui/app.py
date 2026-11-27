"""Streamlit demo of the retail search funnel (run: streamlit run src/ui/app.py)."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.common.utils import resolve
from src.search_service import SearchService

st.set_page_config(page_title="Retail Search Funnel", layout="wide", page_icon="🛍️")

EXAMPLES = {
    "Vocabulary gap: waterproof trail kicks": "waterproof trail kicks",
    "Vocabulary gap: breathable gym tee": "breathable gym tee",
    "Vocabulary gap: office chic heels": "office chic heels",
    "Vocabulary gap: warm winter puffer": "warm winter puffer",
    "Typos + price: gortex snekers under $100": "gortex snekers under $100",
    "Navigational: aeroflux running shoes": "aeroflux running shoes",
    "Attribute: black noise canceling earbuds": "silent commute buds",
    "Category browse: backpacks": "backpacks",
    "Color + price: olive rain jacket below $150": "olive storm proof rain shell below $150",
}
LOCK = threading.Lock()


@st.cache_resource(show_spinner="Loading models (encoder, FAISS, rankers)...")
def get_service():
    return SearchService.load()


svc = get_service()
cfg = svc.cfg
W = svc.world
MET = resolve(cfg, "metrics")


def run(query, **kw):
    with LOCK:
        return svc.search(query, **kw)


def results_df(out):
    rows = []
    for r in out["results"]:
        s = r["scores"]
        rows.append({"#": r["rank"], "title": r["title"], "price": r["price"], "margin %": round(r["gross_margin_pct"] * 100, 1),
                     "return rate": round(r["return_rate"], 3), "stock WH/store": f"{r['stock_online']}/{r['stock_store']}",
                     "route": r["fulfillment"], "cold-start": "🆕" if r["is_cold_start"] else "",
                     "pCTR": round(s["p_click"], 3), "pATC": round(s["p_add_to_cart"], 3), "pCVR": round(s["p_purchase_given_cart"], 3),
                     "pRet": round(s["p_return"], 3), "E[profit] $": round(s["expected_profit"], 3), "flags": ", ".join(r["flags"])})
    return pd.DataFrame(rows)


def lat_chart(lat):
    keys = ["query_understanding_ms", "retrieval_ms", "feature_ms", "ranking_ms", "rerank_ms"]
    fig = go.Figure(go.Bar(x=[lat[k] for k in keys], y=[k.replace("_ms", "") for k in keys], orientation="h",
                           marker_color="#2a6fdb", text=[f"{lat[k]:.1f}" for k in keys], textposition="auto"))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=24, b=0), title=f"Latency breakdown - total {lat['total_ms']:.1f} ms",
                      xaxis_title="ms")
    return fig


st.title("🛍️ GPU-Accelerated Retail Search Funnel")
st.caption("Query understanding → hybrid dense/sparse retrieval → LTR + multi-task scoring → margin-aware business re-ranking")

with st.sidebar:
    st.header("Request")
    ex = st.selectbox("Example queries", list(EXAMPLES))
    q = st.text_input("Query", EXAMPLES[ex], key=f"q_{ex}")
    users = ["(anonymous)"] + list(W.users.user_id.values[:30])
    user = st.selectbox("User", users, index=1)
    store = st.selectbox("Store (BOPIS)", [None] + [s for s in W.store_ids if s != "WH"], index=1,
                         format_func=lambda s: "auto" if s is None else f"{s} - {W.stores.set_index('store_id').city[s]}")
    fulfil = st.radio("Fulfilment", ["any", "ship", "bopis"], horizontal=True)
    ranker = st.selectbox("Ranker", list(svc.models), index=list(svc.models).index(cfg["ranking"]["default_ranker"]))
    fusion = st.radio("Hybrid fusion", ["alpha", "rrf", "dense", "sparse"], horizontal=True)
    mw = st.slider("Margin weight", 0.0, 1.0, float(cfg["rerank"]["margin_weight"]), 0.05)
    c1, c2, c3 = st.columns(3)
    rules = c1.checkbox("Rules", True)
    mopt = c2.checkbox("Margin", True)
    div = c3.checkbox("MMR", True)
    topk = st.slider("Top K", 5, 30, 10)

kw = dict(user_id=None if user == "(anonymous)" else user, store_id=store, fulfillment=fulfil, ranker=ranker, fusion=fusion,
          top_k=topk, margin_weight=mw, apply_rules=rules, margin_opt=mopt, diversify=div)

tabs = st.tabs(["🔎 Search", "🪣 Funnel Explorer", "⚖️ Ranker Comparison", "💰 Business Rules & Margin", "🌱 Cold-Start", "📊 Metrics & Latency"])

# ----------------------------------------------------------------------------------------- search
with tabs[0]:
    out = run(q, **kw)
    u = out["understanding"]
    a, b, c = st.columns([2, 2, 3])
    a.metric("Corrected query", out["corrected_query"])
    b.metric("Intent", u["intent_type"], f"{u['intent_confidence']:.0%} conf")
    c.write("**Facets:** " + ", ".join(f"`{k}`={v}" for k, v in u.items() if k in ("brand", "color", "category", "max_price", "subcategories") and v))
    if out["spell_edits"]:
        st.info("Spell fixes: " + ", ".join(f"**{a_}** → **{b_}**" for a_, b_ in out["spell_edits"]))
    st.write("**Vocabulary-gap expansion** — synonyms: " + (", ".join(u["synonyms"]) or "–") + " | embedding neighbours: "
             + (", ".join(u["embedding_neighbors"]) or "–") + " | learned sparse terms: " + (", ".join(u["learned_sparse_terms"]) or "–"))
    f = out["funnel"]
    st.write(f"Funnel: **{f['retrieved']}** retrieved → **{f['after_hard_filters']}** after hard filters → **{f['returned']}** returned "
             f"({f['cold_start_in_results']} cold-start) — dropped: {f['dropped_reasons'] or 'none'}")
    st.dataframe(results_df(out), use_container_width=True, hide_index=True, height=420)
    st.plotly_chart(lat_chart(out["latency_ms"]), use_container_width=True)

# ------------------------------------------------------------------------------ funnel explorer
with tabs[1]:
    st.subheader(f"Stage by stage for “{q}”")
    ctx = svc.engine.understand(q)
    rows = []
    for mode in ["sparse", "dense", "rrf", "alpha"]:
        cand = svc.engine.retrieve(ctx, mode=mode)
        top = [W.catalog.title.values[i] for i in cand.idx[:5]]
        rows.append({"retriever": mode, "retrieval_ms": round(cand.timings["retrieval_ms"], 1), "candidates": len(cand.idx),
                     "top-5": " | ".join(t.split(" - ")[0][:55] for t in top)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    out2 = run(q, **{**kw, "top_k": 50})
    f = out2["funnel"]
    fig = go.Figure(go.Funnel(y=["Retrieved (hybrid)", "After hard filters", "Final page (MMR top-K)"],
                              x=[f["retrieved"], f["after_hard_filters"], topk]))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.json({"parsed": out2["understanding"], "dropped": f["dropped_reasons"]})

# ------------------------------------------------------------------------------ ranker comparison
with tabs[2]:
    st.subheader("Same candidates, different LTR paradigms")
    res, cols = {}, st.columns(3)
    for i, name in enumerate(svc.models):
        o = run(q, **{**kw, "ranker": name, "top_k": 5, "margin_opt": False, "diversify": False})
        res[name] = o
        with cols[i % 3]:
            st.markdown(f"**{name}** · {getattr(svc.models[name], 'paradigm', '')}")
            for r in o["results"]:
                st.caption(f"{r['rank']}. {r['title'].split(' - ')[0][:60]} — ${r['price']:.0f}")
    rk = MET / "ranking_metrics.csv"
    if rk.exists():
        df = pd.read_csv(rk, index_col=0)
        m = df[["ndcg_rel@10", "expected_revenue@10", "expected_margin@10"]].reset_index().rename(columns={"index": "model"})
        st.plotly_chart(px.bar(m, x="model", y="ndcg_rel@10", title="Offline NDCG@10 (test lists)", range_y=[0.7, 1.0]), use_container_width=True)

# ------------------------------------------------------------------------------ business rules
with tabs[3]:
    st.subheader("Margin weight sweep & rule ablation")
    rows = []
    for w_ in [0.0, 0.2, 0.35, 0.5, 0.7, 0.9]:
        o = run(q, **{**kw, "margin_weight": w_, "margin_opt": True, "top_k": 10})
        it = o["results"]
        if it:
            rows.append({"margin_weight": w_, "avg_margin_%": np.mean([x["gross_margin_pct"] for x in it]) * 100,
                         "avg_price": np.mean([x["price"] for x in it]), "sum_E[profit]": sum(x["scores"]["expected_profit"] for x in it),
                         "avg_return_rate": np.mean([x["return_rate"] for x in it])})
    d = pd.DataFrame(rows)
    st.plotly_chart(px.line(d, x="margin_weight", y=["avg_margin_%", "sum_E[profit]"], markers=True, title="Profit lever (top-10)"), use_container_width=True)
    st.dataframe(d.round(3), hide_index=True, use_container_width=True)
    ab = MET / "rerank_ablation.csv"
    if ab.exists():
        st.write("**Offline re-ranking ablation (test lists)**")
        st.dataframe(pd.read_csv(ab, index_col=0).round(4), use_container_width=True)
    o_off = run(q, **{**kw, "apply_rules": False, "top_k": 10})
    oos = sum(1 for r in o_off["results"] if r["stock_online"] <= 0 and r["stock_store"] <= 0)
    st.warning(f"With business rules OFF, {oos} of the top {len(o_off['results'])} results are unfulfillable.")

# ------------------------------------------------------------------------------ cold start
with tabs[4]:
    st.subheader("Brand-new SKUs (zero interaction history) are retrievable via content embeddings")
    cold = W.catalog[W.catalog.is_cold_start].sample(8, random_state=3)
    pick = st.selectbox("Pick a cold-start SKU", cold.index, format_func=lambda i: W.catalog.title[i][:80])
    rowc = W.catalog.loc[pick]
    qtxt = f"{rowc.color} {rowc.subcategory.lower()} {rowc.brand}"
    qtxt = st.text_input("Query for this SKU", qtxt)
    ctx = svc.engine.understand(qtxt)
    data = []
    for mode in ["sparse", "dense", "alpha"]:
        cand = svc.engine.retrieve(ctx, mode=mode)
        pos = np.where(cand.idx == pick)[0]
        data.append({"retriever": mode, "rank_of_cold_sku": int(pos[0] + 1) if len(pos) else None})
    st.dataframe(pd.DataFrame(data), hide_index=True)
    st.write({"impressions": int(rowc.impressions), "clicks": int(rowc.clicks), "purchases": int(rowc.purchases),
              "created_at": rowc.created_at, "days_since_launch": int(W.days_since_launch[pick])})
    o = run(qtxt, **kw)
    st.write(f"Cold-start items in the final page: **{o['funnel']['cold_start_in_results']}**")
    st.dataframe(results_df(o)[["#", "title", "price", "cold-start", "pCTR", "pCVR", "E[profit] $"]], hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------ metrics
with tabs[5]:
    st.subheader("Offline evaluation & latency")
    rp = MET / "retrieval_metrics.csv"
    if rp.exists():
        st.write("**Retrieval (test queries)**")
        st.dataframe(pd.read_csv(rp, index_col=0).round(4), use_container_width=True)
    rk = MET / "ranking_metrics.csv"
    if rk.exists():
        st.write("**Ranking models (test lists, k=10)**")
        st.dataframe(pd.read_csv(rk, index_col=0).round(4), use_container_width=True)
    lp = MET / "latency.json"
    if lp.exists():
        lat = json.loads(lp.read_text())
        rows = [{"stage": k, **v} for k, v in lat.items() if isinstance(v, dict) and "p50" in v]
        d = pd.DataFrame(rows).round(2)
        st.write(f"**Latency percentiles (ms)** — budget {lat['budget_ms']}")
        st.dataframe(d, hide_index=True, use_container_width=True)
        st.plotly_chart(px.bar(d.melt(id_vars="stage", value_vars=["p50", "p95", "p99"]), x="stage", y="value", color="variable",
                               barmode="group", title="End-to-end latency percentiles"), use_container_width=True)
