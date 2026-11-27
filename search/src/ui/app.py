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
from src.flows import get_flows, run_example
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
    sv = SearchService.load()
    sv.world.load_latent()  # simulator ground truth: used ONLY to score flows, never by serving
    return sv


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

tabs = st.tabs(["🧭 End-to-End Flows", "🔎 Search", "🪣 Funnel Explorer", "⚖️ Ranker Comparison", "💰 Business Rules & Margin", "🌱 Cold-Start", "📊 Metrics & Latency"])


# ------------------------------------------------------------------------------ end-to-end flows
STAGES = ["A retrieval order", "B + LTR ensemble", "C + hard rules & penalties", "D + margin re-rank", "E + MMR (final page)"]


@st.cache_data(show_spinner="Running flow through every stage...")
def flow_results(flow_id: str):
    fl = next(f for f in get_flows(W) if f["id"] == flow_id)
    with LOCK:
        return fl, [run_example(svc, ex) for ex in fl["examples"]]


def scoreboard():
    def j(n):
        p = MET / n
        return json.loads(p.read_text()) if p.exists() else {}

    ret, rk, ab, lat, mt = j("retrieval_metrics.json"), j("ranking_metrics.json"), j("rerank_ablation.json"), j("latency.json"), j("multitask_metrics.json")
    hyb = next((v for k, v in ret.items() if k.startswith("hybrid_alpha")), {})
    sp = next((v for k, v in ret.items() if k.startswith("sparse")), {})
    ens, base = rk.get("ensemble", {}), rk.get("retrieval_order (no LTR)", {})
    full, first = ab.get("4. + MMR diversity (full funnel)", {}), ab.get("1. ranker only (no rules)", {})
    st.markdown("##### Overall funnel scoreboard (offline test split)")
    c = st.columns(6)
    c[0].metric("1-2 · Recall@200 (hybrid)", f"{hyb.get('recall@200', 0):.3f}", f"{hyb.get('recall@200', 0) - sp.get('recall@200', 0):+.3f} vs sparse")
    c[1].metric("Cold-start Recall@50", f"{hyb.get('cold_start_recall@50', 0):.3f}")
    c[2].metric("3 · NDCG@10 (ensemble)", f"{ens.get('ndcg_rel@10', 0):.3f}", f"{ens.get('ndcg_rel@10', 0) - base.get('ndcg_rel@10', 0):+.3f} vs retrieval order")
    c[3].metric("3 · pCTR AUC", f"{mt.get('pCTR_auc', 0):.3f}")
    c[4].metric("4 · Exp. margin@10", f"{full.get('expected_margin@10', 0):.1f}", f"{full.get('expected_margin@10', 0) / max(base.get('expected_margin@10', 1), 1e-9) - 1:+.0%} vs retrieval order")
    c[5].metric("Unfulfillable in top-10", f"{full.get('oos_rate@10', 0):.1%}", f"{full.get('oos_rate@10', 0) - first.get('oos_rate@10', 0):+.1%} vs no rules", delta_color="inverse")
    tot = lat.get("total_ms", {})
    st.caption(f"End-to-end latency p50 / p95 / p99 = {tot.get('p50', 0):.0f} / {tot.get('p95', 0):.0f} / {tot.get('p99', 0):.0f} ms "
               f"(budget {lat.get('budget_ms', {}).get('total', 150)} ms). Sidebar controls do not affect this tab.")


def render_example(r):
    s1, s2, rk_, ru = r["stage1"], r["stage2"], r["ranking"], r["rules"]
    v = r["variants"]
    A, B, E = v[STAGES[0]], v[STAGES[1]], v[STAGES[4]]
    rt = s2["retrievers"]
    ex = r["example"]
    st.caption(f"user **{ex['user_id']}** · store **{ex.get('store_id')}** · fulfilment **{ex.get('fulfillment', 'any')}**"
               + (f" · margin weight **{ex['margin_weight']}**" if ex.get("margin_weight") is not None else ""))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**① Query understanding**")
        st.metric("Corrected", s1["corrected"][:26] + ("…" if len(s1["corrected"]) > 26 else ""))
        st.metric("Intent", s1["intent"], f"{s1['intent_conf']:.0%} conf", delta_color="off")
        st.metric("Facet accuracy vs truth", f"{s1['facet_accuracy']:.0%}")
        st.caption(f"edits: {', '.join(f'{a}→{b}' for a, b in s1['edits']) or 'none'} · synonyms: {', '.join(s1['synonyms'][:4]) or '–'} · "
                   f"neighbours: {', '.join(s1['neighbors'][:2]) or '–'} · {s1['ms']:.0f} ms")
    with c2:
        st.markdown("**② Hybrid retrieval**")
        st.metric("Relevant found in 200", f"{s2['n_relevant_in_candidates']}/{min(s2['n_relevant_in_catalog'], 200)}")
        st.metric("Recall@200 hybrid", f"{rt['hybrid']['recall@200']:.2f}", f"{rt['hybrid']['recall@200'] - max(rt['sparse']['recall@200'], rt['dense']['recall@200']):+.2f} vs best single")
        st.metric("R@50 sparse / dense / hybrid", f"{rt['sparse']['recall@50']:.2f} / {rt['dense']['recall@50']:.2f} / {rt['hybrid']['recall@50']:.2f}")
        st.caption(f"cold-start relevant in candidates: {s2['cold_in_candidates']}/{s2['cold_relevant']} · {s2['ms']:.1f} ms")
    with c3:
        st.markdown("**③ LTR + multi-task**")
        st.metric("NDCG@10", f"{B['ndcg@10']:.3f}", f"{B['ndcg@10'] - A['ndcg@10']:+.3f} vs retrieval order")
        st.metric("Exp. margin@10", f"{B['exp_margin@10']:.1f}", f"{B['exp_margin@10'] - A['exp_margin@10']:+.1f}")
        st.caption(f"ranker: {rk_['ranker']} · mean pCTR {rk_['mean_pCTR@10']:.2f} · pCVR {rk_['mean_pCVR@10']:.2f} · pReturn {rk_['mean_pReturn@10']:.2f} · {rk_['ms']:.0f} ms")
    with c4:
        st.markdown("**④ Business re-rank**")
        st.metric("Retrieved → after filters", f"{ru['retrieved']} → {ru['after_hard_filters']}")
        st.metric("Unfulfillable in top-10", f"{E['unfulfillable@10']}", f"{E['unfulfillable@10'] - B['unfulfillable@10']:+d} vs ranker only", delta_color="inverse")
        st.metric("Exp. margin@10 (final)", f"{E['exp_margin@10']:.1f}", f"{E['exp_margin@10'] / max(A['exp_margin@10'], 1e-9) - 1:+.0%} vs retrieval order")
        st.caption(f"dropped: {ru['dropped'] or 'none'} · high-return flagged {ru['high_return_flagged']} · low-stock flagged {ru['low_stock_flagged']}")
    df = pd.DataFrame(v).T.rename(columns={"ndcg@10": "NDCG@10", "precision@10": "Prec@10", "exp_revenue@10": "Exp rev", "exp_margin@10": "Exp margin",
                                           "unfulfillable@10": "Unfulfillable", "avg_return_prob@10": "Avg P(return)", "avg_price@10": "Avg price",
                                           "pref_brand_share@10": "Pref-brand share", "cold_in_top10": "Cold in top10"}).drop(columns=["shown"])
    st.dataframe(df.round(3), use_container_width=True)
    if "cold_target" in r:
        ct = r["cold_target"]
        st.info(f"Target cold SKU **{ct['title']}** ({ct['impressions']} impressions, {ct['days_since_launch']} days old): "
                f"rank sparse {ct['rank_sparse']} · dense {ct['rank_dense']} · hybrid {ct['rank_hybrid']} · **final page {ct['rank_final']}**")
    fin = pd.DataFrame(r["final"][:6])
    if len(fin):
        fin["flags"] = fin["flags"].apply(lambda x: ", ".join(x))
        fin["cold"] = fin["cold"].map({True: "🆕", False: ""})
        st.dataframe(fin, hide_index=True, use_container_width=True)
    lt = r["latency_ms"]
    st.caption(f"Served latency: total {lt['total_ms']:.0f} ms (understand {lt['query_understanding_ms']:.0f} · retrieve "
               f"{lt['retrieval_ms']:.0f} · rank {lt['ranking_ms']:.0f} · rerank {lt['rerank_ms']:.0f})")


with tabs[0]:
    scoreboard()
    all_flows = get_flows(W)
    fid = st.selectbox("Flow", [f["id"] for f in all_flows], format_func=lambda i: next(f["title"] for f in all_flows if f["id"] == i), key="flow_sel")
    fl, res = flow_results(fid)
    st.markdown(f"### {fl['title']}")
    st.write(fl["blurb"])
    cols_ = ["ndcg@10", "precision@10", "exp_revenue@10", "exp_margin@10", "unfulfillable@10", "avg_return_prob@10"]
    mean = pd.DataFrame({k: {m: np.mean([r["variants"][k][m] for r in res]) for m in cols_} for k in STAGES}).T
    a, b = st.columns([3, 2])
    with a:
        st.markdown(f"**Per-stage mean across {len(res)} examples**")
        st.dataframe(mean.round(3).rename(columns={"ndcg@10": "NDCG@10", "precision@10": "Prec@10", "exp_revenue@10": "Exp rev",
                                                    "exp_margin@10": "Exp margin", "unfulfillable@10": "Unfulfillable", "avg_return_prob@10": "Avg P(return)"}),
                     use_container_width=True)
    with b:
        fig = go.Figure()
        fig.add_bar(x=[s[0] for s in STAGES], y=mean["exp_margin@10"], name="Exp. margin@10", marker_color="#2a6fdb")
        fig.add_scatter(x=[s[0] for s in STAGES], y=mean["ndcg@10"], name="NDCG@10", yaxis="y2", mode="lines+markers", line=dict(color="#e8710a"))
        fig.update_layout(height=270, margin=dict(l=0, r=0, t=10, b=0), yaxis2=dict(overlaying="y", side="right", range=[0, 1.05]),
                          legend=dict(orientation="h", y=1.15), xaxis_title="stage (A retrieval to E final)")
        st.plotly_chart(fig, use_container_width=True)
    for r in res:
        with st.expander(f"Example · {r['example']['label']}  -  query: \"{r['example']['query']}\"", expanded=True):
            render_example(r)

# ----------------------------------------------------------------------------------------- search
with tabs[1]:
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
with tabs[2]:
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
with tabs[3]:
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
with tabs[4]:
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
with tabs[5]:
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
with tabs[6]:
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
