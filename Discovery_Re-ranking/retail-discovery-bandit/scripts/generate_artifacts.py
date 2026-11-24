"""Run the full metric suites, off-policy evaluation, diagnostic plots and Playwright API screenshots.

Requires `scripts/train_and_simulate.py` to have been run first.

    python scripts/generate_artifacts.py [--skip-screenshots] [--ope-requests N]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.synthetic_retail import SimEnvironment
from src.evaluation import metrics as M
from src.evaluation.ope import (ESTIMATORS, clipping_sweep, collect_ope_log, ope_table, run_ope)
from src.system import day_rows, load_system
from src.utils import get_logger, gpu_info, load_config, output_dir, save_json

POLICY_COLORS = {"pure_ranker": "#4C78A8", "epsilon_greedy": "#F58518", "linucb_mmr": "#54A24B", "linucb_high_expl": "#B279A2"}
POLICY_LABELS = {"pure_ranker": "Pure ranker", "epsilon_greedy": "Epsilon-greedy", "linucb_mmr": "LinUCB + MMR",
                 "linucb_high_expl": "High-explore LinUCB"}
DPI = 200
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)


def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pcolors(pols):
    return [POLICY_COLORS[p] for p in pols]


# ----------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------
def plot_data_profile(world, plots: Path):
    it, us, df = world.items, world.users, world.interactions
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    sns.histplot(np.log10(it.price), bins=50, ax=ax[0, 0], color="#4C78A8")
    ax[0, 0].set(title="Catalog price distribution", xlabel="log10(price)")
    sns.histplot(data=it, x="days_since_launch", hue="is_cold", bins=60, ax=ax[0, 1], palette={False: "#4C78A8", True: "#E45756"}, log_scale=(False, True))
    ax[0, 1].set(title="Launch age (red = cold-start, <=14 days)", xlabel="days since launch")
    cnt = df.groupby("item").size().reindex(range(world.n_items), fill_value=0)
    ax[0, 2].hist(cnt[~it.is_cold.values], bins=60, color="#4C78A8", label="warm")
    ax[0, 2].axvline(0, color="#E45756", lw=0)
    ax[0, 2].bar([0], [int(it.is_cold.sum())], width=max(cnt.max() / 60, 1), color="#E45756", label=f"cold-start ({int(it.is_cold.sum())} SKUs, 0 impr.)")
    ax[0, 2].set(title="Impressions per SKU (long tail)", xlabel="impressions in 90 days", yscale="log"); ax[0, 2].legend()
    cs = it.groupby("category_name").size().sort_values()
    ax[1, 0].barh(cs.index, cs.values, color="#4C78A8"); ax[1, 0].set(title="SKUs per category (15 x 4 subcategories)")
    pj = df.assign(persona=us.persona_name.values[df["user"].values]).groupby("persona").agg(ctr=("click", "mean"), buy=("buy", "mean"))
    pj.plot.bar(ax=ax[1, 1], color=["#4C78A8", "#54A24B"], rot=15); ax[1, 1].set(title="Historical CTR / buy rate by persona", xlabel="")
    sns.boxplot(data=us, x="persona_name", y="avg_item_price", ax=ax[1, 2], color="#9ecae9"); ax[1, 2].set(title="Avg item price by persona", xlabel="")
    ax[1, 2].tick_params(axis="x", rotation=15)
    savefig(fig, plots / "01_data_profile.png")


def plot_retrieval(ret_metrics, curve, plots: Path):
    r = ret_metrics["stage1_retrieval"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(curve["epoch"], curve["loss"], color="#4C78A8", marker="o"); ax[0].set(title="Two-tower training loss (in-batch softmax + logQ)", xlabel="epoch")
    v = curve.dropna(subset=["val_recall@200"])
    ax[1].plot(v["epoch"], v["val_recall@50"], marker="o", label="Recall@50", color="#54A24B")
    ax[1].plot(v["epoch"], v["val_recall@200"], marker="o", label="Recall@200", color="#F58518")
    ax[1].set(title="Validation recall (best checkpoint kept)", xlabel="epoch"); ax[1].legend()
    names = ["Recall@50", "Recall@200", "HitRate@200", "ColdStart_Recall@200", "ColdStart_HitRate@200"]
    vals = [r[n] for n in names]
    ax[2].barh(names, vals, color=["#4C78A8"] * 3 + ["#E45756"] * 2)
    for i, x in enumerate(vals): ax[2].text(x + 0.005, i, f"{x:.3f}", va="center")
    ax[2].set(title=f"Stage-1 test metrics (p99 retrieval {r['latency_ms']['p99_ms']:.2f} ms)", xlim=(0, max(vals) * 1.2)); ax[2].invert_yaxis()
    savefig(fig, plots / "02_retrieval_metrics.png")


def plot_ranker(sys_, plots: Path, gap, rm):
    world, builder, ranker = sys_["world"], sys_["builder"], sys_["ranker"]
    te = day_rows(world, *sys_["cfg"]["data"]["ranker_test_window"])
    pc, pb = ranker.predict(builder.build(te.user.values, te.item.values))
    cold = world.cold_eval
    pcc, pbc = ranker.predict(builder.build(cold.user.values, cold.item.values))
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for a, (y, p, name) in zip(ax[0, :2], [(te.click.values, pc, "click"), (te.buy.values, pb, "buy")]):
        cc = M.calibration_curve(y, p, 10)
        a.plot([0, cc.mean_pred.max()], [0, cc.mean_pred.max()], "--", color="grey", label="perfect")
        a.plot(cc.mean_pred, cc.frac_pos, marker="o", color="#4C78A8", label="XGBoost (isotonic)")
        a.set(title=f"P({name}) calibration, warm test window", xlabel="predicted", ylabel="observed"); a.legend()
    ax[0, 2].hist(pc, bins=60, alpha=.65, density=True, color="#4C78A8", label=f"warm (mean {pc.mean():.3f})")
    ax[0, 2].hist(pcc, bins=60, alpha=.65, density=True, color="#E45756", label=f"cold-start (mean {pcc.mean():.3f})")
    ax[0, 2].set(title="Predicted P(click): cold-start SKUs score lower", xlabel="p_click", xlim=(0, .3)); ax[0, 2].legend()
    b = pd.DataFrame(gap["matched_relevance_bins"])
    w = 0.38; x = np.arange(len(b))
    ax[1, 0].bar(x - w / 2, b.true_warm, w, label="true (warm)", color="#bbbbbb"); ax[1, 0].bar(x + w / 2, b.true_cold, w, label="true (cold)", color="#999999", hatch="//")
    ax[1, 0].plot(x - w / 2, b.pred_warm, "o-", color="#4C78A8", label="pred (warm)"); ax[1, 0].plot(x + w / 2, b.pred_cold, "s-", color="#E45756", label="pred (cold)")
    ax[1, 0].set(title="Equal true relevance, unequal scores (cold-start gap)", xlabel="true P(click) quintile", ylabel="P(click)", xticks=x); ax[1, 0].legend(fontsize=8)
    ax[1, 1].bar(x, 100 * b.rel_gap, color="#E45756"); ax[1, 1].axhline(0, color="k", lw=.8)
    ax[1, 1].set(title=f"Relative score gap cold vs warm (mean {100 * gap['mean_relative_gap_matched']:.1f}%)", xlabel="true P(click) quintile", ylabel="% gap", xticks=x)
    fi = pd.DataFrame(rm["feature_importance"]["click"], index=["gain"]).T.sort_values("gain").tail(12)
    ax[1, 2].barh(fi.index, fi.gain, color="#4C78A8"); ax[1, 2].set(title="Click-ranker feature importance (gain)")
    savefig(fig, plots / "03_ranker_calibration_and_cold_gap.png")


def plot_policy_comparison(summ: pd.DataFrame, plots: Path):
    pols = list(summ.index)
    panels = [("ctr", "CTR"), ("buy_rate_per_impression", "Buy rate / impression"), ("avg_reward_per_impression", "Avg reward / impression"),
              ("realized_profit_per_slate", "Realised profit / slate"), ("avg_margin_per_slate", "Avg margin / slate"),
              ("repeat_purchase_rate_7d", "7-day repeat purchase rate"),
              ("cold_start_impression_share", "Cold-start impression share"), ("cold_start_conversion_rate", "Cold-start conversion"),
              ("discovery_conversion_rate", "Discovery conversion"), ("discovery_purchase_share", "Discovery share of purchases"),
              ("catalog_coverage_pct", "Catalog coverage (%)"), ("ild", "Intra-list diversity"),
              ("gini_exposure", "Gini (exposure inequality)"), ("shannon_entropy_normalised", "Normalised Shannon entropy"),
              ("serendipity_score", "Serendipity score"), ("add_to_cart_rate", "Add-to-cart rate")]
    fig, axes = plt.subplots(4, 4, figsize=(18, 13))
    for a, (col, title) in zip(axes.ravel(), panels):
        a.bar(range(len(pols)), summ[col].values, color=pcolors(pols))
        for i, v in enumerate(summ[col].values): a.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        a.set(title=title, xticks=range(len(pols)), xticklabels=[POLICY_LABELS[p].replace(" ", "\n", 1) for p in pols]); a.tick_params(axis="x", labelsize=7)
    savefig(fig, plots / "04_policy_comparison_metrics.png")


def plot_exposure(rec_all: pd.DataFrame, world, plots: Path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for p, g in rec_all.groupby("policy", sort=False):
        e = np.sort(np.bincount(g["item"].values, minlength=world.n_items)).astype(float)
        cum = np.cumsum(e) / e.sum()
        ax[0].plot(np.linspace(0, 1, len(e)), cum, color=POLICY_COLORS[p], label=f"{POLICY_LABELS[p]} (Gini {M.gini(e):.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].set(title="Exposure Lorenz curve (all 10k SKUs)", xlabel="cumulative share of SKUs", ylabel="cumulative share of impressions"); ax[0].legend(fontsize=8)
    for p, g in rec_all.groupby("policy", sort=False):
        e = np.bincount(g["item"].values, minlength=world.n_items)
        ax[1].plot(np.arange(1, 1 + (e > 0).sum()), np.sort(e[e > 0])[::-1], color=POLICY_COLORS[p], label=POLICY_LABELS[p])
    ax[1].set(title="Rank-frequency of exposure", xlabel="SKU rank", ylabel="impressions", yscale="log", xscale="log"); ax[1].legend(fontsize=8)
    cat = world.items.groupby("category_name").size() / world.n_items
    width = 0.2
    for k, (p, g) in enumerate(rec_all.groupby("policy", sort=False)):
        sh = g.assign(cn=world.items.category_name.values[g["item"].values]).groupby("cn").size() / len(g)
        ratio = (sh / cat).reindex(cat.index).fillna(0)
        ax[2].bar(np.arange(len(cat)) + k * width, ratio.values, width, color=POLICY_COLORS[p], label=POLICY_LABELS[p])
    ax[2].axhline(1, color="k", lw=.8, ls="--"); ax[2].set_xticks(np.arange(len(cat)) + 1.5 * width); ax[2].set_xticklabels(cat.index, rotation=70, ha="right", fontsize=8)
    ax[2].set(title="Category exposure ratio (exposure share / catalog share)")
    savefig(fig, plots / "05_exposure_fairness.png")


def plot_learning(rec_all: pd.DataFrame, plots: Path):
    rec_all = rec_all.assign(bucket=rec_all["request"] // 200)
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
    spec = [("reward", "Mean reward / impression"), ("is_cold", "Cold-start impression share"), ("buy", "Buy rate / impression"),
            ("exploration_bonus", "Mean exploration bonus  alpha*sqrt(x'A^-1 x)"), ("p_buy_true", "True P(buy) of shown items"), ("tier", "Mean novelty tier of shown items")]
    for a, (col, title) in zip(ax.ravel(), spec):
        for p, g in rec_all.groupby("policy", sort=False):
            s = g.groupby("bucket")[col].mean()
            a.plot(s.index * 200, s.values, color=POLICY_COLORS[p], marker="o", ms=3, label=POLICY_LABELS[p])
        a.set(title=title, xlabel="request #")
    ax[0, 0].legend(fontsize=8)
    savefig(fig, plots / "06_online_learning_curves.png")


def plot_slate(rec_all: pd.DataFrame, plots: Path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    pols = list(rec_all.policy.unique())
    comp = rec_all.groupby(["policy", "slot"])[["base_utility", "learned_residual", "exploration_bonus"]].mean()
    for k, p in enumerate(pols):
        c = comp.loc[p]
        bottom = np.zeros(len(c)); xs = np.arange(len(c)) + k * 0.2 - 0.3
        for col, hatch, alpha in [("base_utility", None, 1.0), ("learned_residual", "//", .7), ("exploration_bonus", "xx", .5)]:
            v = c[col].values
            ax[0].bar(xs, v, 0.18, bottom=bottom, color=POLICY_COLORS[p], hatch=hatch, alpha=alpha, edgecolor="white", label=POLICY_LABELS[p] if col == "base_utility" else None)
            bottom += np.where(v > 0, v, 0)
    ax[0].set(title="Mean score decomposition per slot (solid=utility, //=learned residual, xx=bonus)", xlabel="slot"); ax[0].legend(fontsize=7)
    tiers = rec_all.groupby(["policy", "tier"]).size().unstack(fill_value=0)
    tiers = tiers.div(tiers.sum(1), axis=0).reindex(pols)
    names = {0: "previously bought", 1: "new SKU", 2: "unseen brand", 3: "unseen subcategory"}
    bottom = np.zeros(len(pols))
    for t, col in zip(sorted(tiers.columns), ["#bdbdbd", "#9ecae9", "#fdae6b", "#e6550d"]):
        ax[1].bar(range(len(pols)), tiers[t].values, bottom=bottom, color=col, label=names[t]); bottom += tiers[t].values
    ax[1].set(title="Novelty-tier mix of displayed items", xticks=range(len(pols)), xticklabels=[POLICY_LABELS[p].replace(" ", "\n", 1) for p in pols]); ax[1].legend(fontsize=8)
    ild = [rec_all[rec_all.policy == p].groupby("request")["ild"].first().values for p in pols]
    parts = ax[2].violinplot(ild, showmedians=True)
    for body, p in zip(parts["bodies"], pols): body.set_facecolor(POLICY_COLORS[p]); body.set_alpha(.7)
    ax[2].set(title="Intra-list diversity (cosine distance) per slate", xticks=range(1, len(pols) + 1), xticklabels=[POLICY_LABELS[p].replace(" ", "\n", 1) for p in pols])
    savefig(fig, plots / "07_slate_diagnostics.png")


def plot_ope(res, res0, df, sweep, plots: Path):
    pols = list(res.keys())
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
    for a, (r, title) in zip(ax[0, :2], [(res, "all slots"), (res0, "slot-0 only (exact propensities)")]):
        for i, p in enumerate(pols):
            for j, k in enumerate(ESTIMATORS):
                e = r[p][k]
                a.errorbar(i + (j - 2) * 0.13, e["estimate"], yerr=[[e["estimate"] - e["ci_low"]], [e["ci_high"] - e["estimate"]]], fmt="o", ms=5,
                           color=sns.color_palette("deep")[j], capsize=3, label=k if i == 0 else None)
            tv = r[p]["DM"]["true_value"]; a.hlines(tv, i - 0.4, i + 0.4, color="k", lw=2, label="ground truth" if i == 0 else None)
        a.set(title=f"OPE estimates with 95% bootstrap CI - {title}", xticks=range(len(pols)), xticklabels=[POLICY_LABELS[p].replace(" ", "\n", 1) for p in pols], ylabel="expected reward / slot")
        a.legend(fontsize=7, ncol=3)
    err = pd.DataFrame({k: [res[p][k]["rel_error_pct"] for p in pols] for k in ESTIMATORS}, index=[POLICY_LABELS[p] for p in pols])
    err.plot.bar(ax=ax[0, 2], rot=15, color=sns.color_palette("deep")[:5]); ax[0, 2].axhline(0, color="k", lw=.8); ax[0, 2].set(title="Relative error vs ground truth (%)"); ax[0, 2].legend(fontsize=7)
    ess = [res[p]["diagnostics"]["ess"] for p in pols]; n = res[pols[0]]["diagnostics"]["n_records"]
    ax[1, 0].bar(range(len(pols)), ess, color=pcolors(pols)); ax[1, 0].axhline(n, color="grey", ls="--", label=f"n={n}")
    for i, v in enumerate(ess): ax[1, 0].text(i, v, f"{v:.0f}\n({100 * v / n:.0f}%)", ha="center", va="bottom", fontsize=8)
    ax[1, 0].set(title="Effective sample size", xticks=range(len(pols)), xticklabels=[POLICY_LABELS[p].replace(" ", "\n", 1) for p in pols]); ax[1, 0].legend()
    for p in pols:
        w = df[f"pi_{p}"] / df["mu"]
        ax[1, 1].hist(np.log10(w.clip(lower=1e-3)), bins=70, histtype="step", color=POLICY_COLORS[p], lw=1.8, label=f"{POLICY_LABELS[p]} (max {w.max():.1f})")
    ax[1, 1].axvline(np.log10(res[pols[0]]["diagnostics"]["clip_threshold"]), color="k", ls="--", label="clip"); ax[1, 1].set(title="Importance weights pi/mu", xlabel="log10(w)"); ax[1, 1].legend(fontsize=7)
    for p in pols:
        s = sweep[sweep.policy == p]
        ax[1, 2].plot(s['clip'], s.IPS, "o--", color=POLICY_COLORS[p], alpha=.6); ax[1, 2].plot(s['clip'], s.DR, "s-", color=POLICY_COLORS[p]); ax[1, 2].hlines(s.true.iloc[0], 2, 100, color=POLICY_COLORS[p], lw=1, ls=":")
    ax[1, 2].set(title="Clipping sweep (dashed = IPS, solid = DR, dotted = truth)", xlabel="clip threshold", xscale="log")
    savefig(fig, plots / "08_off_policy_evaluation.png")


def plot_latency(lat: pd.DataFrame, gpu: dict, plots: Path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    stages = [s for s in sorted(lat.stage.unique()) if s != "total"]
    pols = list(lat.policy.unique()); colors = sns.color_palette("viridis", len(stages))
    bottom = np.zeros(len(pols))
    for s, c in zip(stages, colors):
        v = np.array([lat[(lat.policy == p) & (lat.stage == s)]["p50"].values[0] for p in pols]); ax[0].bar(range(len(pols)), v, bottom=bottom, color=c, label=s); bottom += v
    ax[0].set(title="Median latency by stage (ms)", xticks=range(len(pols)), xticklabels=[POLICY_LABELS[p].replace(" ", "\n", 1) for p in pols]); ax[0].legend(fontsize=7, ncol=2)
    tot = lat[lat.stage == "total"].set_index("policy").loc[pols]
    x = np.arange(len(pols)); w = .26
    for j, q in enumerate(["p50", "p95", "p99"]): ax[1].bar(x + (j - 1) * w, tot[q], w, label=q)
    ax[1].set(title="End-to-end latency percentiles (ms)", xticks=x, xticklabels=[POLICY_LABELS[p].replace(" ", "\n", 1) for p in pols]); ax[1].legend()
    ax[2].axis("off")
    txt = "\n".join([f"{k}: {v}" for k, v in gpu.items()])
    ax[2].text(0, 1, "GPU / VRAM\n\n" + txt, va="top", family="monospace", fontsize=10)
    savefig(fig, plots / "09_latency_and_gpu.png")


# ----------------------------------------------------------------------------------------
# Playwright
# ----------------------------------------------------------------------------------------
def api_screenshots(cfg, shots: Path, logger):
    import httpx
    from playwright.sync_api import sync_playwright
    host, port = cfg["api"]["host"], cfg["api"]["port"]
    base = f"http://{host}:{port}"
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", host, "--port", str(port)], cwd=root,
                            stdout=open(output_dir(cfg, "logs") / "api_server.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                if httpx.get(base + "/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("API did not start")
        # exercise the API so /metrics and /dashboard are populated
        rng = np.random.default_rng(0)
        transcript = []
        for u in rng.integers(0, 2000, size=40):
            r = httpx.post(base + "/discover", json={"user_id": int(u), "k": 5}, timeout=30).json()
            for s in r["slate"]:
                reward = float(rng.choice([0.0, 0.3, 1.3], p=[.9, .07, .03]))
                fb = httpx.post(base + "/feedback", json={"user_id": int(u), "item_id": s["item_id"], "slot": s["slot"], "reward": reward,
                                                           "propensity_score": s["propensity_score"]}, timeout=30).json()
            if len(transcript) < 3:
                transcript.append(r)
        sample = httpx.post(base + "/discover", json={"user_id": 7, "k": 5, "alpha": 0.8, "lambda": 0.6}, timeout=30).json()
        save_json({"discover_example": sample, "feedback_example": fb, "health": httpx.get(base + "/health").json(),
                   "metrics": httpx.get(base + "/metrics").json()}, output_dir(cfg, "metrics") / "api_examples.json")
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
            page.goto(base + "/docs"); page.wait_for_selector(".opblock-summary"); page.screenshot(path=str(shots / "01_swagger_docs.png"), full_page=True)
            try:   # drive Swagger UI: Try it out -> Execute POST /discover
                page.click("#operations-default-discover_discover_post .opblock-summary")
                page.click("#operations-default-discover_discover_post button.try-out__btn")
                page.fill("#operations-default-discover_discover_post textarea.body-param__text", json.dumps({"user_id": 7, "k": 5, "alpha": 0.8, "lambda": 0.6}, indent=2))
                page.click("#operations-default-discover_discover_post button.execute")
                page.wait_for_selector("#operations-default-discover_discover_post .live-responses-table", timeout=20000)
                page.wait_for_timeout(800)
                page.locator("#operations-default-discover_discover_post").screenshot(path=str(shots / "02_swagger_discover_execute.png"))
            except Exception as e:
                logger.warning("swagger interaction screenshot skipped: %s", e)
            for u, name in [(7, "03_demo_shelf_user7.png"), (1234, "04_demo_shelf_user1234.png")]:
                page.goto(f"{base}/demo?user_id={u}"); page.wait_for_selector(".card"); page.screenshot(path=str(shots / name), full_page=True)
            page.goto(base + "/demo?user_id=7&policy=pure_ranker"); page.wait_for_selector(".card"); page.screenshot(path=str(shots / "05_demo_shelf_pure_ranker_user7.png"), full_page=True)
            page.goto(base + "/dashboard"); page.wait_for_selector("#grid .tile"); page.wait_for_timeout(500); page.screenshot(path=str(shots / "06_metrics_dashboard.png"), full_page=True)
            page.goto(base + "/health"); page.screenshot(path=str(shots / "07_health_json.png"))
            browser.close()
        logger.info("screenshots saved to %s", shots)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


# ----------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-screenshots", action="store_true")
    ap.add_argument("--ope-requests", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    if args.ope_requests:
        cfg["ope"]["n_requests"] = args.ope_requests
    log_dir, mdir_m = output_dir(cfg, "logs"), output_dir(cfg, "metrics")
    plots, shots = output_dir(cfg, "plots"), output_dir(cfg, "screenshots")
    logger = get_logger("artifacts", log_dir / "artifacts.log")
    t0 = time.time()
    sys_ = load_system(cfg, seed=cfg["seed"])
    sys_["cfg"] = cfg
    world, pipe = sys_["world"], sys_["pipeline"]
    env = SimEnvironment(world, cfg["seed"])

    # ---- stage reports from training ----
    ret = json.load(open(mdir_m / "stage1_retrieval_metrics.json"))
    rm = json.load(open(mdir_m / "stage3_ranker_metrics.json"))
    gap = json.load(open(mdir_m / "cold_start_bottleneck.json"))
    curve = pd.read_csv(mdir_m / "two_tower_training_curve.csv")
    summ = pd.DataFrame(json.load(open(mdir_m / "stage4_policy_summary.json"))).T
    rec_all = pd.read_parquet(mdir_m / "sim_records_all.parquet")
    lat = pd.read_csv(mdir_m / "latency_by_stage_policy.csv")

    # ---- slices & exposure ----
    n_stock = int(world.items.in_stock.sum())
    for by in ("persona", "is_cold", "category"):
        frames = []
        for p, g in rec_all.groupby("policy", sort=False):
            s = M.slice_metrics(g, by, world.n_items, n_stock); s.insert(0, "policy", p); frames.append(s)
        pd.concat(frames).to_csv(mdir_m / f"slice_metrics_by_{by}.csv", index=False)
    exp_frames = []
    for p, g in rec_all.groupby("policy", sort=False):
        e = M.exposure_by_group(g.assign(category_name=world.items.category_name.values[g["item"].values]), "category_name",
                                world.items, "category_name"); e.insert(0, "policy", p); exp_frames.append(e)
    pd.concat(exp_frames).to_csv(mdir_m / "exposure_by_category.csv", index=False)

    # ---- OPE ----
    logger.info("collecting OPE log (%d requests, behaviour = Boltzmann(tau=%.2f) over pure-ranker utility)", cfg["ope"]["n_requests"], cfg["ope"]["behavior_tau"])
    frozen = sys_["state"].clone()
    from src.reranker.bandit import load_bandit_states
    trained = pipe.bandits
    df = collect_ope_log(pipe, env, frozen, trained, cfg, cfg["ope"]["n_requests"], logger=logger)
    df.to_parquet(mdir_m / "ope_log.parquet", index=False)
    pols = list(pipe.policies.keys())
    res, res0 = run_ope(df, pols, cfg), run_ope(df, pols, cfg, slot0_only=True)
    save_json({"all_slots": res, "slot0_only": res0}, mdir_m / "ope_results.json")
    ope_table(res).to_csv(mdir_m / "ope_results_all_slots.csv", index=False)
    ope_table(res0).to_csv(mdir_m / "ope_results_slot0_only.csv", index=False)
    sweep = pd.concat([clipping_sweep(df, p, switch_tau=cfg["ope"]["switch_threshold"]) for p in pols])
    sweep.to_csv(mdir_m / "ope_clipping_sweep.csv", index=False)
    for p in pols:
        e = res[p]
        logger.info("OPE %-17s truth=%.4f | IPS=%.4f SNIPS=%.4f DR=%.4f SwDR=%.4f DM=%.4f | ESS=%.0f (%.1f%%) max w=%.1f",
                    p, e["DM"]["true_value"], e["IPS"]["estimate"], e["SNIPS"]["estimate"], e["DR"]["estimate"], e["Switch-DR"]["estimate"],
                    e["DM"]["estimate"], e["diagnostics"]["ess"], 100 * e["diagnostics"]["ess_fraction"], e["diagnostics"]["max_weight"])

    # ---- extra system metrics ----
    gi = gpu_info()
    save_json({"gpu": gi, "latency_total_ms": lat[lat.stage == "total"].to_dict("records")}, mdir_m / "system_performance.json")
    full = {"stage1_retrieval": ret["stage1_retrieval"], "stage3_ranker": rm["stage3_ranker"], "cold_start_gap": {k: v for k, v in gap.items() if k != "matched_relevance_bins"},
            "stage4_policies": summ.to_dict("index"), "ope_all_slots": {p: {k: res[p][k] for k in ESTIMATORS} for p in pols},
            "system": {"gpu": gi}}
    save_json(full, mdir_m / "full_metrics_report.json")

    # ---- plots ----
    logger.info("rendering plots")
    plot_data_profile(world, plots)
    plot_retrieval(ret, curve, plots)
    plot_ranker(sys_, plots, gap, rm)
    plot_policy_comparison(summ.astype(float), plots)
    plot_exposure(rec_all, world, plots)
    plot_learning(rec_all, plots)
    plot_slate(rec_all, plots)
    plot_ope(res, res0, df, sweep, plots)
    plot_latency(lat, gi, plots)

    # ---- screenshots ----
    if not args.skip_screenshots:
        try:
            api_screenshots(cfg, shots, logger)
        except Exception as e:
            logger.error("screenshots failed: %s", e)
    manifest = sorted(str(p.relative_to(plots.parents[0])) for p in plots.parents[0].rglob("*") if p.is_file())
    save_json(manifest, mdir_m / "artifact_manifest.json")
    logger.info("artifacts complete in %.1fs (%d files in output/)", time.time() - t0, len(manifest))


if __name__ == "__main__":
    main()
