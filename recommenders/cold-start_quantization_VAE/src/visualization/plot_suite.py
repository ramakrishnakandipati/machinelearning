"""Publication-grade charts (200 DPI) written to outputs/visualizations/.

Colour: validated categorical palette, assigned in fixed slot order (blue, orange, aqua, yellow, magenta, green,
violet, red); neutrals for baselines; thin marks, recessive grid, legends on every multi-series panel.
"""
from __future__ import annotations

import json
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score

from src.config import LOGS_DIR, VIZ_DIR

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NEUTRALS = {"popularity": "#8a8984", "random": "#c3c2b7"}
DPI = 200

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8,
    "figure.titlesize": 14, "figure.titleweight": "bold", "lines.linewidth": 2.0,
})

MAIN = ["semantic_trie", "generative_seq", "codebook_bridge", "hybrid"]
LABEL = {"semantic_trie": "1 Semantic Trie", "generative_seq": "2 Generative Seq", "codebook_bridge": "3 Codebook Bridge",
         "hybrid": "4 Hybrid", "popularity": "Popularity (log-based)", "random": "Random",
         "raw_content_knn": "Raw-feature kNN", "trie_vq": "Trie (VQ-VAE)", "bridge_vq": "Bridge (VQ-VAE)",
         "hybrid_vq": "Hybrid (VQ-VAE)", "bridge_dot": "Bridge: dot only", "bridge_overlap": "Bridge: overlap only"}
REGIME_LABEL = {"warm": "Warm", "item_cold": "Item cold", "user_cold": "User cold", "system_cold": "System cold (Day-0)"}
COLOR = {a: SERIES[i] for i, a in enumerate(MAIN)}
COLOR.update(NEUTRALS)


def _save(fig, name: str) -> str:
    path = VIZ_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _panel(ax, letter: str, title: str):
    ax.set_title(f"{letter}  {title}")


# =============================================================================== 01
def plot_data_distributions(ds: dict) -> str:
    items, users, inter = ds["items"], ds["users"], ds["interactions"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Omnichannel retail dataset: splits, skew and cold-start ratios", x=0.01, ha="left")

    ax = axs[0, 0]
    order = ["train", "val", "test", "item_cold_test", "user_cold_history", "user_cold_test"]
    cnt = inter["split"].value_counts().reindex(order).fillna(0)
    ax.bar(range(len(cnt)), cnt.values, color=[SERIES[0]] * 3 + [SERIES[1]] + [SERIES[2]] * 2, width=0.7)
    ax.set_xticks(range(len(cnt))); ax.set_xticklabels(["train", "val", "test", "item-cold\ntest", "user-cold\nhistory", "user-cold\ntest"])
    for i, v in enumerate(cnt.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_ylabel("interactions")
    ax.legend(handles=[Patch(color=SERIES[0], label="warm"), Patch(color=SERIES[1], label="cold item"),
                       Patch(color=SERIES[2], label="cold user")], loc="upper right")
    _panel(ax, "a", "Interactions by evaluation split")

    ax = axs[0, 1]
    ch = inter.groupby(["channel"]).size()
    ch = ch.reindex(["web", "mobile_app", "in_store"])
    ax.bar(ch.index, ch.values, color=SERIES[:3], width=0.6)
    for i, v in enumerate(ch.values):
        ax.text(i, v, f"{v / ch.sum():.0%}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_ylabel("interactions")
    _panel(ax, "b", "Channel mix (web / app / in-store POS)")

    ax = axs[0, 2]
    cats = ds["categories"]
    tab = items.groupby(["category", "is_cold"]).size().unstack(fill_value=0).reindex(cats)
    ax.bar(range(len(cats)), tab[False], color=SERIES[0], width=0.7, label="warm items")
    ax.bar(range(len(cats)), tab[True], bottom=tab[False], color=SERIES[1], width=0.7, label="cold items (new SKUs)")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.set_ylabel("items"); ax.legend()
    _panel(ax, "c", "Category skew of the catalog")

    ax = axs[1, 0]
    per_user = inter.groupby("user_id").size().reindex(users["user_id"]).fillna(0)
    bins = np.arange(0, per_user.max() + 2, 2)
    ax.hist(per_user[~users["is_cold"].values], bins=bins, color=SERIES[0], label="warm users", alpha=0.95)
    ax.hist(per_user[users["is_cold"].values], bins=bins, color=SERIES[2], label="cold users", alpha=0.95)
    ax.set_xlabel("interactions per user"); ax.set_ylabel("users"); ax.legend()
    _panel(ax, "d", "Activity distribution")

    ax = axs[1, 1]
    pop = inter[inter["split"] == "train"]["item_id"].value_counts()
    ax.loglog(np.arange(1, len(pop) + 1), pop.values, color=SERIES[0])
    ax.axvline(len(pop), color=SERIES[1], lw=1.2, ls="--")
    ax.text(len(pop) * 0.93, pop.values.max() * 0.5, "200 cold items\nhave 0 training\ninteractions", ha="right", color=INK2, fontsize=8)
    ax.set_xlabel("item rank (log)"); ax.set_ylabel("train interactions (log)")
    _panel(ax, "e", "Long-tail popularity")

    ax = axs[1, 2]
    vals = {"cold items": items["is_cold"].mean(), "cold users": users["is_cold"].mean(),
            "interactions on\ncold items": (inter["split"] == "item_cold_test").mean(),
            "interactions of\ncold users": inter["split"].isin(["user_cold_history", "user_cold_test"]).mean()}
    ax.barh(list(vals.keys()), [v * 100 for v in vals.values()], color=[SERIES[1], SERIES[2], SERIES[1], SERIES[2]], height=0.55)
    for i, v in enumerate(vals.values()):
        ax.text(v * 100 + 0.2, i, f"{v:.1%}", va="center", color=INK2)
    ax.set_xlabel("% of total"); ax.invert_yaxis()
    _panel(ax, "f", "Cold-start ratios (system cold = no logs at Day-0)")
    fig.tight_layout()
    return _save(fig, "01_data_distributions.png")


# =============================================================================== 02
def plot_feature_correlations(ds: dict) -> str:
    items, X = ds["items"], ds["item_feats"]
    blocks = ds["feature_blocks"]
    cats = ds["categories"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Item feature space before quantization", x=0.01, ha="left")

    ax = axs[0, 0]
    num = X[:, slice(*blocks["numeric"])]
    txt = X[:, slice(*blocks["text"])][:, :6]
    img = X[:, slice(*blocks["image"])][:, :6]
    D = np.concatenate([num, txt, img], 1)
    names = ["log price", "margin"] + [f"txt{i}" for i in range(6)] + [f"img{i}" for i in range(6)]
    sns.heatmap(np.corrcoef(D.T), ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1, xticklabels=names, yticklabels=names,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"}, linewidths=0.2, linecolor=SURFACE)
    ax.grid(False); ax.tick_params(labelsize=7)
    _panel(ax, "a", "Correlation: numeric / text / image features")

    ax = axs[0, 1]
    data = [np.log10(items.loc[items.category == c, "price"]) for c in cats]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color=INK, lw=1.2),
                    flierprops=dict(markersize=2, markerfacecolor=INK2, markeredgecolor="none"))
    for p, col in zip(bp["boxes"], SERIES):
        p.set_facecolor(col); p.set_edgecolor(SURFACE); p.set_alpha(0.95)
    ax.set_xticks(range(1, len(cats) + 1)); ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.set_ylabel("log10 price")
    _panel(ax, "b", "Price distribution by category")

    ax = axs[0, 2]
    data = [items.loc[items.category == c, "margin"] for c in cats]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color=INK, lw=1.2),
                    flierprops=dict(markersize=2, markerfacecolor=INK2, markeredgecolor="none"))
    for p, col in zip(bp["boxes"], SERIES):
        p.set_facecolor(col); p.set_edgecolor(SURFACE); p.set_alpha(0.95)
    ax.set_xticks(range(1, len(cats) + 1)); ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.set_ylabel("gross margin")
    _panel(ax, "c", "Margin by category (business signal)")

    pca = PCA(n_components=min(30, X.shape[1])).fit(X)
    Z = pca.transform(X)
    cat_idx = items["cat_idx"].values
    handles = [Line2D([], [], marker="o", ls="", color=SERIES[i], label=c, markersize=6) for i, c in enumerate(cats)]

    ax = axs[1, 0]
    ax.scatter(Z[:, 0], Z[:, 1], c=[SERIES[i] for i in cat_idx], s=9, edgecolors=SURFACE, linewidths=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.legend(handles=handles, ncol=2, fontsize=7, loc="best")
    _panel(ax, "d", "PCA of raw multimodal item features")

    ax = axs[1, 1]
    T = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z[:, :30])
    ax.scatter(T[:, 0], T[:, 1], c=[SERIES[i] for i in cat_idx], s=9, edgecolors=SURFACE, linewidths=0.3)
    cold = items["is_cold"].values
    ax.scatter(T[cold, 0], T[cold, 1], facecolors="none", edgecolors=INK, s=26, linewidths=0.7)
    ax.legend(handles=handles[:1] + [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK, label="cold item (ringed)")],
              loc="best", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    _panel(ax, "e", "t-SNE: categories cluster; cold items sit inside them")

    ax = axs[1, 2]
    ev = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(np.arange(1, len(ev) + 1), ev * 100, color=SERIES[0])
    ax.set_xlabel("principal components"); ax.set_ylabel("cumulative variance (%)")
    _panel(ax, "f", f"PCA spectrum (input dim {X.shape[1]})")
    fig.tight_layout()
    return _save(fig, "02_feature_correlations.png")


# =============================================================================== 03
def plot_training_curves() -> str:
    tr = json.loads((LOGS_DIR / "training_logs.json").read_text())
    va = json.loads((LOGS_DIR / "validation_logs.json").read_text())
    col = {"vq_vae": SERIES[0], "rq_vae": SERIES[1]}
    nm = {"vq_vae": "VQ-VAE (K=256)", "rq_vae": "RQ-VAE (M=4, K=256)"}
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("VQ-VAE vs RQ-VAE: training dynamics", x=0.01, ha="left")

    def line(ax, key, field, src, ls="-", label_suffix=""):
        for m in ("vq_vae", "rq_vae"):
            rows = src[m] if src is tr else src[m]["epochs"]
            ax.plot([r["epoch"] for r in rows], [r[field] for r in rows], color=col[m], ls=ls,
                    label=nm[m] + label_suffix)

    ax = axs[0, 0]
    line(ax, "recon", "recon", tr, "-", " train"); line(ax, "recon", "recon", va, "--", " val")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("reconstruction MSE"); ax.legend(fontsize=7)
    _panel(ax, "a", "Reconstruction loss (solid train, dashed val)")

    ax = axs[0, 1]
    line(ax, "commit", "commit", tr); ax.set_xlabel("epoch"); ax.set_ylabel("commitment loss (beta * ||z_e - sg[z_q]||^2)"); ax.legend()
    _panel(ax, "b", "Commitment loss (summed over RQ levels)")

    ax = axs[0, 2]
    line(ax, "ppl", "perplexity", tr); ax.set_xlabel("epoch"); ax.set_ylabel("codebook perplexity (mean per level)")
    ax.axhline(256, color=INK2, lw=0.8, ls=":"); ax.text(2, 250, "K = 256", va="top", fontsize=8, color=INK2); ax.legend()
    _panel(ax, "c", "Codebook perplexity")

    ax = axs[1, 0]
    line(ax, "util", "codebook_utilization_pct", va); ax.set_xlabel("epoch"); ax.set_ylabel("codes used (%)"); ax.legend()
    _panel(ax, "d", "Codebook utilization (dead-code reset keeps it high)")

    ax = axs[1, 1]
    line(ax, "coll", "collision_rate", va); ax.set_xlabel("epoch"); ax.set_ylabel("items sharing a full Semantic ID")
    ax.legend(); _panel(ax, "e", "Semantic-ID collision rate")

    ax = axs[1, 2]
    line(ax, "reset", "dead_code_resets", tr); ax.set_xlabel("epoch"); ax.set_ylabel("cumulative dead-code resets"); ax.legend()
    _panel(ax, "f", "Dead-code expiration events")
    fig.tight_layout()
    return _save(fig, "03_vq_vs_rq_training_curves.png")


# =============================================================================== 04
def plot_codebook_utilization(bundle) -> str:
    rq, vq = bundle.rq, bundle.vq
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Codebook usage and hierarchical balance", x=0.01, ha="left")
    K = rq.K

    def usage(ax, codes, color, title, letter):
        cnt = np.bincount(codes, minlength=K)
        ax.bar(np.arange(K), np.sort(cnt)[::-1], color=color, width=1.0)
        ax.set_xlabel("codes (sorted by usage)"); ax.set_ylabel("items")
        used = (cnt > 0).sum()
        ax.text(0.97, 0.93, f"{used}/{K} codes used\nperplexity {np.exp(-(cnt[cnt>0]/cnt.sum() * np.log(cnt[cnt>0]/cnt.sum())).sum()):.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
        _panel(ax, letter, title)

    usage(axs[0, 0], vq.codes[:, 0], SERIES[0], "VQ-VAE single codebook", "a")
    for l, (r, c) in enumerate([(0, 1), (0, 2), (1, 0), (1, 1)]):
        usage(axs[r, c], rq.codes[:, l], SERIES[1], f"RQ-VAE level {l + 1} codebook", "bcde"[l])

    ax = axs[1, 2]
    c1 = rq.codes[:, 0]
    n_children = [len(np.unique(rq.codes[c1 == c, 1])) for c in np.unique(c1)]
    items_per = np.bincount(c1)[np.unique(c1)]
    ax.scatter(items_per, n_children, s=22, color=SERIES[1], edgecolors=SURFACE, linewidths=0.5, label="level-1 node")
    ax.plot([0, items_per.max()], [0, items_per.max()], color=INK2, lw=0.8, ls=":", label="1 child per item")
    ax.set_xlabel("items under level-1 node"); ax.set_ylabel("distinct level-2 children")
    h = rq.prefix_ids[-1]
    ax.text(0.03, 0.93, f"{len(np.unique(c1))} level-1 nodes\n{len(np.unique(rq.prefix_ids[1]))} (c1,c2) nodes\n"
                        f"{len(np.unique(h))} unique full IDs / {rq.N} items", transform=ax.transAxes, va="top", fontsize=8, color=INK2)
    ax.legend(loc="lower right")
    _panel(ax, "f", "Hierarchy balance: tree fan-out from level 1")
    fig.tight_layout()
    return _save(fig, "04_codebook_utilization.png")


# =============================================================================== 05
def _top_color(codes: np.ndarray, n: int = 8):
    vals, cnt = np.unique(codes, return_counts=True)
    top = vals[np.argsort(-cnt)[:n]]
    col = np.array(["#c3c2b7"] * len(codes), dtype=object)
    for i, v in enumerate(top):
        col[codes == v] = SERIES[i]
    return col, top


def plot_semantic_clusters(bundle) -> str:
    rq, ds = bundle.rq, bundle.dataset
    items = ds["items"]
    T = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(rq.ze)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("RQ-VAE Semantic IDs on the encoder latent space (t-SNE of z_e)", x=0.01, ha="left")
    nmi1 = normalized_mutual_info_score(items["sub_idx"], rq.codes[:, 0])
    nmi_cat = normalized_mutual_info_score(items["cat_idx"], rq.codes[:, 0])
    nmi2 = normalized_mutual_info_score(items["sub_idx"], rq.prefix_ids[1])

    ax = axs[0]
    ax.scatter(T[:, 0], T[:, 1], c=[SERIES[i] for i in items["cat_idx"]], s=10, edgecolors=SURFACE, linewidths=0.3)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=SERIES[i], label=c, markersize=6) for i, c in enumerate(ds["categories"])],
              ncol=2, fontsize=7, loc="best")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    _panel(ax, "a", "Ground truth: product category")

    rng = np.random.default_rng(3)
    for ax, codes, title, letter in ((axs[1], rq.codes[:, 0], f"Level-1 code c1  (NMI vs category {nmi_cat:.2f})", "b"),
                                     (axs[2], rq.prefix_ids[1], f"Level-2 prefix (c1,c2)  (NMI vs sub-category {nmi2:.2f})", "c")):
        uniq = np.unique(codes)
        pal = sns.color_palette("husl", len(uniq))
        order = rng.permutation(len(uniq))                       # neighbours in id-space get unrelated hues
        cmap = {u: pal[order[i]] for i, u in enumerate(uniq)}
        ax.scatter(T[:, 0], T[:, 1], c=[cmap[c] for c in codes], s=11, edgecolors=SURFACE, linewidths=0.3)
        sizes = np.bincount(np.searchsorted(uniq, codes), minlength=len(uniq))
        if letter == "b":                                         # label the biggest level-1 nodes at their centroid
            for j in np.argsort(-sizes)[:14]:
                m = codes == uniq[j]
                ax.text(T[m, 0].mean(), T[m, 1].mean(), str(uniq[j]), fontsize=7, ha="center", va="center", color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
        ax.text(0.01, 0.01, f"{len(uniq)} nodes | colour = node id", transform=ax.transAxes, fontsize=8, color=INK2)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        _panel(ax, letter, title)
    fig.tight_layout()
    return _save(fig, "05_semantic_id_clusters.png")


# =============================================================================== 06
def plot_model_comparison(df: pd.DataFrame) -> str:
    regimes = ["warm", "item_cold", "user_cold", "system_cold"]
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3)
    fig.suptitle("Benchmark: four Semantic-ID cold-start approaches vs baselines", x=0.01, ha="left")
    approaches = MAIN + ["popularity", "random"]
    tab = df.set_index(["approach", "regime"])

    def grouped(ax, metric, ylabel, title, letter, log=False, approaches=approaches):
        w = 0.8 / len(approaches)
        for j, a in enumerate(approaches):
            vals = [tab.loc[(a, r), metric] if (a, r) in tab.index else np.nan for r in regimes]
            ax.bar(np.arange(len(regimes)) + (j - (len(approaches) - 1) / 2) * w, vals, w * 0.88, color=COLOR[a],
                   edgecolor=SURFACE, linewidth=0.8, label=LABEL[a])
        ax.set_xticks(range(len(regimes))); ax.set_xticklabels([REGIME_LABEL[r].replace(" (", "\n(") for r in regimes], fontsize=8)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        _panel(ax, letter, title)

    ax = fig.add_subplot(gs[0, 0]); grouped(ax, "recall@10", "Recall@10", "Recall@10", "a"); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[0, 1]); grouped(ax, "ndcg@10", "NDCG@10", "NDCG@10", "b")
    ax = fig.add_subplot(gs[0, 2]); grouped(ax, "coverage@10", "catalog coverage", "Coverage@10 (share of candidates ever recommended)", "c")
    ax = fig.add_subplot(gs[1, 0]); grouped(ax, "latency_ms", "ms / request (log)", "Inference latency", "d", log=True)

    ax = fig.add_subplot(gs[1, 1], projection="polar")
    dims = ["Recall@10", "NDCG@10", "MRR", "Coverage", "Speed"]
    mean = df[df.approach.isin(MAIN)].groupby("approach")[["recall@10", "ndcg@10", "mrr", "coverage@10", "latency_ms"]].mean()
    norm = mean.copy()
    for c in ["recall@10", "ndcg@10", "mrr", "coverage@10"]:
        norm[c] = mean[c] / mean[c].max()
    norm["latency_ms"] = np.log10(mean["latency_ms"].min() + 1e-3) / np.log10(mean["latency_ms"] + 1e-3)
    norm["latency_ms"] = (1 / (1 + np.log10(1 + mean["latency_ms"])))
    norm["latency_ms"] = norm["latency_ms"] / norm["latency_ms"].max()
    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist(); ang += ang[:1]
    for a in MAIN:
        v = norm.loc[a].tolist(); v += v[:1]
        ax.plot(ang, v, color=COLOR[a], label=LABEL[a], lw=2); ax.fill(ang, v, color=COLOR[a], alpha=0.08)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(dims, fontsize=8); ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=7)
    ax.set_title("e  Profile averaged over regimes (relative)", loc="left", pad=18)

    ax = fig.add_subplot(gs[1, 2])
    rows = ["semantic_trie", "trie_vq", "generative_seq", "codebook_bridge", "bridge_vq", "bridge_dot", "bridge_overlap",
            "hybrid", "hybrid_vq", "raw_content_knn", "popularity", "random"]
    hm = pd.DataFrame({REGIME_LABEL[r]: [tab.loc[(a, r), "recall@10"] if (a, r) in tab.index else np.nan for a in rows] for r in regimes},
                      index=[LABEL[a] for a in rows])
    sns.heatmap(hm, annot=True, fmt=".3f", cmap=sns.light_palette(SERIES[0], as_cmap=True), ax=ax, cbar=False,
                linewidths=1, linecolor=SURFACE, annot_kws={"fontsize": 8})
    ax.grid(False); ax.set_ylabel(""); ax.tick_params(labelsize=8); ax.set_xticklabels([t.get_text().replace(" (", "\n(") for t in ax.get_xticklabels()])
    _panel(ax, "f", "Recall@10: all variants incl. VQ-VAE and ablations")
    fig.tight_layout()
    return _save(fig, "06_model_comparison.png")


def make_all(bundle, eval_df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    out["01"] = plot_data_distributions(bundle.dataset)
    out["02"] = plot_feature_correlations(bundle.dataset)
    out["03"] = plot_training_curves()
    out["04"] = plot_codebook_utilization(bundle)
    out["05"] = plot_semantic_clusters(bundle)
    out["06"] = plot_model_comparison(eval_df)
    return out
