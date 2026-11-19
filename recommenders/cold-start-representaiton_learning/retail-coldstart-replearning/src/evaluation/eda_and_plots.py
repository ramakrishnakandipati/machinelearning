"""Automated EDA + training/evaluation/embedding charts (high-DPI PNG).

Palette: categorical slots 1..6 in fixed order (blue, orange, aqua, magenta, yellow, green), light
chart surface, recessive grid, no top/right spines, legend always present for >= 2 series.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#008300", "#6250d6", "#e34948"]
SLICE_COLOR = {"warm": CAT[0], "user_cold": CAT[1], "item_cold": CAT[2], "system_cold": CAT[3]}
SLICE_LABEL = {"warm": "Warm", "user_cold": "User cold-start", "item_cold": "Item cold-start", "system_cold": "System cold-start"}
METHOD_ORDER = ["popularity", "two_tower_multimodal", "inductive_graphsage", "maml_user_adapter", "contrastive_ssl", "unified_ensemble"]
METHOD_COLOR = {"popularity": MUTED, "two_tower_multimodal": CAT[0], "inductive_graphsage": CAT[1], "maml_user_adapter": CAT[2],
                "contrastive_ssl": CAT[3], "unified_ensemble": CAT[5]}
DPI = 200


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _fig(nr, nc, size):
    fig, axes = plt.subplots(nr, nc, figsize=size, facecolor=SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _legend(ax, **kw):
    kw.setdefault("fontsize", 8)
    leg = ax.legend(frameon=False, labelcolor=INK2, **kw)
    return leg


def plot_eda(items, users, inter, manifest, out: Path):
    fig, axes = _fig(2, 2, (13, 9))
    # (a) long-tail of item popularity (train vs all), warm vs cold items
    ax = axes[0, 0]
    cnt = inter.groupby("item_id").size().reindex(items.item_id).fillna(0).values
    order = np.argsort(-cnt)
    cold = items.is_cold_item.values[order]
    rank = np.arange(1, len(cnt) + 1)
    ax.scatter(rank[~cold], cnt[order][~cold] + 0.8, s=6, color=CAT[0], label="Warm items", linewidths=0)
    ax.scatter(rank[cold], cnt[order][cold] + 0.8, s=6, color=CAT[2], label="Zero-shot (cold) items", linewidths=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Item popularity rank"); ax.set_ylabel("Interactions (all splits)")
    ax.set_title("Item interaction long tail", loc="left", fontsize=11)
    _legend(ax)
    # (b) user activity distribution by group
    ax = axes[0, 1]
    ucnt = inter.groupby("user_id").size().reindex(users.user_id).fillna(0)
    bins = np.arange(0, 41, 2)
    grp_color = {"warm": CAT[0], "few_shot": CAT[1], "zero_shot": CAT[2], "system_cold": CAT[3]}
    for g, c in grp_color.items():
        ax.hist(np.clip(ucnt[users.user_group.values == g], 0, 40), bins=bins, histtype="step", linewidth=1.6, color=c,
                label=g.replace("_", " "))
    ax.set_xlabel("Interactions per user (clipped at 40)"); ax.set_ylabel("Users")
    ax.set_title("User activity by cold-start group", loc="left", fontsize=11)
    _legend(ax)
    # (c) evaluation slice sizes (val / test)
    ax = axes[1, 0]
    sl = list(SLICE_COLOR)
    x = np.arange(len(sl)); w = 0.38
    for k, sp in enumerate(("val", "test")):
        vals = [manifest["slices"][s][sp] for s in sl]
        bars = ax.bar(x + (k - 0.5) * w, vals, w - 0.04, color=[SLICE_COLOR[s] for s in sl], alpha=1.0 if sp == "val" else 0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([SLICE_LABEL[s] for s in sl], fontsize=8)
    ax.set_ylabel("Evaluation interactions")
    ax.set_title("Evaluation slices (solid = validation, light = test)", loc="left", fontsize=11)
    # (d) event types by channel
    ax = axes[1, 1]
    ct = pd.crosstab(inter.channel, inter.event_type)
    bottom = np.zeros(len(ct))
    for k, ev in enumerate(ct.columns):
        ax.bar(ct.index, ct[ev].values, bottom=bottom, color=CAT[k], label=ev.replace("_", " "), width=0.6, edgecolor=SURFACE, linewidth=1.5)
        bottom += ct[ev].values
    ax.set_ylabel("Interactions"); ax.set_title("Omnichannel events", loc="left", fontsize=11)
    _legend(ax)
    fig.suptitle("Retail cold-start dataset: long tail and cold-start slice breakdown", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_loss_curves(histories: dict, out: Path):
    names = {"two_tower": "Two-tower (InfoNCE)", "graphsage": "Inductive GraphSAGE (InfoNCE link loss)",
             "maml": "MAML adapter (query NLL)", "ssl": "Contrastive SSL (NT-Xent)"}
    fig, axes = _fig(2, 2, (12, 8))
    for ax, (k, title) in zip(axes.ravel(), names.items()):
        h = pd.DataFrame(histories.get(k, []))
        if h.empty:
            continue
        ax.plot(h.epoch, h.train_loss, color=CAT[0], linewidth=2, label="train")
        ax.plot(h.epoch, h.val_loss, color=CAT[1], linewidth=2, label="validation")
        if k == "maml" and "val_loss_prior_only" in h:
            ax.plot(h.epoch, h.val_loss_prior_only, color=MUTED, linewidth=1.5, linestyle="--", label="validation, prior only (no adaptation)")
        if k == "ssl":
            ax.set_title(title + " - validation on held-out cold items", loc="left", fontsize=10)
        else:
            ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        i = int(h.val_loss.idxmin())
        ax.scatter([h.epoch[i]], [h.val_loss[i]], s=36, color=CAT[1], zorder=5, edgecolor=SURFACE, linewidths=1.5)
        _legend(ax)
    fig.suptitle("Training / validation convergence", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def project_2d(X: np.ndarray, seed: int = 42):
    try:
        import umap  # type: ignore
        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=20, min_dist=0.2, metric="cosine").fit_transform(X), "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=seed, init="pca", perplexity=30, metric="cosine").fit_transform(X), "t-SNE"


def plot_embedding_space(E: np.ndarray, items: pd.DataFrame, out: Path, title_extra: str = ""):
    xy, name = project_2d(E)
    fig, axes = _fig(1, 2, (14, 6.2))
    cold = items.is_cold_item.values
    l1 = items.l1.values
    cats = sorted(set(l1))
    ax = axes[0]
    for k, c in enumerate(cats):
        m = (l1 == c) & ~cold
        ax.scatter(xy[m, 0], xy[m, 1], s=9, color=CAT[k], label=c, linewidths=0, alpha=0.85)
        m = (l1 == c) & cold
        ax.scatter(xy[m, 0], xy[m, 1], s=34, color=CAT[k], marker="D", edgecolor=INK, linewidths=0.7, alpha=1.0)
    ax.scatter([], [], s=34, color=MUTED, marker="D", edgecolor=INK, linewidths=0.7, label="zero-shot cold item (diamond)")
    ax.set_title(f"{name}: unified item space coloured by category", loc="left", fontsize=11)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    _legend(ax, loc="best", ncol=2, markerscale=1.0)
    ax = axes[1]
    ax.scatter(xy[~cold, 0], xy[~cold, 1], s=9, color=CAT[0], label="warm items", linewidths=0, alpha=0.6)
    ax.scatter(xy[cold, 0], xy[cold, 1], s=22, color=CAT[1], label="zero-shot cold items", linewidths=0.6, edgecolor=SURFACE)
    ax.set_title(f"{name}: warm vs zero-shot items share one space", loc="left", fontsize=11)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    _legend(ax)
    fig.suptitle("Representation space (d=128, L2-normalised)" + title_extra, x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return name


def plot_slice_comparison(df: pd.DataFrame, out: Path):
    metrics = [("recall@20", "Recall@20"), ("ndcg@10", "NDCG@10"), ("cold_items_recommended_pct", "% of top-10 slots that are cold items")]
    fig, axes = _fig(1, 3, (17, 5.2))
    sl = [s for s in SLICE_COLOR if s in set(df["slice"])]
    methods = [m for m in METHOD_ORDER if m in set(df.method)]
    x = np.arange(len(sl)); w = 0.8 / len(methods)
    for ax, (m, lab) in zip(axes, metrics):
        for k, me in enumerate(methods):
            vals = [float(df[(df.method == me) & (df["slice"] == s)][m].iloc[0]) for s in sl]
            ax.bar(x + (k - (len(methods) - 1) / 2) * w, vals, w * 0.88, color=METHOD_COLOR[me], label=me.replace("_", " "))
        ax.set_xticks(x); ax.set_xticklabels([SLICE_LABEL[s] for s in sl], fontsize=8)
        ax.set_title(lab, loc="left", fontsize=11)
    _legend(axes[0], fontsize=7)
    fig.suptitle("Stratified offline evaluation (test split)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_ann_tradeoff(bench: dict, out: Path):
    fig, axes = _fig(1, 2, (12, 5))
    df = pd.DataFrame(bench["results"])
    ax = axes[0]
    for idx, c in (("IndexHNSWFlat", CAT[0]), ("IndexIVFPQ", CAT[1])):
        d = df[df["index"] == idx]
        ax.plot(d["p95_ms"], d["recall@10_vs_exact"], marker="o", color=c, linewidth=2, markersize=6, label=idx)
    f = df[df["index"] == "IndexFlatIP"].iloc[0]
    ax.scatter([f.p95_ms], [f["recall@10_vs_exact"]], color=CAT[5], s=60, label="IndexFlatIP (exact)", zorder=5)
    ax.set_xlabel("p95 latency per query (ms)"); ax.set_ylabel("Recall@10 vs exact")
    ax.set_title("ANN recall vs latency", loc="left", fontsize=11)
    _legend(ax)
    ax = axes[1]
    d = df[df["index"] == "IndexHNSWFlat"]
    for col, c in (("p50_ms", CAT[0]), ("p95_ms", CAT[1]), ("p99_ms", CAT[3])):
        ax.plot(d["efSearch"], d[col], marker="o", color=c, linewidth=2, label=col.split("_")[0])
    ax.set_xscale("log", base=2); ax.set_xlabel("HNSW efSearch"); ax.set_ylabel("Latency (ms)")
    ax.set_title("HNSW latency percentiles", loc="left", fontsize=11)
    _legend(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
