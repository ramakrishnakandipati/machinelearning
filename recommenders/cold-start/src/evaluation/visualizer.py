"""Matplotlib charts: EDA, training curves, strategy comparison and coverage/novelty trade-off."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order (colour-blind-safe, never cycled): one colour per strategy
STRATEGY_COLORS = {
    "Random": "#9AA0A6", "Popularity": "#E69F00", "Content-Based": "#009E73",
    "Two-Tower (ID Dropout)": "#0072B2", "Hybrid Meta-Router": "#CC3311",
}
INK, MUTED, GRID = "#1F2328", "#656D76", "#E4E7EB"


def _style(ax: plt.Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def plot_cold_start_distribution(inter: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, path: Path) -> None:
    """Power-law interaction curves with warm vs cold user/item segments highlighted."""
    tr = inter[inter.split == "train"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ucnt = tr.groupby("user_id").size().reindex(users.user_id).fillna(0).values
    order = np.sort(ucnt)[::-1]
    ax = axes[0]
    x = np.arange(1, len(order) + 1)
    ax.plot(x[order > 0], order[order > 0], color="#0072B2", lw=2, label="Warm users (train history)")
    ax.fill_between(x[order > 0], order[order > 0], color="#0072B2", alpha=0.08)
    n_zero = int((order == 0).sum())
    ax.axvspan(len(order) - n_zero, len(order), color="#CC3311", alpha=0.25,
               label=f"Cold users: 0 train interactions ({n_zero:,})")
    semi = int(((order >= 1) & (order < 5)).sum())
    ax.set_yscale("log"); ax.set_xlabel("User rank"); ax.set_ylabel("Training interactions (log)")
    ax.set_title(f"User activity: power law, {semi:,} semi-cold users (1-4 interactions)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    ax = axes[1]
    icnt = tr.groupby("item_id").size().reindex(items.item_id).fillna(0).values
    order_i = np.argsort(-icnt)
    s = icnt[order_i]
    cold = items.is_cold_item.values[order_i]
    xi = np.arange(1, len(s) + 1)
    ax.plot(xi[~cold], np.maximum(s[~cold], 0.8), color="#0072B2", lw=2, label="Warm items")
    ax.scatter(xi[cold], np.full(cold.sum(), 0.8), s=8, color="#CC3311",
               label=f"Cold items: new SKUs, 0 train sales ({int(cold.sum())})")
    ax.set_yscale("log"); ax.set_xlabel("Item rank"); ax.set_ylabel("Training interactions (log)")
    ax.set_title("Catalog long tail and newly launched SKUs", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_training_curve(history: List[Dict[str, float]], path: Path) -> None:
    """Train vs validation loss and validation Recall@10 across epochs."""
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(h.epoch, h.train_loss, color="#0072B2", lw=2, marker="o", ms=5, label="Train loss (with ID dropout)")
    ax.plot(h.epoch, h.val_loss, color="#E69F00", lw=2, marker="o", ms=5, label="Validation loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Sampled-softmax loss"); ax.legend(frameon=False)
    ax.set_title("Two-Tower loss", loc="left", fontsize=11); _style(ax)
    ax = axes[1]
    ax.plot(h.epoch, h.val_recall_at_10, color="#009E73", lw=2, marker="o", ms=5)
    b = h.val_recall_at_10.idxmax()
    ax.annotate(f"best {h.val_recall_at_10[b]:.3f}", (h.epoch[b], h.val_recall_at_10[b]), textcoords="offset points",
                xytext=(0, 9), ha="center", color=INK, fontsize=9)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Recall@10")
    ax.set_title("Validation Recall@10", loc="left", fontsize=11); _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_strategy_comparison(df: pd.DataFrame, path: Path, metric: str = "recall@10") -> None:
    """Grouped bars: strategies x {warm, cold_user, cold_item} (+ session)."""
    tiers = [t for t in ["warm", "cold_user", "cold_item", "session"] if t in set(df.tier)]
    strategies = [s for s in STRATEGY_COLORS if s in set(df.strategy)]
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.8 / len(strategies)
    for j, s in enumerate(strategies):
        vals = [float(df[(df.strategy == s) & (df.tier == t)][metric].iloc[0]) for t in tiers]
        xs = np.arange(len(tiers)) + (j - (len(strategies) - 1) / 2) * w
        bars = ax.bar(xs, vals, w * 0.92, color=STRATEGY_COLORS[s], label=s)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(np.arange(len(tiers)))
    ax.set_xticklabels([t.replace("_", "-").title() for t in tiers])
    ax.set_ylabel(metric.title()); ax.set_title(f"Strategy comparison by coldness tier ({metric})", loc="left", fontsize=12)
    ax.legend(frameon=False, ncol=len(strategies), loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=9)
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_coverage_novelty(points: pd.DataFrame, path: Path) -> None:
    """Trade-off curve: cold-item Recall@10 vs cold-item catalog coverage as the new-item share grows."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(points.cold_item_coverage, points.recall, color="#CC3311", lw=2, marker="o", ms=7)
    for r in points.itertuples():
        ax.annotate(f"{r.label}", (r.cold_item_coverage, r.recall), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color=INK)
    ax.set_xlabel("Cold-item catalog coverage (share of new SKUs ever recommended)")
    ax.set_ylabel("Recall@10 (all-item ranking, warm users)")
    ax.set_title("Exploration trade-off: new-item slots vs accuracy", loc="left", fontsize=12)
    _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
