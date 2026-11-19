"""Held-out test metrics (overall + by segment), popularity baseline, training/validation/embedding charts."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from .config import CHARTS, METRICS, MODELS  # noqa: E402
from .logger import get_logger, stage_timer  # noqa: E402
from .train import I_KEYS, U_KEYS, all_item_emb, device_of, metrics_from_ranks, ranks_of_positive, to_dev  # noqa: E402
from .model import TwoTower  # noqa: E402

log = get_logger("evaluate", "03_model_training.log")


def load_model(prep, dev) -> TwoTower:
    ck = torch.load(MODELS / "two_tower_best.pt", map_location=dev)
    m = TwoTower(ck["sizes"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def test_metrics(prep, model, dev) -> dict:
    te = to_dev(prep.splits["test"], U_KEYS + ["item"], dev)
    item_t = to_dev(prep.item_table, I_KEYS, dev)
    emb = all_item_emb(model, item_t)
    n = len(te["item"])
    ranks = ranks_of_positive(model, te, torch.arange(n, device=dev), emb).cpu()
    # popularity baseline: rank by train positive frequency
    cnt = np.exp(prep.log_q)
    pop_rank = torch.as_tensor((cnt[None, :] > cnt[prep.splits["test"]["item"]][:, None]).sum(1))
    sp = prep.splits["test"]
    seg = {"overall": np.ones(n, bool), "warm_users": ~sp["unseen_user"], "cold_start_users": sp["unseen_user"]}
    for c in sorted(set(sp["pref_channel"])):
        seg[f"channel_{c}"] = sp["pref_channel"] == c
    out = {"n_test_examples": n, "catalog_size": int(len(prep.items)), "random_recall_50_expected": 50 / len(prep.items),
           "model": {}, "popularity_baseline": {}}
    for name, m in seg.items():
        mt = torch.as_tensor(m)
        if mt.sum() == 0:
            continue
        out["model"][name] = {"n": int(mt.sum()), **metrics_from_ranks(ranks[mt])}
        out["popularity_baseline"][name] = {"n": int(mt.sum()), **metrics_from_ranks(pop_rank[mt])}
    return out


def charts(prep, model, dev):
    sns.set_theme(style="whitegrid")
    h = pd.read_csv(METRICS / "training_history.csv")
    be = int(h.best_epoch.iloc[0])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(h.epoch, h.train_loss, "o-", label="train loss")
    ax.plot(h.epoch, h.val_loss, "s-", label="validation loss")
    ax.axvline(be, color="gray", ls="--", label=f"best checkpoint / early-stop reference (epoch {be})")
    ax.set(xlabel="epoch", ylabel="InfoNCE loss (in-batch, logQ-corrected)", title="Training vs validation loss")
    ax.legend()
    fig.tight_layout(); fig.savefig(CHARTS / "training_validation_loss.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for c in ("val_recall_10", "val_recall_50"):
        ax[0].plot(h.epoch, h[c], "o-", label=c)
    ax[0].set(title="Validation recall", xlabel="epoch"); ax[0].legend()
    for c in ("val_ndcg_10", "val_mrr_10"):
        ax[1].plot(h.epoch, h[c], "o-", label=c)
    ax[1].set(title="Validation NDCG@10 / MRR@10", xlabel="epoch"); ax[1].legend()
    for a in ax:
        a.axvline(be, color="gray", ls="--")
    fig.tight_layout(); fig.savefig(CHARTS / "validation_retrieval_metrics.png", dpi=150); plt.close(fig)

    emb = np.load(MODELS / "item_embeddings.npy")
    cat = prep.items.category_l1.to_numpy()
    p2 = PCA(2, random_state=0).fit_transform(emb)
    t2 = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(emb)
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    for a, z, t in ((ax[0], p2, "PCA"), (ax[1], t2, "t-SNE")):
        sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=cat, s=10, linewidth=0, ax=a, legend=a is ax[1])
        a.set(title=f"Item Tower embeddings - {t} (colored by category_l1)")
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), markerscale=2)
    fig.tight_layout(); fig.savefig(CHARTS / "embedding_space_umap_pca.png", dpi=150); plt.close(fig)


def main(prep=None, model=None):
    from . import dataset
    dev = device_of()
    if prep is None:
        users, items, ev = (pd.read_parquet(f"outputs/data/{n}.parquet") for n in ("users", "items", "events"))
        prep = dataset.prepare(users, items, ev)
    model = model or load_model(prep, dev)
    with stage_timer(log, "test evaluation + charts"):
        res = test_metrics(prep, model, dev)
        (METRICS / "offline_test_metrics.json").write_text(json.dumps(res, indent=2))
        o, p = res["model"]["overall"], res["popularity_baseline"]["overall"]
        log.info("TEST overall: R@10=%.4f R@50=%.4f NDCG@10=%.4f MRR@10=%.4f | popularity R@50=%.4f | random R@50=%.4f",
                 o["recall_10"], o["recall_50"], o["ndcg_10"], o["mrr_10"], p["recall_50"], res["random_recall_50_expected"])
        for k, v in res["model"].items():
            log.info("  segment %-18s n=%-6d R@10=%.4f R@50=%.4f NDCG@10=%.4f", k, v["n"], v["recall_10"], v["recall_50"], v["ndcg_10"])
        charts(prep, model, dev)
    return res


if __name__ == "__main__":
    main()
