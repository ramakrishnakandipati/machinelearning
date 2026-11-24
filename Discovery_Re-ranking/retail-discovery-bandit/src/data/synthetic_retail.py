"""Synthetic omnichannel grocery/retail world.

Generates a 10k-SKU catalog (15 categories x 4 subcategories = 60), 2,000 users with
shopping personas, a 90-day impression/click/buy history, 15% cold-start SKUs with *zero*
historical interactions, and a hidden ground-truth simulator (latent user-item affinity,
price fit, curiosity/discovery effect and 7-day repeat-purchase probability).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

CATEGORIES = [
    "produce", "dairy_eggs", "bakery", "meat_poultry", "seafood", "frozen", "pantry_staples",
    "snacks", "beverages", "household", "personal_care", "baby_kids", "pet", "natural_organic",
    "deli_gourmet",
]
PERSONAS = ["organic_plant_based", "family_bulk", "gourmet_specialty", "budget_conscious"]
_PERSONA_PREFS = {
    "organic_plant_based": ["produce", "natural_organic", "dairy_eggs", "pantry_staples", "beverages"],
    "family_bulk": ["pantry_staples", "frozen", "household", "baby_kids", "meat_poultry", "snacks"],
    "gourmet_specialty": ["deli_gourmet", "seafood", "bakery", "meat_poultry", "natural_organic"],
    "budget_conscious": ["pantry_staples", "frozen", "snacks", "household", "produce", "beverages"],
}
_PERSONA_PRICE_PREF = {"organic_plant_based": 1.15, "family_bulk": 0.85, "gourmet_specialty": 1.9, "budget_conscious": 0.6}
_PERSONA_PROB = [0.28, 0.27, 0.15, 0.30]


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


# ----------------------------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------------------------
class GroundTruth:
    """Hidden simulator. Models must never read from it (only the environment does)."""

    def __init__(self, u_lat, i_lat, user_price_pref, item_price_rel, item_pop_z, item_is_cold, tcfg):
        self.u_lat = torch.as_tensor(u_lat, dtype=torch.float32)
        self.i_lat = torch.as_tensor(i_lat, dtype=torch.float32)
        self.user_price_pref = torch.as_tensor(user_price_pref, dtype=torch.float32)
        self.item_price_rel = torch.as_tensor(item_price_rel, dtype=torch.float32)
        self.item_pop_z = torch.as_tensor(item_pop_z, dtype=torch.float32)
        self.item_is_cold = torch.as_tensor(item_is_cold, dtype=torch.bool)
        self.t = tcfg

    def to(self, device) -> "GroundTruth":
        g = GroundTruth.__new__(GroundTruth)
        g.t = self.t
        for k in ["u_lat", "i_lat", "user_price_pref", "item_price_rel", "item_pop_z", "item_is_cold"]:
            setattr(g, k, getattr(self, k).to(device))
        return g

    # -- core formulae (tensor in / tensor out) -----------------------------------------
    def _terms(self, u_idx: torch.Tensor, i_idx: torch.Tensor):
        cos = (self.u_lat[u_idx] * self.i_lat[i_idx]).sum(-1)
        pen = (torch.log(self.item_price_rel[i_idx]) - torch.log(self.user_price_pref[u_idx])).abs()
        pop = self.item_pop_z[i_idx]
        return cos, pen, pop

    def logits(self, u_idx, i_idx, unseen_subcat=None):
        cos, pen, pop = self._terms(u_idx, i_idx)
        t = self.t
        lc = t["click_bias"] + t["click_cos"] * cos + t["click_pop"] * pop - t["click_price"] * pen
        if unseen_subcat is not None:
            lc = lc + t["curiosity_bonus"] * unseen_subcat.float() * torch.sigmoid(4.0 * cos)
        lb = t["buy_bias"] + t["buy_cos"] * cos - t["buy_price"] * pen
        return lc, lb, cos

    def probs_t(self, u_idx, i_idx, unseen_subcat=None):
        lc, lb, cos = self.logits(u_idx, i_idx, unseen_subcat)
        p_click = torch.sigmoid(lc)
        p_buy = p_click * torch.sigmoid(lb)
        return p_click, p_buy, cos

    def probs(self, u_idx, i_idx, unseen_subcat=None) -> Tuple[np.ndarray, np.ndarray]:
        dev = self.u_lat.device
        u = torch.as_tensor(np.asarray(u_idx), dtype=torch.long, device=dev)
        i = torch.as_tensor(np.asarray(i_idx), dtype=torch.long, device=dev)
        un = None if unseen_subcat is None else torch.as_tensor(np.asarray(unseen_subcat), device=dev)
        pc, pb, _ = self.probs_t(u, i, un)
        return pc.cpu().numpy(), pb.cpu().numpy()

    def repeat_prob(self, u_idx, i_idx) -> np.ndarray:
        """P(7-day repeat purchase | bought). Higher for relevant newly-launched items (discovery value)."""
        dev = self.u_lat.device
        u = torch.as_tensor(np.asarray(u_idx), dtype=torch.long, device=dev)
        i = torch.as_tensor(np.asarray(i_idx), dtype=torch.long, device=dev)
        cos = (self.u_lat[u] * self.i_lat[i]).sum(-1).clamp(min=0.0)
        cold = self.item_is_cold[i].float()
        p = 0.12 + 0.40 * cos + 0.35 * cold * torch.sigmoid(8.0 * (cos - 0.25))
        return p.clamp(0.0, 0.95).cpu().numpy()


# ----------------------------------------------------------------------------------------
# World container
# ----------------------------------------------------------------------------------------
@dataclass
class RetailWorld:
    cfg: Dict[str, Any]
    items: pd.DataFrame
    users: pd.DataFrame
    item_emb: np.ndarray                      # observed semantic/attribute embeddings (I, D)
    interactions: pd.DataFrame                # warm-item impressions: user, item, day, click, buy
    cold_eval: pd.DataFrame                   # held-out cold-start impressions (labels from ground truth)
    truth: GroundTruth
    n_items: int = 0
    n_users: int = 0
    n_subcats: int = 0
    n_cats: int = 0
    n_brands: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "RetailWorld":
        with open(path, "rb") as f:
            return pickle.load(f)


# ----------------------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------------------
def generate_world(cfg: Dict[str, Any], device: Optional[torch.device] = None, logger=None) -> RetailWorld:
    d = cfg["data"]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    I, U, D = d["n_items"], d["n_users"], d["emb_dim"]
    C, SPC = d["n_categories"], d["subcats_per_category"]
    S, B = C * SPC, d["n_brands"]

    # ---- catalog ------------------------------------------------------------------------
    cat_c = _norm(rng.normal(size=(C, D)))
    sub_c = _norm(0.7 * cat_c[np.repeat(np.arange(C), SPC)] + 0.7 * rng.normal(size=(S, D)))
    subcat = rng.integers(0, S, size=I)
    category = subcat // SPC
    i_lat = _norm(sub_c[subcat] + 0.45 * rng.normal(size=(I, D))).astype(np.float32)
    item_emb = _norm(i_lat + 0.35 * rng.normal(size=(I, D))).astype(np.float32)

    # brands: B/C per category, zipf-weighted
    bpc = B // C
    brand = np.zeros(I, dtype=np.int64)
    zipf = 1.0 / np.arange(1, bpc + 1)
    zipf /= zipf.sum()
    for c in range(C):
        idx = np.where(category == c)[0]
        brand[idx] = c * bpc + rng.choice(bpc, size=idx.size, p=zipf)

    cat_base = np.exp(rng.normal(np.log(4.0), 0.55, size=C))
    price = cat_base[category] * np.exp(rng.normal(0, 0.45, size=I))
    cat_median = np.array([np.median(price[category == c]) for c in range(C)])
    price_rel = price / cat_median[category]
    cat_margin = rng.uniform(0.12, 0.40, size=C)
    margin = np.clip(cat_margin[category] + rng.normal(0, 0.05, size=I), 0.05, 0.60)

    n_cold = int(round(d["cold_frac"] * I))
    is_cold = np.zeros(I, dtype=bool)
    is_cold[rng.choice(I, size=n_cold, replace=False)] = True
    days = np.where(is_cold, rng.integers(0, d["cold_max_days"] + 1, size=I),
                    d["cold_max_days"] + 1 + rng.exponential(220, size=I).astype(int))
    in_stock = rng.random(I) < d["in_stock_rate"]
    pop_z = rng.normal(0, 1.0, size=I)
    pop_z = np.where(is_cold, 0.0, pop_z)       # truth: new items are average, but have no history

    items = pd.DataFrame({
        "item_id": np.arange(I), "category": category, "subcat": subcat, "brand": brand,
        "category_name": [CATEGORIES[c] for c in category],
        "price": price.round(2), "margin": margin.round(4), "days_since_launch": days,
        "is_cold": is_cold, "in_stock": in_stock,
    })

    # ---- users --------------------------------------------------------------------------
    persona = rng.choice(len(PERSONAS), size=U, p=_PERSONA_PROB)
    persona_c = np.zeros((len(PERSONAS), D))
    for p, name in enumerate(PERSONAS):
        persona_c[p] = _norm(sum(cat_c[CATEGORIES.index(c)] for c in _PERSONA_PREFS[name]) + 0.2 * rng.normal(size=D))
    fav = np.stack([rng.choice(S, size=2, replace=False) for _ in range(U)])
    u_lat = _norm(0.8 * persona_c[persona] + 0.3 * rng.normal(size=(U, D)) + 0.45 * sub_c[fav].mean(1)).astype(np.float32)
    user_price_pref = np.array([_PERSONA_PRICE_PREF[PERSONAS[p]] for p in persona]) * np.exp(rng.normal(0, 0.15, size=U))

    truth = GroundTruth(u_lat, i_lat, user_price_pref, price_rel, pop_z, is_cold, d["truth"]).to(device)

    # ---- 90-day history on warm items ---------------------------------------------------
    t = d["truth"]
    warm_idx = torch.as_tensor(np.where(~is_cold)[0], device=device)
    n_imp = d["impressions_per_user"]
    g = torch.Generator(device=device)
    g.manual_seed(cfg["seed"])
    rows_u, rows_i, rows_p = [], [], []
    for start in range(0, U, 500):
        u_ids = torch.arange(start, min(U, start + 500), device=device)
        cos = truth.u_lat[u_ids] @ truth.i_lat[warm_idx].T
        pen = (torch.log(truth.item_price_rel[warm_idx])[None, :] - torch.log(truth.user_price_pref[u_ids])[:, None]).abs()
        logit = t["expo_pop"] * truth.item_pop_z[warm_idx][None, :] + t["expo_cos"] * cos - t["expo_price"] * pen
        probs = torch.softmax(logit, dim=1)
        pick = torch.multinomial(probs, n_imp, replacement=True, generator=g)
        rows_u.append(u_ids[:, None].expand_as(pick).reshape(-1))
        rows_i.append(warm_idx[pick].reshape(-1))
    iu = torch.cat(rows_u)
    ii = torch.cat(rows_i)
    pc, pb, _ = truth.probs_t(iu, ii, None)
    click = torch.rand(pc.shape, device=device, generator=g) < pc
    buy_given_click = (pb / pc.clamp(min=1e-9))
    buy = click & (torch.rand(pc.shape, device=device, generator=g) < buy_given_click)
    day = torch.randint(0, d["history_days"], iu.shape, device=device, generator=g)
    inter = pd.DataFrame({
        "user": iu.cpu().numpy().astype(np.int32), "item": ii.cpu().numpy().astype(np.int32),
        "day": day.cpu().numpy().astype(np.int16), "click": click.cpu().numpy().astype(np.int8),
        "buy": buy.cpu().numpy().astype(np.int8),
    }).sort_values(["day", "user"]).reset_index(drop=True)

    # ---- held-out cold-start impressions (never used for training any model) -------------
    cold_ids = np.where(is_cold)[0]
    cu = rng.integers(0, U, size=(cold_ids.size, d["cold_eval_users_per_item"]))
    cu_t = torch.as_tensor(cu.reshape(-1), device=device)
    ci_t = torch.as_tensor(np.repeat(cold_ids, d["cold_eval_users_per_item"]), device=device)
    cpc, cpb, _ = truth.probs_t(cu_t, ci_t, None)
    cclick = torch.rand(cpc.shape, device=device, generator=g) < cpc
    cbuy = cclick & (torch.rand(cpc.shape, device=device, generator=g) < cpb / cpc.clamp(min=1e-9))
    cold_eval = pd.DataFrame({
        "user": cu_t.cpu().numpy().astype(np.int32), "item": ci_t.cpu().numpy().astype(np.int32),
        "click": cclick.cpu().numpy().astype(np.int8), "buy": cbuy.cpu().numpy().astype(np.int8),
        "p_click_true": cpc.cpu().numpy(), "p_buy_true": cpb.cpu().numpy(),
    })

    # ---- user table (observed attributes only) ------------------------------------------
    buys = inter[inter.buy == 1]
    avg_price = buys.assign(price=price[buys["item"].values]).groupby("user")["price"].mean()
    fallback = np.array([np.mean(price) * p for p in user_price_pref])
    avg_item_price = np.array([avg_price.get(u, fallback[u]) for u in range(U)])
    users = pd.DataFrame({
        "user_id": np.arange(U), "persona": persona, "persona_name": [PERSONAS[p] for p in persona],
        "avg_item_price": avg_item_price.round(3),
        "avg_basket_value": (avg_item_price * rng.uniform(8, 25, size=U)).round(2),
        "n_buys_90d": buys.groupby("user").size().reindex(range(U), fill_value=0).values,
        "n_impressions_90d": inter.groupby("user").size().reindex(range(U), fill_value=0).values,
    })

    world = RetailWorld(cfg=cfg, items=items, users=users, item_emb=item_emb, interactions=inter,
                        cold_eval=cold_eval, truth=truth.to("cpu"), n_items=I, n_users=U,
                        n_subcats=S, n_cats=C, n_brands=B,
                        meta={"personas": PERSONAS, "categories": CATEGORIES})
    if logger:
        logger.info("world: %d items (%d cold, %d out of stock) | %d users | %d impressions | CTR=%.4f BuyRate=%.4f | cold_eval=%d (CTR=%.4f)",
                    I, n_cold, int((~in_stock).sum()), U, len(inter), inter.click.mean(), inter.buy.mean(),
                    len(cold_eval), cold_eval.click.mean())
    return world


# ----------------------------------------------------------------------------------------
# Online environment (uses ground truth)
# ----------------------------------------------------------------------------------------
class SimEnvironment:
    """Draws click / buy / 7-day-repeat outcomes from the hidden ground truth."""

    def __init__(self, world: RetailWorld, seed: int = 0):
        self.world = world
        self.truth = world.truth
        self.rng = np.random.default_rng(seed)
        r = world.cfg["reward"]
        self.r_click, self.r_buy = r["click"], r["buy"]

    def reward_value(self, click: np.ndarray, buy: np.ndarray) -> np.ndarray:
        # buy implies click; reward in {0, r_click, r_click + r_buy}
        return click * self.r_click + buy * self.r_buy

    def expected_reward(self, u_idx, i_idx, unseen_subcat) -> np.ndarray:
        pc, pb = self.truth.probs(u_idx, i_idx, unseen_subcat)
        return pc * self.r_click + pb * self.r_buy

    def respond(self, u_idx: int, items: np.ndarray, unseen_subcat: np.ndarray) -> Dict[str, np.ndarray]:
        u = np.full(len(items), u_idx)
        pc, pb = self.truth.probs(u, items, unseen_subcat)
        click = (self.rng.random(len(items)) < pc).astype(np.int8)
        cond = pb / np.maximum(pc, 1e-9)
        buy = (click.astype(bool) & (self.rng.random(len(items)) < cond)).astype(np.int8)
        rp = self.truth.repeat_prob(u, items)
        repeat = (buy.astype(bool) & (self.rng.random(len(items)) < rp)).astype(np.int8)
        return {"click": click, "buy": buy, "repeat": repeat, "p_click": pc, "p_buy": pb,
                "reward": self.reward_value(click, buy)}
