"""End-to-end Stage 1-4 orchestrator with Boltzmann propensity scoring and structured logging.

Stage 1  GPU two-tower retrieval (top-200)         Stage 2  hard filters (stock, recent purchases)
Stage 3  XGBoost P(click)/P(buy)                    Stage 4  utility -> guardrails -> bandit -> MMR -> propensity
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.state import TIER_NAMES, UserState, novelty_tier_pairs
from src.evaluation.metrics import intra_list_diversity
from src.models.primary_ranker import FEATURES, FeatureBuilder, PrimaryRanker
from src.models.two_tower import TwoTowerRetriever
from src.reranker.bandit import LinUCBBandit
from src.reranker.diversity_mmr import mmr_select
from src.reranker.guardrails import Guardrails
from src.reranker.utility import novelty_from_tier, utility, utility_weights
from src.utils import JsonlLogger, StageTimer

CONTEXT_FEATURES = [
    "bias", "p_click", "p_buy", "p_buy_x_margin", "novelty", "abs_price_logratio", "is_cold", "log_age",
    "tt_cos", "content_cos", "margin", "unseen_subcat", "unseen_brand", "utility",
    "persona_0", "persona_1", "persona_2", "persona_3",
] + [f"uixemb_{j}" for j in range(8)]
SHARED_CONTEXT_IDX = [0, 1, 3, 4, 6, 8]       # hybrid LinUCB shared features z
_F = {n: i for i, n in enumerate(FEATURES)}


@dataclass
class PolicySpec:
    name: str
    type: str                     # pure_ranker | epsilon_greedy | linucb | thompson
    tau: float = 0.06
    alpha: float = 0.35
    epsilon: float = 0.0
    mmr: bool = True
    lam: float = 0.75
    use_novelty: bool = True

    @staticmethod
    def from_cfg(name: str, c: Dict, cfg: Dict) -> "PolicySpec":
        t = c["type"]
        return PolicySpec(name=name, type=t, tau=c.get("tau", cfg["propensity"]["tau"]),
                          alpha=c.get("alpha", cfg["bandit"]["alpha"]), epsilon=c.get("epsilon", 0.0),
                          mmr=c.get("mmr", True), lam=c.get("lambda", cfg["mmr"]["lambda"]) if c.get("mmr", True) else 1.0,
                          use_novelty=t in ("linucb", "thompson"))

    @property
    def uses_bandit(self) -> bool:
        return self.type in ("linucb", "thompson")


@dataclass
class CandidatePack:
    """Everything computed once per request and shared by all policies (so policies are scored on identical pools)."""
    user: int
    day: int
    items: np.ndarray                 # surviving candidate item ids
    pc: np.ndarray
    pb: np.ndarray
    tier: np.ndarray
    novelty: np.ndarray
    margin: np.ndarray
    price: np.ndarray
    subcat: np.ndarray
    is_cold: np.ndarray
    unseen_sub: np.ndarray
    ctx: torch.Tensor                 # (n, d) bandit context (device)
    groups: torch.Tensor              # (n,)
    emb: torch.Tensor                 # (n, d_emb) normalised (device)
    subcat_t: torch.Tensor
    u_rel: torch.Tensor               # utility w/o novelty
    u_full: torch.Tensor              # utility with novelty
    ranker_topk: np.ndarray           # membership in pure-ranker top-K
    stage_counts: Dict[str, Any] = field(default_factory=dict)
    latency: Dict[str, float] = field(default_factory=dict)


class DiscoveryPipeline:
    def __init__(self, world, retriever: TwoTowerRetriever, builder: FeatureBuilder, ranker: PrimaryRanker,
                 state: UserState, cfg: Dict, device: torch.device, bandits: Optional[Dict[str, LinUCBBandit]] = None,
                 policies: Optional[Dict[str, PolicySpec]] = None, log_dir=None, seed: int = 0):
        self.world, self.retriever, self.builder, self.ranker = world, retriever, builder, ranker
        self.state, self.cfg, self.dev = state, cfg, device
        self.guard = Guardrails(cfg)
        self.timer = StageTimer()
        self.rng = np.random.default_rng(seed)
        self.lock = threading.RLock()
        it = world.items
        self.i_cat = it["category"].values
        self.i_sub = it["subcat"].values
        self.i_margin = it["margin"].values.astype(np.float32)
        self.i_price = it["price"].values.astype(np.float32)
        self.i_cold = it["is_cold"].values
        self.i_stock = it["in_stock"].values
        self.i_days = it["days_since_launch"].values.astype(np.float32)
        self.user_avg_price = world.users["avg_item_price"].values.astype(np.float32)
        self.user_persona = world.users["persona"].values
        self.item_emb_t = torch.as_tensor(world.item_emb, device=device)
        self.policies = policies or {n: PolicySpec.from_cfg(n, c, cfg) for n, c in cfg["policies"].items()}
        self.bandits = bandits if bandits is not None else {}
        self.ctx_dim = len(CONTEXT_FEATURES)
        self.k = cfg["mmr"]["k"]
        self.top_n = cfg["retrieval"]["top_n"]
        self.rw = cfg["reward"]
        self.ctx_cache: "OrderedDict[Tuple[int,int], Tuple[np.ndarray,int,float,str]]" = OrderedDict()
        self.ctx_cache_size = cfg["api"]["context_cache"]
        self.drop_totals: Dict[str, int] = {"retrieved": 0, "out_of_stock": 0, "recent_purchase": 0, "kept_after_filter": 0}
        self.prop_log: Optional[JsonlLogger] = None
        self.inference_log: Optional[JsonlLogger] = None
        self.guard_log: Optional[JsonlLogger] = None
        if log_dir is not None:
            self.prop_log = JsonlLogger(log_dir / "propensity_log.jsonl")
            self.inference_log = JsonlLogger(log_dir / "inference_log.jsonl")
            self.guard_log = JsonlLogger(log_dir / "guardrail_log.jsonl")
        self.request_counter = 0

    # ------------------------------------------------------------------------------------
    def new_bandits(self) -> Dict[str, LinUCBBandit]:
        b = self.cfg["bandit"]
        out = {}
        for name, p in self.policies.items():
            if p.type == "linucb":
                out[name] = LinUCBBandit(self.ctx_dim, self.world.n_cats, self.dev, alpha=p.alpha, lam=b["lambda_reg"],
                                         hybrid=b["hybrid"], shared_idx=SHARED_CONTEXT_IDX, refresh_every=b["refresh_every"])
            elif p.type == "thompson":
                from src.reranker.bandit import ThompsonBandit
                out[name] = ThompsonBandit(self.ctx_dim, self.world.n_cats, self.dev, v=b["thompson_v"], lam=b["lambda_reg"])
        return out

    # ------------------------------------------------------------------------------------
    def make_context(self, u_arr: np.ndarray, i_arr: np.ndarray, X: np.ndarray, pc: np.ndarray, pb: np.ndarray,
                     tier: np.ndarray, unseen_sub: np.ndarray, u_full: np.ndarray) -> np.ndarray:
        """Vectorised bandit context x_{u,i} (scaled to O(1))."""
        n = len(i_arr)
        margin = self.i_margin[i_arr]
        nov = novelty_from_tier(tier, self.cfg)
        ut = torch.as_tensor(u_arr, device=self.dev)
        it = torch.as_tensor(i_arr, device=self.dev)
        uie = (self.retriever.user_matrix[ut][:, :8] * self.retriever.item_matrix[it][:, :8] * 8.0).cpu().numpy()
        persona = np.zeros((n, 4), dtype=np.float32)
        persona[np.arange(n), self.user_persona[u_arr]] = 1.0
        brand_unseen = (tier == 2).astype(np.float32)
        cols = [np.ones(n, np.float32), pc * 5.0, pb * 10.0, pb * margin * 20.0, nov,
                np.abs(X[:, _F["price_logratio"]]), self.i_cold[i_arr].astype(np.float32),
                np.log1p(self.i_days[i_arr]) / np.log(800.0), X[:, _F["tt_cos"]], X[:, _F["content_cos"]], margin,
                unseen_sub.astype(np.float32), brand_unseen, u_full * 4.0]
        return np.concatenate([np.stack(cols, axis=1), persona, uie], axis=1).astype(np.float32)

    # ------------------------------------------------------------------------------------
    def prepare(self, u: int, day: Optional[int] = None, k: Optional[int] = None) -> CandidatePack:
        """Stages 1-3 + candidate-level Stage 4 pieces (utility, guardrail prefilter, contexts)."""
        k = k or self.k
        day = self.state.today if day is None else day
        T = self.timer
        fcfg = self.cfg["filtering"]
        # ---- Stage 1: retrieval ----
        with T.stage("1_retrieval"):
            _, top = self.retriever.retrieve(torch.as_tensor([u], device=self.dev), self.top_n)
            items = top[0].cpu().numpy()
        # ---- Stage 2: hard filters ----
        with T.stage("2_filtering"):
            oos = ~self.i_stock[items]
            recent = self.state.recently_bought(u, items, day, fcfg["recent_purchase_days"])
            keep = ~oos & ~recent
            counts = {"retrieved": int(len(items)), "out_of_stock": int(oos.sum()),
                      "recent_purchase_14d": int((recent & ~oos).sum()), "kept": int(keep.sum())}
            items = items[keep]
            self.drop_totals["retrieved"] += counts["retrieved"]
            self.drop_totals["out_of_stock"] += counts["out_of_stock"]
            self.drop_totals["recent_purchase"] += counts["recent_purchase_14d"]
            self.drop_totals["kept_after_filter"] += counts["kept"]
        # ---- Stage 3: primary ranker ----
        with T.stage("3_ranking"):
            u_arr = np.full(len(items), u)
            X = self.builder.build(u_arr, items)
            pc, pb = self.ranker.predict(X)
        # ---- Stage 4a: utility & novelty ----
        with T.stage("4a_utility"):
            tier = self.state.novelty_tier(u, items)
            nov = novelty_from_tier(tier, self.cfg)
            unseen_sub = self.state.unseen_subcat(u, items)
            margin, price = self.i_margin[items], self.i_price[items]
            pc_t, pb_t = torch.as_tensor(pc, device=self.dev), torch.as_tensor(pb, device=self.dev)
            m_t, n_t = torch.as_tensor(margin, device=self.dev), torch.as_tensor(nov, device=self.dev)
            w1, w2, _ = utility_weights(self.cfg, True)
            u_rel = utility(pc_t, pb_t, m_t, n_t, *utility_weights(self.cfg, False))
            u_full = utility(pc_t, pb_t, m_t, n_t, *utility_weights(self.cfg, True))
        # ---- Stage 4b: guardrails (candidate level) ----
        with T.stage("4b_guardrails"):
            keep2, ginfo = self.guard.filter(self.state, u, items, tier, self.i_cold[items], price,
                                             float(self.user_avg_price[u]), day, k)
            idx = np.where(keep2)[0]
            sel = torch.as_tensor(idx, device=self.dev)
            items, X, pc, pb, tier, nov = items[idx], X[idx], pc[idx], pb[idx], tier[idx], nov[idx]
            unseen_sub, margin, price = unseen_sub[idx], margin[idx], price[idx]
            u_rel, u_full = u_rel[sel], u_full[sel]
        # ---- context ----
        with T.stage("4c_context"):
            u_arr = np.full(len(items), u)
            ctx = self.make_context(u_arr, items, X, pc, pb, tier, unseen_sub, u_full.cpu().numpy())
            ctx_t = torch.as_tensor(ctx, device=self.dev)
            groups = torch.as_tensor(self.i_cat[items], device=self.dev)
            emb = self.item_emb_t[torch.as_tensor(items, device=self.dev)]
            sub_t = torch.as_tensor(self.i_sub[items], device=self.dev)
            topk = np.zeros(len(items), dtype=bool)
            topk[torch.topk(u_rel, min(k, len(items))).indices.cpu().numpy()] = True
        return CandidatePack(user=u, day=day, items=items, pc=pc, pb=pb, tier=tier, novelty=nov, margin=margin,
                             price=price, subcat=self.i_sub[items], is_cold=self.i_cold[items], unseen_sub=unseen_sub,
                             ctx=ctx_t, groups=groups, emb=emb, subcat_t=sub_t, u_rel=u_rel, u_full=u_full,
                             ranker_topk=topk, stage_counts={"filter": counts, "guardrails": ginfo})

    # ------------------------------------------------------------------------------------
    def score(self, pack: CandidatePack, pol: PolicySpec, alpha: Optional[float] = None
              ) -> Dict[str, torch.Tensor]:
        """Score_bandit = U + [learned residual] + alpha*sqrt(x^T A^-1 x); returns decomposition tensors."""
        base = pack.u_full if pol.use_novelty else pack.u_rel
        zero = torch.zeros_like(base)
        if not pol.uses_bandit:
            return {"base": base, "mean": zero, "bonus": zero, "score": base}
        b = self.bandits[pol.name]
        mean, bonus = b.scores(pack.ctx, pack.groups, alpha=pol.alpha if alpha is None else alpha)
        if not self.cfg["bandit"]["use_mean"]:
            mean = zero
        else:
            mean = mean * self.cfg["bandit"]["mean_scale"]
        return {"base": base, "mean": mean, "bonus": bonus, "score": base + mean + bonus}

    def policy_probs(self, scores: torch.Tensor, pol: PolicySpec, tau: Optional[float] = None) -> torch.Tensor:
        """Item-level Boltzmann distribution P(i|u) = softmax(score/tau) over the guard-railed candidate set."""
        tau = pol.tau if tau is None else tau
        p = torch.softmax(scores / tau, dim=0)
        if pol.type == "epsilon_greedy":
            p = (1.0 - pol.epsilon) * p + pol.epsilon / p.shape[0]
        return p

    # ------------------------------------------------------------------------------------
    def _select(self, pack: CandidatePack, pol: PolicySpec, sc: Dict[str, torch.Tensor], probs: torch.Tensor,
                k: int, lam: float, sample: bool) -> Tuple[List[int], List[float]]:
        mcfg = self.cfg["mmr"]
        cap = self.cfg["guardrails"]["max_per_subcategory"]
        if sample:       # stochastic behaviour policy: sequential Boltzmann sampling with the subcategory cap
            p = probs.detach().cpu().numpy().astype(np.float64)
            chosen, counts = [], {}
            avail = np.ones(len(p), dtype=bool)
            for _ in range(min(k, len(p))):
                w = p * avail
                for s, c in counts.items():
                    if c >= cap:
                        w = np.where(pack.subcat == s, 0.0, w)
                if w.sum() <= 0:
                    w = p * avail
                j = int(self.rng.choice(len(p), p=w / w.sum()))
                chosen.append(j)
                avail[j] = False
                counts[pack.subcat[j]] = counts.get(pack.subcat[j], 0) + 1
            return chosen, [0.0] * len(chosen)
        picks, pens, _ = mmr_select(sc["score"], pack.emb, pack.subcat_t, k, lam, mcfg["w_embed_sim"],
                                    mcfg["w_subcat_sim"], cap)
        if pol.type == "epsilon_greedy":
            pens = list(pens)
            for s in range(len(picks)):
                if self.rng.random() < pol.epsilon:
                    cnt = {}
                    for j in picks:
                        cnt[pack.subcat[j]] = cnt.get(pack.subcat[j], 0) + 1
                    cand = [j for j in range(len(pack.items)) if j not in picks and cnt.get(pack.subcat[j], 0) < cap]
                    if cand:
                        picks[s] = int(self.rng.choice(cand))
        return picks, pens

    # ------------------------------------------------------------------------------------
    def recommend(self, u: int, policy: str = "linucb_mmr", k: Optional[int] = None, day: Optional[int] = None,
                  alpha: Optional[float] = None, lam: Optional[float] = None, sample: bool = False,
                  tau: Optional[float] = None, request_id: Optional[int] = None) -> Dict[str, Any]:
        with self.lock:
            k = k or self.k
            pol = self.policies[policy]
            T = self.timer
            pack = self.prepare(u, day, k)
            with T.stage("4d_bandit"):
                sc = self.score(pack, pol, alpha)
            with T.stage("4e_propensity"):
                probs = self.policy_probs(sc["score"], pol, tau)
            lam_eff = (pol.lam if pol.mmr else 1.0) if lam is None else lam
            with T.stage("4f_mmr"):
                picks, pens = self._select(pack, pol, sc, probs, k, lam_eff, sample)
            lat = T.commit()
            self.request_counter += 1
            rid = self.request_counter if request_id is None else request_id
            base = sc["base"].cpu().numpy(); mean = sc["mean"].cpu().numpy(); bonus = sc["bonus"].cpu().numpy()
            score = sc["score"].cpu().numpy(); pr = probs.cpu().numpy()
            sub_list = [int(pack.subcat[j]) for j in picks]
            viol = self.guard.validate_slate(sub_list)
            ild = intra_list_diversity(pack.emb[torch.as_tensor(picks, device=self.dev)].cpu().numpy())
            ctx_np = pack.ctx.cpu().numpy()
            slate = []
            for slot, j in enumerate(picks):
                item = int(pack.items[j])
                rec = {
                    "slot": slot, "item_id": item, "category": int(self.i_cat[item]), "subcat": int(pack.subcat[j]),
                    "price": float(pack.price[j]), "margin": float(pack.margin[j]),
                    "p_click": float(pack.pc[j]), "p_buy": float(pack.pb[j]),
                    "base_utility": float(base[j]), "ranker_q": float(self.rw["click"] * pack.pc[j] + self.rw["buy"] * pack.pb[j]),
                    "learned_residual": float(mean[j]),
                    "exploration_bonus": float(bonus[j]), "mmr_penalty": float(pens[slot]),
                    "bandit_score": float(score[j]), "propensity_score": float(pr[j]),
                    "novelty_tier": TIER_NAMES[int(pack.tier[j])], "novelty_tier_id": int(pack.tier[j]),
                    "novelty": float(pack.novelty[j]), "is_cold_start": bool(pack.is_cold[j]),
                    "unseen_subcat": bool(pack.unseen_sub[j]), "ranker_topk": bool(pack.ranker_topk[j]),
                }
                slate.append(rec)
                with self.lock:
                    self.ctx_cache[(u, item)] = (ctx_np[j], int(pack.groups[j]), slate[-1]["ranker_q"], policy)
                    if len(self.ctx_cache) > self.ctx_cache_size:
                        self.ctx_cache.popitem(last=False)
            out = {"request": rid, "user_id": u, "policy": policy, "day": pack.day, "slate": slate, "ild": ild,
                   "latency_ms": lat, "stage_counts": pack.stage_counts, "cap_violations": viol,
                   "n_candidates": int(len(pack.items)), "_pack": pack, "_picks": picks, "_ctx": ctx_np}
            if self.inference_log is not None:
                self.inference_log.log({"ts": time.time(), "request": rid, "user_id": u, "policy": policy,
                                        "day": pack.day, "items": [s["item_id"] for s in slate], "ild": ild,
                                        "latency_ms": lat, "filter": pack.stage_counts["filter"]})
            if self.guard_log is not None:
                self.guard_log.log({"ts": time.time(), "request": rid, "user_id": u, **pack.stage_counts["guardrails"],
                                    "slate_cap_violations": viol})
            return out

    # ------------------------------------------------------------------------------------
    def log_propensity(self, out: Dict[str, Any], outcomes: Optional[Dict[str, np.ndarray]] = None,
                       event: str = "impression") -> None:
        if self.prop_log is None:
            return
        for s, rec in enumerate(out["slate"]):
            ctx = out["_ctx"][out["_picks"][s]]
            row = {"ts": time.time(), "event": event, "policy": out["policy"], "request": out["request"],
                   "user_id": out["user_id"], "item_id": rec["item_id"], "slot": rec["slot"],
                   "propensity_score": rec["propensity_score"],
                   "context": {n: round(float(v), 5) for n, v in zip(CONTEXT_FEATURES, ctx)},
                   "reward": None if outcomes is None else float(outcomes["reward"][s])}
            if outcomes is not None:
                row["click"] = int(outcomes["click"][s]); row["buy"] = int(outcomes["buy"][s])
            self.prop_log.log(row)

    # ------------------------------------------------------------------------------------
    def bandit_update(self, policy: str, ctx: np.ndarray, group: int, reward: float, baseline: float) -> None:
        """Online incremental update. With residual_target the bandit regresses (reward - ranker expected reward),
        i.e. it learns the *correction* to the primary ranker (large for cold-start SKUs)."""
        pol = self.policies[policy]
        if not pol.uses_bandit:
            return
        target = reward - baseline if self.cfg["bandit"]["residual_target"] else reward
        with self.lock:
            self.bandits[policy].update(torch.as_tensor(ctx, device=self.dev), group, float(target))

    def feedback(self, user_id: int, item_id: int, reward: float, policy: str, day: Optional[int] = None,
                 clicked: Optional[bool] = None, purchased: Optional[bool] = None) -> Dict[str, Any]:
        """API feedback path: updates LinUCB covariance and the user's shopping state."""
        with self.lock:
            entry = self.ctx_cache.get((user_id, item_id))
            recomputed = False
            if entry is None:   # context not cached -> recompute it deterministically
                pack = self.prepare(user_id, day)
                self.timer.commit()
                hit = np.where(pack.items == item_id)[0]
                if len(hit) == 0:
                    return {"updated": False, "reason": "item not in current candidate pool"}
                j = int(hit[0])
                entry = (pack.ctx[j].cpu().numpy(), int(pack.groups[j]),
                         float(self.rw["click"] * pack.pc[j] + self.rw["buy"] * pack.pb[j]), policy)
                recomputed = True
            ctx, group, base_u, pol_name = entry
            pol_name = policy if policy in self.policies else pol_name
            self.bandit_update(pol_name, ctx, group, reward, base_u)
            click = reward > 0 if clicked is None else clicked
            buy = reward >= self.rw["buy"] if purchased is None else purchased
            self.state.record(user_id, np.array([item_id]), np.array([int(click)]), np.array([int(buy)]),
                              self.state.today if day is None else day)
            n_upd = self.bandits[pol_name].n_updates if pol_name in self.bandits else 0
            return {"updated": pol_name in self.bandits, "policy": pol_name, "recomputed_context": recomputed,
                    "bandit_updates": n_upd}
