"""Learning-to-rank zoo: pointwise, pairwise (LambdaMART), listwise (YetiRank / rank:ndcg / ListNet) and a
neural Deep & Cross Network v2. All models share fit(train, val) / predict(X) and run on GPU when possible."""
from __future__ import annotations

import time

import joblib
import numpy as np
from loguru import logger

from src.common.utils import get_device
from src.evaluation.offline_metrics import quick_ndcg


def _qid(data):
    return np.repeat(np.arange(len(data["sizes"])), data["sizes"])


class LTRModel:
    name = "base"
    paradigm = ""

    def fit(self, tr: dict, va: dict, cfg: dict): ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------------------------- XGBoost
class XGBPointwise(LTRModel):
    name, paradigm = "xgb_pointwise", "pointwise"

    def fit(self, tr, va, cfg):
        import xgboost as xgb

        g = cfg["ranking"]["gbdt"]
        self.m = xgb.XGBRegressor(n_estimators=g["n_estimators"], learning_rate=g["learning_rate"], max_depth=g["max_depth"],
                                  tree_method="hist", device=get_device(), subsample=0.85, colsample_bytree=0.85,
                                  objective="reg:squarederror")
        self.m.fit(tr["X"], tr["y_eng"].astype(float), eval_set=[(va["X"], va["y_eng"].astype(float))], verbose=False)
        return self

    def predict(self, X):
        return self.m.predict(X)


class XGBRank(LTRModel):
    def __init__(self, name, objective, paradigm):
        self.name, self.objective, self.paradigm = name, objective, paradigm

    def fit(self, tr, va, cfg):
        import xgboost as xgb

        g = cfg["ranking"]["gbdt"]
        kw = dict(n_estimators=g["n_estimators"], learning_rate=g["learning_rate"], max_depth=g["max_depth"],
                  tree_method="hist", device=get_device(), subsample=0.85, colsample_bytree=0.85, objective=self.objective)
        if self.objective == "rank:ndcg":
            kw.update(lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=8)
        self.m = xgb.XGBRanker(**kw)
        self.m.fit(tr["X"], tr["y_eng"], qid=_qid(tr), eval_set=[(va["X"], va["y_eng"])], eval_qid=[_qid(va)], verbose=False)
        return self

    def predict(self, X):
        return self.m.predict(X)


# ---------------------------------------------------------------------------------------------- LightGBM
class LGBMPointwise(LTRModel):
    name, paradigm = "lgbm_pointwise", "pointwise"

    def fit(self, tr, va, cfg):
        import lightgbm as lgb

        g = cfg["ranking"]["gbdt"]
        self.m = lgb.LGBMClassifier(n_estimators=g["n_estimators"], learning_rate=g["learning_rate"], num_leaves=48,
                                    subsample=0.85, subsample_freq=1, colsample_bytree=0.85, verbose=-1)
        self.m.fit(tr["X"], (tr["y_eng"] >= 2).astype(int))
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


class LGBMLambdaMART(LTRModel):
    name, paradigm = "lgbm_lambdamart", "pairwise (LambdaMART)"

    def fit(self, tr, va, cfg):
        import lightgbm as lgb

        g = cfg["ranking"]["gbdt"]
        self.m = lgb.LGBMRanker(objective="lambdarank", n_estimators=g["n_estimators"], learning_rate=g["learning_rate"],
                                num_leaves=48, subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
                                lambdarank_truncation_level=20, verbose=-1)
        self.m.fit(tr["X"], tr["y_eng"], group=tr["sizes"], eval_set=[(va["X"], va["y_eng"])], eval_group=[va["sizes"]],
                   eval_at=[10])
        return self

    def predict(self, X):
        return self.m.predict(X)


# ---------------------------------------------------------------------------------------------- CatBoost
class CatYetiRank(LTRModel):
    name, paradigm = "cat_yetirank", "listwise (YetiRank)"

    def fit(self, tr, va, cfg):
        from catboost import CatBoostRanker, Pool

        g = cfg["ranking"]["gbdt"]
        kw = dict(loss_function="YetiRank", iterations=g["n_estimators"], learning_rate=g["learning_rate"] * 1.3,
                  depth=6, verbose=False, random_seed=cfg["seed"])
        trp = Pool(tr["X"], tr["y_eng"], group_id=_qid(tr))
        vap = Pool(va["X"], va["y_eng"], group_id=_qid(va))
        try:
            self.m = CatBoostRanker(task_type="GPU" if get_device() == "cuda" else "CPU", **kw)
            self.m.fit(trp, eval_set=vap)
        except Exception as e:  # pragma: no cover
            logger.warning("CatBoost GPU failed ({}); falling back to CPU", e)
            self.m = CatBoostRanker(task_type="CPU", **kw)
            self.m.fit(trp, eval_set=vap)
        return self

    def predict(self, X):
        return self.m.predict(X)


# ---------------------------------------------------------------------------------------------- Torch
def _prep_cols(Xtr):
    return np.where(np.abs(Xtr).max(0) > 50)[0]


class _Scaler:
    def fit(self, X):
        self.log_cols = _prep_cols(X)
        Z = self._log(X)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        return self

    def _log(self, X):
        Z = X.copy()
        Z[:, self.log_cols] = np.sign(Z[:, self.log_cols]) * np.log1p(np.abs(Z[:, self.log_cols]))
        return Z

    def __call__(self, X):
        return np.clip((self._log(X) - self.mu) / self.sd, -6, 6).astype(np.float32)


def _pad(data, scaler, y_key="y_eng", max_len=None):
    import torch

    off = data["offsets"]
    nl = len(off) - 1
    L = max_len or int(max(data["sizes"]))
    F = data["X"].shape[1]
    Xs = scaler(data["X"])
    Xp = np.zeros((nl, L, F), dtype=np.float32)
    Y = np.zeros((nl, L), dtype=np.float32)
    M = np.zeros((nl, L), dtype=bool)
    for i in range(nl):
        s, e = off[i], off[i + 1]
        n = min(e - s, L)
        Xp[i, :n] = Xs[s:s + n]
        Y[i, :n] = data[y_key][s:s + n]
        M[i, :n] = True
    return torch.from_numpy(Xp), torch.from_numpy(Y), torch.from_numpy(M)


def _make_net(kind, F, cfg):
    import torch
    from torch import nn

    d = cfg["ranking"]["dcn"]

    class DCNv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross = nn.ModuleList([nn.Linear(F, F) for _ in range(d["cross_layers"])])
            dims, layers = [F] + d["hidden"], []
            for a, b in zip(dims, dims[1:]):
                layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(d["embed_dropout"])]
            self.deep = nn.Sequential(*layers)
            self.out = nn.Linear(F + dims[-1], 1)

        def forward(self, x):
            x0, xl = x, x
            for lin in self.cross:  # DCN-v2: x_{l+1} = x0 * (W x_l + b) + x_l
                xl = x0 * lin(xl) + xl
            return self.out(torch.cat([xl, self.deep(x)], -1)).squeeze(-1)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(F, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return DCNv2() if kind == "dcnv2" else MLP()


class TorchListwise(LTRModel):
    """Neural LTR trained with a listwise (ListNet top-1) loss on gain-weighted targets."""

    def __init__(self, kind: str):
        self.kind = kind
        self.name = kind
        self.paradigm = "neural DCNv2 (listwise loss)" if kind == "dcnv2" else "listwise (ListNet)"

    def fit(self, tr, va, cfg):
        import torch

        d = cfg["ranking"]["dcn"]
        dev = get_device()
        torch.manual_seed(cfg["seed"])
        self.scaler = _Scaler().fit(tr["X"])
        Xt, Yt, Mt = _pad(tr, self.scaler)
        Xt, Yt, Mt = Xt.to(dev), Yt.to(dev), Mt.to(dev)
        F = Xt.shape[-1]
        self.net = _make_net(self.kind, F, cfg).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=d["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=d["epochs"])
        nl = Xt.shape[0]
        best, best_state = -1, None
        for ep in range(d["epochs"] * (2 if self.kind == "dcnv2" else 1)):
            self.net.train()
            perm = torch.randperm(nl, device=dev)
            for b in range(0, nl, d["batch_lists"]):
                ix = perm[b:b + d["batch_lists"]]
                x, y, m = Xt[ix], Yt[ix], Mt[ix]
                s = self.net(x).masked_fill(~m, -1e9)
                tgt = torch.pow(2.0, y) - 1.0
                tgt = tgt.masked_fill(~m, 0.0)
                tgt = tgt / (tgt.sum(1, keepdim=True) + 1e-9)
                loss = -(tgt * torch.log_softmax(s, 1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            if ep % 2 == 1:
                self.net.eval()
                v = quick_ndcg(va, self._predict_np(va["X"]), 10)
                if v > best:
                    best, best_state = v, {k: x.detach().clone() for k, x in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self.net_cpu_state = {k: x.cpu() for k, x in self.net.state_dict().items()}
        return self

    def _predict_np(self, X):
        import torch

        dev = next(self.net.parameters()).device
        with torch.no_grad():
            return self.net(torch.from_numpy(self.scaler(X)).to(dev)).cpu().numpy()

    def predict(self, X):
        return self._predict_np(X)

    def __getstate__(self):
        st = dict(self.__dict__)
        st.pop("net", None)
        st["F"] = len(self.scaler.mu)
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)


def reload_torch(model: TorchListwise, cfg: dict):
    net = _make_net(model.kind, model.F, cfg).to(get_device())
    net.load_state_dict(model.net_cpu_state)
    net.eval()
    model.net = net
    return model


# ---------------------------------------------------------------------------------------------- ensemble
class Ensemble(LTRModel):
    name, paradigm = "ensemble", "z-score blend of top-4 models by validation NDCG"

    def __init__(self, members: dict, weights: dict):
        self.members, w = members, np.array([weights[k] for k in members], float)
        self.w = w / w.sum()

    @staticmethod
    def _z(s):
        return (s - s.mean()) / (s.std() + 1e-9)

    def predict(self, X):
        return sum(w * self._z(m.predict(X)) for w, m in zip(self.w, self.members.values()))


REGISTRY = {
    "xgb_pointwise": lambda: XGBPointwise(),
    "lgbm_pointwise": lambda: LGBMPointwise(),
    "xgb_pairwise": lambda: XGBRank("xgb_pairwise", "rank:pairwise", "pairwise"),
    "lgbm_lambdamart": lambda: LGBMLambdaMART(),
    "cat_yetirank": lambda: CatYetiRank(),
    "xgb_listwise": lambda: XGBRank("xgb_listwise", "rank:ndcg", "listwise (LambdaRank NDCG)"),
    "torch_listnet": lambda: TorchListwise("listnet"),
    "dcnv2": lambda: TorchListwise("dcnv2"),
}


def train_all(tr, va, cfg):
    models, times = {}, {}
    for name in cfg["ranking"]["models"]:
        t0 = time.time()
        try:
            models[name] = REGISTRY[name]().fit(tr, va, cfg)
        except Exception as e:
            logger.exception("Training {} failed: {}", name, e)
            continue
        times[name] = time.time() - t0
        v = quick_ndcg(va, models[name].predict(va["X"]), 10)
        logger.info("  {:<18} trained in {:5.1f}s  val NDCG@10(rel)={:.4f}", name, times[name], v)
    vals_all = {k: quick_ndcg(va, m.predict(va["X"]), 10, key="y_eng") for k, m in models.items()}
    want = cfg["ranking"]["ensemble_members"]
    if want == "auto":
        want = sorted(vals_all, key=vals_all.get, reverse=True)[: cfg["ranking"].get("ensemble_size", 4)]
    members = {k: models[k] for k in want if k in models}
    vals = {k: vals_all[k] for k in members}
    logger.info("  ensemble members: {}", {k: round(v, 4) for k, v in vals.items()})
    models["ensemble"] = Ensemble(members, vals)
    return models, times


def save_models(models: dict, path):
    joblib.dump(models, path)


def load_models(path, cfg):
    models = joblib.load(path)

    def fix(m):
        if isinstance(m, TorchListwise):
            reload_torch(m, cfg)
        if isinstance(m, Ensemble):
            for x in m.members.values():
                fix(x)

    for m in models.values():
        fix(m)
    return models
