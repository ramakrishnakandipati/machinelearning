"""Deep neural contextual bandits on GPU: NeuralEpsilonGreedy, NeuralLinear, NeuralTS, NeuralUCB."""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap

from ..config import DEVICE
from .base import BaseBandit
from .linear import ArmPosterior, arm_stats


class QNet(nn.Module):
    """MLP with BatchNorm + Dropout and K output heads (one reward estimate per action)."""

    def __init__(self, d, K, hidden=(128, 64), dropout=0.1):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, K)
        self.out_dim = prev

    def features(self, x):
        return self.trunk(x)

    def forward(self, x):
        return self.head(self.trunk(x))


def make_qnet(d, K, hp):
    return QNet(d, K, tuple(hp["hidden"]), hp["dropout"]).to(DEVICE)


def train_qnet(net, X, a, r, *, epochs, lr, wd, bs, val_frac=0.1, log=None, tag="", seed=0):
    """Masked-MSE regression on the logged action's reward. Keeps best-val weights when val_frac > 0."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(a)
    Xt = torch.as_tensor(X, device=DEVICE)
    at = torch.as_tensor(a, device=DEVICE)
    rt = torch.as_tensor(r, dtype=torch.float32, device=DEVICE)
    perm = torch.randperm(n, generator=g).to(DEVICE)
    nv = int(val_frac * n) if val_frac > 0 else 0
    vi, ti = perm[:nv], perm[nv:]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        sh = ti[torch.randperm(len(ti), generator=g).to(DEVICE)]
        tot, cnt = 0.0, 0
        for i in range(0, len(sh), bs):
            b = sh[i:i + bs]
            if len(b) < 2:
                continue
            pred = net(Xt[b]).gather(1, at[b, None]).squeeze(1)
            loss = ((pred - rt[b]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
            cnt += len(b)
        rec = {"event": "epoch", "tag": tag, "epoch": ep + 1, "train_loss": tot / max(cnt, 1), "kind": "epoch"}
        if nv:
            net.eval()
            with torch.no_grad():
                vl = ((net(Xt[vi]).gather(1, at[vi, None]).squeeze(1) - rt[vi]) ** 2).mean().item()
            rec["val_loss"] = vl
            if vl < best:
                best, best_state = vl, copy.deepcopy(net.state_dict())
        if log:
            log(rec)
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


@torch.no_grad()
def predict_net(net, X, chunk=8192, mc_dropout=False):
    net.train(mc_dropout)
    if mc_dropout:  # keep BN in eval mode, only dropout stochastic
        for m in net.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
    out = []
    for i in range(0, len(X), chunk):
        out.append(net(torch.as_tensor(X[i:i + chunk], device=DEVICE)))
    net.eval()
    return torch.cat(out)


@torch.no_grad()
def net_features(net, X, chunk=8192):
    net.eval()
    return torch.cat([net.features(torch.as_tensor(X[i:i + chunk], device=DEVICE)) for i in range(0, len(X), chunk)])


NEURAL_DEFAULTS = dict(hidden=(128, 64), dropout=0.1, epochs=40, replay_epochs=3, lr=2e-3, wd=1e-4, bs=512,
                       epsilon=0.0, seed=0)


class _NeuralBase(BaseBandit):
    DEFAULTS = NEURAL_DEFAULTS

    def _train(self, net, X, a, r, warm, tag=""):
        ep = self.hp["replay_epochs"] if warm else self.hp["epochs"]
        return train_qnet(net, X, a, r, epochs=ep, lr=self.hp["lr"] * (0.5 if warm else 1.0), wd=self.hp["wd"],
                          bs=self.hp["bs"], val_frac=0.0 if warm else 0.1, log=(lambda rec: self._log(**rec)) if self.logger else None,
                          tag=tag, seed=self.hp["seed"] + self._rounds)


class NeuralEpsilonGreedy(_NeuralBase):
    default_name = "NeuralEpsGreedy"
    DEFAULTS = {**NEURAL_DEFAULTS, "epsilon": 0.05}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.net = None

    def _fit(self, X, a, r, p, warm=False):
        if self.net is None or not warm:
            torch.manual_seed(self.hp["seed"])
            self.net = make_qnet(self.d, self.K, self.hp)
        self._train(self.net, X, a, r, warm and self.net is not None)

    def _mean(self, X):
        return predict_net(self.net, X).cpu().numpy()

    def _save_weights(self, stem):
        torch.save({"net": self.net.state_dict()}, f"{stem}.pt")

    def _load_weights(self, stem):
        self.net = make_qnet(self.d, self.K, self.hp)
        self.net.load_state_dict(torch.load(f"{stem}.pt", map_location=DEVICE, weights_only=False)["net"])
        self.net.eval()


class NeuralLinearBandit(_NeuralBase):
    """Deep representation learning + exact Bayesian linear last layer (Riquelme et al., 2018)."""
    default_name = "NeuralLinear"
    stochastic = True
    DEFAULTS = {**NEURAL_DEFAULTS, "v": 1.0, "lam": 1.0}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.net, self.post = None, None

    def _phi(self, X):
        f = net_features(self.net, X)
        return torch.cat([f, torch.ones(len(f), 1, device=f.device)], 1)

    def _fit(self, X, a, r, p, warm=False):
        if self.net is None or not warm:
            torch.manual_seed(self.hp["seed"])
            self.net = make_qnet(self.d, self.K, self.hp)
        self._train(self.net, X, a, r, warm and self.net is not None)
        Phi = self._phi(X).double().cpu().numpy()
        self.post = ArmPosterior(*arm_stats(Phi, a, r, self.K, self.hp["lam"]))

    @torch.no_grad()
    def _mean(self, X):
        return self.post.mean(self._phi(X)).cpu().numpy()

    @torch.no_grad()
    def _acting(self, X):
        return self.post.sample_scores(self._phi(X), self.hp["v"]).cpu().numpy()

    def _save_weights(self, stem):
        torch.save({"net": self.net.state_dict(), "post": self.post.state()}, f"{stem}.pt")

    def _load_weights(self, stem):
        z = torch.load(f"{stem}.pt", map_location=DEVICE, weights_only=False)
        self.net = make_qnet(self.d, self.K, self.hp)
        self.net.load_state_dict(z["net"])
        self.net.eval()
        self.post = ArmPosterior.from_state(z["post"])


class NeuralTS(_NeuralBase):
    """Thompson sampling from a deep ensemble whose members use MC-Dropout at decision time."""
    default_name = "NeuralTS"
    stochastic = True
    DEFAULTS = {**NEURAL_DEFAULTS, "n_members": 5, "dropout": 0.15, "epochs": 30}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.nets = None

    def _fit(self, X, a, r, p, warm=False):
        if self.nets is None or not warm:
            self.nets = []
            for m in range(self.hp["n_members"]):
                torch.manual_seed(self.hp["seed"] + 100 * m)
                self.nets.append(make_qnet(self.d, self.K, self.hp))
        for m, net in enumerate(self.nets):
            self._train(net, X, a, r, warm, tag=f"member{m}" if m == 0 else f"member{m}")

    @torch.no_grad()
    def _mean(self, X):
        return torch.stack([predict_net(n, X) for n in self.nets]).mean(0).cpu().numpy()

    @torch.no_grad()
    def _acting(self, X):
        outs = torch.stack([predict_net(n, X, mc_dropout=True) for n in self.nets])  # (M,n,K)
        idx = torch.randint(0, len(self.nets), (outs.shape[1],), device=outs.device)
        return outs[idx, torch.arange(outs.shape[1], device=outs.device)].cpu().numpy()

    def _save_weights(self, stem):
        torch.save({"nets": [n.state_dict() for n in self.nets]}, f"{stem}.pt")

    def _load_weights(self, stem):
        z = torch.load(f"{stem}.pt", map_location=DEVICE, weights_only=False)
        self.nets = []
        for sd in z["nets"]:
            n = make_qnet(self.d, self.K, self.hp)
            n.load_state_dict(sd)
            n.eval()
            self.nets.append(n)


class NeuralUCB(_NeuralBase):
    """UCB with gradient-based confidence bounds (Zhou et al., 2020) using a diagonal Fisher/NTK approximation
    over the last hidden layer and output head: bonus(x,a) = alpha * sqrt( sum_p g_p(x,a)^2 / Z_p )."""
    default_name = "NeuralUCB"
    DEFAULTS = {**NEURAL_DEFAULTS, "alpha": 0.05, "lam": 1.0}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.net, self.Z = None, None

    # parameters whose gradients define the confidence ellipsoid
    def _sub_names(self):
        last_lin = max(i for i, m in enumerate(self.net.trunk) if isinstance(m, nn.Linear))
        return [f"trunk.{last_lin}.weight", f"trunk.{last_lin}.bias", "head.weight", "head.bias"]

    def _jac(self, Xc):
        names = self._sub_names()
        params = {n: p.detach() for n, p in self.net.named_parameters() if n in names}
        frozen = {n: p.detach() for n, p in self.net.named_parameters() if n not in names}
        buffers = dict(self.net.named_buffers())

        def f(prm, x):
            return functional_call(self.net, ({**prm, **frozen}, buffers), (x.unsqueeze(0),)).squeeze(0)

        J = vmap(jacrev(f), in_dims=(None, 0))(params, Xc)
        return torch.cat([J[n].reshape(len(Xc), self.K, -1) for n in names], dim=2)  # (n,K,P)

    def _build_Z(self, X, a):
        self.net.eval()
        Z = None
        Xt = torch.as_tensor(X, device=DEVICE)
        at = torch.as_tensor(a, device=DEVICE)
        for i in range(0, len(X), 512):
            J = self._jac(Xt[i:i + 512])
            g = J[torch.arange(len(J), device=DEVICE), at[i:i + 512]]
            s = (g ** 2).sum(0)
            Z = s if Z is None else Z + s
        self.Z = self.hp["lam"] + Z

    def _fit(self, X, a, r, p, warm=False):
        if self.net is None or not warm:
            torch.manual_seed(self.hp["seed"])
            self.net = make_qnet(self.d, self.K, self.hp)
        self._train(self.net, X, a, r, warm and self.net is not None)
        self._build_Z(X, a)

    @torch.no_grad()
    def _mean(self, X):
        return predict_net(self.net, X).cpu().numpy()

    def _bonus(self, X):
        self.net.eval()
        Xt = torch.as_tensor(X, device=DEVICE)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                J = self._jac(Xt[i:i + 512])
                out.append(((J ** 2) / self.Z).sum(-1).sqrt())
        return torch.cat(out)

    def _acting(self, X):
        return (torch.as_tensor(self._mean(X), device=DEVICE) + self.hp["alpha"] * self._bonus(X)).cpu().numpy()

    def _save_weights(self, stem):
        torch.save({"net": self.net.state_dict(), "Z": self.Z.cpu()}, f"{stem}.pt")

    def _load_weights(self, stem):
        z = torch.load(f"{stem}.pt", map_location=DEVICE, weights_only=False)
        self.net = make_qnet(self.d, self.K, self.hp)
        self.net.load_state_dict(z["net"])
        self.net.eval()
        self.Z = z["Z"].to(DEVICE)
