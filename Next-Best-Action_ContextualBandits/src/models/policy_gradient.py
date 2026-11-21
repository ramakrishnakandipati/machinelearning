"""Direct policy optimisation with a counterfactual Doubly-Robust objective (GPU)."""
import numpy as np
import torch
import torch.nn as nn

from ..config import DEVICE
from .base import BaseBandit
from .neural import NEURAL_DEFAULTS, make_qnet, predict_net, train_qnet


class PolicyNet(nn.Module):
    def __init__(self, d, K, hidden=(128, 64)):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, K))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DoublyRobustPolicyGradient(BaseBandit):
    """Softmax policy pi(a|x) maximising the DR value estimate

        V_DR(pi) = mean_i [ sum_a pi(a|x_i) q(x_i,a) + clip(pi(a_i|x_i)/p_i) (r_i - q(x_i,a_i)) ] + entropy bonus

    with q fitted by 2-fold cross-fitting (out-of-fold predictions only enter the objective)."""
    default_name = "DR-PolicyGradient"
    stochastic = True
    DEFAULTS = {**NEURAL_DEFAULTS, "pg_epochs": 60, "pg_replay_epochs": 5, "pg_lr": 3e-3, "w_clip": 10.0,
                "p_floor": 0.02, "entropy": 0.01, "q_epochs": 30, "epsilon": 0.0}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.qnets, self.policy = None, None

    def _fit(self, X, a, r, p, warm=False):
        hp, K = self.hp, self.K
        warm = warm and self.qnets is not None
        fold = (np.arange(len(a)) % 2)
        if not warm:
            torch.manual_seed(hp["seed"])
            self.qnets = [make_qnet(self.d, K, hp) for _ in range(2)]
            self.policy = PolicyNet(self.d, K, tuple(hp["hidden"])).to(DEVICE)
        q_oof = np.zeros((len(a), K), dtype=np.float32)
        for f in range(2):
            tr, te = fold != f, fold == f
            train_qnet(self.qnets[f], X[tr], a[tr], r[tr], epochs=hp["replay_epochs"] if warm else hp["q_epochs"],
                       lr=hp["lr"] * (0.5 if warm else 1.0), wd=hp["wd"], bs=hp["bs"], val_frac=0.0 if warm else 0.1,
                       log=(lambda rec: self._log(**rec)) if self.logger else None, tag=f"reward_model_fold{f}",
                       seed=hp["seed"] + f)
            q_oof[te] = predict_net(self.qnets[f], X[te]).cpu().numpy()
        self._train_policy(X, a, r, p, q_oof, warm)

    def _train_policy(self, X, a, r, p, q_oof, warm):
        hp = self.hp
        Xt = torch.as_tensor(X, device=DEVICE)
        at = torch.as_tensor(a, device=DEVICE)
        rt = torch.as_tensor(r, dtype=torch.float32, device=DEVICE)
        pt = torch.as_tensor(np.maximum(p, hp["p_floor"]), dtype=torch.float32, device=DEVICE)
        qt = torch.as_tensor(q_oof, device=DEVICE)
        opt = torch.optim.Adam(self.policy.parameters(), lr=hp["pg_lr"] * (0.5 if warm else 1.0))
        epochs = hp["pg_replay_epochs"] if warm else hp["pg_epochs"]
        n, bs = len(a), 1024
        for ep in range(epochs):
            self.policy.train()
            perm = torch.randperm(n, device=DEVICE)
            tot, tv, cnt = 0.0, 0.0, 0
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                pi = torch.softmax(self.policy(Xt[b]), 1)
                pi_a = pi.gather(1, at[b, None]).squeeze(1)
                q_a = qt[b].gather(1, at[b, None]).squeeze(1)
                w = torch.clamp(pi_a / pt[b], max=hp["w_clip"])
                dr = (pi * qt[b]).sum(1) + w * (rt[b] - q_a)
                ent = -(pi * torch.log(pi + 1e-8)).sum(1)
                loss = -(dr.mean() + hp["entropy"] * ent.mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
                tv += dr.mean().item() * len(b)
                cnt += len(b)
            self._log(event="epoch", tag="policy", epoch=ep + 1, train_loss=tot / cnt, dr_value=tv / cnt, kind="epoch")
        self.policy.eval()

    @torch.no_grad()
    def _probs(self, X):
        self.policy.eval()
        out = [torch.softmax(self.policy(torch.as_tensor(X[i:i + 8192], device=DEVICE)), 1)
               for i in range(0, len(X), 8192)]
        return torch.cat(out)

    @torch.no_grad()
    def _mean(self, X):
        return torch.stack([predict_net(q, X) for q in self.qnets]).mean(0).cpu().numpy()

    @torch.no_grad()
    def _acting(self, X):  # Gumbel-max trick == sampling from pi
        logp = torch.log(self._probs(X) + 1e-12)
        g = -torch.log(-torch.log(torch.rand_like(logp).clamp_min(1e-12)))
        return (logp + g).cpu().numpy()

    def action_probs(self, X, n_mc=0):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if not self.fitted:
            return np.full((len(X), self.K), 1.0 / self.K)
        return self._probs(X).cpu().numpy().astype(np.float64)

    def _save_weights(self, stem):
        torch.save({"q": [q.state_dict() for q in self.qnets], "policy": self.policy.state_dict()}, f"{stem}.pt")

    def _load_weights(self, stem):
        z = torch.load(f"{stem}.pt", map_location=DEVICE, weights_only=False)
        self.qnets = []
        for sd in z["q"]:
            q = make_qnet(self.d, self.K, self.hp)
            q.load_state_dict(sd)
            q.eval()
            self.qnets.append(q)
        self.policy = PolicyNet(self.d, self.K, tuple(self.hp["hidden"])).to(DEVICE)
        self.policy.load_state_dict(z["policy"])
        self.policy.eval()
