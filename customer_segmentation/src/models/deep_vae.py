"""β-VAE with KL annealing; clustering is done in the μ latent space."""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import JsonlMetricsLogger, get_logger, grad_norm, sync
from .deep_autoencoder import mlp

log = get_logger()


class BetaVAE(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), latent=8, beta=2.0):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, hidden=list(hidden), latent=latent, beta=beta)
        self.beta = beta
        self.enc = mlp([in_dim, *hidden], last_act=True)
        self.mu = nn.Linear(hidden[-1], latent)
        self.logvar = nn.Linear(hidden[-1], latent)
        self.dec = mlp([latent, *reversed(hidden), in_dim])

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h).clamp(-8, 8)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        return self.dec(z), mu, lv


def train_vae(X, device, cfg, metrics: JsonlMetricsLogger):
    tc, mc = cfg["training"], cfg["model"]
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    vae = BetaVAE(X.shape[1], mc["hidden"], mc["latent_dim"], tc["vae_beta"]).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=tc["lr"])
    hist = {"total": [], "recon": [], "kl": [], "beta_eff": []}
    anneal = max(1, min(tc["vae_anneal_epochs"], tc["vae_epochs"] // 2))
    for ep in range(tc["vae_epochs"]):
        beta_eff = vae.beta * min(1.0, (ep + 1) / anneal)  # linear KL annealing
        vae.train(); t0 = time.perf_counter(); tt = tr = tk = 0.0; gn = 0.0
        idx = torch.randperm(n, device=device)
        for i in range(0, n, tc["batch_size"]):
            x = Xt[idx[i:i + tc["batch_size"]]]
            recon, mu, lv = vae(x)
            rl = F.mse_loss(recon, x, reduction="none").sum(1).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
            loss = rl + beta_eff * kl
            opt.zero_grad(); loss.backward(); gn = grad_norm(vae); opt.step()
            m = len(x)
            tt += loss.item() * m; tr += rl.item() * m; tk += kl.item() * m
        sync(); el = time.perf_counter() - t0
        hist["total"].append(tt / n); hist["recon"].append(tr / n); hist["kl"].append(tk / n)
        hist["beta_eff"].append(beta_eff)
        metrics.log(model="beta_vae", phase="train", epoch=ep + 1, loss=tt / n, recon_loss=tr / n,
                    kl_loss=tk / n, beta_eff=beta_eff, grad_norm=gn, lr=opt.param_groups[0]["lr"],
                    throughput_samples_s=n / el, epoch_time_s=el)
        if ep % 10 == 0 or ep == tc["vae_epochs"] - 1:
            log.info("[beta-VAE] epoch %d/%d elbo=%.3f recon=%.3f kl=%.3f beta=%.2f", ep + 1, tc["vae_epochs"],
                     tt / n, tr / n, tk / n, beta_eff)
    vae.eval()
    return vae, hist


@torch.no_grad()
def vae_latents(vae, X, device):
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    mu, _ = vae.encode(Xt)
    return mu.cpu().numpy()


def save_vae(vae, path, cluster_model_path=None):
    torch.save({"state_dict": vae.state_dict(), "config": vae.cfg, "type": "beta_vae",
                "cluster_model_path": cluster_model_path}, path)


def load_vae(path, device):
    ck = torch.load(path, map_location=device)
    c = ck["config"]
    vae = BetaVAE(c["in_dim"], c["hidden"], c["latent"], c["beta"])
    vae.load_state_dict(ck["state_dict"])
    return vae.to(device).eval()
