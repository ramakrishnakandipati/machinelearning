"""Denoising Autoencoder + Deep Embedded Clustering (DEC) in PyTorch."""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import JsonlMetricsLogger, get_logger, grad_norm, sync
from .classical_ml import KMeansGPU

log = get_logger()


def mlp(dims, last_act=False, dropout=0.0):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers += [nn.BatchNorm1d(dims[i + 1]), nn.LeakyReLU(0.1)]
            if dropout and i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class DenoisingAE(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), latent=8, dropout=0.1, noise_std=0.1):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, hidden=list(hidden), latent=latent, dropout=dropout, noise_std=noise_std)
        self.noise_std = noise_std
        self.enc = mlp([in_dim, *hidden, latent], dropout=dropout)
        self.dec = mlp([latent, *reversed(hidden), in_dim], dropout=dropout)

    def encode(self, x):
        return self.enc(x)

    def forward(self, x):
        xin = x + self.noise_std * torch.randn_like(x) if self.training else x
        z = self.enc(xin)
        return self.dec(z), z


class DEC(nn.Module):
    """AE + learnable cluster centres with Student-t soft assignment."""

    def __init__(self, ae: DenoisingAE, n_clusters=5, alpha=1.0):
        super().__init__()
        self.ae, self.alpha, self.k = ae, alpha, n_clusters
        self.centers = nn.Parameter(torch.zeros(n_clusters, ae.cfg["latent"]))

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.cdist(z, self.centers) ** 2 / self.alpha)
        q = q ** ((self.alpha + 1) / 2)
        return q / q.sum(1, keepdim=True)

    @staticmethod
    def target(q):
        w = q ** 2 / q.sum(0)
        return w / w.sum(1, keepdim=True)

    def forward(self, x):
        recon, z = self.ae(x)
        return recon, z, self.soft_assign(z)


def _batches(n, bs, device, shuffle=True):
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def train_dec(X, n_clusters, device, cfg, metrics: JsonlMetricsLogger, seed=42):
    """Pretrain AE (MSE) then joint DEC fine-tune. Returns (DEC model, history dict)."""
    tc, mc = cfg["training"], cfg["model"]
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    ae = DenoisingAE(X.shape[1], mc["hidden"], mc["latent_dim"]).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=tc["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["ae_pretrain_epochs"])
    hist = {"pre_loss": [], "dec_total": [], "dec_recon": [], "dec_kl": []}

    for ep in range(tc["ae_pretrain_epochs"]):
        ae.train(); t0 = time.perf_counter(); tot = 0.0; gn = 0.0
        for b in _batches(n, tc["batch_size"], device):
            x = Xt[b]
            recon, _ = ae(x)
            loss = F.mse_loss(recon, x)
            opt.zero_grad(); loss.backward(); gn = grad_norm(ae); opt.step()
            tot += loss.item() * len(b)
        sched.step(); sync()
        el = time.perf_counter() - t0
        hist["pre_loss"].append(tot / n)
        metrics.log(model="dec_autoencoder", phase="pretrain", epoch=ep + 1, loss=tot / n, recon_loss=tot / n,
                    grad_norm=gn, lr=opt.param_groups[0]["lr"], throughput_samples_s=n / el, epoch_time_s=el)
        if ep % 10 == 0 or ep == tc["ae_pretrain_epochs"] - 1:
            log.info("[DEC-pretrain] epoch %d/%d mse=%.5f (%.2fs)", ep + 1, tc["ae_pretrain_epochs"], tot / n, el)

    # init centres via K-Means on latent codes
    ae.eval()
    with torch.no_grad():
        Z = ae.encode(Xt)
    km = KMeansGPU(n_clusters, seed=seed, device=device).fit(Z.cpu().numpy())
    dec = DEC(ae, n_clusters).to(device)
    dec.centers.data = km.centers_.clone()
    opt = torch.optim.Adam(dec.parameters(), lr=tc["lr"] * 0.3)
    P = None
    for ep in range(tc["dec_epochs"]):
        if ep % tc["dec_update_interval"] == 0:
            dec.eval()
            with torch.no_grad():
                P = dec.target(dec.soft_assign(ae.encode(Xt)))
        dec.train(); t0 = time.perf_counter(); tt = tr = tk = 0.0; gn = 0.0
        for b in _batches(n, tc["batch_size"], device):
            x = Xt[b]
            recon, z, q = dec(x)
            rl = F.mse_loss(recon, x)
            kl = F.kl_div(q.log(), P[b], reduction="batchmean")
            loss = rl + 0.1 * kl
            opt.zero_grad(); loss.backward(); gn = grad_norm(dec); opt.step()
            tt += loss.item() * len(b); tr += rl.item() * len(b); tk += kl.item() * len(b)
        sync(); el = time.perf_counter() - t0
        hist["dec_total"].append(tt / n); hist["dec_recon"].append(tr / n); hist["dec_kl"].append(tk / n)
        metrics.log(model="dec_autoencoder", phase="dec_finetune", epoch=ep + 1, loss=tt / n,
                    recon_loss=tr / n, kl_loss=tk / n, grad_norm=gn, lr=opt.param_groups[0]["lr"],
                    throughput_samples_s=n / el, epoch_time_s=el)
        if ep % 10 == 0 or ep == tc["dec_epochs"] - 1:
            log.info("[DEC-finetune] epoch %d/%d total=%.5f recon=%.5f kl=%.5f", ep + 1, tc["dec_epochs"],
                     tt / n, tr / n, tk / n)
    dec.eval()
    return dec, hist


@torch.no_grad()
def dec_predict(dec: DEC, X, device):
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    z = dec.ae.encode(Xt)
    q = dec.soft_assign(z)
    return q.argmax(1).cpu().numpy(), q.cpu().numpy(), z.cpu().numpy()


def save_dec(dec: DEC, path):
    torch.save({"state_dict": dec.state_dict(), "ae_config": dec.ae.cfg, "n_clusters": dec.k,
                "alpha": dec.alpha, "type": "dec"}, path)


def load_dec(path, device):
    ck = torch.load(path, map_location=device)
    c = ck["ae_config"]
    ae = DenoisingAE(c["in_dim"], c["hidden"], c["latent"], c["dropout"], c["noise_std"])
    dec = DEC(ae, ck["n_clusters"], ck["alpha"])
    dec.load_state_dict(ck["state_dict"])
    return dec.to(device).eval()
