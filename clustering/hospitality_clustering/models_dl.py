"""Deep clustering: Deep Embedded Clustering (DEC) and beta-VAE + latent GMM (PyTorch, GPU)."""
from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import config as C

log = logging.getLogger("hospitality")
DEV = torch.device(C.DEVICE)


def _log_loss(**row):
    with open(C.LOSS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}) + "\n")


def _mlp(dims, last_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _batches(n, bs, shuffle=True):
    idx = torch.randperm(n, device=DEV) if shuffle else torch.arange(n, device=DEV)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


# --------------------------------------------------------------------------- DEC
class DenoisingAE(nn.Module):
    def __init__(self, d_in, hidden, latent, noise_std):
        super().__init__()
        self.noise_std = noise_std
        self.enc = _mlp([d_in, *hidden, latent])
        self.dec = _mlp([latent, *reversed(hidden), d_in])

    def encode(self, x):
        return self.enc(x)

    def forward(self, x):
        xin = x + self.noise_std * torch.randn_like(x) if self.training else x
        z = self.enc(xin)
        return self.dec(z), z


class DEC(nn.Module):
    """Autoencoder + Student's t soft-assignment layer."""

    def __init__(self, ae: DenoisingAE, k: int, latent: int, alpha: float = 1.0):
        super().__init__()
        self.ae, self.alpha = ae, alpha
        self.centroids = nn.Parameter(torch.zeros(k, latent))

    def soft_assign(self, z):
        d2 = torch.cdist(z, self.centroids).pow(2)
        q = (1.0 + d2 / self.alpha).pow(-(self.alpha + 1) / 2)
        return q / q.sum(1, keepdim=True)


def target_distribution(q):
    w = q.pow(2) / q.sum(0)
    return w / w.sum(1, keepdim=True)


class DECPredictor:
    """Inference wrapper (numpy in / numpy out)."""

    def __init__(self, model: DEC):
        self.model = model.to(DEV).eval()

    @torch.no_grad()
    def embed(self, X):
        z = self.model.ae.encode(torch.as_tensor(X, dtype=torch.float32, device=DEV))
        return z.cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X):
        x = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        q = self.model.soft_assign(self.model.ae.encode(x)).cpu().numpy()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        return q

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def train_dec(X: np.ndarray, k: int, hp: dict | None = None):
    hp = {**C.DL, **(hp or {})}
    d = X.shape[1]
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    n = len(Xt)
    ae = DenoisingAE(d, hp["hidden"], hp["latent_dim"], hp["noise_std"]).to(DEV)

    # Phase 1: denoising autoencoder pretraining (MSE)
    opt = torch.optim.Adam(ae.parameters(), lr=hp["lr"])
    for ep in range(1, hp["ae_epochs"] + 1):
        ae.train()
        tot = 0.0
        for b in _batches(n, hp["batch_size"]):
            recon, _ = ae(Xt[b])
            loss = F.mse_loss(recon, Xt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        _log_loss(model="dec", phase="pretrain", epoch=ep, recon=tot / n, kl=None, total=tot / n)
    log.info("DEC pretrain final recon MSE: %.4f", tot / n)

    # Phase 2: centroid init via K-Means on latent embeddings
    ae.eval()
    with torch.no_grad():
        z0 = ae.encode(Xt).cpu().numpy()
    km = KMeans(k, n_init=20, random_state=C.SEED).fit(z0)
    dec = DEC(ae, k, hp["latent_dim"]).to(DEV)
    dec.centroids.data = torch.as_tensor(km.cluster_centers_, dtype=torch.float32, device=DEV)
    prev = km.labels_

    # Phase 3: joint fine-tuning, KL(P || Q)
    opt = torch.optim.Adam(dec.parameters(), lr=hp["lr"] * 0.3)
    p = None
    for ep in range(1, hp["dec_epochs"] + 1):
        if (ep - 1) % hp["dec_update_interval"] == 0:
            dec.eval()
            with torch.no_grad():
                q_all = dec.soft_assign(ae.encode(Xt))
                p = target_distribution(q_all)
                cur = q_all.argmax(1).cpu().numpy()
            delta = float((cur != prev).mean())
            prev = cur
            if ep > 1 and delta < hp["dec_tol"]:
                log.info("DEC converged at epoch %d (label change %.4f)", ep, delta)
                break
        dec.train()
        tot = rec = 0.0
        for b in _batches(n, hp["batch_size"]):
            z = ae.encode(Xt[b])
            q = dec.soft_assign(z)
            kl = F.kl_div(q.log(), p[b], reduction="batchmean")
            opt.zero_grad(); kl.backward(); opt.step()
            tot += kl.item() * len(b)
            with torch.no_grad():
                rec += F.mse_loss(ae.dec(z), Xt[b]).item() * len(b)
        _log_loss(model="dec", phase="finetune", epoch=ep, recon=rec / n, kl=tot / n, total=tot / n)
    pred = DECPredictor(dec)
    labels = pred.predict(X)
    torch.save(dict(state=dec.state_dict(), d_in=d, k=k, hp={kk: hp[kk] for kk in
               ("hidden", "latent_dim", "noise_std")}), C.MODELS_DIR / "autoencoder_dec.pt")
    return pred, labels


def load_dec(path=None) -> DECPredictor:
    ck = torch.load(path or C.MODELS_DIR / "autoencoder_dec.pt", map_location=DEV, weights_only=False)
    hp = ck["hp"]
    ae = DenoisingAE(ck["d_in"], tuple(hp["hidden"]), hp["latent_dim"], hp["noise_std"])
    dec = DEC(ae, ck["k"], hp["latent_dim"])
    dec.load_state_dict(ck["state"])
    return DECPredictor(dec)


# --------------------------------------------------------------------------- VAE
class BetaVAE(nn.Module):
    def __init__(self, d_in, hidden, latent):
        super().__init__()
        self.body = _mlp([d_in, *hidden], last_act=True)
        self.mu = nn.Linear(hidden[-1], latent)
        self.logvar = nn.Linear(hidden[-1], latent)
        self.dec = _mlp([latent, *reversed(hidden), d_in])

    def encode(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h).clamp(-10, 10)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        return self.dec(z), mu, lv


class VAEPredictor:
    def __init__(self, vae: BetaVAE, gmm: GaussianMixture):
        self.vae, self.gmm = vae.to(DEV).eval(), gmm

    @torch.no_grad()
    def embed(self, X):
        mu, _ = self.vae.encode(torch.as_tensor(X, dtype=torch.float32, device=DEV))
        return mu.cpu().numpy()

    def predict_proba(self, X):
        return self.gmm.predict_proba(self.embed(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def train_vae(X: np.ndarray, k: int, hp: dict | None = None):
    hp = {**C.DL, **(hp or {})}
    d = X.shape[1]
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    n = len(Xt)
    vae = BetaVAE(d, hp["hidden"], hp["latent_dim"]).to(DEV)
    opt = torch.optim.Adam(vae.parameters(), lr=hp["lr"])
    for ep in range(1, hp["vae_epochs"] + 1):
        vae.train()
        beta = hp["beta"] * min(1.0, ep / max(1, hp["kl_warmup"]))  # KL warm-up
        sr = sk = 0.0
        for b in _batches(n, hp["batch_size"]):
            recon, mu, lv = vae(Xt[b])
            rec = F.mse_loss(recon, Xt[b], reduction="none").sum(1).mean()
            kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()
            loss = rec + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            sr += rec.item() * len(b); sk += kl.item() * len(b)
        _log_loss(model="vae", phase="train", epoch=ep, recon=sr / n, kl=sk / n,
                  total=sr / n + beta * sk / n, beta=beta)
    log.info("VAE final recon=%.3f KL=%.3f", sr / n, sk / n)
    vae.eval()
    with torch.no_grad():
        mu = vae.encode(Xt)[0].cpu().numpy()
    gmm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=C.SEED, reg_covar=1e-4).fit(mu)
    torch.save(dict(state=vae.state_dict(), d_in=d, k=k,
                    hp={kk: hp[kk] for kk in ("hidden", "latent_dim")}, gmm_file="vae_gmm.joblib"),
               C.MODELS_DIR / "vae_cluster.pt")
    joblib.dump(gmm, C.MODELS_DIR / "vae_gmm.joblib")
    pred = VAEPredictor(vae, gmm)
    return pred, gmm.predict(mu)


def load_vae(path=None) -> VAEPredictor:
    path = path or C.MODELS_DIR / "vae_cluster.pt"
    ck = torch.load(path, map_location=DEV, weights_only=False)
    vae = BetaVAE(ck["d_in"], tuple(ck["hp"]["hidden"]), ck["hp"]["latent_dim"])
    vae.load_state_dict(ck["state"])
    gmm = joblib.load(C.MODELS_DIR / ck["gmm_file"])
    return VAEPredictor(vae, gmm)


def fit_all(X: np.ndarray, k: int, hp: dict | None = None) -> dict:
    C.LOSS_LOG.write_text("", encoding="utf-8")
    out = {}
    for name, fn in (("dec", train_dec), ("vae_gmm", train_vae)):
        t = time.perf_counter()
        model, labels = fn(X, k, hp)
        log.info("%-8s trained on %s in %.1fs", name, C.DEVICE, time.perf_counter() - t)
        out[name] = dict(model=model, labels=labels)
    return out
