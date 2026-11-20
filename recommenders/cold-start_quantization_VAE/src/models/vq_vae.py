"""VQ-VAE with EMA codebook updates, straight-through estimator and dead-code resetting."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(x)


def make_mlp(in_dim: int, hidden: int, out_dim: int, n_blocks: int, dropout: float) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden)]
    layers += [ResBlock(hidden, dropout) for _ in range(n_blocks)]
    layers += [nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class EMAVectorQuantizer(nn.Module):
    """Codebook E in R^{K x D}, updated with exponential moving averages (no gradient on E)."""

    def __init__(self, num_codes: int, dim: int, beta: float = 0.25, decay: float = 0.99, eps: float = 1e-5,
                 dead_threshold: float = 0.1):
        super().__init__()
        self.K, self.D, self.beta, self.decay, self.eps, self.dead_threshold = num_codes, dim, beta, decay, eps, dead_threshold
        self.register_buffer("embedding", torch.randn(num_codes, dim) * 0.1)
        self.register_buffer("cluster_size", torch.ones(num_codes))
        self.register_buffer("embed_avg", self.embedding.clone())
        self.register_buffer("initted", torch.tensor(False))
        self.register_buffer("n_resets", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def init_codebook(self, codes: torch.Tensor) -> None:
        self.embedding.copy_(codes)
        self.embed_avg.copy_(codes)
        self.cluster_size.fill_(1.0)
        self.initted.fill_(True)

    @torch.no_grad()
    def random_init(self, x: torch.Tensor) -> None:
        idx = torch.randint(0, x.shape[0], (self.K,), device=x.device)
        self.init_codebook(x[idx] + 0.01 * torch.randn(self.K, self.D, device=x.device))

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.embedding)

    def nearest(self, x: torch.Tensor) -> torch.Tensor:
        d = x.pow(2).sum(1, keepdim=True) - 2 * x @ self.embedding.t() + self.embedding.pow(2).sum(1)[None]
        return d.argmin(1)

    @torch.no_grad()
    def _ema_update(self, x: torch.Tensor, idx: torch.Tensor) -> None:
        onehot = F.one_hot(idx, self.K).type(x.dtype)
        counts = onehot.sum(0)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(onehot.t() @ x, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n  # Laplace smoothing
        self.embedding.copy_(self.embed_avg / smoothed.unsqueeze(1))

    @torch.no_grad()
    def _reset_dead_codes(self, x: torch.Tensor) -> int:
        dead = self.cluster_size < self.dead_threshold
        n_dead = int(dead.sum())
        if n_dead > 0:
            pick = x[torch.randint(0, x.shape[0], (n_dead,), device=x.device)]
            pick = pick + 0.01 * torch.randn_like(pick)
            self.embedding[dead] = pick
            self.embed_avg[dead] = pick
            self.cluster_size[dead] = 1.0
            self.n_resets += n_dead
        return n_dead

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.autocast(device_type=x.device.type, enabled=False):   # quantizer always in fp32
            x = x.float()
            if self.training and not bool(self.initted):
                self.random_init(x.detach())
            idx = self.nearest(x.detach())
            q = self.lookup(idx)
            if self.training:
                self._ema_update(x.detach(), idx)
                self._reset_dead_codes(x.detach())
            commit = self.beta * F.mse_loss(x, q.detach())
            q_st = x + (q - x).detach()                           # straight-through estimator
            onehot_mean = F.one_hot(idx, self.K).float().mean(0)
            perplexity = torch.exp(-(onehot_mean * torch.log(onehot_mean + 1e-10)).sum())
        return {"q": q_st, "q_raw": q, "idx": idx, "commit": commit, "perplexity": perplexity}


class VQVAE(nn.Module):
    """Encoder -> single-stage EMA quantizer -> decoder."""

    def __init__(self, in_dim: int, latent_dim: int = 64, hidden: int = 256, n_blocks: int = 2, num_codes: int = 256,
                 beta: float = 0.25, decay: float = 0.99, dead_threshold: float = 0.1, dropout: float = 0.0):
        super().__init__()
        self.in_dim, self.latent_dim, self.num_codes = in_dim, latent_dim, num_codes
        self.n_levels = 1
        self.encoder = make_mlp(in_dim, hidden, latent_dim, n_blocks, dropout)
        self.decoder = make_mlp(latent_dim, hidden, in_dim, n_blocks, dropout)
        self.quantizer = EMAVectorQuantizer(num_codes, latent_dim, beta, decay, dead_threshold=dead_threshold)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def encode_to_codes(self, x: torch.Tensor) -> torch.Tensor:
        """(B, F) -> (B, 1) code indices."""
        return self.quantizer.nearest(self.encoder(x).float())[:, None]

    @torch.no_grad()
    def codes_to_embedding(self, codes: torch.Tensor, upto: int | None = None) -> torch.Tensor:
        return self.quantizer.lookup(codes[:, 0])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.encoder(x)
        qo = self.quantizer(z_e)
        recon = self.decoder(qo["q"])
        recon_loss = F.mse_loss(recon.float(), x.float())
        return {
            "recon": recon, "z_e": z_e, "z_q": qo["q_raw"], "codes": qo["idx"][:, None],
            "recon_loss": recon_loss, "commit_loss": qo["commit"], "loss": recon_loss + qo["commit"],
            "perplexity": qo["perplexity"], "level_perplexity": qo["perplexity"][None],
        }
