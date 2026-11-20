"""Residual-quantized VAE producing hierarchical Semantic IDs (c1, ..., cM)."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq_vae import EMAVectorQuantizer, make_mlp


@torch.no_grad()
def kmeans_pp_init(x: torch.Tensor, k: int, n_iter: int = 10) -> torch.Tensor:
    """K-Means++ seeding followed by a few Lloyd iterations. x: (N, D) -> centroids (K, D)."""
    n = x.shape[0]
    if n <= k:  # fewer or equal points than centroids: pad with jittered copies
        reps = x[torch.randint(0, n, (k - n,), device=x.device)] + 0.01 * torch.randn(k - n, x.shape[1], device=x.device)
        return torch.cat([x, reps])
    centers = [x[int(torch.randint(0, n, (1,)).item())]]
    d2 = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        tot = d2.sum()
        if not torch.isfinite(tot) or tot <= 1e-12:      # all remaining points coincide with a centre
            nxt = int(torch.randint(0, n, (1,)).item())
        else:
            nxt = int(torch.multinomial(d2 / tot, 1).item())
        centers.append(x[nxt])
        d2 = torch.minimum(d2, ((x - x[nxt]) ** 2).sum(1))
    c = torch.stack(centers)
    for _ in range(n_iter):
        assign = torch.cdist(x, c).argmin(1)
        onehot = F.one_hot(assign, k).float()
        counts = onehot.sum(0)
        new_c = (onehot.t() @ x) / counts.clamp_min(1).unsqueeze(1)
        c = torch.where(counts.unsqueeze(1) > 0, new_c, c)
    return c


class RQVAE(nn.Module):
    """M-level residual quantizer. Level l quantizes the residual left over by levels < l, so
    coarse levels capture category-like structure and finer levels capture style / fine attributes."""

    def __init__(self, in_dim: int, latent_dim: int = 64, hidden: int = 256, n_blocks: int = 2, num_codes: int = 256,
                 n_levels: int = 4, beta: float = 0.25, decay: float = 0.99, dead_threshold: float = 0.1,
                 dropout: float = 0.0):
        super().__init__()
        self.in_dim, self.latent_dim, self.num_codes, self.n_levels = in_dim, latent_dim, num_codes, n_levels
        self.encoder = make_mlp(in_dim, hidden, latent_dim, n_blocks, dropout)
        self.decoder = make_mlp(latent_dim, hidden, in_dim, n_blocks, dropout)
        self.quantizers = nn.ModuleList(
            [EMAVectorQuantizer(num_codes, latent_dim, beta, decay, dead_threshold=dead_threshold) for _ in range(n_levels)])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _quantize(self, z_e: torch.Tensor):
        residual = z_e.float()
        total = torch.zeros_like(residual)
        codes, commits, perps = [], [], []
        for qz in self.quantizers:
            if self.training and not bool(qz.initted):          # K-Means++ init on first-batch residuals
                qz.init_codebook(kmeans_pp_init(residual.detach(), qz.K))
            out = qz(residual)
            total = total + out["q_raw"]
            residual = residual - out["q_raw"].detach()
            codes.append(out["idx"])
            commits.append(out["commit"])
            perps.append(out["perplexity"])
        z_q_st = z_e.float() + (total - z_e.float()).detach()     # STE over the summed quantization
        return z_q_st, total, torch.stack(codes, 1), torch.stack(commits), torch.stack(perps)

    @torch.no_grad()
    def encode_to_codes(self, x: torch.Tensor) -> torch.Tensor:
        """(B, F) -> (B, M) hierarchical Semantic IDs."""
        residual = self.encoder(x).float()
        codes = []
        for qz in self.quantizers:
            idx = qz.nearest(residual)
            codes.append(idx)
            residual = residual - qz.lookup(idx)
        return torch.stack(codes, 1)

    @torch.no_grad()
    def codes_to_embedding(self, codes: torch.Tensor, upto: int | None = None) -> torch.Tensor:
        """Sum of codebook vectors of the first `upto` levels (default all)."""
        upto = upto or self.n_levels
        return sum(self.quantizers[l].lookup(codes[:, l]) for l in range(upto))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.encoder(x)
        z_q_st, z_q, codes, commits, perps = self._quantize(z_e)
        recon = self.decoder(z_q_st)
        recon_loss = F.mse_loss(recon.float(), x.float())
        commit = commits.sum()
        return {
            "recon": recon, "z_e": z_e, "z_q": z_q, "codes": codes, "recon_loss": recon_loss,
            "commit_loss": commit, "level_commit": commits, "loss": recon_loss + commit,
            "perplexity": perps.mean(), "level_perplexity": perps,
        }
