"""Two-Tower architecture: independent User / Item MLP towers -> L2-normalised 128-d embeddings."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EMB_DIM


def mlp(dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class UserTower(nn.Module):
    def __init__(self, sizes: dict, d: int = EMB_DIM, dropout: float = 0.1):
        super().__init__()
        self.uid = nn.Embedding(sizes["user"], 64, padding_idx=0)
        self.tier, self.chan, self.region = (nn.Embedding(sizes[k], 8, padding_idx=0) for k in ("tier", "chan", "region"))
        self.hist = nn.Embedding(sizes["item"], 64, padding_idx=0)       # pooled last-N item sequence
        self.dow, self.hour, self.device = nn.Embedding(8, 4), nn.Embedding(25, 8), nn.Embedding(sizes["device"], 4, padding_idx=0)
        self.mlp = mlp([64 + 24 + 4 + 64 + 16, 512, 256, d], dropout)
        self.dense_proj = nn.Linear(4, 4)

    def forward(self, uid, tier, chan, region, dense, hist, dow, hour, device):
        e = self.hist(hist)
        mask = (hist > 0).unsqueeze(-1).to(e.dtype)
        pooled = (e * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        x = torch.cat([self.uid(uid), self.tier(tier), self.chan(chan), self.region(region), self.dense_proj(dense),
                       pooled, self.dow(dow), self.hour(hour), self.device(device)], dim=1)
        return F.normalize(self.mlp(x).float(), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, sizes: dict, d: int = EMB_DIM, dropout: float = 0.1):
        super().__init__()
        self.iid = nn.Embedding(sizes["item"], 64, padding_idx=0)
        self.cat1, self.cat2 = nn.Embedding(sizes["cat1"], 16, padding_idx=0), nn.Embedding(sizes["cat2"], 16, padding_idx=0)
        self.brand, self.season = nn.Embedding(sizes["brand"], 16, padding_idx=0), nn.Embedding(sizes["season"], 4, padding_idx=0)
        self.mlp = mlp([64 + 16 + 16 + 16 + 4 + 4, 256, d], dropout)

    def forward(self, iid, cat1, cat2, brand, season, dense):
        x = torch.cat([self.iid(iid), self.cat1(cat1), self.cat2(cat2), self.brand(brand), self.season(season), dense], dim=1)
        return F.normalize(self.mlp(x).float(), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, sizes: dict, d: int = EMB_DIM, dropout: float = 0.1, init_temperature: float = 0.07):
        super().__init__()
        self.user_tower, self.item_tower = UserTower(sizes, d, dropout), ItemTower(sizes, d, dropout)
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_temperature)))   # learnable temperature

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(0.01, 1.0)

    def forward(self, user_in: tuple, item_in: tuple):
        return self.user_tower(*user_in), self.item_tower(*item_in)
