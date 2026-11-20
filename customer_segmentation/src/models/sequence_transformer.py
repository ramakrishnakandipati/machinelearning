"""Temporal Transformer encoder-decoder autoencoder over each guest's last-10-stay sequence."""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import JsonlMetricsLogger, get_logger, grad_norm, sync

log = get_logger()


class SeqTransformerAE(nn.Module):
    def __init__(self, seq_len=10, n_cont=3, n_props=4, d_model=32, nhead=4, layers=2, latent=16):
        super().__init__()
        self.cfg = dict(seq_len=seq_len, n_cont=n_cont, n_props=n_props, d_model=d_model, nhead=nhead,
                        layers=layers, latent=latent)
        self.inp = nn.Linear(n_cont, d_model)
        self.prop_emb = nn.Embedding(n_props, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.to_latent = nn.Linear(d_model, latent)
        # decoder: latent -> expanded sequence -> transformer -> reconstruct
        self.from_latent = nn.Linear(latent, d_model)
        dec_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, 0.1, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, layers)
        self.out_cont = nn.Linear(d_model, n_cont)
        self.out_prop = nn.Linear(d_model, n_props)

    def embed(self, cont, prop, mask):
        h = self.inp(cont) + self.prop_emb(prop) + self.pos
        h = self.encoder(h, src_key_padding_mask=(mask == 0))
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        return self.to_latent(pooled)

    def forward(self, cont, prop, mask):
        z = self.embed(cont, prop, mask)
        h = self.from_latent(z).unsqueeze(1) + self.pos
        h = self.decoder(h)
        return self.out_cont(h), self.out_prop(h), z


def _loss(model, cont, prop, mask):
    rc, rp, _ = model(cont, prop, mask)
    m = mask.unsqueeze(-1)
    mse = (((rc - cont) ** 2) * m).sum() / (m.sum() * cont.shape[-1]).clamp_min(1)
    ce = (F.cross_entropy(rp.reshape(-1, rp.shape[-1]), prop.reshape(-1), reduction="none")
          * mask.reshape(-1)).sum() / mask.sum().clamp_min(1)
    return mse + 0.3 * ce, mse, ce


def train_seq(cont, prop, mask, device, cfg, metrics: JsonlMetricsLogger, seed=42):
    tc = cfg["training"]
    C, P, M = (torch.as_tensor(a, device=device) for a in (cont, prop, mask))
    n = C.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g).to(device)
    nval = int(n * 0.1)
    va, tr = perm[:nval], perm[nval:]
    model = SeqTransformerAE(cont.shape[1], cont.shape[2], 4, tc["seq_d_model"], latent=cfg["model"]["latent_dim"] * 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["seq_epochs"])
    hist = {"train": [], "val": []}
    for ep in range(tc["seq_epochs"]):
        model.train(); t0 = time.perf_counter(); tot = 0.0; gn = 0.0
        sh = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(sh), tc["batch_size"]):
            b = sh[i:i + tc["batch_size"]]
            loss, _, _ = _loss(model, C[b], P[b], M[b])
            opt.zero_grad(); loss.backward(); gn = grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl, vm, vc = _loss(model, C[va], P[va], M[va])
        sync(); el = time.perf_counter() - t0
        hist["train"].append(tot / len(tr)); hist["val"].append(vl.item())
        metrics.log(model="sequence_transformer", phase="train", epoch=ep + 1, loss=tot / len(tr),
                    val_loss=vl.item(), val_mse=vm.item(), val_ce=vc.item(), grad_norm=gn,
                    lr=opt.param_groups[0]["lr"], throughput_samples_s=len(tr) / el, epoch_time_s=el)
        if ep % 5 == 0 or ep == tc["seq_epochs"] - 1:
            log.info("[SeqTransformer] epoch %d/%d train=%.4f val=%.4f", ep + 1, tc["seq_epochs"], tot / len(tr), vl.item())
    model.eval()
    return model, hist


@torch.no_grad()
def seq_embeddings(model, cont, prop, mask, device, bs=4096):
    out = []
    for i in range(0, len(cont), bs):
        c, p, m = (torch.as_tensor(a[i:i + bs], device=device) for a in (cont, prop, mask))
        out.append(model.embed(c, p, m).cpu().numpy())
    return np.concatenate(out)


def save_seq(model, path):
    torch.save({"state_dict": model.state_dict(), "config": model.cfg, "type": "sequence_transformer"}, path)


def load_seq(path, device):
    ck = torch.load(path, map_location=device)
    m = SeqTransformerAE(**ck["config"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()
