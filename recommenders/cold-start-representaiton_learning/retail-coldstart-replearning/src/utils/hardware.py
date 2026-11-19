"""Device detection, mixed precision helpers and seed management."""
from __future__ import annotations

import contextlib
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        if pref == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if pref == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return torch.device("cpu")
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_info(device: torch.device) -> dict:
    info = {"device": str(device), "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


def autocast_ctx(device: torch.device, enabled: bool = True):
    """AMP context: fp16 autocast on CUDA, bf16 on CPU disabled (no-op) for numerical safety."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_scaler(device: torch.device, enabled: bool = True):
    return torch.amp.GradScaler("cuda", enabled=bool(enabled and device.type == "cuda"))
