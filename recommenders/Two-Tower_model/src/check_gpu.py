"""GPU / CUDA / FAISS diagnostics -> logs/01_gpu_diagnostics.log + metrics/hardware_profile.json."""
from __future__ import annotations

import json
import platform
import subprocess

from .config import METRICS
from .logger import get_logger

log = get_logger("gpu", "01_gpu_diagnostics.log")


def probe() -> dict:
    info: dict = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception as e:
        info["nvidia_smi"] = f"unavailable ({e})"
    try:
        import torch
        info.update(torch_version=torch.__version__, torch_cuda_available=torch.cuda.is_available(),
                    cuda_version=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu_name=p.name, vram_gb=round(p.total_memory / 2**30, 2),
                        compute_capability=f"{p.major}.{p.minor}")
            x = torch.randn(1024, 1024, device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                info["amp_matmul_ok"] = bool(torch.isfinite(x @ x).all())
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        import faiss
        ngpu = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
        info.update(faiss_version=faiss.__version__, faiss_gpu_count=ngpu,
                    faiss_gpu_resources=bool(ngpu > 0 and hasattr(faiss, "StandardGpuResources")))
    except Exception as e:
        info["faiss_error"] = str(e)
    for mod in ("cupy", "cudf"):
        try:
            __import__(mod)
            info[mod] = True
        except Exception:
            info[mod] = False
    info["ann_backend"] = "faiss-gpu" if info.get("faiss_gpu_resources") else "faiss-cpu"
    return info


def main() -> dict:
    info = probe()
    for k, v in info.items():
        log.info("%-22s %s", k, v)
    (METRICS / "hardware_profile.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    main()
