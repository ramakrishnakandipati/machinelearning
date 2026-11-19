import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LOG = logging.getLogger("e2e")
LOG.setLevel(logging.INFO)


def pytest_runtest_logreport(report):
    if report.when == "call":
        LOG.info("%-70s %s (%.2fs)", report.nodeid, report.outcome.upper(), report.duration)


@pytest.fixture(scope="session")
def server():
    from src.config import LOGS
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    logf = open(LOGS / "05_api_server.log", "a", encoding="utf-8")
    logf.write(f"\n=== uvicorn start {time.ctime()} port={port} ===\n")
    logf.flush()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port),
                             "--log-level", "info"], cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(180):
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if proc.poll() is not None:
                raise RuntimeError("uvicorn exited early; see outputs/logs/05_api_server.log")
            time.sleep(0.5)
        else:
            raise RuntimeError("server not healthy in 90s")
        LOG.info("server healthy at %s", base)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.write(f"=== uvicorn stopped {time.ctime()} ===\n")
        logf.close()
