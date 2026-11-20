"""Headless Chromium (Playwright) screenshots of the live API: Swagger /docs and the interactive test harness."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import List

import requests
from playwright.sync_api import sync_playwright

from src.config import ROOT, SHOT_DIR

PORT = 8765


def start_server(port: int = PORT) -> subprocess.Popen:
    """Launch uvicorn in a background subprocess."""
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.app:app", "--host", "127.0.0.1", "--port", str(port),
                             "--log-level", "warning"], cwd=str(ROOT))


def wait_for_health(base: str, timeout: float = 180.0, proc: subprocess.Popen | None = None) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"uvicorn exited early with code {proc.returncode}")
        try:
            r = requests.get(f"{base}/health", timeout=2)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError("API did not become healthy in time")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


HARNESS_SHOTS = [("item-cold", "02_item_cold_start_test.png"), ("user-cold", "03_user_cold_start_test.png"),
                 ("system-cold", "04_system_cold_start_test.png"), ("inspect", "05_semantic_id_inspect_test.png")]


def capture_all(base: str, out_dir: Path = SHOT_DIR) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
        page = ctx.new_page()

        # 01: Swagger / OpenAPI docs with every operation expanded
        page.goto(f"{base}/docs", wait_until="networkidle", timeout=60000)
        page.wait_for_selector(".opblock-summary", timeout=60000)
        for btn in page.locator(".opblock-summary-control").all():
            btn.click()
        page.wait_for_timeout(600)
        path = out_dir / "01_swagger_docs.png"
        page.screenshot(path=str(path), full_page=True)
        saved.append(str(path))

        # 02-05: test harness; click the real "Run test" button and wait for the live response to render
        for tab, fname in HARNESS_SHOTS:
            page.goto(f"{base}/ui/test-harness?tab={tab}", wait_until="networkidle")
            page.wait_for_selector("#health-pills .pill.ok", timeout=30000)
            page.click("#run-btn")
            page.wait_for_selector('#result[data-state="done"]', timeout=60000)
            page.wait_for_timeout(400)
            path = out_dir / fname
            page.screenshot(path=str(path), full_page=True)
            saved.append(str(path))

        # bonus: compare-approaches view
        page.goto(f"{base}/ui/test-harness?tab=compare", wait_until="networkidle")
        page.click("#run-btn")
        page.wait_for_selector('#result[data-state="done"]', timeout=60000)
        page.wait_for_timeout(400)
        path = out_dir / "06_compare_approaches_test.png"
        page.screenshot(path=str(path), full_page=True)
        saved.append(str(path))
        browser.close()
    return saved


def run_with_server(port: int = PORT) -> List[str]:
    base = f"http://127.0.0.1:{port}"
    proc = start_server(port)
    try:
        wait_for_health(base, proc=proc)
        return capture_all(base)
    finally:
        stop_server(proc)


if __name__ == "__main__":
    print("\n".join(run_with_server()))
