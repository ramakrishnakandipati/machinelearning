"""Start the FastAPI server, drive the Swagger UI with Playwright (headless Chromium), save screenshots, stop server."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.logger import get_logger, load_config  # noqa: E402


def wait_ready(base: str, timeout: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("API did not become ready")


def run_operation(page, op_id: str, shot: Path, lg) -> None:
    """Expand an endpoint, 'Try it out' with its prefilled example, execute, wait for a 200, save a full-page shot."""
    block = page.locator(f"#{op_id}")
    block.locator(".opblock-summary").first.click()
    block.locator("button.try-out__btn").click()
    block.locator("button.execute").click()
    block.locator(".live-responses-table .response-col_status", has_text="200").first.wait_for(timeout=60000)
    page.wait_for_timeout(800)
    block.scroll_into_view_if_needed()
    page.screenshot(path=str(shot), full_page=True)
    lg.info(f"saved {shot.name} ({shot.stat().st_size / 1024:.0f} KB)")
    block.locator(".opblock-summary").first.click()  # collapse


def main() -> int:
    from playwright.sync_api import sync_playwright
    cfg = load_config()
    lg = get_logger(cfg["out"])
    host, port = cfg["api"]["host"], cfg["api"]["port"]
    base = f"http://{host}:{port}"
    shots = cfg["out"] / "api_screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app", "--host", host, "--port", str(port),
                             "--log-level", "warning"], cwd=ROOT)
    try:
        wait_ready(base)
        lg.info(f"API up at {base}; health={httpx.get(base + '/health').json()['device']}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=2).new_page()
            page.goto(f"{base}/docs", wait_until="networkidle")
            page.wait_for_selector(".opblock")
            page.screenshot(path=str(shots / "swagger_ui_overview.png"), full_page=True)
            lg.info("saved swagger_ui_overview.png")
            run_operation(page, "operations-prediction-predict_guest_predict_guest_post", shots / "single_guest_prediction.png", lg)
            run_operation(page, "operations-prediction-predict_batch_predict_batch_post", shots / "batch_cohort_prediction.png", lg)
            browser.close()
        bad = [n for n in ("swagger_ui_overview.png", "single_guest_prediction.png", "batch_cohort_prediction.png")
               if not (shots / n).exists() or (shots / n).stat().st_size < 10_000]
        if bad:
            lg.error(f"Screenshot verification failed: {bad}")
            return 1
        lg.info("All screenshots verified")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        lg.info("API server terminated")


if __name__ == "__main__":
    sys.exit(main())
