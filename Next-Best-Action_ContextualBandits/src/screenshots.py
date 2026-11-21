"""Headless Playwright captures of the live FastAPI Swagger UI and the NBA dashboard."""
import subprocess
import sys
import time

import requests

from .config import ROOT, SCREENSHOTS_DIR


def capture(port=8765):
    from playwright.sync_api import sync_playwright

    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app", "--port", str(port), "--log-level", "warning"],
                            cwd=str(ROOT))
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if requests.get(f"{base}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.5)
        else:
            raise RuntimeError("API did not start")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
            page.goto(f"{base}/docs", wait_until="networkidle")
            page.wait_for_selector(".opblock")
            page.locator("#operations-default-predict_nba_predict_nba_post .opblock-summary").click()
            page.wait_for_timeout(600)
            page.screenshot(path=str(SCREENSHOTS_DIR / "fastapi_swagger_docs.png"), full_page=True)

            page.goto(f"{base}/dashboard", wait_until="networkidle")
            page.select_option("#preset", "f")
            page.dispatch_event("#preset", "change")
            page.click("#go")
            page.wait_for_function("document.getElementById('rec').textContent.length > 1 && "
                                   "document.getElementById('rec').textContent !== '—'")
            page.click("#fb")
            page.wait_for_function("document.getElementById('fbout').textContent.length > 2")
            page.screenshot(path=str(SCREENSHOTS_DIR / "nba_dashboard.png"), full_page=True)
            browser.close()
        print(f"      saved screenshots to {SCREENSHOTS_DIR}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    capture()
