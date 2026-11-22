"""Start uvicorn, exercise the endpoints through Swagger UI with Playwright and save screenshots."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config  # noqa: E402

CFG = load_config()
HOST, PORT = CFG["api"]["host"], CFG["api"]["port"]
BASE = f"http://{HOST}:{PORT}"
OUT = ROOT / CFG["paths"]["api_screenshots"]
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE = {
    "loyalty_tier": "Gold", "adr_avg": 210.0, "length_of_stay_avg": 3.0, "lead_time_days": 14.0,
    "days_since_last_stay": 45.0, "total_prior_stays": 9, "ancillary_spend_ratio": 0.22,
    "promo_email_open_rate": 0.45, "app_sessions_30d": 12, "channel_direct": 1,
    "trip_purpose_business": 1, "seasonality_index": 1.05,
}


def wait_for_health(timeout: float = 120.0) -> dict:
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=3) as r:
                return json.loads(r.read())
        except Exception as exc:               # server not up yet
            last = exc
            time.sleep(1.0)
    raise RuntimeError(f"API did not become healthy in {timeout}s: {last}")


def expand(page, path_text: str):
    block = page.locator(".opblock").filter(has_text=path_text).first
    if "is-open" not in (block.get_attribute("class") or ""):
        block.locator(".opblock-summary").first.click()
    block.locator(".opblock-body").wait_for(state="visible")
    return block


def main() -> None:
    log_file = open(ROOT / CFG["paths"]["training_logs"] / "uvicorn.log", "w", encoding="utf-8")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--host", HOST, "--port", str(PORT)],
        cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        health = wait_for_health()
        print("health:", json.dumps({k: health[k] for k in ("status", "active_device", "loaded_models")}))
        assert health["status"] == "ok", f"API degraded: {health.get('error')}"

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
            page.goto(f"{BASE}/docs", wait_until="networkidle")
            page.locator(".opblock").first.wait_for()
            page.screenshot(path=OUT / "01_swagger_overview.png", full_page=True)

            # --- GET /health ---------------------------------------------------------------------
            blk = expand(page, "/health")
            blk.get_by_role("button", name="Try it out").click()
            blk.get_by_role("button", name="Execute").click()
            blk.locator(".live-responses-table").wait_for(state="visible", timeout=30000)
            page.screenshot(path=OUT / "02_health_executed.png", full_page=True)
            blk.locator(".opblock-summary").first.click()          # collapse

            # --- POST /predict/survival ------------------------------------------------------------
            blk = expand(page, "/predict/survival")
            page.screenshot(path=OUT / "03_predict_survival_expanded.png", full_page=True)
            blk.get_by_role("button", name="Try it out").click()
            body = blk.locator("textarea.body-param__text")
            body.fill(json.dumps(SAMPLE, indent=2))
            blk.get_by_role("button", name="Execute").click()
            blk.locator(".live-responses-table").wait_for(state="visible", timeout=30000)
            blk.locator(".live-responses-table .response-col_status", has_text="200").wait_for(timeout=30000)
            time.sleep(0.5)
            page.screenshot(path=OUT / "04_predict_survival_executed.png", full_page=True)
            blk.locator(".live-responses-table").screenshot(path=OUT / "05_predict_survival_response.png")

            # --- raw JSON views ----------------------------------------------------------------------
            page.goto(f"{BASE}/health", wait_until="load")
            page.screenshot(path=OUT / "06_health_json.png", full_page=True)
            browser.close()

        # Also persist the raw JSON of the call for the record
        req = urllib.request.Request(f"{BASE}/predict/survival", data=json.dumps(SAMPLE).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read())
        (OUT / "sample_response.json").write_text(json.dumps(payload, indent=2))
        print("sample response:", json.dumps(payload, indent=2))
        print("screenshots saved to", OUT)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()


if __name__ == "__main__":
    main()
