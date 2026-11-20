"""Launch the API (uvicorn subprocess), hit endpoints and capture screenshots with headless Chromium.

Usage: python tests/capture_api_screenshots.py [--port 8000]
Requires: pip install playwright && playwright install chromium
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "output" / "api_screenshots"

SAMPLE_GUEST = {  # corporate road-warrior style profile
    "loyalty_tier": "Platinum", "tenure_months": 72, "recency_days": 18, "frequency_stays_12m": 9,
    "monetary_total_spend": 14000, "avg_daily_rate": 215, "avg_length_of_stay": 2.0,
    "booking_lead_time_days": 6, "weekend_vs_weekday_ratio": 0.1, "business_center_usage": 0.8,
    "last_minute_booking_ratio": 0.45, "property_type_affinity": "Urban Business"}


def wait_for_server(base, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(f"{base}/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            time.sleep(1)
    return False


def expand_endpoint(page, text):
    page.locator(".opblock-summary", has_text=text).first.click()
    page.wait_for_timeout(400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    SHOTS.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1",
                             "--port", str(args.port)], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_for_server(base):
            raise RuntimeError("API did not start - did you run `python -m src.train_pipeline`?")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1.5)

            # 1. Swagger overview (collapsed, then all endpoints expanded)
            page.goto(f"{base}/docs", wait_until="networkidle")
            page.wait_for_selector(".opblock-summary")
            for t in ["/health", "/predict/segment", "/predict/batch", "/segments/personas"]:
                expand_endpoint(page, t)
            page.screenshot(path=str(SHOTS / "swagger_ui_overview.png"), full_page=True)

            # 2. Try-it-out: POST /predict/segment through the Swagger UI
            page.goto(f"{base}/docs", wait_until="networkidle")
            expand_endpoint(page, "/predict/segment")
            block = page.locator(".opblock", has_text="/predict/segment").first
            block.locator("button.try-out__btn").click()
            ta = block.locator("textarea.body-param__text")
            ta.fill(json.dumps(SAMPLE_GUEST, indent=2))
            block.locator("button.execute").click()
            page.wait_for_selector(".responses-table.live-responses-table .response-col_status", timeout=15000)
            page.wait_for_timeout(800)
            block.scroll_into_view_if_needed()
            page.screenshot(path=str(SHOTS / "swagger_predict_segment_response.png"), full_page=True)

            # 3. Rendered JSON responses for GET endpoints (prettified HTML wrapper)
            for name, path in [("health_response", "/health"), ("segment_personas_response", "/segments/personas")]:
                data = requests.get(base + path, timeout=30).json()
                page.set_content(
                    "<html><body style='font-family:Consolas,monospace;background:#0f172a;color:#e2e8f0;padding:24px'>"
                    f"<h2 style='color:#93c5fd'>GET {path}</h2>"
                    f"<pre style='font-size:14px;white-space:pre-wrap'>{json.dumps(data, indent=2)}</pre></body></html>")
                page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)

            # 4. Live POST /predict/segment & /predict/batch results
            r = requests.post(f"{base}/predict/segment", json=SAMPLE_GUEST, timeout=30).json()
            b = requests.post(f"{base}/predict/batch", json={"guests": [SAMPLE_GUEST] * 5}, timeout=30).json()
            b["predictions"] = b["predictions"][:2] + ["..."]
            page.set_content(
                "<html><body style='font-family:Consolas,monospace;background:#0f172a;color:#e2e8f0;padding:24px'>"
                "<h2 style='color:#93c5fd'>POST /predict/segment</h2>"
                f"<pre style='font-size:14px'>{json.dumps(r, indent=2)}</pre>"
                "<h2 style='color:#93c5fd'>POST /predict/batch (5 guests)</h2>"
                f"<pre style='font-size:14px'>{json.dumps(b, indent=2)}</pre></body></html>")
            page.screenshot(path=str(SHOTS / "predict_results.png"), full_page=True)
            browser.close()
        print("Screenshots saved to", SHOTS)
        for p in sorted(SHOTS.glob("*.png")):
            print("  ", p.name)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
