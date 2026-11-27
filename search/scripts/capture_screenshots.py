"""Launch FastAPI + Streamlit, then capture Swagger and UI screenshots with headless Chromium (Playwright)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests
from playwright.sync_api import sync_playwright

from src.common.utils import load_config, resolve

API, UI = "http://127.0.0.1:8000", "http://127.0.0.1:8501"


def wait(url, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.5)
    raise RuntimeError(f"timeout waiting for {url}")


def settle(page, extra=1.5):
    page.wait_for_load_state("networkidle")
    for _ in range(80):  # wait for Streamlit's running indicator to disappear
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(0.5)
    time.sleep(extra)


def main():
    cfg = load_config()
    out = resolve(cfg, "screenshots")
    logs = resolve(cfg, "logs")
    env_py = sys.executable
    api = subprocess.Popen([env_py, "-m", "uvicorn", "src.api.main:app", "--port", "8000"], cwd=ROOT,
                           stdout=open(logs / "uvicorn.log", "w"), stderr=subprocess.STDOUT)
    ui = subprocess.Popen([env_py, "-m", "streamlit", "run", "src/ui/app.py", "--server.port", "8501", "--server.headless", "true",
                           "--browser.gatherUsageStats", "false"], cwd=ROOT,
                          stdout=open(logs / "streamlit.log", "w"), stderr=subprocess.STDOUT)
    try:
        wait(f"{API}/health"); wait(f"{UI}/_stcore/health")
        # real example responses saved next to screenshots
        samples = {}
        for name, body in {"vocab_gap": {"query": "waterproof trail kicks", "user_id": "U00012", "zip_code": "98101", "top_k": 5},
                           "typo_price": {"query": "gortex snekers under $100", "store_id": "S2", "top_k": 5}}.items():
            samples[name] = requests.post(f"{API}/search", json=body).json()
        (out / "sample_api_responses.json").write_text(json.dumps(samples, indent=2))
        with sync_playwright() as p:
            b = p.chromium.launch()
            ctx = b.new_context(viewport={"width": 1500, "height": 950}, device_scale_factor=1.5)
            page = ctx.new_page()
            # ---- Swagger
            page.goto(f"{API}/docs"); page.wait_for_selector(".opblock-summary")
            page.screenshot(path=str(out / "01_swagger_overview.png"), full_page=True)
            page.click("#operations-search-search_search_post .opblock-summary")
            page.click("button.try-out__btn")
            ta = page.locator(".body-param__text").first
            ta.fill(json.dumps({"query": "gortex snekers under $100", "user_id": "U00012", "zip_code": "98101", "top_k": 5,
                                "ranker": "ensemble", "fusion": "alpha"}, indent=2))
            page.click("button.execute")
            page.wait_for_selector(".live-responses-table .response-col_status", timeout=60000)
            time.sleep(1.5)
            page.locator("#operations-search-search_search_post").scroll_into_view_if_needed()
            page.screenshot(path=str(out / "02_swagger_search_execute.png"), full_page=True)
            # ---- Streamlit
            page.goto(UI); page.wait_for_selector("text=GPU-Accelerated Retail Search Funnel", timeout=240000)
            settle(page, 3)
            page.screenshot(path=str(out / "03_ui_search_vocab_gap.png"), full_page=True)

            def pick_example(label):
                page.get_by_label("Example queries").click()
                page.get_by_role("option", name=label).click()
                settle(page, 2)

            pick_example("Typos + price: gortex snekers under $100")
            page.screenshot(path=str(out / "04_ui_search_typo_price.png"), full_page=True)
            pick_example("Vocabulary gap: office chic heels")
            page.screenshot(path=str(out / "05_ui_search_office_heels.png"), full_page=True)
            for i, (tab, fn) in enumerate([("Funnel Explorer", "06_ui_funnel_explorer"), ("Ranker Comparison", "07_ui_ranker_comparison"),
                                           ("Business Rules & Margin", "08_ui_business_rules_margin"), ("Cold-Start", "09_ui_cold_start"),
                                           ("Metrics & Latency", "10_ui_metrics_latency")]):
                page.get_by_role("tab", name=tab, exact=False).click()
                settle(page, 2.5)
                page.screenshot(path=str(out / f"{fn}.png"), full_page=True)
            b.close()
        print("screenshots:", sorted(x.name for x in out.glob("*.png")))
    finally:
        for pr in (api, ui):
            pr.terminate()
        time.sleep(1)
        for pr in (api, ui):
            if pr.poll() is None:
                pr.kill()


if __name__ == "__main__":
    main()
