"""Launch the FastAPI service + Streamlit dashboard, drive them with headless Playwright, save screenshots.

Run:  python tests/capture_api_screenshots.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)
API, UI = "http://127.0.0.1:8000", "http://127.0.0.1:8501"


def wait(url, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return True
        except Exception:
            time.sleep(1.5)
    return False


def main():
    logs = ROOT / "outputs" / "logs"
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.app:app", "--port", "8000", "--log-level", "warning"],
                           cwd=ROOT, stdout=open(logs / "uvicorn.log", "w"), stderr=subprocess.STDOUT)
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard/streamlit_app.py", "--server.port", "8501",
                           "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                          cwd=ROOT, stdout=open(logs / "streamlit.log", "w"), stderr=subprocess.STDOUT)
    ok = True
    try:
        assert wait(f"{API}/health"), "API did not start"
        assert wait(f"{UI}/_stcore/health"), "Streamlit did not start"
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1500, "height": 1000})
            # ---- Swagger UI
            pg.goto(f"{API}/docs"); pg.wait_for_selector(".opblock-summary")
            pg.screenshot(path=OUT / "api_swagger_docs.png", full_page=True)
            # try-it-out on /predict/guest
            pg.locator(".opblock-summary-path[data-path='/predict/guest']").click(); pg.wait_for_selector("button.try-out__btn")
            pg.click("button.try-out__btn"); pg.click("button.execute"); pg.wait_for_selector(".live-responses-table .response-col_status >> text=200", timeout=60000)
            pg.locator(".opblock.is-open").first.screenshot(path=OUT / "api_predict_guest_response.png")
            # ---- /health JSON
            pg.goto(f"{API}/health"); pg.screenshot(path=OUT / "api_health.png")
            # ---- Streamlit
            pg.goto(UI); pg.wait_for_selector("text=Guest profile", timeout=120000)
            pg.wait_for_selector("text=Propensity scores & recommended actions", timeout=120000)
            time.sleep(2); pg.screenshot(path=OUT / "dashboard_simulator.png", full_page=True)
            pg.get_by_role("tab", name="Model leaderboard").click(); time.sleep(2)
            pg.screenshot(path=OUT / "dashboard_leaderboard.png", full_page=True)
            pg.wait_for_selector("[data-testid=stTab][aria-selected=true] >> text=Model leaderboard"); time.sleep(4)
            pg.get_by_role("tab", name="Evaluation gallery").click(force=True, timeout=60000); time.sleep(4)
            pg.screenshot(path=OUT / "dashboard_gallery.png", full_page=False)
            b.close()
    except Exception as e:
        ok = False
        print("FAILED:", repr(e))
    finally:
        for pr in (api, ui):
            pr.terminate()
    shots = sorted(OUT.glob("api_*.png")) + sorted(OUT.glob("dashboard_*.png"))
    print("screenshots:", [s.name for s in shots])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
