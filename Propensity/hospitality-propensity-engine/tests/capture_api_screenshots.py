"""Launch the FastAPI service + Streamlit dashboard, drive them with headless Playwright, save screenshots.

Run:  python tests/capture_api_screenshots.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "screenshots"
STR, APIO = OUT / "streamlit", OUT / "api"
for d in (STR, APIO):
    d.mkdir(parents=True, exist_ok=True)
EXAMPLES = json.loads((ROOT / "outputs" / "examples" / "examples.json").read_text())
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
            pg.screenshot(path=APIO / "swagger_docs.png", full_page=True)
            # try-it-out on /predict/guest
            pg.locator(".opblock-summary-path[data-path='/predict/guest']").click(); pg.wait_for_selector("button.try-out__btn")
            pg.click("button.try-out__btn"); pg.click("button.execute"); pg.wait_for_selector(".live-responses-table .response-col_status >> text=200", timeout=60000)
            pg.locator(".opblock.is-open").first.screenshot(path=APIO / "predict_guest_response.png")
            # ---- /health JSON
            pg.goto(f"{API}/health"); pg.screenshot(path=APIO / "health.png")
            # ---- Streamlit: every flow x every example, then the other views
            pg.set_viewport_size({"width": 1500, "height": 1650})
            n = 0
            for scen, exs in EXAMPLES.items():
                for k in range(1, len(exs) + 1):
                    pg.goto(f"{UI}/?view=flows&scenario={scen}&example={k}")
                    pg.wait_for_selector("text=Next-best action", timeout=120000)
                    pg.wait_for_selector("text=Request payload", timeout=60000)
                    time.sleep(1.5)
                    pg.screenshot(path=STR / f"flow_{scen}_{k}.png", full_page=True); n += 1
            pg.set_viewport_size({"width": 1500, "height": 1250})
            for view in ("sim", "board", "gallery"):
                pg.goto(f"{UI}/?view={view}")
                pg.wait_for_selector({"sim": "text=Propensity scores & recommended actions", "board": "text=Master leaderboard", "gallery": "text=ROC / PR"}[view], timeout=120000)
                time.sleep(2.5)
                pg.screenshot(path=STR / f"view_{ {'sim': 'simulator', 'board': 'leaderboard', 'gallery': 'gallery'}[view] }.png", full_page=True)
            print("flow screenshots:", n)
            b.close()
    except Exception as e:
        ok = False
        print("FAILED:", repr(e))
    finally:
        for pr in (api, ui):
            pr.terminate()
    print("api:", sorted(p.name for p in APIO.glob("*.png")), "| streamlit:", len(list(STR.glob("*.png"))))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
