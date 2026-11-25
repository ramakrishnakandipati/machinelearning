"""Hi-res crops of the Streamlit 'Scenario flows' page (case card + SHAP drivers) for the deck.
Run from the project root:  python presentation/capture_story_shots.py"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "story_shots"
UI = "http://127.0.0.1:8521"
PICK = {"ota_to_direct": 4}          # OTA flow tells the uplift story (persuadable); others use example 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ex = json.loads((ROOT / "outputs" / "examples" / "examples.json").read_text())
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard/streamlit_app.py", "--server.port", "8521",
                           "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT,
                          stdout=open(ROOT / "outputs" / "logs" / "streamlit_story.log", "w"), stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while time.time() - t0 < 120:
            try:
                if requests.get(f"{UI}/_stcore/health", timeout=3).status_code == 200:
                    break
            except Exception:
                time.sleep(1.5)
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_context(viewport={"width": 1500, "height": 1700}, device_scale_factor=2).new_page()
            for s in ex:
                k = PICK.get(s, 1)
                pg.goto(f"{UI}/?view=flows&scenario={s}&example={k}")
                pg.wait_for_selector("text=Request payload", timeout=120000)
                time.sleep(1.5)
                top = pg.get_by_text(re.compile(r"^Case \d+ ")).first.bounding_box()["y"] - 12
                bot = pg.get_by_text("Request payload").first.bounding_box()["y"] - 6
                pg.screenshot(path=OUT / f"{s}.png", clip={"x": 372, "y": top, "width": 1060, "height": bot - top})
                print(s, k, round(bot - top))
            b.close()
    finally:
        ui.terminate()


if __name__ == "__main__":
    main()
