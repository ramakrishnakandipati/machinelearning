"""Crop the Streamlit End-to-End Flows tab into story shots (scoreboard, per-flow summary, first example).

Run from the project root:  python presentation/capture_story_shots.py
"""
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "story_shots"
UI = "http://127.0.0.1:8502"
TITLES = {"vocab_gap": "1 · Vocabulary gap", "typo_price": "2 · Typos + price constraint", "navigational": "3 · Navigational (brand) intent",
          "regional": "4 · Regional stock & BOPIS", "cold_start": "5 · Cold-start SKUs", "margin": "6 · Margin-aware re-ranking",
          "personalization": "7 · Personalisation"}


def settle(page, extra=2.0):
    page.wait_for_load_state("networkidle")
    for _ in range(100):
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(0.4)
    time.sleep(extra)


def box(loc):
    return loc.first.bounding_box()


def main():
    OUT.mkdir(exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src/ui/app.py", "--server.port", "8502", "--server.headless", "true",
                             "--browser.gatherUsageStats", "false"], cwd=ROOT, stdout=open(OUT / "_streamlit.log", "w"), stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            try:
                if requests.get(f"{UI}/_stcore/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(1.5)
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_context(viewport={"width": 1600, "height": 3600}, device_scale_factor=1.4).new_page()
            page.goto(UI)
            page.wait_for_selector("text=Overall funnel scoreboard", timeout=240000)
            settle(page, 4)
            sb = box(page.get_by_text("Overall funnel scoreboard"))
            sel = box(page.get_by_text("Flow", exact=True))
            ex = box(page.locator('[data-testid="stExpander"]'))
            page.screenshot(path=str(OUT / "scoreboard.png"), clip={"x": ex["x"], "y": sb["y"] - 6, "width": ex["width"], "height": sel["y"] - sb["y"] - 4})
            for n, (fid, title) in enumerate(TITLES.items(), 1):
                if n > 1:
                    page.get_by_role("combobox", name="Flow").click()
                    page.get_by_role("option", name=title).click()
                    settle(page, 3)
                head = box(page.get_by_text("Per-stage mean across"))
                exps = page.locator('[data-testid="stExpander"]')
                e0 = exps.first.bounding_box()
                page.screenshot(path=str(OUT / f"{fid}_summary.png"), clip={"x": e0["x"], "y": head["y"] - 8, "width": e0["width"], "height": e0["y"] - head["y"] - 4})
                exps.first.screenshot(path=str(OUT / f"{fid}_example.png"))
                exps.first.locator('[data-testid="stHorizontalBlock"]').first.screenshot(path=str(OUT / f"{fid}_panels.png"))
                exps.first.locator('[data-testid="stDataFrame"]').first.screenshot(path=str(OUT / f"{fid}_variants.png"))
            b.close()
        print(sorted(x.name for x in OUT.glob("*.png")))
    finally:
        proc.terminate()
        time.sleep(1)
        if proc.poll() is None:
            proc.kill()


if __name__ == "__main__":
    main()
