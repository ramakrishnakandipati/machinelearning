"""Headless Playwright verification: boots the real uvicorn server, drives Swagger UI, saves screenshots."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "outputs" / "screenshots"

pw = pytest.importorskip("playwright.sync_api")
if not (ROOT / "outputs" / "models" / "two_tower.pt").exists():
    pytest.skip("trained artifacts missing - run `python run_pipeline.py` first", allow_module_level=True)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.serving.app:app", "--port", str(port), "--log-level", "warning"], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < 180:
        try:
            urllib.request.urlopen(base + "/health", timeout=2)
            break
        except Exception:
            if proc.poll() is not None:
                pytest.fail("uvicorn exited during start-up")
            time.sleep(1)
    else:
        proc.kill()
        pytest.fail("server did not become healthy")
    yield base
    proc.terminate()
    try:
        proc.wait(10)
    except Exception:
        proc.kill()


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch(headless=True)
        except Exception:
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)
            try:
                b = p.chromium.launch(headless=True)
            except Exception as e:
                pytest.skip(f"chromium unavailable: {e}")
        yield b
        b.close()


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def test_swagger_docs_and_live_call(server, browser):
    SHOTS.mkdir(parents=True, exist_ok=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(server + "/docs")
    try:
        page.wait_for_selector(".swagger-ui .opblock", timeout=20000)
    except Exception:
        page.screenshot(path=str(SHOTS / "01_swagger_docs_unrendered.png"))
        pytest.skip("Swagger UI assets (CDN) not reachable from this machine")
    page.screenshot(path=str(SHOTS / "01_swagger_docs.png"), full_page=True)
    assert page.locator(".opblock-post").count() >= 2
    # drive the real UI: expand POST /recommend, Try it out, execute a brand-new-user request
    op = "#operations-recommend-recommend_recommend_post"
    page.locator(f"{op} .opblock-summary").click()
    page.locator(f"{op} .try-out__btn").click()
    page.locator(f"{op} textarea.body-param__text").fill(
        json.dumps({"profile": {"acquisition_channel": "social", "home_store": "S03", "quiz_tags": ["likes_audio"]}, "k": 5}))
    page.locator(f"{op} .execute").click()
    page.wait_for_selector(f"{op} .live-responses-table", timeout=30000)
    body = page.locator(f"{op} .live-responses-table .microlight").first.inner_text()
    assert '"strategy": "content+graph"' in body and '"telemetry"' in body
    page.locator(f"{op} .responses-wrapper").screenshot(path=str(SHOTS / "02_swagger_recommend_response.png"))
    page.close()


def test_rendered_scenario_responses(server, browser):
    """Render each cold-start scenario response as a readable page and capture a screenshot."""
    SHOTS.mkdir(parents=True, exist_ok=True)
    scen = {
        "zero_shot_profile": {"profile": {"acquisition_channel": "paid_search", "quiz_tags": ["likes_sneakers"]}, "k": 5},
        "few_shot_session": {"profile": {"acquisition_channel": "organic"}, "recent_item_ids": ["I0001", "I0002"], "k": 5},
        "cold_items_only": {"profile": {"loyalty_tier": "gold"}, "filters": {"only_cold_items": True}, "k": 5},
    }
    cards = []
    for name, body in scen.items():
        j = post(server, "/recommend", body)
        rows = "".join(f"<tr><td>{i['item_id']}</td><td>{i['title']}</td><td>{i['l3']}</td><td>{'cold' if i['is_cold_item'] else 'warm'}</td>"
                       f"<td>{i['score']:.3f}</td></tr>" for i in j["items"])
        t = j["telemetry"]
        cards.append(f"<h3>{name}: {j['strategy']} / {j['cold_start_type']} &middot; {t['total_ms']:.1f} ms</h3>"
                     f"<table border=1 cellpadding=4>{rows}</table>")
    page = browser.new_page(viewport={"width": 1000, "height": 900})
    page.set_content("<body style='font-family:sans-serif'><h2>Live API responses</h2>" + "".join(cards) + "</body>")
    page.screenshot(path=str(SHOTS / "03_scenario_responses.png"), full_page=True)
    assert "content+graph" in page.content() and "meta_adapter+graph" in page.content()
    page.goto(server + "/metrics")
    page.screenshot(path=str(SHOTS / "04_metrics_endpoint.png"))
    page.goto(server + "/openapi.json")
    assert "/recommend" in page.content()
    page.close()
