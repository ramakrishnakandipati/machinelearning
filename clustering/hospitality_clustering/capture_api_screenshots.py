"""Launch the API in a subprocess and capture Swagger / endpoint screenshots with headless Chromium."""
from __future__ import annotations

import html
import json
import logging
import socket
import subprocess
import sys
import time

import httpx

import config as C
from api import EXAMPLE

log = logging.getLogger("hospitality")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _json_page(title: str, url: str, payload: dict, status: int) -> str:
    body = html.escape(json.dumps(payload, indent=2))
    return (f"<html><body style='font-family:Segoe UI,Arial;background:#fafafa;padding:24px'>"
            f"<h2>{html.escape(title)}</h2><p><b>{html.escape(url)}</b> &rarr; HTTP {status}</p>"
            f"<pre style='background:#1e1e1e;color:#d4d4d4;padding:16px;border-radius:8px;font-size:14px'>"
            f"{body}</pre></body></html>")


def capture(timeout: float = 90.0) -> bool:
    from playwright.sync_api import sync_playwright

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1",
                             "--port", str(port), "--log-level", "warning"],
                            cwd=str(C.BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if proc.poll() is not None:
                raise RuntimeError("uvicorn exited early:\n" + proc.stdout.read().decode(errors="replace"))
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            raise TimeoutError("API did not become healthy in time")
        log.info("API up at %s", base)

        health = httpx.get(f"{base}/health")
        pred = httpx.post(f"{base}/predict", json=EXAMPLE)
        log.info("/predict -> %s", pred.json())

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            # 01: Swagger UI
            page.goto(f"{base}/docs", wait_until="networkidle")
            try:
                page.wait_for_selector(".opblock-summary", timeout=15000)
            except Exception:
                log.warning("Swagger UI did not render (CDN blocked?) - screenshot may be blank")
            page.screenshot(path=str(C.API_SHOTS_DIR / "01_swagger_docs.png"), full_page=True)

            # 02: health
            page.set_content(_json_page("GET /health", f"{base}/health", health.json(), health.status_code))
            page.screenshot(path=str(C.API_SHOTS_DIR / "02_health_check.png"), full_page=True)

            # 03: live Swagger "Try it out" on /predict; fall back to rendered response
            done = False
            try:
                page.goto(f"{base}/docs", wait_until="networkidle")
                page.locator("#operations-default-predict_predict_post .opblock-summary").click(timeout=10000)
                page.get_by_role("button", name="Try it out").click(timeout=5000)
                page.get_by_role("button", name="Execute").click(timeout=5000)
                page.wait_for_selector(".live-responses-table .response-col_status", timeout=15000)
                page.locator(".responses-wrapper").scroll_into_view_if_needed()
                page.screenshot(path=str(C.API_SHOTS_DIR / "03_predict_cluster_response.png"), full_page=True)
                done = True
            except Exception as e:
                log.warning("Swagger interaction failed (%s); rendering response page instead", e)
            if not done:
                page.set_content(_json_page("POST /predict", f"{base}/predict", pred.json(), pred.status_code))
                page.screenshot(path=str(C.API_SHOTS_DIR / "03_predict_cluster_response.png"), full_page=True)
            browser.close()
        return True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    C.setup_logging()
    capture()
