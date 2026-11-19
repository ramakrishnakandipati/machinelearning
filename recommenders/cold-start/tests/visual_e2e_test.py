"""Headless Playwright visual verification.

Starts the FastAPI server, exercises the 4 cold-start scenarios, renders each request/response as a formatted
HTML page and captures full-page screenshots (plus /health and the Swagger /docs UI) into outputs/screenshots/.

Run:  python tests/visual_e2e_test.py   (needs: pip install playwright && python -m playwright install chromium)
"""
from __future__ import annotations

import html
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "outputs" / "models"
SHOTS = ROOT / "outputs" / "screenshots"

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1F2328;background:#fff;max-width:1180px}
h1{font-size:22px;margin:0 0 4px} .sub{color:#656D76;margin-bottom:16px}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;color:#fff;margin-right:6px}
.cold_zero_history{background:#CC3311}.semi_cold_session{background:#E69F00;color:#1F2328}.warm_personalized{background:#0072B2}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px} th,td{padding:6px 8px;border-bottom:1px solid #E4E7EB;text-align:left}
th{color:#656D76;font-weight:600}.tag{display:inline-block;background:#EEF2F6;border-radius:4px;padding:1px 6px;margin:1px 3px 1px 0;font-size:11.5px}
.new{background:#E6F4EA;color:#0B5D2A;font-weight:600} pre{background:#F6F8FA;padding:12px;border-radius:6px;font-size:12px;overflow:auto}
.row{display:flex;gap:16px}.row>div{flex:1;min-width:0} h2{font-size:14px;color:#656D76;margin:16px 0 4px}
"""


def render(title: str, subtitle: str, req: dict, resp: dict) -> str:
    rows = ""
    for it in resp.get("items", []):
        tags = "".join(f'<span class="tag{" new" if "New-Item" in t else ""}">{html.escape(t)}</span>' for t in it["explain"])
        rows += (f"<tr><td>{it['rank']}</td><td>{html.escape(it['item_id'])}{' <b>NEW</b>' if it['is_new_item'] else ''}</td>"
                 f"<td>{html.escape(it['title'])}</td><td>{html.escape(it['subcat'])}</td><td>{it['price']:.2f}</td>"
                 f"<td>{it['score']:.3f}</td><td>{tags}</td></tr>")
    strat = resp.get("strategy", "n/a")
    return f"""<!doctype html><meta charset=utf-8><style>{CSS}</style>
<h1>{html.escape(title)}</h1><div class=sub>{html.escape(subtitle)}</div>
<span class="badge {strat}">{strat}</span><span class="badge" style="background:#656D76">interactions: {resp.get('n_interactions')}</span>
<span class="badge" style="background:#009E73">{resp.get('latency_ms')} ms</span>
<table><tr><th>#</th><th>Item</th><th>Title</th><th>Sub-category</th><th>Price</th><th>Score</th><th>Why (explainability)</th></tr>{rows}</table>
<div class=row><div><h2>POST body</h2><pre>{html.escape(json.dumps(req, indent=2))}</pre></div>
<div><h2>Routing weights &amp; warnings</h2><pre>{html.escape(json.dumps({k: resp.get(k) for k in ('weights', 'warnings', 'cached')}, indent=2))}</pre></div></div>"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    from playwright.sync_api import sync_playwright

    if not (MODELS / "serving_state.pkl").exists():
        print("model artifacts missing - run `python run_pipeline.py` first", file=sys.stderr)
        return 2
    SHOTS.mkdir(parents=True, exist_ok=True)
    (SHOTS / "html").mkdir(exist_ok=True)
    port = free_port()
    env = dict(os.environ, COLDSTART_CONFIG=str(MODELS / "config_used.yaml"), COLDSTART_MODELS=str(MODELS),
               COLDSTART_REDIS_URL="", PYTHONPATH=str(ROOT))
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.serving.app:app", "--host", "127.0.0.1",
                            "--port", str(port), "--log-level", "warning"], cwd=ROOT, env=env)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if httpx.get(f"{base}/health", timeout=1).json().get("status") == "ok":
                    break
            except Exception:
                pass
            if srv.poll() is not None:
                raise RuntimeError("server exited during startup")
            time.sleep(0.5)
        else:
            raise RuntimeError("server did not become healthy")
        items = pd.read_csv(MODELS / "items_catalog.csv")
        session_ids = items[items.subcat == "Running Shoes"].item_id.head(3).tolist()
        new_sku = dict(item_id="SKU_LIVE_001", title="Aero Trail Runner Pro", brand="Nova Works",
                       description="Lightweight trail running shoe with grippy sole, cushion and breathable mesh.",
                       dept="Footwear", subcat="Running Shoes", price=119.0, leaf="trail runner", color="teal", style="pro")
        scenarios = []
        scenarios.append(("1_anonymous_zero_history", "Anonymous Zero-History User",
                          "No id, no onboarding: contextual popularity + exploration",
                          {"user_context": {"region": "north", "device": "ios", "channel": "app", "hour": 20}, "k": 8}))
        scenarios.append(("2_onboarding_new_user", "Onboarding New User",
                          "Zero purchases; onboarding categories + acquisition channel drive the two-tower",
                          {"user_context": {"region": "west", "device": "android", "acquisition_channel": "paid_social",
                                            "onboarding_categories": ["Audio", "Wearables"], "hour": 9}, "k": 8}))
        scenarios.append(("3_session_clickstream_user", "Session Clickstream User",
                          "Anonymous shopper with 3 real-time clicks on running shoes",
                          {"session_item_ids": session_ids, "user_context": {"channel": "web", "hour": 14}, "k": 8}))
        shots = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            def snap(name: str, title: str, sub: str, req: dict, resp: dict) -> None:
                f = SHOTS / "html" / f"{name}.html"
                f.write_text(render(title, sub, req, resp), encoding="utf-8")
                page.goto(f.resolve().as_uri())
                page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)
                shots.append(f"{name}.png")

            for name, title, sub, body in scenarios:
                r = httpx.post(f"{base}/recommend", json=body, timeout=30)
                r.raise_for_status()
                snap(name, title, sub, body, r.json())
            o = httpx.post(f"{base}/items/onboard", json=new_sku, timeout=30)
            o.raise_for_status()
            body = {"user_context": {"region": "south", "onboarding_categories": ["Running Shoes"], "hour": 7}, "k": 8,
                    "include_new_items_boost": True}
            r = httpx.post(f"{base}/recommend", json=body, timeout=30).json()
            r["warnings"] = r.get("warnings", []) + [f"onboarded in {o.json()['embed_and_inject_ms']} ms; "
                                                     f"index size now {o.json()['index_size']}"]
            snap("4_new_sku_injection", "Newly Onboarded SKU Injection",
                 "POST /items/onboard embeds the SKU and injects it into the live FAISS index; /recommend surfaces it", body, r)
            assert any(i["item_id"] == "SKU_LIVE_001" for i in r["items"]) or any(i["is_new_item"] for i in r["items"]), \
                "no new-item surfaced"
            page.goto(f"{base}/health")
            page.screenshot(path=str(SHOTS / "5_health.png"), full_page=True); shots.append("5_health.png")
            page.goto(f"{base}/docs", wait_until="networkidle", timeout=30000)
            try:
                page.wait_for_selector(".swagger-ui", timeout=8000)
            except Exception:
                print("note: Swagger UI assets did not load (offline CDN) - screenshot may be blank")
            page.screenshot(path=str(SHOTS / "6_swagger_docs.png"), full_page=True); shots.append("6_swagger_docs.png")
            browser.close()
        print("screenshots:", ", ".join(shots))
        return 0
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    raise SystemExit(main())
