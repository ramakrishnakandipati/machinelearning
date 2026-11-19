"""Screenshot helpers: Playwright (headless Chromium) with a Pillow fallback when no browser is available."""
from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """
body{margin:0;background:#12161e;color:#e6ebf5;font-family:Segoe UI,Arial,sans-serif;padding:28px;width:1340px}
h1{font-size:24px;margin:0 0 18px} .ex{background:#1c222e;border-radius:10px;padding:16px;margin-bottom:20px}
.bar{display:flex;align-items:center;gap:12px;margin-bottom:10px;font-weight:600}
.m{background:#468cdc;border-radius:6px;padding:4px 10px} .ok{background:#2ea05a;border-radius:6px;padding:4px 10px}
.bad{background:#c84646;border-radius:6px;padding:4px 10px} .ms{color:#e6aa3c} .lbl{color:#8c96aa;font-size:12px;margin:8px 0 4px}
pre{background:#12161e;border-radius:8px;padding:12px;margin:0;font:13px/1.45 Consolas,monospace;white-space:pre-wrap;word-break:break-all}
.tel{display:flex;gap:10px;margin-top:10px} .tel div{background:#12161e;border-radius:8px;padding:8px 14px}
.tel b{display:block;font-size:20px;color:#e6aa3c}
"""


def abbreviate(obj, n: int = 5):
    if isinstance(obj, list):
        out = [abbreviate(x, n) for x in obj[:n]]
        return out + [f"... ({len(obj) - n} more items)"] if len(obj) > n else out
    if isinstance(obj, dict):
        return {k: abbreviate(v, n) for k, v in obj.items()}
    return obj


def _pretty(o) -> str:
    return json.dumps(abbreviate(o), indent=2) if o is not None else "(no request body)"


def _html(title: str, exchanges: list[dict]) -> str:
    parts = [f"<html><head><style>{CSS}</style></head><body><h1>{html.escape(title)}</h1>"]
    for ex in exchanges:
        cls = "ok" if ex.get("ok", 200 <= ex["status"] < 300) else "bad"
        parts.append(f'<div class="ex"><div class="bar"><span class="m">{ex["method"]}</span><span>{html.escape(ex["url"])}</span>'
                     f'<span class="{cls}">{ex["status"]} {html.escape(ex.get("reason", ""))}</span>'
                     f'<span class="ms">client latency {ex["latency_ms"]:.1f} ms</span></div>'
                     f'<div class="lbl">REQUEST</div><pre>{html.escape(_pretty(ex["request"]))}</pre>'
                     f'<div class="lbl">RESPONSE</div><pre>{html.escape(_pretty(ex["response"]))}</pre>')
        tel = ex["response"].get("telemetry") if isinstance(ex["response"], dict) else None
        if tel:
            parts.append('<div class="tel">' + "".join(f"<div>{k}<b>{v:.2f} ms</b></div>" for k, v in tel.items()) + "</div>")
        parts.append("</div>")
    return "".join(parts) + "</body></html>"


def _pillow(path: Path, title: str, exchanges: list[dict]) -> None:
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont
    fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    mono, bold = ImageFont.truetype(str(fd / "DejaVuSansMono.ttf"), 14), ImageFont.truetype(str(fd / "DejaVuSans-Bold.ttf"), 17)
    lines: list[tuple[str, str]] = [("title", title), ("", "")]
    for ex in exchanges:
        lines.append(("hdr", f'{ex["method"]} {ex["url"]}   [{ex["status"]}]   {ex["latency_ms"]:.1f} ms'))
        lines += [("lbl", "REQUEST")] + [("txt", l[:170]) for l in _pretty(ex["request"]).splitlines()]
        lines += [("lbl", "RESPONSE")] + [("txt", l[:170]) for l in _pretty(ex["response"]).splitlines()] + [("", "")]
    img = Image.new("RGB", (1500, 60 + 20 * len(lines)), (18, 22, 30))
    d = ImageDraw.Draw(img)
    y = 24
    for kind, t in lines:
        d.text((30, y), t, font=bold if kind in ("title", "hdr", "lbl") else mono,
               fill={"title": (230, 235, 245), "hdr": (230, 170, 60), "lbl": (140, 150, 170)}.get(kind, (210, 220, 235)))
        y += 20
    img.save(path)


def render_cards(path: Path, title: str, exchanges: list[dict]) -> str:
    """Returns the renderer used ('playwright' or 'pillow')."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1400, "height": 800}, device_scale_factor=2)
            pg.set_content(_html(title, exchanges))
            pg.screenshot(path=str(path), full_page=True)
            b.close()
        return "playwright"
    except Exception:
        _pillow(path, title, exchanges)
        return "pillow"


def capture_swagger(base_url: str, path: Path, openapi: dict) -> str:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
            pg.goto(f"{base_url}/docs", wait_until="networkidle", timeout=30000)
            pg.wait_for_selector(".opblock", timeout=20000)
            pg.screenshot(path=str(path), full_page=True)
            b.close()
        return "playwright-swagger"
    except Exception:       # offline / no browser: render the OpenAPI endpoint table instead
        rows = [{"method": m.upper(), "url": p, "status": 200, "latency_ms": 0.0, "request": None,
                 "response": {"summary": spec.get("summary") or spec.get("operationId")}}
                for p, ops in openapi["paths"].items() for m, spec in ops.items()]
        return "fallback-" + render_cards(path, "Swagger / OpenAPI endpoints (/docs unavailable offline)", rows)
