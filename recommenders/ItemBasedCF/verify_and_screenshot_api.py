"""Spin up the FastAPI service, run 5 endpoint scenarios with assertions, and render a
PNG "screenshot" card per scenario into output/api_tests/ (Pillow only - no browser needed).

Run:  python verify_and_screenshot_api.py   (after python train_pipeline.py)
"""
from __future__ import annotations

import json
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import matplotlib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ART = ROOT / "output" / "artifacts"
OUT = ROOT / "output" / "api_tests"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

W = 1760
BG, PANEL, FG, MUTED = (18, 22, 30), (28, 34, 46), (230, 235, 245), (140, 150, 170)
GREEN, RED, BLUE, AMBER = (46, 160, 90), (200, 70, 70), (70, 130, 220), (230, 170, 60)
LINE_H = 20
MAX_LINES = 34
MAX_CHARS = 186


# ----------------------------------------------------------------------- rendering
def font(name: str, size: int):
    return ImageFont.truetype(str(FONT_DIR / name), size)


F_MONO, F_BOLD, F_TITLE = font("DejaVuSansMono.ttf", 15), font("DejaVuSans-Bold.ttf", 18), font("DejaVuSans-Bold.ttf", 26)


def fmt_json(obj, indent=0) -> list[str]:
    """Readable JSON; lists of dicts are rendered one compact object per line."""
    pad = "  " * indent
    if isinstance(obj, dict):
        lines = [pad + "{"]
        for n, (k, v) in enumerate(obj.items()):
            comma = "," if n < len(obj) - 1 else ""
            if isinstance(v, (dict, list)) and v:
                sub = fmt_json(v, indent + 1)
                sub[0] = pad + f'  "{k}": ' + sub[0].strip()
                sub[-1] += comma
                lines += sub
            else:
                lines.append(pad + f'  "{k}": {json.dumps(v)}{comma}')
        return lines + [pad + "}"]
    if isinstance(obj, list):
        if not obj:
            return [pad + "[]"]
        if all(isinstance(x, dict) and not any(isinstance(y, (dict, list)) for y in x.values()) for x in obj):
            lines = [pad + "["] + [pad + "  " + json.dumps(x) + ("," if n < len(obj) - 1 else "") for n, x in enumerate(obj)]
        else:
            lines = [pad + "["] + [pad + "  " + json.dumps(x) + ("," if n < len(obj) - 1 else "") for n, x in enumerate(obj)]
        return lines + [pad + "]"]
    return [pad + json.dumps(obj)]


def clip(lines: list[str]) -> list[str]:
    if len(lines) > MAX_LINES:
        lines = lines[: MAX_LINES - 1] + [f"... ({len(lines) - MAX_LINES + 1} more lines truncated)"]
    return [l if len(l) <= MAX_CHARS else l[: MAX_CHARS - 3] + "..." for l in lines]


def render_card(path: Path, title: str, exchanges: list[dict], assertions: list[tuple[bool, str]]):
    blocks = []
    for ex in exchanges:
        req = clip(fmt_json(ex["request"])) if ex["request"] is not None else ["(no request body)"]
        res = clip(fmt_json(ex["response"])) if isinstance(ex["response"], (dict, list)) else [str(ex["response"])]
        blocks.append((ex, req, res))
    h = 90 + sum(60 + 44 + (len(r) + 1) * LINE_H + 30 + 44 + (len(s) + 1) * LINE_H + 40 for _, r, s in blocks) + 60 + 28 * len(assertions)
    img = Image.new("RGB", (W, h), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 28), title, font=F_TITLE, fill=FG)
    y = 90
    for ex, req, res in blocks:
        ok = ex["expected_status"] == ex["status"]
        d.rounded_rectangle((40, y, W - 40, y + 50), 8, fill=PANEL)
        d.rounded_rectangle((52, y + 8, 76 + len(ex["method"]) * 12, y + 42), 6, fill=BLUE)
        d.text((62, y + 13), ex["method"], font=F_BOLD, fill="white")
        x = 100 + len(ex["method"]) * 12
        d.text((x, y + 13), ex["url"], font=F_BOLD, fill=FG)
        badge = f"{ex['status']} {ex['reason']}"
        bx = W - 640
        d.rounded_rectangle((bx, y + 8, bx + 330, y + 42), 6, fill=GREEN if ok else RED)
        d.text((bx + 14, y + 13), badge, font=F_BOLD, fill="white")
        d.text((bx + 346, y + 13), f"{ex['latency_ms']:.1f} ms", font=F_BOLD, fill=AMBER)
        y += 62
        for label, lines in (("REQUEST", req), ("RESPONSE", res)):
            d.text((48, y), label, font=F_BOLD, fill=MUTED)
            y += 28
            ph = (len(lines) + 1) * LINE_H
            d.rounded_rectangle((40, y, W - 40, y + ph), 8, fill=PANEL)
            for n, l in enumerate(lines):
                d.text((56, y + 10 + n * LINE_H), l, font=F_MONO, fill=FG)
            y += ph + 14
        y += 24
    d.text((48, y), "ASSERTIONS", font=F_BOLD, fill=MUTED)
    y += 30
    for okk, msg in assertions:
        d.text((56, y), ("PASS  " if okk else "FAIL  ") + msg, font=F_MONO, fill=GREEN if okk else RED)
        y += 28
    img.crop((0, 0, W, y + 20)).save(path)


# --------------------------------------------------------------------------- helpers
class Runner:
    def __init__(self, client: httpx.Client):
        self.c = client

    def call(self, method, url, body=None, expected=200):
        t0 = time.perf_counter()
        r = self.c.request(method, url, json=body)
        ms = (time.perf_counter() - t0) * 1000
        try:
            data = r.json()
        except ValueError:
            data = r.text
        return {"method": method, "url": str(self.c.base_url) + url.lstrip("/") if False else self.c.base_url.join(url).__str__(),
                "request": body, "status": r.status_code, "reason": r.reason_phrase, "latency_ms": ms,
                "response": data, "expected_status": expected}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    if not (ART / "model_meta.json").exists():
        print("Artifacts missing - run `python train_pipeline.py` first.")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_parquet(ART / "catalog.parquet")
    sim = pd.read_parquet(ART / "item_sim_map.parquet")
    hist = pd.read_parquet(ART / "user_history.parquet")
    meta = json.loads((ART / "model_meta.json").read_text())
    stock = catalog.set_index("item_id")["in_stock"]
    durable = set(catalog.loc[~catalog["is_consumable"], "item_id"])
    rnd = random.Random(7)

    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(port),
                             "--log-level", "warning"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results, failures = [], 0
    try:
        base = f"http://127.0.0.1:{port}"
        client = httpx.Client(base_url=base, timeout=10)
        for _ in range(120):
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if proc.poll() is not None:
                print("Server exited early.")
                return 3
            time.sleep(0.5)
        else:
            print("Server did not become healthy in 60s.")
            return 3
        print(f"Server up on {base}")
        R = Runner(client)

        # warm-up load so /metrics latency percentiles are meaningful
        item_pool = [i for i in catalog["item_id"] if stock[i]]
        users = hist["user_id"].unique().tolist()
        for n in range(300):
            if n % 2:
                client.get(f"/v1/recommend/item/{rnd.choice(item_pool)}?limit=10")
            else:
                client.post("/v1/recommend/user", json={"user_id": rnd.choice(users), "limit": 10})

        def finish(name, title, exchanges, asserts):
            nonlocal failures
            asserts = [(bool(ok), msg) for ok, msg in asserts]
            asserts += [(ex["status"] == ex["expected_status"], f"{ex['method']} {ex['url'].split(str(port))[-1]} -> HTTP {ex['expected_status']}")
                        for ex in exchanges]
            render_card(OUT / name, title, exchanges, asserts)
            bad = [m for ok, m in asserts if not ok]
            failures += len(bad)
            print(f"  [{'PASS' if not bad else 'FAIL'}] {name}" + ("".join(f"\n        x {m}" for m in bad)))
            results.append({"screenshot": name, "passed": not bad, "assertions": [{"ok": ok, "msg": m} for ok, m in asserts]})

        def subcat_ok(items, cap=3):
            counts = pd.Series([i["sub_category"] for i in items]).value_counts()
            return bool((counts <= cap).all()) if len(counts) else True

        # ---- test 1: health + metrics
        h = R.call("GET", "/health")
        m = R.call("GET", "/metrics")
        mm = m["response"]
        p99 = (mm.get("latency_ms_recommend_overall") or {}).get("p99_ms")
        finish("test_01_health_and_metrics.png", "Test 1 - GET /health and GET /metrics", [h, m], [
            (h["response"].get("status") == "ok", "health status == ok"),
            (mm.get("model_version") == meta["model_version"], "metrics model_version matches artifact meta"),
            (mm.get("artifact_status") == "loaded", "artifact_status == loaded"),
            (0 < mm.get("catalog_coverage", 0) <= 1, f"catalog_coverage in (0,1]: {mm.get('catalog_coverage')}"),
            (p99 is not None and p99 < 30, f"recommend p99 latency < 30 ms (observed {p99} ms over warm-up)"),
        ])

        # ---- test 2: PDP item recommendations
        counts = sim.groupby("item_id").size()
        seed = next(i for i in item_pool if counts.get(i, 0) >= 10)
        r2 = R.call("GET", f"/v1/recommend/item/{seed}?limit=8&in_stock_only=true")
        it = r2["response"]["items"]
        sc = [x["score"] for x in it]
        finish("test_02_item_pdp_recommendations.png", f"Test 2 - Item PDP recommendations (seed {seed})", [r2], [
            (len(it) > 0, f"returned {len(it)} items"),
            (seed not in {x["item_id"] for x in it}, "seed item not recommended to itself"),
            (sc == sorted(sc, reverse=True), "scores sorted descending"),
            (r2["response"]["strategy"] == "item_similarity", "strategy == item_similarity"),
            (all(x["in_stock"] for x in it), "all items in stock"),
            (subcat_ok(it), "sub-category diversity cap (<=3) holds"),
        ])

        # ---- test 3: personalized + session + durable suppression
        pur = hist[hist["n_purchases"] > 0]
        durable_users = pur[pur["item_id"].isin(durable)].groupby("user_id").size().sort_values(ascending=False)
        user = durable_users.index[0]
        bought_durable = set(pur[(pur["user_id"] == user) & pur["item_id"].isin(durable)]["item_id"])
        base_call = client.post("/v1/recommend/user", json={"user_id": user, "limit": 10}).json()
        top1 = base_call["items"][0]["item_id"]
        recent = rnd.sample(item_pool, 2)
        body = {"user_id": user, "recent_item_ids": recent, "exclude_item_ids": [top1], "limit": 10}
        r3 = R.call("POST", "/v1/recommend/user", body)
        it = r3["response"]["items"]
        ids = {x["item_id"] for x in it}
        finish("test_03_user_personalized_with_session.png", f"Test 3 - Personalized + real-time session ({user})", [r3], [
            (r3["response"]["strategy"] == "personalized", "strategy == personalized"),
            (len(it) > 0, f"returned {len(it)} items"),
            (not (ids & bought_durable), f"durable purchases suppressed ({len(bought_durable)} durable items bought by user)"),
            (not (ids & set(recent)), "session items not echoed back"),
            (top1 not in ids, "explicit exclude_item_ids honored"),
            (all(x["in_stock"] for x in it), "all items in stock"),
            (subcat_ok(it), "sub-category diversity cap (<=3) holds"),
        ])

        # ---- test 4: cold start
        body = {"user_id": "U_UNSEEN_99999", "limit": 10}
        r4 = R.call("POST", "/v1/recommend/user", body)
        it = r4["response"]["items"]
        finish("test_04_cold_start_fallback.png", "Test 4 - Cold-start fallback (unseen user)", [r4], [
            (r4["response"]["strategy"] == "fallback", 'strategy == "fallback"'),
            (len(it) == 10, f"returned {len(it)} items"),
            (all(x["in_stock"] for x in it), "all fallback items in stock"),
            (all(x["reason"] == "popular_margin_weighted" for x in it), "reason == popular_margin_weighted"),
            (subcat_ok(it), "sub-category diversity cap (<=3) holds"),
        ])

        # ---- test 5: guardrails + validation
        oos = set(catalog.loc[~catalog["in_stock"], "item_id"])
        pdp_seed, r5b = None, None
        for cand in sim[sim["neighbor_id"].isin(oos) & (sim["rank"] <= 8)]["item_id"].drop_duplicates():
            t = client.get(f"/v1/recommend/item/{cand}?limit=10&in_stock_only=false").json()
            if any(not x["in_stock"] for x in t["items"]):
                pdp_seed = cand
                break
        assert pdp_seed, "no seed with out-of-stock neighbor found"
        r5a = R.call("GET", f"/v1/recommend/item/{pdp_seed}?limit=10&in_stock_only=false")
        r5b = R.call("GET", f"/v1/recommend/item/{pdp_seed}?limit=10&in_stock_only=true")
        r5c = R.call("POST", "/v1/recommend/user", {"user_id": "", "limit": 0, "recent_item_ids": "nope"}, expected=422)
        resp_c = r5c["response"]
        finish("test_05_guardrails_out_of_stock_and_validation_error.png",
               "Test 5 - Out-of-stock guardrail + structured 422 validation error", [r5a, r5b, r5c], [
            (any(not x["in_stock"] for x in r5a["response"]["items"]), "in_stock_only=false CAN surface out-of-stock items (control)"),
            (all(x["in_stock"] for x in r5b["response"]["items"]), "in_stock_only=true excludes every out_of_stock item"),
            (resp_c.get("error") == "validation_error", '422 body has error == "validation_error"'),
            (isinstance(resp_c.get("details"), list) and len(resp_c["details"]) >= 2, f"structured details list ({len(resp_c.get('details', []))} issues)"),
        ])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    (OUT / "api_test_summary.json").write_text(json.dumps({"failures": failures, "results": results}, indent=1))
    print(f"\n{'ALL TESTS PASSED' if not failures else str(failures) + ' ASSERTION(S) FAILED'} -> {OUT}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
