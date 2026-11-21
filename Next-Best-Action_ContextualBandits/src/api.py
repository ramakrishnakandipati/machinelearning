"""FastAPI serving layer for the Next-Best-Action engine.

Run:  uvicorn src.api:app --port 8000      (docs: /docs, dashboard: /dashboard)
"""
import json
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import ACTIONS, ACTION_SHORT, DEVICE, LOGS_DIR, MODELS_DIR
from .models import load_model
from .preprocessing import load_preprocessor, transform
from .telemetry import log_jsonl

API_LOG = LOGS_DIR / "api_requests.jsonl"
EXPLORATION_EPS = 0.05   # uniform exploration mixed into the served policy => propensities always >= eps/K
UPDATE_EVERY = 25        # apply an online model update after this many reward feedbacks

STATE: dict = {}
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    reg = json.loads((MODELS_DIR / "model_registry.json").read_text())
    STATE["registry"] = reg
    STATE["pre"] = load_preprocessor(MODELS_DIR / "preprocessor.joblib")
    STATE["model_name"] = reg["best_model"]
    STATE["model"] = load_model(MODELS_DIR, reg["best_model"])
    STATE["pending"] = OrderedDict()   # request_id -> (x, action, propensity)
    STATE["feedback"] = []             # (x, a, r, p) awaiting the next online update
    STATE["n_updates"] = 0
    STATE["rng"] = np.random.default_rng(0)
    yield


app = FastAPI(
    title="Next-Best-Action Engine - Hospitality Contextual Bandits",
    description="Margin-aware offer recommendation for resort guests using the best off-policy-validated "
                "contextual bandit. `/predict_nba` logs the serving propensity; `/update_reward` closes the loop.",
    version="1.0.0", lifespan=lifespan)


class GuestContext(BaseModel):
    loyalty_tier: Literal["Bronze", "Silver", "Gold", "Platinum"] = "Gold"
    lead_time: int = Field(30, ge=0, le=365, description="days between booking and arrival")
    los: int = Field(3, ge=1, le=21, description="length of stay (nights)")
    booking_channel: Literal["Web", "App", "OTA"] = "Web"
    hist_spend: float = Field(1500.0, ge=0, description="historical spend ($)")
    ancillary_rate: float = Field(0.3, ge=0, le=1, description="past ancillary attachment rate")
    month: int = Field(7, ge=1, le=12)
    day_of_week: Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] = "Fri"
    party_size: int = Field(2, ge=1, le=10)
    guest_archetype: Literal["Business", "Family", "Couple", "Solo"] = "Couple"


class PredictResponse(BaseModel):
    request_id: str
    action: int
    action_name: str
    propensity: float
    explored: bool
    expected_margin: Dict[str, float]
    probabilities: Dict[str, float]
    model: str
    device: str
    latency_ms: float


class RewardFeedback(BaseModel):
    request_id: str
    reward: float = Field(..., description="realised net contribution margin ($); 0 if the guest did not book")
    converted: Optional[bool] = None


def _row(ctx: GuestContext):
    return transform(STATE["pre"], pd.DataFrame([ctx.model_dump()]))


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE.get("model_name"), "device": str(DEVICE),
            "pending_requests": len(STATE.get("pending", {})), "online_updates": STATE.get("n_updates", 0)}


@app.get("/model_info")
def model_info():
    reg = STATE["registry"]
    return {"serving_model": STATE["model_name"], "doubly_robust_value": reg["dr_value"],
            "true_value_simulator": reg["true_value"], "actions": ACTIONS, "exploration_eps": EXPLORATION_EPS}


@app.post("/predict_nba", response_model=PredictResponse)
def predict_nba(ctx: GuestContext):
    t0 = time.perf_counter()
    model = STATE["model"]
    x = _row(ctx)
    with _lock:
        pi = model.action_probs(x)[0]
        scores = model.mean_scores(x)[0]
        K = len(pi)
        probs = (1 - EXPLORATION_EPS) * pi + EXPLORATION_EPS / K
        a = int(STATE["rng"].choice(K, p=probs / probs.sum()))
        prop = float(probs[a] / probs.sum())
        rid = uuid.uuid4().hex[:12]
        STATE["pending"][rid] = (x[0], a, prop)
        while len(STATE["pending"]) > 20000:
            STATE["pending"].popitem(last=False)
    resp = PredictResponse(
        request_id=rid, action=a, action_name=ACTIONS[a], propensity=prop, explored=bool(a != int(np.argmax(pi))),
        expected_margin={ACTION_SHORT[k]: float(scores[k]) for k in range(K)},
        probabilities={ACTION_SHORT[k]: float(probs[k]) for k in range(K)},
        model=STATE["model_name"], device=str(DEVICE), latency_ms=round((time.perf_counter() - t0) * 1000, 2))
    log_jsonl(API_LOG, {"type": "predict", "request_id": rid, "context": ctx.model_dump(), "action": a,
                        "propensity": prop, "model": STATE["model_name"], "latency_ms": resp.latency_ms})
    return resp


@app.post("/update_reward")
def update_reward(fb: RewardFeedback):
    with _lock:
        item = STATE["pending"].pop(fb.request_id, None)
        if item is None:
            raise HTTPException(404, f"unknown or already-rewarded request_id {fb.request_id}")
        x, a, p = item
        STATE["feedback"].append((x, a, fb.reward, p))
        applied = False
        if len(STATE["feedback"]) >= UPDATE_EVERY:
            X = np.stack([f[0] for f in STATE["feedback"]])
            A = np.array([f[1] for f in STATE["feedback"]])
            R = np.array([f[2] for f in STATE["feedback"]])
            P = np.array([f[3] for f in STATE["feedback"]])
            STATE["model"].update(X, A, R, P)
            STATE["feedback"].clear()
            STATE["n_updates"] += 1
            applied = True
        buffered = len(STATE["feedback"])
    log_jsonl(API_LOG, {"type": "reward", "request_id": fb.request_id, "action": a, "propensity": p,
                        "reward": fb.reward, "converted": fb.converted, "model_updated": applied})
    return {"status": "recorded", "model_updated": applied, "buffered_feedback": buffered,
            "online_updates": STATE["n_updates"]}


DASHBOARD_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>NBA Decision Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--bg:#0f172a;--card:#1e293b;--ink:#e2e8f0;--mut:#94a3b8;--acc:#38bdf8;--ok:#34d399}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:18px 28px;border-bottom:1px solid #334155}h1{margin:0;font-size:22px}header span{color:var(--mut);font-size:13px}
main{display:grid;grid-template-columns:360px 1fr;gap:20px;padding:20px 28px}
.card{background:var(--card);border-radius:12px;padding:18px}label{display:block;font-size:12px;color:var(--mut);margin:10px 0 3px}
input,select{width:100%;padding:7px 9px;border-radius:6px;border:1px solid #475569;background:#0f172a;color:var(--ink)}
button{margin-top:14px;width:100%;padding:10px;border:0;border-radius:8px;background:var(--acc);color:#04141f;font-weight:700;cursor:pointer}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.rec{font-size:24px;font-weight:700;color:var(--ok);margin:4px 0}
.bar{display:grid;grid-template-columns:170px 1fr 90px;align-items:center;gap:8px;margin:7px 0;font-size:13px}
.track{background:#0f172a;border-radius:5px;height:18px;overflow:hidden}.fill{height:100%;background:#475569}
.fill.best{background:var(--ok)}.kv{color:var(--mut);font-size:13px}.kv b{color:var(--ink)}
@media(max-width:900px){main{grid-template-columns:1fr}}</style></head><body>
<header><h1>Next-Best-Action Decision Dashboard</h1><span id="sub">contextual-bandit offer optimisation &middot; hospitality &amp; resorts</span></header>
<main><div class="card"><b>Guest context</b>
<label>Preset</label><select id="preset"><option value="">custom</option><option value="b">Business traveller</option>
<option value="f">Family vacation</option><option value="c">Couple's getaway</option><option value="s">Solo budget (OTA)</option></select>
<div class="row"><div><label>Loyalty tier</label><select id="loyalty_tier"><option>Bronze</option><option>Silver</option><option selected>Gold</option><option>Platinum</option></select></div>
<div><label>Archetype</label><select id="guest_archetype"><option>Business</option><option>Family</option><option selected>Couple</option><option>Solo</option></select></div>
<div><label>Channel</label><select id="booking_channel"><option>Web</option><option>App</option><option>OTA</option></select></div>
<div><label>Arrival day</label><select id="day_of_week"><option>Mon</option><option>Tue</option><option>Wed</option><option>Thu</option><option selected>Fri</option><option>Sat</option><option>Sun</option></select></div>
<div><label>Lead time (days)</label><input id="lead_time" type="number" value="30"></div>
<div><label>Nights</label><input id="los" type="number" value="3"></div>
<div><label>Party size</label><input id="party_size" type="number" value="2"></div>
<div><label>Month (1-12)</label><input id="month" type="number" value="7"></div>
<div><label>Historical spend ($)</label><input id="hist_spend" type="number" value="1500"></div>
<div><label>Ancillary attach rate</label><input id="ancillary_rate" type="number" step="0.05" value="0.3"></div></div>
<button id="go">Recommend next best action</button></div>
<div class="card"><div class="kv">Recommended action</div><div class="rec" id="rec">&mdash;</div>
<div class="kv" id="meta">Submit a guest context to score all six actions.</div><hr style="border-color:#334155;margin:14px 0">
<b>Expected net margin per action ($)</b><div id="bars"></div><hr style="border-color:#334155;margin:14px 0">
<b>Close the loop</b><div class="row"><div><label>Realised margin ($, 0 if no booking)</label><input id="reward" type="number" value="450"></div>
<div><button id="fb" style="margin-top:24px">Send reward feedback</button></div></div><div class="kv" id="fbout"></div></div></main>
<script>
const F=["loyalty_tier","lead_time","los","booking_channel","hist_spend","ancillary_rate","month","day_of_week","party_size","guest_archetype"];
const P={b:{loyalty_tier:"Gold",guest_archetype:"Business",booking_channel:"App",day_of_week:"Tue",lead_time:5,los:2,party_size:1,month:10,hist_spend:4200,ancillary_rate:0.35},
f:{loyalty_tier:"Silver",guest_archetype:"Family",booking_channel:"Web",day_of_week:"Sat",lead_time:70,los:6,party_size:4,month:7,hist_spend:1800,ancillary_rate:0.45},
c:{loyalty_tier:"Gold",guest_archetype:"Couple",booking_channel:"Web",day_of_week:"Fri",lead_time:40,los:4,party_size:2,month:2,hist_spend:2600,ancillary_rate:0.3},
s:{loyalty_tier:"Bronze",guest_archetype:"Solo",booking_channel:"OTA",day_of_week:"Wed",lead_time:20,los:2,party_size:1,month:5,hist_spend:350,ancillary_rate:0.1}};
const $=i=>document.getElementById(i);let rid=null;
$("preset").onchange=e=>{const p=P[e.target.value];if(p)for(const k in p)$(k).value=p[k]};
function ctx(){const o={};F.forEach(k=>{const v=$(k).value;o[k]=isNaN(v)||v===""?v:Number(v)});return o}
$("go").onclick=async()=>{const r=await fetch("/predict_nba",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(ctx())});
const d=await r.json();if(!r.ok){$("meta").textContent="Error: "+JSON.stringify(d.detail);return}
rid=d.request_id;$("rec").textContent=d.action_name;
$("meta").innerHTML=`model <b>${d.model}</b> &middot; propensity <b>${d.propensity.toFixed(3)}</b> &middot; ${d.explored?"exploration pick":"greedy pick"} &middot; <b>${d.latency_ms} ms</b> on <b>${d.device}</b> &middot; request <b>${d.request_id}</b>`;
const e=d.expected_margin,mx=Math.max(...Object.values(e)),mn=Math.min(0,...Object.values(e));
$("bars").innerHTML=Object.entries(e).map(([k,v])=>{const best=v===mx;return `<div class="bar"><span>${k}</span><div class="track"><div class="fill ${best?"best":""}" style="width:${Math.max(2,(v-mn)/(mx-mn||1)*100)}%"></div></div><span>$${v.toFixed(0)} <small style="color:#94a3b8">(${(d.probabilities[k]*100).toFixed(0)}%)</small></span></div>`}).join("")};
$("fb").onclick=async()=>{if(!rid){$("fbout").textContent="Get a recommendation first.";return}
const r=await fetch("/update_reward",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({request_id:rid,reward:Number($("reward").value)})});
$("fbout").textContent=JSON.stringify(await r.json())};
</script></body></html>"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML
