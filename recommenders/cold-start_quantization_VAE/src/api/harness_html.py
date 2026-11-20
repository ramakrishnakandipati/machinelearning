"""Self-contained HTML/JS test dashboard served at /ui/test-harness (real fetch() calls against the live API)."""
import json

_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cold-Start API Test Harness</title>
<style>
:root{
  --surface:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --line:#e4e3df; --accent:#2a78d6;
  --ok:#0a7a3d; --bad:#c0392b; --l1:#2a78d6; --l2:#eb6834; --l3:#1baf7a; --l4:#eda100; --code-bg:#f4f3ef;
  color-scheme: light;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --surface:#1a1a19; --panel:#222221; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8a8984; --line:#363532; --accent:#3987e5;
    --ok:#4cc58a; --bad:#e66767; --l1:#3987e5; --l2:#d95926; --l3:#199e70; --l4:#c98500; --code-bg:#2a2a28; color-scheme: dark;
  }
}
:root[data-theme="dark"]{
  --surface:#1a1a19; --panel:#222221; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8a8984; --line:#363532; --accent:#3987e5;
  --ok:#4cc58a; --bad:#e66767; --l1:#3987e5; --l2:#d95926; --l3:#199e70; --l4:#c98500; --code-bg:#2a2a28; color-scheme: dark;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:center;justify-content:space-between;padding:16px 24px;border-bottom:1px solid var(--line)}
header h1{font-size:18px;margin:0;font-weight:650}
header .sub{color:var(--ink2);font-size:12.5px}
.pills{display:flex;flex-wrap:wrap;gap:8px}
.pill{border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:12px;color:var(--ink2);background:var(--panel)}
.pill b{color:var(--ink);font-weight:600}
.pill.ok::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--ok);margin-right:6px}
.layout{display:grid;grid-template-columns:230px 1fr;gap:20px;padding:20px 24px;max-width:1500px;margin:0 auto}
nav{display:flex;flex-direction:column;gap:6px}
nav button{all:unset;cursor:pointer;padding:10px 12px;border-radius:8px;border:1px solid transparent;color:var(--ink2);font-weight:550}
nav button small{display:block;font-weight:400;color:var(--muted);font-size:11.5px}
nav button:hover{background:var(--panel);border-color:var(--line)}
nav button[aria-selected="true"]{background:var(--panel);border-color:var(--line);color:var(--ink);box-shadow:inset 3px 0 0 var(--accent)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}
h2{margin:0 0 4px;font-size:17px}
.desc{color:var(--ink2);margin:0 0 12px}
.endpoint{display:inline-flex;gap:8px;align-items:center;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;background:var(--code-bg);padding:4px 10px;border-radius:6px;margin-bottom:12px}
.method{font-weight:700;color:var(--accent)}
.row{display:grid;grid-template-columns:minmax(300px,420px) 1fr;gap:18px;align-items:start}
textarea{width:100%;min-height:250px;resize:vertical;font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace;background:var(--code-bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:10px}
.run{all:unset;cursor:pointer;margin-top:10px;background:var(--accent);color:#fff;padding:8px 18px;border-radius:8px;font-weight:600}
.run:hover{filter:brightness(1.08)}
.status{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.chip{font-size:12px;padding:3px 10px;border-radius:6px;background:var(--code-bg);color:var(--ink2)}
.chip b{color:var(--ink)}
.chip.good b{color:var(--ok)} .chip.err b{color:var(--bad)}
.sid{display:inline-flex;gap:3px;vertical-align:middle}
.sid span{font:600 11.5px ui-monospace,Menlo,Consolas,monospace;padding:1px 6px;border-radius:5px;color:var(--ink);border-left:3px solid var(--c);background:color-mix(in srgb,var(--c) 16%,var(--panel))}
.sid span:nth-child(1){--c:var(--l1)} .sid span:nth-child(2){--c:var(--l2)} .sid span:nth-child(3){--c:var(--l3)} .sid span:nth-child(4){--c:var(--l4)}
.sid.big span{font-size:20px;padding:4px 12px;border-left-width:4px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{font-weight:600;text-align:left;color:var(--ink2);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid var(--line);padding:6px 8px}
td{padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.tag{font-size:10.5px;font-weight:650;padding:1px 6px;border-radius:4px;background:color-mix(in srgb,var(--l3) 22%,var(--panel));margin-left:6px}
.muted{color:var(--muted)} .why{color:var(--ink2);font-size:12px;max-width:340px}
h3{font-size:13px;margin:16px 0 6px;color:var(--ink2);text-transform:uppercase;letter-spacing:.05em}
.grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.mini{border:1px solid var(--line);border-radius:10px;padding:10px;background:var(--surface)}
.mini h4{margin:0 0 2px;font-size:13.5px} .mini .lat{font-size:12px;color:var(--ink2);margin-bottom:6px}
.mini ol{margin:0;padding:0;list-style:none} .mini li{padding:5px 0;border-top:1px solid var(--line)}
.mini li .t{font-size:12.5px;display:block;margin-bottom:2px}
.tree .lv{display:grid;grid-template-columns:70px 70px 1fr 160px;gap:10px;align-items:center;padding:7px 0;border-bottom:1px solid var(--line)}
.bar{height:10px;border-radius:5px;background:var(--code-bg);overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
pre{margin:0;max-height:300px;overflow:auto;background:var(--code-bg);padding:12px;border-radius:8px;font:12px/1.45 ui-monospace,Menlo,Consolas,monospace}
.k{color:var(--accent)} .s{color:var(--l3)} .n{color:var(--l2)} .b{color:var(--l4)}
#result[data-state="idle"]{color:var(--muted)}
.jac{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
@media (max-width:1000px){.layout{grid-template-columns:1fr}.row{grid-template-columns:1fr}.grid4{grid-template-columns:1fr 1fr}nav{flex-direction:row;flex-wrap:wrap}}
</style>
</head>
<body>
<header>
  <div><h1>Omnichannel Cold-Start Recommender &mdash; Test Harness</h1>
  <div class="sub">VQ-VAE / RQ-VAE Semantic IDs &middot; live <code>fetch()</code> calls against this server</div></div>
  <div class="pills" id="health-pills"><span class="pill">checking health&hellip;</span></div>
</header>
<div class="layout">
  <nav id="tabs" role="tablist"></nav>
  <section class="card" id="panel">
    <h2 id="p-title"></h2><p class="desc" id="p-desc"></p>
    <div class="endpoint"><span class="method" id="p-method"></span><span id="p-path"></span></div>
    <div class="row">
      <div><textarea id="req" spellcheck="false" aria-label="request body"></textarea><br>
        <button class="run" id="run-btn">Run test</button></div>
      <div id="result" data-state="idle">Press <b>Run test</b> to call the endpoint.</div>
    </div>
  </section>
</div>
<script>
const SAMPLES = __SAMPLES__;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const sid = (codes, big) => `<span class="sid${big?' big':''}">${codes.map(c=>`<span>${c}</span>`).join('')}</span>`;
const hl = o => esc(JSON.stringify(o, null, 2)).replace(/(&quot;[^&]*?&quot;)(\s*:)?|\b(true|false|null)\b|(-?\d+\.?\d*(?:[eE][+-]?\d+)?)/g,
  (m, str, colon, bool, num) => str ? (colon ? `<span class="k">${str}</span>${colon}` : `<span class="s">${str}</span>`) : bool ? `<span class="b">${m}</span>` : `<span class="n">${m}</span>`);

function itemsTable(list, withWhy = true, extra = null) {
  return `<table><thead><tr><th>#</th><th>Item</th><th>Semantic ID</th><th class="num">Price</th><th class="num">Margin</th>
    <th class="num">Score</th>${extra?`<th class="num">${extra.h}</th>`:''}${withWhy?'<th>Why</th>':''}</tr></thead><tbody>` +
    list.map((r, i) => `<tr><td>${i+1}</td><td><b>${esc(r.title)}</b>${r.new_launch?'<span class="tag">NEW</span>':''}<br>
      <span class="muted">${esc(r.category)} / ${esc(r.subcategory)} &middot; #${r.item_id}</span></td><td>${sid(r.semantic_id)}</td>
      <td class="num">$${r.price.toFixed(2)}</td><td class="num">${(r.margin*100).toFixed(0)}%</td>
      <td class="num">${r.score===undefined?'':r.score.toFixed(3)}</td>${extra?`<td class="num">${extra.f(r)}</td>`:''}
      ${withWhy?`<td class="why">${esc(r.explanation||'')}</td>`:''}</tr>`).join('') + '</tbody></table>';
}
const chips = (d, http) => `<div class="status"><span class="chip ${http<300?'good':'err'}">HTTP <b>${http}</b></span>
  <span class="chip">round-trip <b>${d._rtt.toFixed(0)} ms</b></span>${d.latency_ms!==undefined?`<span class="chip">server inference <b>${d.latency_ms} ms</b></span>`:''}
  ${d.approach?`<span class="chip">approach <b>${d.approach}</b></span>`:''}${d.regime?`<span class="chip">regime <b>${d.regime}</b></span>`:''}</div>`;
const rawJson = d => { const c = {...d}; delete c._rtt; return `<h3>Raw JSON response</h3><pre>${hl(c)}</pre>`; };

const R = {
  itemCold: d => `<h3>Semantic ID assigned on the fly by RQ-VAE</h3>${sid(d.semantic_id, true)}
    <p class="muted">VQ-VAE single code: <b>${d.vq_vae_code}</b> &middot; ${d.is_novel_full_id?'novel full ID (no catalog item shares all 4 tokens)':'full ID already present in catalog'} &middot;
    catalog items sharing prefix: ${Object.entries(d.catalog_items_sharing_prefix).map(([k,v])=>`${k.replace('depth_','c1..c')}: <b>${v}</b>`).join(' &middot; ')}</p>
    <h3>Target user segments</h3><table><thead><tr><th>Region</th><th>Age band</th><th>Intent category</th><th class="num">Affinity (z)</th></tr></thead><tbody>
    ${d.target_segments.map(s=>`<tr><td>${s.region}</td><td>${s.age_band}</td><td>${s.intent_category}</td><td class="num">${s.affinity_z}</td></tr>`).join('')}</tbody></table>
    <h3>Semantically similar catalog items</h3>${itemsTable(d.similar_items, false, {h:'Shared prefix', f:r=>`c1..c${r.shared_prefix_depth}`})}`,
  userCold: d => `<h3>Predicted Semantic-ID prefix for this visitor</h3><p>${d.predicted_semantic_prefix.slice(0,2).map(l=>
      `level ${l.level}: ${l.top_codes.map(c=>`<span class="sid"><span>${c.code}</span></span> ${(c.prob*100).toFixed(1)}%`).join(' ')}`).join(' &nbsp;|&nbsp; ')}</p>
    <h3>Top-${d.recommendations.length} recommendations</h3>${itemsTable(d.recommendations)}`,
  systemCold: d => `<p class="muted">${esc(d.strategy)}</p><h3>Semantic clusters represented</h3><table><thead><tr><th>Level-1 code</th><th class="num">Items</th>
    <th class="num">Avg margin</th><th>Representative</th></tr></thead><tbody>${d.semantic_clusters.map(c=>`<tr><td>${sid([c.level1_code])}</td>
    <td class="num">${c.n_items}</td><td class="num">${(c.avg_margin*100).toFixed(0)}%</td><td>${esc(c.representative)}</td></tr>`).join('')}</tbody></table>
    <h3>Diversified high-margin recommendations (mean margin ${(d.avg_margin*100).toFixed(0)}%)</h3>${itemsTable(d.recommendations)}`,
  compare: d => `<p class="muted">scenario <b>${d.scenario}</b> &middot; pool: ${d.candidate_pool}</p><div class="grid4">${Object.entries(d.approaches).map(([n,a])=>
      `<div class="mini"><h4>${n.replace('_',' ')}</h4><div class="lat">${a.latency_ms} ms</div><ol>${a.recommendations.map((r,i)=>
      `<li><span class="t">${i+1}. ${esc(r.title)}</span>${sid(r.semantic_id)}</li>`).join('')}</ol></div>`).join('')}</div>
    <h3>Jaccard overlap between lists</h3><div class="jac">${Object.entries(d.jaccard_overlap).map(([k,v])=>`<span class="chip">${k.replace(/_/g,' ')} <b>${v}</b></span>`).join('')}</div>`,
  inspect: d => `<h3>${esc(d.item.title)} &middot; ${esc(d.item.category)} / ${esc(d.item.subcategory)}</h3>${sid(d.semantic_id, true)}
    <p class="muted">VQ-VAE single code: <b>${d.vq_vae_code}</b></p><h3>Hierarchical tree path (coarse &rarr; fine)</h3><div class="tree">
    ${d.tree_path.map(t=>`<div class="lv"><b>level ${t.level}</b><span class="sid"><span style="--c:var(--l${t.level})">${t.code}</span></span>
      <div><div class="bar"><i style="width:${Math.min(100,100*Math.log(1+t.node_size)/Math.log(1+d.tree_path[0].node_size))}%"></i></div>
      <span class="muted">${t.node_size} items &middot; ${t.dominant_category} (${(t.category_purity*100).toFixed(0)}% pure)</span></div>
      <span class="muted">residual ${t.residual_pct_of_encoder_norm}% of latent norm</span></div>`).join('')}</div>
    <h3>Look-alikes</h3>${itemsTable(d.similar_items, false)}`
};
const TABS = {
  'item-cold': {name:'Item cold-start', sub:'brand-new SKU, 0 interactions', method:'POST', path:'/recommend/item-cold-start', body:SAMPLES.item_cold, render:R.itemCold,
    desc:'Raw attributes of a new SKU are encoded on the fly by the RQ-VAE into a hierarchical Semantic ID; returns target user segments and look-alikes.'},
  'user-cold': {name:'User cold-start', sub:'context only, 0-2 interactions', method:'POST', path:'/recommend/user-cold-start', body:SAMPLES.user_cold, render:R.userCold,
    desc:'A first-time visitor described by channel / device / region / intent and an initial click gets a personalised top-K with Semantic-ID explanations.'},
  'system-cold': {name:'System cold-start', sub:'Day-0 anonymous session', method:'POST', path:'/recommend/system-cold-start', body:SAMPLES.system_cold, render:R.systemCold,
    desc:'No interaction logs exist: diversified, high-margin semantic cluster representatives for the landing category.'},
  'compare': {name:'Compare approaches', sub:'all four side by side', method:'POST', path:'/recommend/compare-approaches', body:SAMPLES.compare, render:R.compare,
    desc:'Runs semantic trie, generative seq-to-Semantic-ID, codebook-bridged two-tower and hybrid on the same request with per-approach latency.'},
  'inspect': {name:'Semantic ID inspector', sub:'RQ-VAE tokens & look-alikes', method:'GET', path:'/semantic-id/inspect/{item_id}', body:{item_id:SAMPLES.inspect.item_id, k:6}, render:R.inspect,
    desc:'Walks an item down the residual-quantization tree: node sizes, category purity, residual energy per level, siblings and look-alikes.'}
};
let current = null;
function select(key) {
  current = key; const t = TABS[key];
  document.querySelectorAll('#tabs button').forEach(b => b.setAttribute('aria-selected', b.dataset.tab === key));
  $('p-title').textContent = t.name; $('p-desc').textContent = t.desc; $('p-method').textContent = t.method; $('p-path').textContent = t.path;
  $('req').value = JSON.stringify(t.body, null, 2);
  $('result').dataset.state = 'idle'; $('result').innerHTML = 'Press <b>Run test</b> to call the endpoint.';
  history.replaceState(null, '', '?tab=' + key);
}
async function run() {
  const t = TABS[current]; const res = $('result'); res.dataset.state = 'loading'; res.textContent = 'Running…';
  let body; try { body = JSON.parse($('req').value); } catch (e) { res.dataset.state = 'error'; res.innerHTML = `<div class="chip err">Invalid JSON: <b>${esc(e.message)}</b></div>`; return; }
  const t0 = performance.now(); let url = t.path, init = {method: t.method};
  if (t.method === 'GET') { const {item_id, ...q} = body; url = `/semantic-id/inspect/${item_id}?` + new URLSearchParams(q); }
  else { init.headers = {'Content-Type':'application/json'}; init.body = JSON.stringify(body); }
  try {
    const r = await fetch(url, init); const d = await r.json(); d._rtt = performance.now() - t0;
    if (!r.ok) { res.dataset.state = 'error'; res.innerHTML = chips(d, r.status) + rawJson(d); return; }
    res.innerHTML = chips(d, r.status) + t.render(d) + rawJson(d); res.dataset.state = 'done';
  } catch (e) { res.dataset.state = 'error'; res.innerHTML = `<div class="chip err">Network error: <b>${esc(e.message)}</b></div>`; }
}
(async function init() {
  $('tabs').innerHTML = Object.entries(TABS).map(([k,t]) => `<button role="tab" data-tab="${k}" id="tab-${k}">${t.name}<small>${t.sub}</small></button>`).join('');
  document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => select(b.dataset.tab));
  $('run-btn').onclick = run;
  const q = new URLSearchParams(location.search); select(TABS[q.get('tab')] ? q.get('tab') : 'item-cold');
  try { const h = await (await fetch('/health')).json();
    $('health-pills').innerHTML = `<span class="pill ok"><b>${h.status}</b></span><span class="pill">device <b>${h.device}</b></span>
      <span class="pill">codebooks <b>${h.n_levels} &times; ${h.codebook_size}</b></span><span class="pill">catalog <b>${h.catalog_items}</b> items (${h.new_launch_items} new)</span>
      <span class="pill">unique Semantic IDs <b>${h.unique_semantic_ids}</b></span>`;
  } catch (e) { $('health-pills').innerHTML = '<span class="pill">health check failed</span>'; }
  if (q.get('autorun')) run();
})();
</script>
</body>
</html>
"""


def render_harness(samples: dict) -> str:
    return _HTML.replace("__SAMPLES__", json.dumps(samples))
