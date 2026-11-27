// Builds Retail_Search_Funnel.pptx from deck_data.json + story_shots/. Run from this folder: node build_deck.js
// Needs pptxgenjs, react, react-dom, react-icons, sharp on NODE_PATH and the pptx skill's apply_theme.js (PPTX_SKILL_DIR).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const SKILL = process.env.PPTX_SKILL_DIR || "C:/Users/RamaKrishna/.claude/skills/synced/c4087e75-6e8c-4493-8472-1adcd2a2603e_2c02675d-fb05-4126-b416-349c05b8005e/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));
const fa = require("react-icons/fa");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const SHOTS = path.join(__dirname, "story_shots");
const OUTFILE = path.join(__dirname, "Retail_Search_Funnel.pptx");

// ---------------------------------------------------------------- theme
const THEME = {
  name: "Retail Funnel", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B2430", lt1: "FFFFFF", dk2: "14213D", lt2: "F1F4F8", accent1: "1B998B", accent2: "F4A261", accent3: "E76F51",
            accent4: "3D5A80", accent5: "8D99AE", accent6: "6A4C93", hlink: "1B998B", folHlink: "6A4C93" },
};
const HEX = { teal: "1B998B", amber: "F4A261", coral: "E76F51", steel: "3D5A80", gray: "8D99AE", plum: "6A4C93", ink: "14213D", text: "1B2430", light: "F1F4F8" };
const CH = [HEX.teal, HEX.amber, HEX.steel, HEX.coral, HEX.gray, HEX.plum];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "GPU-Accelerated Retail Search Funnel";
pres.author = "ML Search Team";
const C = pres.SchemeColor;
const W = 13.33, H = 7.5, MX = 0.6;

pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: HEX.ink },
  objects: [],
  slideNumber: undefined,
});
pres.defineSlideMaster({
  title: "SECTION_DARK", background: { color: HEX.ink },
  objects: [],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  objects: [
    { text: { text: "GPU-Accelerated Retail Search Funnel", options: { x: MX, y: 7.08, w: 6, h: 0.3, fontSize: 10, color: C.accent5, margin: 0, isTextBox: true } } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.32, w: W - 2 * MX, h: 0.85, fontFace: "Cambria", fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - MX - 0.6, y: 7.08, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

const sections = new Set();
function newSlide(master, section) {
  if (section && !sections.has(section)) { pres.addSection({ title: section }); sections.add(section); }
  return pres.addSlide({ masterName: master, sectionTitle: section });
}
function content(title, section, notes) {
  const s = newSlide("CONTENT", section);
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function divider(num, title, sub, section) {
  const s = newSlide("SECTION_DARK", section);
  s.addText(num, { x: MX, y: 2.3, w: 3, h: 1.0, fontFace: "Cambria", fontSize: 60, bold: true, color: C.accent2, margin: 0, isTextBox: true });
  s.addText(title, { x: MX, y: 3.3, w: 11.5, h: 0.9, fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, margin: 0, isTextBox: true });
  s.addText(sub, { x: MX, y: 4.25, w: 10.5, h: 0.9, fontSize: 18, color: C.accent5, margin: 0, isTextBox: true, valign: "top" });
  return s;
}

// ---------------------------------------------------------------- helpers
const f2 = (x, d = 2) => Number(x).toFixed(d);
const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
const sgn = (x, d = 1) => (x >= 0 ? "+" : "") + Number(x).toFixed(d);
const MODEL_NAME = { xgb_pointwise: "XGB pointwise", lgbm_pointwise: "LGBM pointwise", xgb_pairwise: "XGB pairwise", lgbm_lambdamart: "LGBM LambdaMART",
  cat_yetirank: "CatBoost YetiRank", xgb_listwise: "XGB listwise (NDCG)", torch_listnet: "ListNet (torch)", dcnv2: "DCNv2 (torch)", ensemble: "Ensemble (default)" };
const MODELS = D.models;

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
async function dims(file) { const m = await sharp(file).metadata(); return { w: m.width, h: m.height }; }

function stat(s, x, y, w, big, label, color = C.accent1, bigSize = 36) {
  s.addText(big, { x, y, w, h: 0.75, fontFace: "Cambria", fontSize: bigSize, bold: true, color, margin: 0, isTextBox: true, valign: "bottom" });
  s.addText(label, { x, y: y + 0.78, w, h: 0.55, fontSize: 13, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}
function card(s, x, y, w, h, heading, lines, opts = {}) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: opts.fill || C.background2 }, line: { color: opts.fill || C.background2, width: 0 }, objectName: "card" });
  s.addText(heading, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.45, fontFace: "Cambria", fontSize: opts.hs || 16, bold: true, color: opts.hc || C.text2, margin: 0, isTextBox: true, valign: "middle" });
  const runs = lines.map((t, i) => ({ text: t, options: { bullet: opts.bullet !== false, breakLine: i < lines.length - 1, paraSpaceAfter: 4 } }));
  s.addText(runs, { x: x + 0.2, y: y + 0.62, w: w - 0.4, h: h - 0.72, fontSize: opts.fs || 13, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}
function para(s, x, y, w, h, items, fs = 14, color = C.text1) {
  const runs = items.map((t, i) => (typeof t === "string"
    ? { text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 } }
    : { text: t.text, options: { bullet: t.bullet !== false, breakLine: i < items.length - 1, paraSpaceAfter: 6, bold: !!t.bold, color: t.color } }));
  s.addText(runs, { x, y, w, h, fontSize: fs, color, margin: 0, isTextBox: true, valign: "top" });
}
function takeaway(s, text, y = 6.45) {
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: W - 2 * MX, h: 0.52, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "takeaway" });
  s.addText(text, { x: MX + 0.2, y, w: W - 2 * MX - 0.4, h: 0.52, fontSize: 14, bold: true, color: C.background1, margin: 0, isTextBox: true, valign: "middle" });
}
function heat(v, lo, hi) { // teal scale for table cells
  const t = hi === lo ? 0.5 : Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  const a = [0xEA, 0xF6, 0xF4], b = [0x7F, 0xCF, 0xC4];
  return a.map((x, i) => Math.round(x + (b[i] - x) * t).toString(16).padStart(2, "0")).join("").toUpperCase();
}
function table(s, rows, o) {
  const fs = o.fs || 11;
  const out = rows.map((r, ri) => r.map((c, ci) => {
    const cell = typeof c === "object" && c !== null && "text" in c ? c : { text: String(c) };
    const opt = Object.assign({ fontSize: fs, color: C.text1, align: ci === 0 ? "left" : (o.align || "center"), valign: "middle", margin: [0.03, 0.07, 0.03, 0.07] }, cell.options || {});
    if (ri === 0) Object.assign(opt, { bold: true, color: C.background1, fill: { color: C.accent4 } });
    else if (!opt.fill) opt.fill = { color: ri % 2 ? C.background1 : C.background2 };
    return { text: cell.text, options: opt };
  }));
  s.addTable(out, { x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH || 0.3, border: { type: "solid", color: "DDE3EA", pt: 0.5 }, autoPage: false });
}
function chartBase(extra) {
  return Object.assign({
    catAxisLabelColor: "4B5563", valAxisLabelColor: "4B5563", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
    dataLabelFontFace: "+mn-lt", dataLabelFontSize: 10, dataLabelColor: "1B2430", legendFontFace: "+mn-lt", legendFontSize: 11,
    legendColor: "1B2430", titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: "1B2430", showTitle: true,
  }, extra);
}
function barChart(s, series, labels, x, y, w, h, o = {}) {
  const data = series.map((se) => ({ name: se.name, labels, values: se.values }));
  s.addChart(o.horizontal ? pres.charts.BAR : pres.charts.BAR, data, chartBase(Object.assign({
    x, y, w, h, barDir: o.horizontal ? "bar" : "col", barGrouping: o.stacked ? "stacked" : "clustered", chartColors: o.colors || CH,
    showLegend: series.length > 1, legendPos: "b", showValue: true, dataLabelPosition: o.stacked ? "ctr" : "outEnd",
    dataLabelFormatCode: o.fmt || "0.00", valAxisMinVal: o.min, valAxisMaxVal: o.max, barGapWidthPct: o.gap || 60, title: o.title || "",
    showTitle: !!o.title, valAxisHidden: !!o.hideVal, valAxisLabelFormatCode: o.fmt || "0.00",
  }, o.extra || {})));
}

// ================================================================= build
(async function main() {
  const ic = {
    search: await icon(fa.FaSearch, "FFFFFF"), scale: await icon(fa.FaBalanceScale, "FFFFFF"), store: await icon(fa.FaStore, "FFFFFF"),
    seed: await icon(fa.FaSeedling, "FFFFFF"), bolt: await icon(fa.FaBolt, "FFFFFF"), db: await icon(fa.FaDatabase, "FFFFFF"),
    filter: await icon(fa.FaFilter, "FFFFFF"), chart: await icon(fa.FaChartLine, "FFFFFF"), brain: await icon(fa.FaBrain, "FFFFFF"),
    layer: await icon(fa.FaLayerGroup, "FFFFFF"), flask: await icon(fa.FaFlask, "FFFFFF"), shield: await icon(fa.FaShieldAlt, "FFFFFF"),
  };
  const iconDot = (s, img, x, y, color, d = 0.62) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color, width: 0 }, objectName: "icon-circle" });
    s.addImage({ data: img, x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56, objectName: "icon" });
  };
  const R = D.ranking, AB = D.ablation, RT = D.retrieval, LAT = D.latency;
  const base = R["retrieval_order (no LTR)"];
  const ens = R["ensemble"];
  const abA = AB["1. ranker only (no rules)"], abC = AB["2. + hard filters & soft penalties"], abD = AB["3. + margin-aware re-rank"], abE = AB["4. + MMR diversity (full funnel)"];
  const hyb = RT["hybrid_alpha=0.6"], spr = RT["sparse_only (BM25+char+learned)"], dns = RT["dense_only (bge+multimodal)"], rrf = RT["hybrid_rrf"];
  const flowById = Object.fromEntries(D.flows.map((f) => [f.id, f]));
  const ST = ["A retrieval order", "B + LTR ensemble", "C + hard rules & penalties", "D + margin re-rank", "E + MMR (final page)"];
  const meanFacet = D.flows.reduce((a, f) => a + f.facet_accuracy, 0) / D.flows.length;

  // ---------------------------------------------------------------- 1 title
  {
    const s = newSlide("TITLE_DARK", "Introduction");
    s.addText("GPU-Accelerated Retail Search Funnel", { x: MX, y: 1.7, w: 11.8, h: 1.4, fontFace: "Cambria", fontSize: 48, bold: true, color: C.background1, margin: 0, isTextBox: true, valign: "top" });
    s.addText("End-to-end flow, uplift, all nine rankers compared (eight models plus an ensemble), and seven shopper scenarios told through the Streamlit app", { x: MX, y: 3.25, w: 10.5, h: 0.9, fontSize: 20, color: C.accent5, margin: 0, isTextBox: true, valign: "top" });
    const t = [[`${D.world.n_skus.toLocaleString()}`, "SKUs"], [`${D.funnel.impressions.toLocaleString()}`, "simulated impressions"], ["9", "rankers compared"], ["7", "scenario flows"], [`${f2(LAT.total_ms.p95, 0)} ms`, "p95 end-to-end"]];
    t.forEach(([b, l], i) => {
      s.addText(b, { x: MX + i * 2.45, y: 5.05, w: 2.3, h: 0.7, fontFace: "Cambria", fontSize: 30, bold: true, color: C.accent2, margin: 0, isTextBox: true });
      s.addText(l, { x: MX + i * 2.45, y: 5.75, w: 2.3, h: 0.4, fontSize: 14, color: C.background1, margin: 0, isTextBox: true });
    });
    s.addNotes("Synthetic data throughout; ground truth comes from a latent simulator that serving never sees.");
  }

  // ---------------------------------------------------------------- 2 executive summary
  {
    const s = content("The funnel lifts margin by ~80% over retrieval order", "Introduction", "Headline numbers are from the offline test split (300 lists). Uplift is relative to ranking by the fused retrieval score.");
    stat(s, MX, 1.5, 2.9, `${f2(hyb["recall@200"], 3)}`, "Recall@200, hybrid retrieval", C.accent1);
    stat(s, MX + 3.1, 1.5, 2.9, `${f2(base["ndcg_rel@10"], 2)} → ${f2(abE["ndcg_rel@10"], 2)}`, "NDCG@10: retrieval order → full funnel", C.accent4, 32);
    stat(s, MX + 6.2, 1.5, 2.9, `${sgn((abE["expected_margin@10"] / base["expected_margin@10"] - 1) * 100, 0)}%`, "expected margin per page vs retrieval order", C.accent3);
    stat(s, MX + 9.3, 1.5, 2.9, `${pct(base["oos_rate@10"], 1)} → 0%`, "unfulfillable results in top-10", C.accent2, 32);
    card(s, MX, 3.35, 3.9, 3.0, "What we built", ["4 stages: understand, retrieve, rank, re-rank", "Dense + sparse + multimodal retrieval", "8 LTR models plus a blended ensemble", "Multi-task heads: click, cart, purchase, return"], { fs: 13 });
    card(s, MX + 4.1, 3.35, 3.9, 3.0, "What the data shows", [`Every learned ranker beats retrieval order (NDCG ${f2(base["ndcg_rel@10"], 2)} → 0.89–0.93)`, "Margin step adds the largest revenue and margin lift", "Hard rules remove all unfulfillable results", "Model ordering is within noise"], { fs: 13 });
    card(s, MX + 8.2, 3.35, 3.93, 3.0, "How to read this deck", ["Stages 1–4, one design and metric slide each", "Overall, stage-level and uplift view", "Flow-level metrics and model heat-maps", "Seven Streamlit stories, two slides each"], { fs: 13 });
  }

  // ---------------------------------------------------------------- 3 realities
  {
    const s = content("Four retail search realities drive the design", "Introduction");
    const items = [
      [ic.search, HEX.teal, "Vocabulary gap", "\"waterproof trail kicks\" never appears in the catalog.", "Spell fix, synonyms, embedding neighbours, click-learned sparse terms, dense retrieval."],
      [ic.scale, HEX.steel, "Multi-objective", "Revenue and margin matter, not clicks.", "E[profit] = pCTR · pATC · pCVR · margin · (1 − pReturn), blended with relevance."],
      [ic.store, HEX.coral, "Hard rules and stock", "Out-of-stock and wrong-region items break trust.", "Stock, BOPIS radius, brand and price filters plus return-rate penalties."],
      [ic.seed, HEX.plum, "Long tail and cold start", "Brand-new SKUs have no history.", "Text + image + tabular embeddings make them retrievable on day 0."],
    ];
    items.forEach(([img, col, h, prob, sol], i) => {
      const x = MX + (i % 2) * 6.15, y = 1.5 + Math.floor(i / 2) * 2.7;
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: 5.95, h: 2.5, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, objectName: "card" });
      iconDot(s, img, x + 0.25, y + 0.25, col, 0.7);
      s.addText(h, { x: x + 1.1, y: y + 0.25, w: 4.6, h: 0.7, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, margin: 0, isTextBox: true, valign: "middle" });
      s.addText([{ text: "Problem: ", options: { bold: true } }, { text: prob, options: { breakLine: true } }, { text: "Answer: ", options: { bold: true, color: col } }, { text: sol }],
        { x: x + 0.25, y: y + 1.1, w: 5.45, h: 1.3, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 6 });
    });
  }

  // ---------------------------------------------------------------- 4 architecture
  {
    const s = content("End-to-end flow: four stages, one 150 ms budget", "Introduction", "Latencies are p50 of the served path on an RTX 3090, ensemble ranker, faiss-cpu HNSW.");
    const stages = [
      ["1", "Query understanding", HEX.teal, ["Spell correction (SymSpell)", "NER + intent classifier", "Synonym + embedding expansion"], LAT.query_understanding_ms.p50, "budget 30 ms"],
      ["2", "Hybrid retrieval", HEX.steel, ["BM25F + char n-gram + learned terms", "bge-small multimodal vectors, FAISS HNSW", "Alpha / RRF fusion, N = 200"], LAT.retrieval_ms.p50, "budget 25 ms"],
      ["3", "LTR + multi-task", HEX.amber, ["~50 query/doc/user/cross features", "8 rankers + ensemble", "pCTR, pATC, pCVR, pReturn heads"], LAT.feature_ms.p50 + LAT.ranking_ms.p50, "budget 60 ms"],
      ["4", "Business re-rank", HEX.coral, ["Stock / BOPIS / price / brand filters", "Return & low-stock penalties", "Margin optimiser + MMR diversity"], LAT.rerank_ms.p50, "budget 15 ms"],
    ];
    stages.forEach(([n, t, col, lines, ms, bud], i) => {
      const x = MX + i * 3.08, w = 2.78;
      s.addShape(pres.shapes.RECTANGLE, { x, y: 1.55, w, h: 3.55, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, objectName: "stage-card" });
      s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: 1.7, w: 0.55, h: 0.55, fill: { color: col }, line: { color: col, width: 0 }, objectName: "stage-badge" });
      s.addText(n, { x: x + 0.2, y: 1.7, w: 0.55, h: 0.55, fontFace: "Cambria", fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(t, { x: x + 0.85, y: 1.7, w: w - 0.95, h: 0.55, fontFace: "Cambria", fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true, valign: "middle" });
      para(s, x + 0.2, 2.45, w - 0.4, 1.8, lines, 12.5);
      s.addText(`${f2(ms, 1)} ms`, { x: x + 0.2, y: 4.2, w: w - 0.4, h: 0.5, fontFace: "Cambria", fontSize: 24, bold: true, color: col, margin: 0, isTextBox: true });
      s.addText(`p50 · ${bud}`, { x: x + 0.2, y: 4.68, w: w - 0.4, h: 0.3, fontSize: 11, color: C.accent5, margin: 0, isTextBox: true });
      if (i < 3) s.addText("›", { x: x + w - 0.02, y: 2.9, w: 0.34, h: 0.6, fontSize: 32, bold: true, color: C.accent5, align: "center", margin: 0, isTextBox: true });
    });
    const tot = LAT.total_ms;
    s.addText([{ text: "Total: ", options: { bold: true } }, { text: `p50 ${f2(tot.p50, 0)} ms · p95 ${f2(tot.p95, 0)} ms · p99 ${f2(tot.p99, 0)} ms against a ${LAT.budget_ms.total} ms budget. ` }, { text: "Offline: ", options: { bold: true } }, { text: "the same stages are replayed against a latent simulator to score every stage." }],
      { x: MX, y: 5.35, w: W - 2 * MX, h: 0.8, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
    takeaway(s, "Every stage can be switched on or off, so its contribution is measurable on its own.", 6.3);
  }

  // ---------------------------------------------------------------- 5 data
  {
    const s = content("Synthetic world with a hidden ground-truth simulator", "Introduction");
    const tiles = [[D.world.n_skus.toLocaleString(), "SKUs, 19 product types, 18 brands"], [D.world.n_cold.toLocaleString(), "cold-start SKUs, zero history"], [D.world.n_users.toLocaleString(), "users with brand/price affinity"], [D.world.n_queries.toLocaleString(), "colloquial, typo, brand, formal queries"]];
    tiles.forEach(([b, l], i) => stat(s, MX, 1.45 + i * 1.2, 4.2, b, l, [C.accent1, C.accent2, C.accent4, C.accent3][i], 30));
    barChart(s, [{ name: "Events", values: [D.funnel.impressions, D.funnel.clicks, D.funnel.add_to_cart, D.funnel.purchases, D.funnel.returns] }],
      ["Impressions", "Clicks", "Add to cart", "Purchases", "Returns"], 5.2, 1.4, 7.5, 3.6, { title: "Simulated clickstream funnel (100k sessions)", fmt: "#,##0", colors: [HEX.teal], hideVal: true });
    para(s, 5.2, 5.1, 7.5, 1.4, ["5 stores plus an online warehouse with per-SKU stock, some deliberate stockouts", "8% of SKUs have misleading titles: they win clicks but are returned more", "Latent truth lives in a separate file used only by evaluation"], 13);
  }

  // ================================================================= SECTION: stages
  divider("01", "Stage by stage", "Design, models and metrics for each of the four funnel stages", "Stage by stage");

  // ---------------------------------------------------------------- 6 stage 1
  {
    const s = content("Stage 1: understand the query in about 19 ms", "Stage by stage", "Spell, NER and expansion cost about 1 ms; the rest of the stage is the GPU query embedding.");
    const ex = flowById.typo_price.examples[0];
    card(s, MX, 1.45, 3.85, 2.55, "Spell correction", ["SymSpell on catalog vocabulary", "Protects price words (under, below)", "Fixes domain terms: gortex → gore-tex"], { fs: 13 });
    card(s, MX + 4.0, 1.45, 3.85, 2.55, "NER and intent", ["Gazetteers: brand, colour, category", "Regex: max price from \"under $100\"", "TF-IDF + logistic intent classifier"], { fs: 13 });
    card(s, MX + 8.0, 1.45, 4.13, 2.55, "Query expansion", ["Curated synonym graph (kicks → sneakers)", "Embedding nearest-neighbour terms", "Click-learned sparse terms (train queries)"], { fs: 13 });
    s.addText("Live example", { x: MX, y: 4.2, w: 3, h: 0.35, fontFace: "Cambria", fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
    table(s, [["Raw query", "Corrected", "Intent (conf.)", "Facet accuracy", "Stage time"],
      [ex.query, ex.stage1.corrected, `${ex.stage1.intent} (${pct(ex.stage1.intent_conf)})`, pct(ex.stage1.facet_accuracy), `${f2(ex.stage1.ms, 0)} ms`]],
      { x: MX, y: 4.6, w: 7.6, colW: [1.9, 2.1, 1.7, 1.0, 0.9], fs: 11, rowH: 0.42 });
    stat(s, 8.6, 4.3, 1.9, `${pct(meanFacet)}`, "facet accuracy, mean over 7 flows", C.accent1, 30);
    stat(s, 10.7, 4.3, 1.9, `${f2(LAT.query_understanding_ms.p95, 0)} ms`, "p95 stage time incl. encode", C.accent4, 30);
    takeaway(s, "Facets were right in every flow example; weak spot is intent confidence on generic category queries.", 6.3);
  }

  // ---------------------------------------------------------------- 7 retrieval design
  {
    const s = content("Stage 2: dense and sparse retrieval, fused", "Stage by stage");
    const boxes = [
      [MX, 1.5, HEX.steel, "Sparse", ["BM25F on a sparse matrix: title 3x, brand 2x, attributes 1.5x, description 1x", "Char 3–5-gram TF-IDF for model numbers and partial words", "Learned expansion from purchases (SPLADE-style)"]],
      [MX + 4.2, 1.5, HEX.teal, "Dense multimodal", ["bge-small-en-v1.5 on CUDA, fp16", "Image + tabular vectors projected into text space (ridge)", "FAISS HNSW index, cold SKUs indexed on content only"]],
      [MX + 8.4, 1.5, HEX.amber, "Fusion", ["Alpha: α·dense + (1−α)·sparse, min-max normalised", "RRF as the rank-based alternative", `α = 0.6 tuned on validation recall@50`]],
    ];
    boxes.forEach(([x, y, col, h, lines]) => {
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: 3.95, h: 3.6, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, objectName: "card" });
      s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.2, w: 0.2, h: 0.2, fill: { color: col }, line: { color: col, width: 0 }, objectName: "dot" });
      s.addText(h, { x: x + 0.5, y: y + 0.1, w: 3.2, h: 0.4, fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true, valign: "middle" });
      para(s, x + 0.2, y + 0.7, 3.55, 2.8, lines, 13);
    });
    stat(s, MX, 5.35, 3.6, `${f2(hyb.latency_ms.p95, 0)} ms`, "p95 understanding + retrieval (offline run)", C.accent4, 28);
    stat(s, MX + 4.2, 5.35, 3.6, `${f2(LAT.retrieval_ms.p50, 1)} ms`, "p50 retrieval only (sparse + dense + fuse)", C.accent1, 28);
    stat(s, MX + 8.4, 5.35, 3.6, "200", "candidates handed to ranking", C.accent2, 28);
  }

  // ---------------------------------------------------------------- 8 retrieval comparison
  {
    const s = content("Hybrid retrieval beats either retriever alone", "Stage by stage");
    const names = ["Sparse", "Dense", "Hybrid RRF", "Hybrid α=0.6"], R4 = [spr, dns, rrf, hyb];
    barChart(s, [{ name: "Recall@50", values: R4.map((r) => r["recall@50"]) }, { name: "Recall@200", values: R4.map((r) => r["recall@200"]) },
      { name: "Cold-start R@50", values: R4.map((r) => r["cold_start_recall@50"]) }, { name: "Cold-start R@200", values: R4.map((r) => r["cold_start_recall"]) }],
      names, MX, 1.4, 7.7, 4.8, { title: "Test-split retrieval recall", min: 0, max: 1, fmt: "0.00", gap: 50, extra: { dataLabelFontSize: 9 } });
    card(s, 8.6, 1.45, 4.13, 4.75, "Reading the chart", [`Hybrid α=0.6 reaches recall@50 ${f2(hyb["recall@50"])} vs ${f2(spr["recall@50"])} sparse and ${f2(dns["recall@50"])} dense`, `Cold-start recall@50 rises from ${f2(spr["cold_start_recall@50"])} (sparse) to ${f2(hyb["cold_start_recall@50"])}`, "Dense carries cold SKUs because it uses content embeddings, not history", "At 200 candidates all variants are within 0.03: recall@200 is easy on this small catalog", "Recall is normalised by min(|relevant|, N)"], { fs: 13 });
    takeaway(s, "Alpha fusion wins at the cut-off that matters for ranking; RRF is a close, tuning-free fallback.");
  }

  // ---------------------------------------------------------------- 9 features
  {
    const s = content("Stage 3 inputs: ~50 features in four groups", "Stage by stage");
    const g = [["Query", "length, tokens, price cap, brand/colour flags, intent, subcategory entropy, expansion count, top dense/BM25"], ["Document", "price, margin, Bayesian CTR/CVR, return rate, days since launch, cold flag, stock, brand tier"], ["User", "brand affinity, category affinity, price distance to median spend, recently bought durable"], ["Query × doc", "dense cosine, BM25, char sim, ranks, brand/colour/subcategory match, attribute overlap, price ratio"]];
    g.forEach(([h, t], i) => card(s, MX, 1.45 + i * 1.22, 5.6, 1.1, h, [t], { bullet: false, fs: 12, hs: 14 }));
    const imp = Object.entries(D.importance).slice(0, 10).reverse();
    barChart(s, [{ name: "Gain share", values: imp.map((e) => e[1]) }], imp.map((e) => e[0]), 6.5, 1.4, 6.2, 4.9, { horizontal: true, title: "Top features by gain (LambdaMART)", fmt: "0.00", colors: [HEX.teal], hideVal: false, gap: 40 });
    takeaway(s, "Retrieval signals and history priors dominate; user features add personalisation on top.");
  }

  // ---------------------------------------------------------------- 10 model zoo
  {
    const s = content("Nine rankers across every LTR paradigm", "Stage by stage");
    const meta = {
      xgb_pointwise: ["Pointwise", "XGBoost reg:squarederror", "GPU"], lgbm_pointwise: ["Pointwise", "LightGBM binary (cart or better)", "CPU"],
      xgb_pairwise: ["Pairwise", "XGBoost rank:pairwise", "GPU"], lgbm_lambdamart: ["Pairwise", "LightGBM LambdaMART", "CPU"],
      cat_yetirank: ["Listwise", "CatBoost YetiRank", "GPU"], xgb_listwise: ["Listwise", "XGBoost rank:ndcg", "GPU"],
      torch_listnet: ["Listwise", "PyTorch MLP, ListNet loss", "GPU"], dcnv2: ["Neural", "PyTorch DCNv2, listwise softmax", "GPU"],
      ensemble: ["Blend", "z-score blend of top-4 by val NDCG", "mixed"],
    };
    const rows = [["Model", "Paradigm", "Implementation", "Device", "Train (s)", "Infer (ms/list)"]];
    MODELS.forEach((m) => rows.push([MODEL_NAME[m], meta[m][0], meta[m][1], meta[m][2], R[m].train_s ? f2(R[m].train_s, 1) : "–", R[m].infer_ms_per_list ? f2(R[m].infer_ms_per_list, 1) : "–"]));
    table(s, rows, { x: MX, y: 1.45, w: W - 2 * MX, colW: [2.4, 1.5, 4.1, 1.1, 1.4, 1.63], fs: 12, rowH: 0.44 });
    s.addText("Labels: engagement stage per candidate (0 none, 1 click, 2 cart, 3 purchase) from simulated impressions. Training uses 1,800 query×user lists of 200 candidates; the ensemble members are chosen on validation NDCG.", { x: MX, y: 6.0, w: W - 2 * MX, h: 0.7, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
  }

  // ---------------------------------------------------------------- 11 model quality
  {
    const s = content("All rankers beat retrieval order on relevance", "Stage by stage");
    const names = ["Retrieval order"].concat(MODELS.map((m) => MODEL_NAME[m].replace(" (default)", "")));
    const get = (k) => [base[k]].concat(MODELS.map((m) => R[m][k]));
    barChart(s, [{ name: "NDCG@10 (graded relevance)", values: get("ndcg_rel@10") }, { name: "NDCG@10 (engagement)", values: get("ndcg_eng@10") }], names, MX, 1.4, 8.4, 4.9,
      { title: "Test-split ranking quality, k = 10", min: 0.5, max: 1.0, fmt: "0.00", gap: 40, extra: { catAxisLabelFontSize: 9, dataLabelFontSize: 8 } });
    const best = MODELS.slice().sort((a, b) => R[b]["ndcg_rel@10"] - R[a]["ndcg_rel@10"]);
    card(s, 9.2, 1.45, 3.53, 4.8, "Takeaways", [`Best relevance: ${MODEL_NAME[best[0]]} (${f2(R[best[0]]["ndcg_rel@10"], 3)}), then ${MODEL_NAME[best[1]]} (${f2(R[best[1]]["ndcg_rel@10"], 3)})`, `Gap to retrieval order: +${f2(R[best[0]]["ndcg_rel@10"] - base["ndcg_rel@10"], 2)} NDCG`, `Engagement NDCG: ensemble leads (${f2(ens["ndcg_eng@10"], 3)})`, "Spread among learned models is small: single seed, 300 lists"], { fs: 12 });
    takeaway(s, "Pick on business metrics and stability, not on 0.01 NDCG differences.");
  }

  // ---------------------------------------------------------------- 12 model business value
  {
    const s = content("Expected revenue and margin per ranker", "Stage by stage");
    const names = ["Retrieval order"].concat(MODELS.map((m) => MODEL_NAME[m].replace(" (default)", "")));
    const get = (k) => [base[k]].concat(MODELS.map((m) => R[m][k]));
    barChart(s, [{ name: "Expected revenue@10", values: get("expected_revenue@10") }, { name: "Expected margin@10", values: get("expected_margin@10") }], names, MX, 1.4, 8.4, 4.9,
      { title: "Per-page expected value (latent truth, position-discounted)", fmt: "0", gap: 40, colors: [HEX.steel, HEX.teal], extra: { catAxisLabelFontSize: 9, dataLabelFontSize: 8 } });
    const bm = MODELS.slice().sort((a, b) => R[b]["expected_margin@10"] - R[a]["expected_margin@10"])[0];
    card(s, 9.2, 1.45, 3.53, 4.8, "Takeaways", [`Best margin: ${MODEL_NAME[bm]} (${f2(R[bm]["expected_margin@10"], 1)}) vs retrieval order ${f2(base["expected_margin@10"], 1)}`, `Ensemble: ${f2(ens["expected_margin@10"], 1)} (${sgn((ens["expected_margin@10"] / base["expected_margin@10"] - 1) * 100, 0)}%)`, `Learned rankers span ${f2(Math.min(...MODELS.map((m) => R[m]["expected_margin@10"])), 1)}–${f2(Math.max(...MODELS.map((m) => R[m]["expected_margin@10"])), 1)} margin: a spread of ~${Math.round((Math.max(...MODELS.map((m) => R[m]["expected_margin@10"])) / Math.min(...MODELS.map((m) => R[m]["expected_margin@10"])) - 1) * 100)}%`, `Re-ranking (Stage 4) then adds a further ${Math.round((abE["expected_margin@10"] / abA["expected_margin@10"] - 1) * 100)}%`], { fs: 12 });
    takeaway(s, `Learned ranking alone is worth ${sgn((ens["expected_margin@10"] / base["expected_margin@10"] - 1) * 100, 0)}% expected margin before any business rule is applied.`);
  }

  // ---------------------------------------------------------------- 13 multi-task
  {
    const s = content("Multi-task heads turn clicks into expected profit", "Stage by stage");
    const mt = D.multitask;
    const heads = [["pCTR", "click | shown", f2(mt.pCTR_auc, 3), "AUC"], ["pATC", "cart | click", f2(mt["pATC_auc(|click)"], 3), "AUC"], ["pCVR", "purchase | cart", f2(mt["pCVR_auc(|cart)"], 3), "AUC"], ["pReturn", "return | purchase", f2(mt.pReturn_corr_with_latent, 2), "corr. with truth"]];
    heads.forEach(([h, d, v, m], i) => {
      const x = MX + i * 3.08;
      s.addShape(pres.shapes.RECTANGLE, { x, y: 1.5, w: 2.88, h: 2.3, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, objectName: "card" });
      s.addText(h, { x: x + 0.2, y: 1.6, w: 2.5, h: 0.45, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, margin: 0, isTextBox: true });
      s.addText(d, { x: x + 0.2, y: 2.05, w: 2.5, h: 0.35, fontSize: 13, color: C.accent5, margin: 0, isTextBox: true });
      s.addText(v, { x: x + 0.2, y: 2.45, w: 2.5, h: 0.75, fontFace: "Cambria", fontSize: 38, bold: true, color: [HEX.teal, HEX.steel, HEX.amber, HEX.coral][i], margin: 0, isTextBox: true });
      s.addText(m, { x: x + 0.2, y: 3.2, w: 2.5, h: 0.4, fontSize: 12, color: C.text1, margin: 0, isTextBox: true });
    });
    s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 4.1, w: W - 2 * MX, h: 1.0, fill: { color: C.accent4 }, line: { color: C.accent4, width: 0 }, objectName: "formula" });
    s.addText("E[profit] = pCTR · pATC|click · pCVR|cart · margin $ · (1 − pReturn)", { x: MX + 0.3, y: 4.1, w: W - 2 * MX - 0.6, h: 1.0, fontFace: "Cambria", fontSize: 24, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    para(s, MX, 5.35, W - 2 * MX, 1.0, ["Each head is a GPU XGBoost model trained only where the previous stage happened (weighted by event count)", "Conditional heads learn from fewer events, so AUC falls down the funnel: cart-to-purchase is the hardest to separate"], 13);
  }

  // ---------------------------------------------------------------- 14 rerank design
  {
    const s = content("Stage 4: business rules, margin and diversity", "Stage by stage");
    const items = [
      [ic.shield, HEX.coral, "Hard filters", ["Out of stock or not in region", "BOPIS within 60 km of the store", "Price cap (also parsed from the query)", "Brand allow / deny lists"]],
      [ic.filter, HEX.amber, "Soft penalties", ["Return rate above 22%", "Low stock (5 or fewer)", "Recently bought durables"]],
      [ic.chart, HEX.teal, "Margin optimiser", ["Blend rank-normalised relevance with √E[profit]", "Relevance floor 0.35: no profit credit below it", "Margin weight is a request parameter"]],
      [ic.layer, HEX.steel, "MMR diversity", ["λ = 0.92 over fused embeddings", "Max 4 per brand, 12 per subcategory", "Brand cap lifted for brand-intent queries"]],
    ];
    items.forEach(([img, col, h, lines], i) => {
      const x = MX + (i % 2) * 6.15, y = 1.45 + Math.floor(i / 2) * 2.45;
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: 5.95, h: 2.3, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, objectName: "card" });
      iconDot(s, img, x + 0.2, y + 0.2, col, 0.6);
      s.addText(h, { x: x + 0.95, y: y + 0.2, w: 4.8, h: 0.6, fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true, valign: "middle" });
      para(s, x + 0.25, y + 0.95, 5.5, 1.3, lines, 13);
    });
    takeaway(s, "Rules run after ranking, so the ranker stays purely about relevance and value.", 6.4);
  }

  // ---------------------------------------------------------------- 15 ablation
  {
    const s = content("Each re-ranking step, measured on its own", "Stage by stage");
    const names = ["A Retrieval", "B + LTR", "C + Rules", "D + Margin", "E + MMR"];
    const stg = [base, abA, abC, abD, abE];
    barChart(s, [{ name: "Expected margin@10", values: stg.map((r) => r["expected_margin@10"]) }], names, MX, 1.4, 6.4, 3.3, { title: "Expected margin per page", fmt: "0.0", colors: [HEX.teal] });
    barChart(s, [{ name: "NDCG@10", values: stg.map((r) => r["ndcg_rel@10"]) }], names, MX, 4.0 + 0.7, 6.4, 1.65, { title: "", fmt: "0.00", colors: [HEX.steel], min: 0.8, max: 1, extra: { showTitle: false } });
    const rows = [["Stage", "NDCG@10", "Revenue", "Margin", "Unfulf.", "P(return)"]];
    stg.forEach((r, i) => rows.push([names[i], f2(r["ndcg_rel@10"], 3), f2(r["expected_revenue@10"], 1), f2(r["expected_margin@10"], 1), pct(r["oos_rate@10"], 1), r["avg_return_prob@10"] != null ? f2(r["avg_return_prob@10"], 3) : "–"]));
    table(s, rows, { x: 7.3, y: 1.5, w: 5.43, colW: [1.35, 0.85, 0.85, 0.8, 0.75, 0.83], fs: 11, rowH: 0.42 });
    const dl = (a, b, k) => ((b[k] / a[k] - 1) * 100);
    para(s, 7.3, 4.2, 5.43, 2.1, [`Ranker (B): ${sgn(dl(base, abA, "expected_margin@10"), 0)}% margin, ${sgn(dl(base, abA, "expected_revenue@10"), 0)}% revenue vs retrieval order`, `Rules (C): unfulfillable ${pct(abA["oos_rate@10"], 1)} → 0, revenue ${sgn(dl(abA, abC, "expected_revenue@10"), 1)}%, return risk ${f2(abA["avg_return_prob@10"], 3)} → ${f2(abC["avg_return_prob@10"], 3)}`, `Margin step (D): ${sgn(dl(abC, abD, "expected_margin@10"), 0)}% margin, NDCG ${sgn(abD["ndcg_rel@10"] - abC["ndcg_rel@10"], 3)}`, `MMR (E): diversity for ${sgn(dl(abD, abE, "expected_margin@10"), 1)}% margin`], 13);
    takeaway(s, "Margin optimisation is the single biggest lever after the ranker; rules buy trust at a small cost.");
  }

  // ---------------------------------------------------------------- 16 latency
  {
    const s = content("Latency: 66 ms median, 72 ms at p95", "Stage by stage");
    const keys = [["query_understanding_ms", "Understanding"], ["retrieval_ms", "Retrieval"], ["feature_ms", "Features"], ["ranking_ms", "Ranking"], ["rerank_ms", "Re-rank"]];
    barChart(s, keys.map(([k, n]) => ({ name: n, values: [LAT[k].p50, LAT[k].p95] })), ["p50", "p95"], MX, 1.4, 6.6, 4.9, { stacked: true, title: "Stacked stage latency (ms)", fmt: "0", gap: 45, colors: [HEX.teal, HEX.steel, HEX.gray, HEX.amber, HEX.coral] });
    const rows = [["Stage", "p50", "p95", "p99", "Budget"]];
    const bud = { query_understanding_ms: 30, retrieval_ms: 25, feature_ms: "–", ranking_ms: 60, rerank_ms: 15 };
    keys.forEach(([k, n]) => rows.push([n, f2(LAT[k].p50, 1), f2(LAT[k].p95, 1), f2(LAT[k].p99, 1), String(bud[k])]));
    rows.push([{ text: "Total", options: { bold: true } }, { text: f2(LAT.total_ms.p50, 1), options: { bold: true } }, { text: f2(LAT.total_ms.p95, 1), options: { bold: true } }, { text: f2(LAT.total_ms.p99, 1), options: { bold: true } }, { text: "150", options: { bold: true } }]);
    table(s, rows, { x: 7.5, y: 1.5, w: 5.23, colW: [1.7, 0.85, 0.85, 0.85, 0.98], fs: 12, rowH: 0.4 });
    para(s, 7.5, 4.3, 5.23, 2.0, ["Ranking dominates: four-model ensemble plus four heads, with a CPU↔GPU copy per call", "Spec target of <10 ms for understanding holds only without the query embedding", "FAISS runs on CPU (no Windows GPU wheel); HNSW search is ~1 ms anyway"], 13);
  }

  // ================================================================= SECTION: overall & uplift
  divider("02", "Overall metrics and uplift", "What each stage adds over retrieval order, and how that would be proven online", "Overall and uplift");

  // ---------------------------------------------------------------- 17 overall scoreboard
  {
    const s = content("Overall scoreboard", "Overall and uplift", "Screenshot is the scoreboard at the top of the Streamlit End-to-End Flows tab.");
    const sb = path.join(SHOTS, "scoreboard.png"), d = await dims(sb);
    const w = W - 2 * MX, h = w * d.h / d.w;
    s.addImage({ path: sb, x: MX, y: 1.4, w, h, objectName: "scoreboard-screenshot" });
    const y0 = 1.4 + h + 0.35;
    const rows = [["Metric", "Retrieval order", "Full funnel", "Change"],
      ["NDCG@10 (relevance)", f2(base["ndcg_rel@10"], 3), f2(abE["ndcg_rel@10"], 3), sgn(abE["ndcg_rel@10"] - base["ndcg_rel@10"], 3)],
      ["Expected revenue / page", f2(base["expected_revenue@10"], 1), f2(abE["expected_revenue@10"], 1), `${sgn((abE["expected_revenue@10"] / base["expected_revenue@10"] - 1) * 100, 0)}%`],
      ["Expected margin / page", f2(base["expected_margin@10"], 1), f2(abE["expected_margin@10"], 1), `${sgn((abE["expected_margin@10"] / base["expected_margin@10"] - 1) * 100, 0)}%`],
      ["Unfulfillable in top-10", pct(base["oos_rate@10"], 1), pct(abE["oos_rate@10"], 1), `${sgn((abE["oos_rate@10"] - base["oos_rate@10"]) * 100, 1)} pts`]];
    table(s, rows, { x: MX, y: y0, w: 7.6, colW: [2.9, 1.7, 1.5, 1.5], fs: 12, rowH: 0.4 });
    card(s, 8.6, y0 - 0.05, 4.13, 2.2, "Also on the board", [`Cold-start recall@50: ${f2(hyb["cold_start_recall@50"], 3)}`, `pCTR AUC: ${f2(D.multitask.pCTR_auc, 3)}`, `p95 latency: ${f2(LAT.total_ms.p95, 0)} ms of 150`], { fs: 13 });
  }

  // ---------------------------------------------------------------- 18 uplift waterfall
  {
    const s = content("Stage-level uplift versus retrieval order", "Overall and uplift", "Uplift = (stage value / retrieval-order value) - 1, measured on the same 300 test lists.");
    const names = ["B + LTR", "C + Rules", "D + Margin", "E + MMR"], stg = [abA, abC, abD, abE];
    const up = (k) => stg.map((r) => (r[k] / base[k] - 1) * 100);
    barChart(s, [{ name: "Expected revenue", values: up("expected_revenue@10") }, { name: "Expected margin", values: up("expected_margin@10") }, { name: "NDCG@10", values: up("ndcg_rel@10") }],
      names, MX, 1.4, 7.6, 4.9, { title: "Cumulative uplift vs retrieval order (%)", fmt: "0", gap: 50, colors: [HEX.steel, HEX.teal, HEX.amber] });
    const inc = (k, i) => ((i === 0 ? stg[0][k] / base[k] : stg[i][k] / stg[i - 1][k]) - 1) * 100;
    const rows = [["Step", "Δ revenue", "Δ margin", "Δ NDCG"]];
    ["B + LTR ensemble", "C + hard rules", "D + margin re-rank", "E + MMR"].forEach((n, i) => rows.push([n, `${sgn(inc("expected_revenue@10", i), 1)}%`, `${sgn(inc("expected_margin@10", i), 1)}%`, `${sgn(inc("ndcg_rel@10", i), 1)}%`]));
    s.addText("Incremental effect of each step", { x: 8.5, y: 1.45, w: 4.2, h: 0.4, fontFace: "Cambria", fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
    table(s, rows, { x: 8.5, y: 1.9, w: 4.23, colW: [1.75, 0.85, 0.8, 0.83], fs: 11, rowH: 0.4 });
    para(s, 8.5, 4.2, 4.23, 2.1, ["Learning to rank carries relevance and a first margin gain", "Rules trade a little revenue for zero unfulfillable results", "Margin optimisation recovers and extends value"], 12.5);
    takeaway(s, "Uplift compounds: ranker first, margin second, rules and diversity protect the experience.");
  }

  // ---------------------------------------------------------------- 19 uplift flow
  {
    const s = content("Uplift end-to-end: from offline lift to online proof", "Overall and uplift", "High-level documentation of how uplift is defined and would be validated. Only the offline replay is implemented in this repo; shadow and A/B are the proposed path.");
    const steps = [
      ["1", "Define baseline", "Retrieval order (or the current production ranker) is the control.", HEX.gray],
      ["2", "Offline replay", "Score each stage on simulator truth: NDCG, E[revenue], E[margin], unfulfillable rate.", HEX.teal],
      ["3", "Per-flow cut", "Break uplift down by scenario so regressions in one flow are visible.", HEX.steel],
      ["4", "Shadow traffic", "Run the funnel on live queries without serving; compare to logged outcomes.", HEX.amber],
      ["5", "Online A/B", "Randomise users; read margin per session with guardrails.", HEX.coral],
    ];
    steps.forEach(([n, h, t, col], i) => {
      const x = MX + i * 2.46, w = 2.28;
      s.addShape(pres.shapes.RECTANGLE, { x, y: 1.5, w, h: 3.1, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, objectName: "step-card" });
      s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: 1.65, w: 0.5, h: 0.5, fill: { color: col }, line: { color: col, width: 0 }, objectName: "step-badge" });
      s.addText(n, { x: x + 0.2, y: 1.65, w: 0.5, h: 0.5, fontFace: "Cambria", fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(h, { x: x + 0.2, y: 2.25, w: w - 0.4, h: 0.45, fontFace: "Cambria", fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true, valign: "middle" });
      s.addText(t, { x: x + 0.2, y: 2.75, w: w - 0.4, h: 1.7, fontSize: 12.5, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
      if (i < 4) s.addText("›", { x: x + w - 0.04, y: 2.4, w: 0.32, h: 0.6, fontSize: 28, bold: true, color: C.accent5, align: "center", margin: 0, isTextBox: true });
    });
    card(s, MX, 4.85, 5.95, 1.65, "Definition", ["Uplift = metric(stage) / metric(baseline) − 1", "Primary: margin and revenue per search session", "Same candidates, same users, only the stage differs"], { fs: 12, hs: 15 });
    card(s, MX + 6.15, 4.85, 5.98, 1.65, "Status", ["Done: offline replay and per-flow cut (this deck)", "Not done: shadow traffic and A/B (needs real data)", "Offline lifts are optimistic: labels come from the simulator"], { fs: 12, hs: 15 });
  }

  // ---------------------------------------------------------------- 20 uplift guardrails
  {
    const s = content("Uplift decision framework and guardrails", "Overall and uplift", "Thresholds in the right-hand column are proposed starting points, not measured results.");
    const rows = [["Metric", "Role", "Offline result (full funnel vs retrieval order)", "Proposed online rule"],
      ["Margin per session", "Primary", `${sgn((abE["expected_margin@10"] / base["expected_margin@10"] - 1) * 100, 0)}%`, "Ship if lift is positive with 95% CI"],
      ["Revenue per session", "Primary", `${sgn((abE["expected_revenue@10"] / base["expected_revenue@10"] - 1) * 100, 0)}%`, "Must not fall; watch the rules step (−4%)"],
      ["NDCG@10 / click-through", "Health", `${sgn(abE["ndcg_rel@10"] - base["ndcg_rel@10"], 3)} NDCG`, "No drop beyond noise"],
      ["Unfulfillable results", "Guardrail", `${pct(base["oos_rate@10"], 1)} → ${pct(abE["oos_rate@10"], 1)}`, "Stay at 0 in served pages"],
      ["Return probability", "Guardrail", `${f2(abA["avg_return_prob@10"], 3)} → ${f2(abE["avg_return_prob@10"], 3)} (ranker → full)`, "Return rate must not rise"],
      ["Cold-start exposure", "Fairness", "see cold-start flow", "Share of cold SKUs in top-10 stable"],
      ["p95 latency", "Guardrail", `${f2(LAT.total_ms.p95, 0)} ms`, "Under 150 ms"]];
    table(s, rows, { x: MX, y: 1.45, w: W - 2 * MX, colW: [2.5, 1.2, 4.3, 4.13], fs: 12, rowH: 0.52 });
    takeaway(s, "Uplift is only claimed when primary metrics rise and every guardrail holds.", 6.2);
  }

  // ================================================================= SECTION: flow level
  divider("03", "Flow level metrics", "Seven shopper scenarios, 24 examples, every stage and every model", "Flow level");

  // ---------------------------------------------------------------- 21 flow overview table
  {
    const s = content("Flow-level metrics at a glance", "Flow level", "Each row is the mean over that flow's examples, scored against simulator ground truth.");
    const rows = [["Flow", "Ex.", "Facet acc.", "R@200 hybrid", "NDCG A → E", "Margin A → E", "Margin lift", "Unfulf. A → E", "Served ms"]];
    D.flows.forEach((f) => {
      const A = f.stages[ST[0]], E = f.stages[ST[4]];
      rows.push([f.title, String(f.n_examples), pct(f.facet_accuracy), f2(f.recall200.hybrid), `${f2(A["ndcg@10"], 2)} → ${f2(E["ndcg@10"], 2)}`, `${f2(A["exp_margin@10"], 0)} → ${f2(E["exp_margin@10"], 0)}`, `${sgn((E["exp_margin@10"] / A["exp_margin@10"] - 1) * 100, 0)}%`, `${f2(A["unfulfillable@10"], 1)} → ${f2(E["unfulfillable@10"], 1)}`, f2(f.latency_total, 0)]);
    });
    const A = D.stage_means_all[ST[0]], E = D.stage_means_all[ST[4]];
    rows.push([{ text: "All flows", options: { bold: true } }, String(D.all_examples), "", "", `${f2(A["ndcg@10"], 2)} → ${f2(E["ndcg@10"], 2)}`, `${f2(A["exp_margin@10"], 0)} → ${f2(E["exp_margin@10"], 0)}`, `${sgn((E["exp_margin@10"] / A["exp_margin@10"] - 1) * 100, 0)}%`, `${f2(A["unfulfillable@10"], 1)} → ${f2(E["unfulfillable@10"], 1)}`, ""]);
    table(s, rows, { x: MX, y: 1.45, w: W - 2 * MX, colW: [3.1, 0.55, 1.0, 1.25, 1.35, 1.35, 1.0, 1.4, 1.13], fs: 12, rowH: 0.47 });
    const lifts = D.flows.map((f) => [f.title.replace(/^\d · /, ""), f.stages[ST[4]]["exp_margin@10"] / f.stages[ST[0]]["exp_margin@10"] - 1]).sort((a, b) => b[1] - a[1]);
    takeaway(s, `Margin lifts in every flow: largest in ${lifts[0][0]} (${sgn(lifts[0][1] * 100, 0)}%), smallest in ${lifts[lifts.length - 1][0]} (${sgn(lifts[lifts.length - 1][1] * 100, 0)}%).`, 6.3);
  }

  // ---------------------------------------------------------------- 22 per-flow stage chart
  {
    const s = content("Stage-by-stage margin across all seven flows", "Flow level");
    barChart(s, ST.map((st) => ({ name: st.replace(/^[A-E] /, (m) => m), values: D.flows.map((f) => f.stages[st]["exp_margin@10"]) })), D.flows.map((f) => f.title.replace(/^\d · /, "")),
      MX, 1.4, 8.6, 4.9, { title: "Expected margin@10 by stage and flow", fmt: "0", gap: 40, colors: [HEX.gray, HEX.steel, HEX.amber, HEX.teal, HEX.coral], extra: { catAxisLabelFontSize: 9, dataLabelShow: false, showValue: false } });
    const rows = [["Stage", "NDCG", "Margin", "Unf."]];
    ST.forEach((st) => rows.push([st.replace(" (final page)", ""), f2(D.stage_means_all[st]["ndcg@10"], 3), f2(D.stage_means_all[st]["exp_margin@10"], 1), f2(D.stage_means_all[st]["unfulfillable@10"], 1)]));
    s.addText("Mean over all 24 examples", { x: 9.4, y: 1.45, w: 3.3, h: 0.4, fontFace: "Cambria", fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true });
    table(s, rows, { x: 9.4, y: 1.9, w: 3.33, colW: [1.5, 0.65, 0.65, 0.53], fs: 10.5, rowH: 0.45 });
    para(s, 9.4, 4.8, 3.33, 1.5, ["Margin rises at stage B in every flow", "Stage D raises margin again", "NDCG falls at stage B only for cold-start (flow 5)"], 12);
  }

  // ---------------------------------------------------------------- 23/24 model heat tables
  function heatSlide(title, sub, key, stageKey, fmt, notes) {
    const s = content(title, "Flow level", notes);
    const hdr = ["Model"].concat(D.flows.map((f) => f.title.replace(/^\d · /, "").replace(" (brand) intent", "").replace(" constraint", "").replace("Margin-aware re-ranking", "Margin-aware").replace("Regional stock & BOPIS", "Regional/BOPIS"))).concat(["Mean"]);
    const vals = MODELS.map((m) => D.flows.map((f) => f.models[m][stageKey][key]));
    const all = vals.flat(), lo = Math.min(...all), hi = Math.max(...all);
    const means = vals.map((v) => v.reduce((a, b) => a + b, 0) / v.length);
    const rows = [hdr];
    MODELS.forEach((m, i) => {
      const r = [MODEL_NAME[m]];
      vals[i].forEach((v, j) => {
        const colMax = Math.max(...MODELS.map((_, k) => vals[k][j]));
        r.push({ text: Number(v).toFixed(fmt), options: { fill: { color: heat(v, lo, hi) }, bold: v === colMax } });
      });
      r.push({ text: Number(means[i]).toFixed(fmt), options: { bold: true, fill: { color: heat(means[i], lo, hi) } } });
      rows.push(r);
    });
    table(s, rows, { x: MX, y: 1.4, w: W - 2 * MX, colW: [2.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.5, 1.23], fs: 11.5, rowH: 0.45 });
    s.addText(sub, { x: MX, y: 6.0, w: W - 2 * MX, h: 0.8, fontSize: 12.5, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
    return { s, vals, means };
  }
  heatSlide("Rankers across flows: NDCG@10, ranker only", "Same 200 candidates per example, ranker order only (stage B). Darker = higher; bold = best in the flow column. Cold-start is where most rankers lose to retrieval order.", "ndcg@10", "B", 3);
  heatSlide("Rankers across flows: expected margin, full funnel", "Each ranker feeds the full Stage-4 funnel (rules, margin step, MMR). Darker = higher; bold = best in the flow column. Margin differences between rankers shrink once re-ranking is applied.", "exp_margin@10", "E", 1);

  // ---------------------------------------------------------------- 25 model takeaways
  {
    const s = content("Which ranker wins where", "Flow level");
    const rows = [["Flow", "Best NDCG (ranker only)", "Best margin (full funnel)", "Ensemble margin rank"]];
    D.flows.forEach((f) => {
      const sortedN = MODELS.slice().sort((a, b) => f.models[b].B["ndcg@10"] - f.models[a].B["ndcg@10"]);
      const byN = sortedN[0], nTie = sortedN.filter((m) => Math.abs(f.models[m].B["ndcg@10"] - f.models[byN].B["ndcg@10"]) < 1e-9).length;
      const sortedM = MODELS.slice().sort((a, b) => f.models[b].E["exp_margin@10"] - f.models[a].E["exp_margin@10"]);
      rows.push([f.title, nTie > 1 ? `${nTie} models tie (${f2(f.models[byN].B["ndcg@10"], 3)})` : `${MODEL_NAME[byN]} (${f2(f.models[byN].B["ndcg@10"], 3)})`, `${MODEL_NAME[sortedM[0]]} (${f2(f.models[sortedM[0]].E["exp_margin@10"], 1)})`, `${sortedM.indexOf("ensemble") + 1} of 9`]);
    });
    table(s, rows, { x: MX, y: 1.4, w: 7.9, colW: [2.6, 2.1, 2.1, 1.1], fs: 11, rowH: 0.5, align: "left" });
    const mm = D.model_means_all;
    const bestN = MODELS.slice().sort((a, b) => mm[b].B["ndcg@10"] - mm[a].B["ndcg@10"])[0];
    const bestM = MODELS.slice().sort((a, b) => mm[b].E["exp_margin@10"] - mm[a].E["exp_margin@10"])[0];
    card(s, 8.8, 1.4, 3.93, 4.9, "Across all 24 examples", [`Best ranker-only NDCG: ${MODEL_NAME[bestN]} (${f2(mm[bestN].B["ndcg@10"], 3)})`, `Best full-funnel margin: ${MODEL_NAME[bestM]} (${f2(mm[bestM].E["exp_margin@10"], 1)})`, `Ensemble: NDCG ${f2(mm.ensemble.B["ndcg@10"], 3)}, margin ${f2(mm.ensemble.E["exp_margin@10"], 1)}`, "No model wins every flow: winners change with the scenario", "Small flows (3–4 examples) make per-flow winners noisy"], { fs: 12.5 });
    takeaway(s, "Keep the ensemble as default: it ranks 2nd–4th of 9 on margin in every flow and avoids betting on one paradigm.", 6.3);
  }

  // ================================================================= SECTION: stories
  divider("04", "Scenarios as stories", "Seven shopper scenarios told with the Streamlit End-to-End Flows tab: story slide, then stage and model metrics", "Stories");

  const STORY = {
    vocab_gap: ["A shopper types words the catalog has never seen.", "Spell fix and synonyms turn \"kicks\" into sneakers; dense retrieval finds trail shoes even when the title says Gore-Tex.", "Hybrid retrieval finds the relevant items; the ranker and margin step raise value, not relevance, since NDCG is already high."],
    typo_price: ["A shopper mistypes and sets a budget.", "Spell correction restores \"gore-tex sneakers\" and the $100 cap becomes a hard filter, so most of the 200 candidates are removed.", "This is where ranking matters most: NDCG roughly doubles from retrieval order to the final page."],
    navigational: ["A shopper names a brand.", "Brand match dominates the features and the brand cap is lifted, so a page of one brand is allowed.", "Relevance is saturated; the funnel's gain is in margin and removing unfulfillable items."],
    regional: ["A shopper picks up in store.", "Only items in stock at or near that store may appear; the ranker alone would show several unfulfillable results.", "Rules cost a little NDCG but remove every unfulfillable item."],
    cold_start: ["A brand-new SKU has no history.", "Content embeddings make it retrievable in the top ranks; the ranker, which leans on history, pushes it down; the margin step and MMR partly recover.", "The honest finding: cold-start gets found but under-ranked, so it needs an exploration bonus."],
    margin: ["The same relevant items, ordered by value.", "At margin weight 0.6 the optimiser promotes profitable relevant items without crossing the relevance floor.", "Margin per page rises sharply while NDCG stays near its ceiling."],
    personalization: ["Three shoppers, one query.", "User brand affinity and price distance reorder the same candidates; preferred-brand share shows the effect.", "Personalisation is a re-ordering of relevant items, so NDCG barely moves while per-user value does."],
  };
  const FLOW_ORDER = ["vocab_gap", "typo_price", "navigational", "regional", "cold_start", "margin", "personalization"];
  for (const fid of FLOW_ORDER) {
    const f = flowById[fid], ex = f.examples[0], v = ex.variants, A = v[ST[0]], B = v[ST[1]], E = v[ST[4]];
    const num = f.title.split(" · ")[0];
    // --- story slide
    {
      const s = content(`${f.title}`, "Stories", `Screenshot is a crop of the Streamlit End-to-End Flows tab (first example of this flow). Row detail: ${ex.label}.`);
      s.addText([{ text: `Story ${num}: `, options: { bold: true, color: C.accent1 } }, { text: `${STORY[fid][0]} `, options: { bold: true, color: C.text2 } },
        { text: `Query "${ex.query}" · user ${ex.user} · store ${ex.store}${ex.fulfillment !== "any" ? " · " + ex.fulfillment.toUpperCase() : ""}`, options: { color: C.text1 } }],
        { x: MX, y: 1.2, w: W - 2 * MX, h: 0.5, fontSize: 15, margin: 0, isTextBox: true, valign: "middle" });
      const pn = path.join(SHOTS, `${fid}_panels.png`), dp = await dims(pn);
      let iw = W - 2 * MX, ih = iw * dp.h / dp.w;
      if (ih > 4.55) { ih = 4.55; iw = ih * dp.w / dp.h; }
      s.addImage({ path: pn, x: MX + (W - 2 * MX - iw) / 2, y: 1.8, w: iw, h: ih, objectName: "streamlit-stage-panels" });
      s.addText(STORY[fid][1] + " " + STORY[fid][2], { x: MX, y: 1.8 + ih + 0.12, w: W - 2 * MX, h: 0.75, fontSize: 13, italic: true, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
    }
    // --- metrics slide
    {
      const s = content(`${f.title}: stage and model metrics`, "Stories");
      barChart(s, [{ name: "Expected margin@10", values: ST.map((st) => f.stages[st]["exp_margin@10"]) }], ["A Retr.", "B LTR", "C Rules", "D Margin", "E MMR"], MX, 1.3, 5.4, 2.15, { title: "Mean over examples: expected margin by stage", fmt: "0.0", colors: [HEX.teal], gap: 45 });
      const rows = [["Stage", "NDCG", "Prec.", "Revenue", "Margin", "Unf."]];
      ST.forEach((st) => { const r = f.stages[st]; rows.push([st.replace(" (final page)", "").replace(" & penalties", ""), f2(r["ndcg@10"], 3), f2(r["precision@10"], 2), f2(r["exp_revenue@10"], 0), f2(r["exp_margin@10"], 0), f2(r["unfulfillable@10"], 1)]); });
      table(s, rows, { x: MX, y: 3.55, w: 5.4, colW: [2.1, 0.65, 0.6, 0.7, 0.7, 0.65], fs: 10, rowH: 0.27 });
      const ex_rows = [["Example", "NDCG A→E", "Margin A→E"]];
      f.examples.forEach((e) => ex_rows.push([e.label.length > 34 ? e.label.slice(0, 33) + "…" : e.label, `${f2(e.variants[ST[0]]["ndcg@10"], 2)} → ${f2(e.variants[ST[4]]["ndcg@10"], 2)}`, `${f2(e.variants[ST[0]]["exp_margin@10"], 0)} → ${f2(e.variants[ST[4]]["exp_margin@10"], 0)}`]));
      table(s, ex_rows, { x: MX, y: 5.3, w: 5.4, colW: [2.9, 1.25, 1.25], fs: 10, rowH: 0.27, align: "center" });
      // models table
      const mrows = [["Model", "NDCG (B)", "Margin (B)", "NDCG (E)", "Margin (E)"]];
      const bestB = Math.max(...MODELS.map((m) => f.models[m].B["ndcg@10"])), bestE = Math.max(...MODELS.map((m) => f.models[m].E["exp_margin@10"]));
      MODELS.forEach((m) => {
        const mb = f.models[m].B, me = f.models[m].E;
        mrows.push([MODEL_NAME[m], { text: f2(mb["ndcg@10"], 3), options: { bold: mb["ndcg@10"] === bestB } }, f2(mb["exp_margin@10"], 1), f2(me["ndcg@10"], 3), { text: f2(me["exp_margin@10"], 1), options: { bold: me["exp_margin@10"] === bestE } }]);
      });
      s.addText("All rankers on this flow (B = ranker only, E = full funnel)", { x: 6.3, y: 1.35, w: 6.4, h: 0.4, fontFace: "Cambria", fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true });
      table(s, mrows, { x: 6.3, y: 1.8, w: 6.43, colW: [2.2, 1.0, 1.1, 1.0, 1.13], fs: 11.5, rowH: 0.42 });
      const A = f.stages[ST[0]], E = f.stages[ST[4]];
      s.addText(`Flow summary: facet accuracy ${pct(f.facet_accuracy)}, hybrid recall@200 ${f2(f.recall200.hybrid)} (sparse ${f2(f.recall200.sparse)}, dense ${f2(f.recall200.dense)}), margin ${f2(A["exp_margin@10"], 0)} → ${f2(E["exp_margin@10"], 0)} (${sgn((E["exp_margin@10"] / A["exp_margin@10"] - 1) * 100, 0)}%).`,
        { x: 6.3, y: 6.15, w: 6.43, h: 0.8, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
    }
  }

  // ================================================================= closing
  {
    const s = content("What to trust, and what not to", "Wrap-up");
    card(s, MX, 1.45, 5.95, 4.9, "Strengths", ["Whole funnel measured per stage, per flow, per model", "Hybrid retrieval, multi-task value scoring, rules and margin in one served path", "Latency well inside the budget on one GPU", "Every number on these slides is generated from repo artifacts"], { fs: 14 });
    card(s, MX + 6.15, 1.45, 5.98, 4.9, "Limitations", ["Synthetic data: absolute metrics are optimistic; compare stages, not to production", "Cold-start items are found but under-ranked by the ranker", "Intent confidence is low on generic category queries", "Model differences are within noise (one seed, small flows)", "Image vectors are synthetic; FAISS runs on CPU on Windows", "Uplift is offline only: no shadow or A/B run"], { fs: 14, hc: C.accent3 });
  }
  {
    const s = content("Next steps", "Wrap-up");
    const steps = [[ic.flask, HEX.teal, "Prove uplift online", "Shadow traffic, then an A/B test with the guardrails on the decision-framework slide."], [ic.seed, HEX.plum, "Fix cold-start ranking", "Exploration bonus or content-based prior so new SKUs are not pushed down."], [ic.brain, HEX.steel, "Better multi-task model", "Shared-bottom or ESMM-style network; calibrate heads."], [ic.bolt, HEX.amber, "Productionise", "GPU FAISS on Linux, request batching, feature store, monitoring."], [ic.db, HEX.coral, "Real data", "Replace the simulator with logged queries, clicks, inventory and returns."]];
    steps.forEach(([img, col, h, t], i) => {
      const y = 1.45 + i * 0.98;
      iconDot(s, img, MX, y, col, 0.62);
      s.addText(h, { x: MX + 0.85, y, w: 3.6, h: 0.62, fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true, valign: "middle" });
      s.addText(t, { x: MX + 4.6, y, w: 7.5, h: 0.62, fontSize: 15, color: C.text1, margin: 0, isTextBox: true, valign: "middle" });
    });
  }
  {
    const s = content("Appendix: run it yourself", "Wrap-up");
    s.addText([["python scripts/run_pipeline.py --regen", "data, index, train, evaluate"], ["python scripts/run_flows.py", "per-stage metrics per flow"], ["streamlit run src/ui/app.py", "End-to-End Flows tab"], ["python presentation/prepare_deck_data.py", "collect deck numbers"], ["python presentation/capture_story_shots.py", "Streamlit crops"], ["node presentation/build_deck.js", "build this deck"]]
      .map(([c, d], i, arr) => ({ text: c + "   # " + d, options: { breakLine: i < arr.length - 1 } })),
      { x: MX, y: 1.45, w: 7.7, h: 2.6, fontFace: "Courier New", fontSize: 11, color: C.background1, fill: { color: C.accent4 }, margin: 12, isTextBox: true, valign: "top", paraSpaceAfter: 4 });
    card(s, 8.5, 1.45, 4.23, 2.6, "Stack", ["torch, sentence-transformers, FAISS", "XGBoost, LightGBM, CatBoost", "FastAPI, Streamlit, Plotly", "Playwright for screenshots"], { fs: 13 });
    card(s, MX, 4.3, W - 2 * MX, 2.0, "Repo map", ["src/data_gen, query_understanding, retrieval, ranking, reranking, evaluation: one package per stage", "src/search_service.py: served funnel; src/flows.py: per-stage scoring of scenario examples", "outputs/metrics: every JSON/CSV behind these slides; outputs/screenshots: full-page UI shots"], { fs: 13 });
  }

  await pres.writeFile({ fileName: OUTFILE });
  await applyTheme(OUTFILE, THEME);
  console.log("wrote", OUTFILE);
})();
