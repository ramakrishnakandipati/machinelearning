// Builds Discovery_Engine_Flow_and_Metrics.pptx from the metrics in ../retail-discovery-bandit/output/metrics
// Run:  NODE_PATH=<dir with node_modules/pptxgenjs> node build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.APPLY_THEME || "./apply_theme.js");

const M = path.join(__dirname, "..", "retail-discovery-bandit", "output", "metrics");
const J = (f) => JSON.parse(fs.readFileSync(path.join(M, f), "utf8"));
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const SUM = J("stage4_policy_summary.json");
const OPE = J("ope_results.json").all_slots;
const FULL = J("full_metrics_report.json");
const GPU = J("gpu_utilization.json");
const COLD = J("cold_start_bottleneck.json");

const P = ["pure_ranker", "epsilon_greedy", "linucb_mmr", "linucb_high_expl"];
const L = ["Pure ranker", "Epsilon-greedy", "LinUCB + MMR", "High-explore LinUCB"];
const f = (x, d = 3) => Number(x).toFixed(d);
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";

const THEME = {
  name: "Discovery Engine",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A2F", lt1: "FFFFFF", dk2: "0B3D3A", lt2: "EEF4F2",
    accent1: "2A9D8F", accent2: "E4572E", accent3: "F4A259", accent4: "6B7A8F", accent5: "264653", accent6: "8AB17D",
    hlink: "2A9D8F", folHlink: "6B7A8F",
  },
};
const HEX = THEME.colors;
const POL_HEX = [HEX.accent4, HEX.accent3, HEX.accent1, HEX.accent2];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Discovery Engine: flow, mechanics and discovery metrics";
pres.author = "Retail Discovery Bandit project";
const C = pres.SchemeColor;
const W = 13.33, MX = 0.6, CW = W - 2 * MX;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: C.text2 },
  objects: [],
  slideNumber: undefined,
});
pres.defineSlideMaster({
  title: "SECTION_DARK", background: { color: C.text2 },
  objects: [],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { text: { text: "Retail Discovery Engine  |  contextual-bandit re-ranking", options: { x: MX, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.35, w: CW, h: 0.8, fontSize: 30, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - MX - 0.6, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
});

// ---------------------------------------------------------------- helpers
let curSection = "Introduction";
function content(title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: curSection });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function txt(s, text, o) { s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontFace: THEME.bodyFontFace, color: C.text1, valign: "top" }, o)); }
function bullets(s, items, o) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 16 }, breakLine: i < items.length - 1, paraSpaceAfter: 7 } }));
  s.addText(runs, Object.assign({ isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: "top", fontFace: THEME.bodyFontFace }, o));
}
function card(s, x, y, w, h, fill, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 }, line: { type: "none" }, objectName: name || "card" });
}
function stat(s, x, y, w, big, label, color, size) {
  card(s, x, y, w, 1.55, C.background2, "stat card");
  txt(s, big, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: 0.8, fontSize: size || 40, bold: true, color: color || C.accent1, fontFace: THEME.headFontFace, valign: "middle" });
  txt(s, label, { x: x + 0.2, y: y + 0.95, w: w - 0.4, h: 0.5, fontSize: 13, color: C.text1 });
}
function arrow(s, x1, y1, x2, y2, color) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1,
    line: { color: color || C.accent4, width: 2, endArrowType: "triangle" }, objectName: "arrow" });
}
function flowBox(s, x, y, w, h, head, sub, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: o.fill || C.background2 }, line: { type: "none" }, objectName: "flow box" });
  txt(s, head, { x: x + 0.15, y: y + 0.12, w: w - 0.3, h: 0.5, fontSize: o.headSize || 14, bold: true, color: o.headColor || C.text2, fontFace: THEME.headFontFace });
  txt(s, sub, { x: x + 0.15, y: y + 0.62, w: w - 0.3, h: h - 0.7, fontSize: o.subSize || 13, color: o.subColor || C.text1 });
}
function badge(s, x, y, n, fill) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color: fill || C.accent2 }, line: { type: "none" }, objectName: "step badge" });
  txt(s, String(n), { x, y, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
}
const CHART_BASE = () => ({
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent4, dataLabelColor: HEX.dk1,
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 11, legendFontSize: 12, titleFontSize: 14, titleColor: HEX.dk1,
  valGridLine: { color: "D9E2E0", size: 0.75 }, catGridLine: { style: "none" },
});
function section(title, sub, notes) {
  curSection = title;
  pres.addSection({ title });
  const s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: title });
  txt(s, title, { x: 0.9, y: 2.6, w: 11.5, h: 1.2, fontSize: 44, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "middle" });
  txt(s, sub, { x: 0.9, y: 3.95, w: 11, h: 0.9, fontSize: 20, color: C.accent3 });
  if (notes) s.addNotes(notes);
}

// ================================================================ 1 TITLE
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  txt(s, "Discovery Engine", { x: 0.9, y: 1.7, w: 11.5, h: 1.3, fontSize: 56, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "middle" });
  txt(s, "From relevance to exploration: flow, mechanics and discovery metrics", { x: 0.9, y: 3.1, w: 11.5, h: 0.8, fontSize: 26, color: C.accent3 });
  txt(s, "Contextual-bandit re-ranking for omnichannel grocery and retail product discovery", { x: 0.9, y: 4.1, w: 11.5, h: 0.5, fontSize: 18, color: C.background2 });
  txt(s, "GPU pipeline  |  10,000 SKUs  |  2,000 users  |  4 policies  |  3,000 simulated requests each", { x: 0.9, y: 6.3, w: 11.5, h: 0.4, fontSize: 14, color: C.accent6 });
  s.addNotes("This deck explains how the discovery engine fits into the retrieval-ranking pipeline, how each of its five steps works, and how discovery should be measured. All numbers come from the shipped run (seed 42) on a synthetic retail world with a hidden ground truth, so they show mechanism and relative effects rather than production lift.");
}

// ================================================================ 2 EXEC SUMMARY
{
  const s = content("Executive summary: discovery at a small relevance cost",
    "Four headline numbers. The cold-start gap motivates the work; the three others show what the engine delivers. Reward retention compares average reward per impression of LinUCB+MMR with the pure ranker (0.0855 vs 0.0885).");
  const rr = SUM.linucb_mmr.avg_reward_per_impression / SUM.pure_ranker.avg_reward_per_impression;
  const cs = SUM.linucb_mmr.cold_start_impression_share / SUM.pure_ranker.cold_start_impression_share;
  const bw = CW / 4 - 0.15;
  stat(s, MX, 1.5, bw, pct(-COLD.mean_relative_gap_matched, 0), "lower score for new SKUs at equal true relevance", C.accent2);
  stat(s, MX + (bw + 0.2), 1.5, bw, pct(rr, 0), "of the pure ranker's reward kept by LinUCB + MMR", C.accent1);
  stat(s, MX + 2 * (bw + 0.2), 1.5, bw, f(cs, 0) + "x", "more cold-start impressions (0.14% to 1.7%)", C.accent1);
  stat(s, MX + 3 * (bw + 0.2), 1.5, bw, pct(SUM.linucb_mmr.discovery_purchase_share, 0), "of purchases from discovery items (pure ranker: " + pct(SUM.pure_ranker.discovery_purchase_share, 0) + ")", C.accent1);
  bullets(s, [
    "A relevance-only ranker starves new products: no history, low score, no exposure, still no history.",
    "The discovery engine adds exploration in five steps: utility, guardrails, LinUCB bonus, MMR slate, propensity logging.",
    "Discovery is judged on its own metrics: cold-start share and conversion, discovery conversion, breadth, diversity, serendipity.",
    "Off-policy evaluation recovers the simulator's truth within 1.6%, so discovery policies can be compared before an A/B test.",
    "Honest limits: weak ranker (AUC 0.63), exposure still concentrated (Gini 0.99), synthetic data.",
  ], { x: MX, y: 3.35, w: CW, h: 3.5, fontSize: 16 });
}

// ================================================================ 3 PROBLEM
{
  const s = content("The problem: relevance rankers starve new products",
    "On the right: for each quintile of true click relevance, the ranker's predicted P(click) for warm items follows relevance, while for cold-start items it is flat and lower. That is the bottleneck the discovery engine is designed to correct.");
  txt(s, "The cold-start feedback loop", { x: MX, y: 1.45, w: 5.2, h: 0.4, fontSize: 20, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  bullets(s, [
    "History features (impression counts, smoothed CTR and CVR, buys) are empty for new SKUs",
    "The ranker therefore scores them systematically lower",
    "Lower scores mean no exposure, so no clicks, so no history",
    "Result: " + f(COLD.mean_pred_click_cold, 3) + " mean predicted P(click) for cold vs " + f(COLD.mean_pred_click_warm, 3) + " for warm SKUs",
    "At equal true relevance the gap averages " + pct(COLD.mean_relative_gap_matched, 0),
  ], { x: MX, y: 1.95, w: 5.2, h: 4.6, fontSize: 15 });
  const bins = D.cold_bins;
  const labels = bins.map((b, i) => "Q" + (i + 1));
  s.addChart(pres.charts.BAR, [
    { name: "Predicted, warm SKUs", labels, values: bins.map((b) => +b.pred_warm.toFixed(4)) },
    { name: "Predicted, cold-start SKUs", labels, values: bins.map((b) => +b.pred_cold.toFixed(4)) },
    { name: "True P(click)", labels, values: bins.map((b) => +b.true_cold.toFixed(4)) },
  ], Object.assign(CHART_BASE(), {
    x: 6.2, y: 1.4, w: 6.5, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent1, HEX.accent2, "B8C4C2"],
    showTitle: true, title: "Predicted vs true P(click) by true-relevance quintile", showLegend: true, legendPos: "b",
    showValue: false, valAxisLabelFormatCode: "0.00", catAxisTitle: "True relevance quintile (Q5 = most relevant)", showCatAxisTitle: true, catAxisTitleFontSize: 11, catAxisTitleColor: HEX.accent4,
  }));
}

// ================================================================ 4 FLOW
{
  const s = content("End-to-end flow: four stages, one feedback loop",
    "Stages 1-3 are conventional retrieve-filter-rank. Stage 4, the discovery engine, is where exploration, diversity and business rules are applied and where propensities are logged. Outcomes flow back as rank-1 bandit updates, and the propensity log feeds off-policy evaluation.");
  const bw = 2.15, gap = 0.34, y = 1.5, h = 2.0;
  const boxes = [
    ["1  Retrieval", "Two-tower model, GPU matmul + top-k. 200 candidates per user", C.background2, false],
    ["2  Filtering", "Drop out-of-stock and items bought in the last 14 days", C.background2, false],
    ["3  Primary ranker", "XGBoost on GPU: calibrated P(click), P(buy)", C.background2, false],
    ["4  Discovery engine", "Utility, guardrails, LinUCB bonus, MMR slate, propensities", C.accent2, true],
    ["Slate of 5", "Score decomposition and propensity per item, served to the shopper", C.background2, false],
  ];
  boxes.forEach((b, i) => {
    const x = MX + i * (bw + gap);
    flowBox(s, x, y, bw, h, b[0], b[1], b[3] ? { fill: b[2], headColor: C.background1, subColor: C.background1 } : { fill: b[2] });
    if (i < 4) arrow(s, x + bw + 0.02, y + h / 2, x + bw + gap - 0.02, y + h / 2);
  });
  const facts = ["Recall@200 0.185, p99 0.17 ms", "2.4M retrieved, 4.9% out of stock", "Click AUC 0.63, isotonic-calibrated", "192 candidates to 65 to 5 (example)", "ILD 0.97, 0 cap violations"];
  facts.forEach((t, i) => txt(s, t, { x: MX + i * (bw + gap), y: y + h + 0.12, w: bw, h: 0.6, fontSize: 12, color: C.accent4 }));
  // feedback lane
  const ly = 4.85;
  flowBox(s, MX + 0.3, ly, 5.6, 1.45, "Feedback loop (online)", "Click / buy reward updates A^-1 and b by Sherman-Morrison rank-1; user novelty and fatigue state refresh immediately", { headSize: 15 });
  flowBox(s, MX + 6.5, ly, 5.6, 1.45, "Logging and evaluation", "(user, item, slot, propensity, context, reward) to JSONL; IPS, SNIPS, DR, Switch-DR score any policy offline", { headSize: 15 });
  arrow(s, MX + 3.1, ly, MX + 3.1, y + h + 0.75);
  arrow(s, MX + 9.3, y + h + 0.75, MX + 9.3, ly);
  txt(s, "reward", { x: MX + 3.2, y: ly - 0.5, w: 1.0, h: 0.3, fontSize: 11, color: C.accent4 });
  txt(s, "propensities", { x: MX + 9.4, y: ly - 0.5, w: 1.5, h: 0.3, fontSize: 11, color: C.accent4 });
}

// ================================================================ 5 STAGES 1-2
{
  const s = content("Stages 1 and 2: retrieve broadly, filter hard",
    "Retrieval is a single GPU matrix multiply and top-k, so latency is a non-issue. Item-ID dropout during training forces the item tower to produce usable content-only embeddings, which is how cold-start items can be retrieved at all. Cold-start recall is still weak; the ceiling has not been measured yet.");
  txt(s, "Two-tower retrieval", { x: MX, y: 1.45, w: 5.8, h: 0.4, fontSize: 20, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  bullets(s, [
    "User tower: ID, persona, price level and click-history content vector",
    "Item tower: ID, subcategory, category, price, margin and the semantic embedding",
    "In-batch negatives with logQ popularity correction",
    "40% item-ID dropout so new SKUs get content-only embeddings",
    "Top-200 by cosine similarity in one CUDA matmul",
  ], { x: MX, y: 1.95, w: 5.8, h: 3.0, fontSize: 15 });
  txt(s, "Hard filters (Stage 2), per 200 retrieved", { x: MX, y: 5.0, w: 5.8, h: 0.4, fontSize: 18, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  const reqs = 2400000 / 200;
  const fd = J("filter_and_guardrail_drop_counts.json").filter_drop_counts_all_requests;
  bullets(s, [
    "Out of stock: " + f(fd.out_of_stock / reqs, 1) + " dropped (" + pct(fd.out_of_stock / fd.retrieved) + ")",
    "Bought in last 14 days: " + f(fd.recent_purchase / reqs, 2) + " dropped (" + pct(fd.recent_purchase / fd.retrieved, 2) + ")",
  ], { x: MX, y: 5.45, w: 5.8, h: 1.4, fontSize: 15 });
  const r = FULL.stage1_retrieval;
  const sw = 2.9;
  stat(s, 6.9, 1.5, sw, f(r["Recall@200"], 3), "Recall@200 (warm test positives)");
  stat(s, 6.9 + sw + 0.2, 1.5, sw, f(r["HitRate@200"], 3), "HitRate@200");
  stat(s, 6.9, 3.25, sw, f(r["ColdStart_Recall@200"], 3), "ColdStart_Recall@200", C.accent2);
  stat(s, 6.9 + sw + 0.2, 3.25, sw, f(r.latency_ms.p99_ms, 2) + " ms", "p99 retrieval latency (target < 10 ms)");
  txt(s, "Recall is bounded by the data: ranking by the true hidden affinity reaches only about 0.28 Recall@200 because impressions are popularity-biased and noisy.", { x: 6.9, y: 5.2, w: 5.9, h: 1.2, fontSize: 14, color: C.accent4 });
}

// ================================================================ 6 STAGE 3
{
  const s = content("Stage 3: calibrated probabilities, blind to new items",
    "Calibration matters because the utility multiplies P(buy) by margin and the propensities are computed from scores. The ranker is modest (AUC 0.63) because the latent affinity is only partly visible through features; that is a known weak spot, documented in the README.");
  const rc = FULL.stage3_ranker.click, rb = FULL.stage3_ranker.buy, rk = FULL.stage3_ranker.ranking;
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, align: "center", fontSize: 13 } });
  const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 13, align: "center" }, o) });
  s.addTable([
    [hdr("Metric"), hdr("P(click)"), hdr("P(buy)")],
    [cell("ROC-AUC", { align: "left" }), cell(f(rc.roc_auc)), cell(f(rb.roc_auc))],
    [cell("PR-AUC", { align: "left" }), cell(f(rc.pr_auc)), cell(f(rb.pr_auc))],
    [cell("LogLoss", { align: "left" }), cell(f(rc.logloss)), cell(f(rb.logloss))],
    [cell("Brier score", { align: "left" }), cell(f(rc.brier)), cell(f(rb.brier))],
    [cell("Mean prediction vs base rate", { align: "left" }), cell(f(rc.mean_pred) + " vs " + f(rc.base_rate)), cell(f(rb.mean_pred) + " vs " + f(rb.base_rate))],
    [cell("NDCG@5 / NDCG@10", { align: "left" }), { text: f(rk["ndcg@5"]) + " / " + f(rk["ndcg@10"]), options: { fontSize: 13, align: "center", colspan: 2 } }],
  ], { x: MX, y: 1.5, w: 6.4, colW: [2.8, 1.8, 1.8], rowH: 0.45, border: { type: "solid", color: "D9E2E0", pt: 0.75 }, fontFace: THEME.bodyFontFace, color: HEX.dk1 });
  txt(s, "Why new SKUs lose", { x: 7.5, y: 1.45, w: 5.2, h: 0.4, fontSize: 20, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  bullets(s, [
    "21 user, item and cross features, including the two-tower similarity",
    "Four item-history features are zero or prior-only for cold SKUs",
    "Cold SKUs reach a mean score of " + f(COLD.mean_pred_click_cold, 3) + " vs " + f(COLD.mean_pred_click_warm, 3) + " for warm",
    "Cold predictions barely move with true relevance, so the ranker cannot tell good new items from bad",
    "Cold-start P(buy) is over-predicted " + f(COLD.calibration_ratio_buy_cold, 1) + "x on average, warm is well calibrated (" + f(COLD.calibration_ratio_buy_warm, 2) + "x)",
  ], { x: 7.5, y: 1.95, w: 5.2, h: 4.8, fontSize: 15 });
  txt(s, "Warm test window, 33,372 impressions. Calibration: isotonic on a held-out window.", { x: MX, y: 5.0, w: 6.4, h: 0.6, fontSize: 12, color: C.accent4 });
}

// ================================================================ 7 SECTION
section("Inside the Discovery Engine", "Five steps turn a ranked candidate list into an explorable, diverse, auditable slate",
  "The next six slides walk through one request, step by step, in the order the code executes.");

// ================================================================ 8 FIVE STEPS
{
  const s = content("One request, five steps",
    "The order matters. Guardrails run before scoring so that the Boltzmann denominator only contains eligible items. The subcategory cap is enforced inside the greedy MMR loop. Counts in the strip are from a real request for user 7.");
  const bw = 2.2, gap = 0.275, y = 1.5, h = 3.1;
  const steps = [
    ["Utility", "U = w1 p(click) + w2 p(buy) margin + w3 novelty", "utility.py"],
    ["Guardrails", "Drop fatigued items; keep explored items inside the +-30% price band", "guardrails.py"],
    ["LinUCB bonus", "Score = U + alpha sqrt(x' A^-1 x), per-category groups", "bandit.py"],
    ["MMR slate", "Greedy pick with a similarity penalty; max 2 per subcategory", "diversity_mmr.py"],
    ["Propensity", "P(i|u) = softmax(Score / tau), logged per slot", "pipeline.py"],
  ];
  steps.forEach((st, i) => {
    const x = MX + i * (bw + gap);
    card(s, x, y, bw, h, C.background2, "step card");
    badge(s, x + 0.15, y + 0.18, i + 1);
    txt(s, st[0], { x: x + 0.75, y: y + 0.18, w: bw - 0.85, h: 0.5, fontSize: 15, bold: true, color: C.text2, fontFace: THEME.headFontFace, valign: "middle" });
    txt(s, st[1], { x: x + 0.18, y: y + 0.9, w: bw - 0.36, h: 1.7, fontSize: 14 });
    txt(s, st[2], { x: x + 0.18, y: y + h - 0.45, w: bw - 0.36, h: 0.3, fontSize: 11, color: C.accent4, fontFace: "Courier New" });
    if (i < 4) arrow(s, x + bw + 0.02, y + h / 2, x + bw + gap - 0.02, y + h / 2);
  });
  const ex = D.example;
  const fd = ex.filter_drop_counts, gd = ex.guardrails;
  const fy = 5.0;
  const chips = [[String(fd.retrieved), "retrieved"], [String(fd.kept), "after stock and 14-day filters"], [String(gd.n_out), "after guardrails"], ["5", "slate slots"]];
  chips.forEach((c, i) => {
    const x = MX + i * 3.05;
    txt(s, c[0], { x, y: fy, w: 1.1, h: 0.9, fontSize: 40, bold: true, color: i === 3 ? C.accent2 : C.accent1, fontFace: THEME.headFontFace, valign: "middle" });
    txt(s, c[1], { x: x + 1.15, y: fy + 0.05, w: 1.7, h: 0.8, fontSize: 13, valign: "middle" });
  });
  txt(s, "Candidate funnel for one real request (user " + ex.user_id + ").", { x: MX, y: 6.1, w: CW, h: 0.4, fontSize: 12, color: C.accent4 });
}

// ================================================================ 9 UTILITY
{
  const s = content("Step 1: multi-objective utility with a novelty term",
    "Novelty is computed from the user's live state: interacted subcategories and brands, and bought items. The fourth tier, new SKU, is an extension to the three tiers in the brief. The novelty weight was lowered from 0.10 to 0.04 because at 0.10 it overwhelmed relevance and cut reward sharply.");
  card(s, MX, 1.45, CW, 1.0, C.accent5, "formula band");
  txt(s, "U(u, i) = w1 * p(click) + w2 * p(buy) * margin + w3 * novelty(u, i)", { x: MX + 0.3, y: 1.45, w: CW - 0.6, h: 1.0, fontSize: 24, bold: true, color: C.background1, fontFace: "Courier New", valign: "middle" });
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 13 } });
  const c = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 13 }, o) });
  s.addTable([
    [hdr("Novelty tier"), hdr("Rule (live user state)"), hdr("Value")],
    [c("Unseen subcategory"), c("User never clicked or bought in that subcategory"), c("1.00", { bold: true })],
    [c("Unseen brand"), c("Known subcategory, new brand"), c("0.50", { bold: true })],
    [c("New SKU"), c("Known subcategory and brand, not bought (extension)"), c("0.25", { bold: true })],
    [c("Previously bought"), c("User has bought this SKU before"), c("0.00", { bold: true })],
  ], { x: MX, y: 2.8, w: 7.4, colW: [2.0, 4.4, 1.0], rowH: 0.5, border: { type: "solid", color: "D9E2E0", pt: 0.75 }, fontFace: THEME.bodyFontFace, color: HEX.dk1 });
  txt(s, "Weights in this run", { x: 8.5, y: 2.75, w: 4.2, h: 0.4, fontSize: 18, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  bullets(s, [
    "w1 = 1.0 (click), w2 = 2.0 (buy x margin)",
    "w3 = 0.04 (novelty); the pure ranker and epsilon-greedy use w3 = 0",
    "Reward for learning: 0.3 per click, 1.0 per purchase",
  ], { x: 8.5, y: 3.2, w: 4.2, h: 2.6, fontSize: 15 });
  txt(s, "Tuning lesson: with w3 = 0.10 a novelty tier of 1.0 outweighed typical click-probability differences and reward fell about 25%.", { x: MX, y: 5.6, w: 7.4, h: 0.9, fontSize: 14, color: C.accent4 });
}

// ================================================================ 10 GUARDRAILS
{
  const s = content("Step 2: guardrails shape the candidate pool",
    "The price band is by far the most active rule. About 132 of roughly 190 candidates per request are removed by it, because every cold-start or unseen-subcategory item outside +-30% of the user's average price is dropped. Fatigue and cap rules almost never fire in this simulation. The price band deserves tuning: it is the main reason the candidate pool is small.");
  const gp = D.guard_per_req.linucb_mmr;
  const cw = (CW - 0.4) / 3;
  const rules = [
    ["Impression fatigue", "Exclude items with more than 3 unclicked impressions in 7 days", f(gp.fatigue, 2) + " dropped per request"],
    ["Price band on explored items", "Cold-start or unseen items must be within +-30% of the user's average item price", f(gp.price_band, 0) + " dropped per request"],
    ["Subcategory cap", "At most 2 items per subcategory, enforced inside the MMR loop", "0 violations in 60,000 impressions"],
  ];
  rules.forEach((r, i) => {
    const x = MX + i * (cw + 0.2);
    card(s, x, 1.5, cw, 2.6, C.background2, "rule card");
    txt(s, r[0], { x: x + 0.2, y: 1.65, w: cw - 0.4, h: 0.5, fontSize: 17, bold: true, color: C.text2, fontFace: THEME.headFontFace });
    txt(s, r[1], { x: x + 0.2, y: 2.2, w: cw - 0.4, h: 1.1, fontSize: 14 });
    txt(s, r[2], { x: x + 0.2, y: 3.4, w: cw - 0.4, h: 0.5, fontSize: 15, bold: true, color: i === 1 ? C.accent2 : C.accent1 });
  });
  s.addChart(pres.charts.BAR, [{ name: "Items per request", labels: ["Retrieved", "After stock and recency filters", "After guardrails (mean)", "Slate"], values: [200, 190, +(190 - gp.price_band - gp.fatigue).toFixed(0), 5] }],
    Object.assign(CHART_BASE(), { x: MX, y: 4.3, w: 7.2, h: 2.65, barDir: "bar", chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, catAxisOrientation: "maxMin" }));
  txt(s, "Fallbacks", { x: 8.2, y: 4.35, w: 4.5, h: 0.4, fontSize: 18, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  bullets(s, [
    "If fewer than 12 candidates survive, relax the price band, then fatigue",
    "Fallbacks fired 0 times in this run",
    "Every drop is logged per request (guardrail_log.jsonl)",
  ], { x: 8.2, y: 4.8, w: 4.5, h: 2.1, fontSize: 14 });
}

// ================================================================ 11 BANDIT
{
  const s = content("Step 3: hybrid LinUCB adds a decaying exploration bonus",
    "Arms are grouped by category so a new SKU inherits statistics from its group and from shared hybrid parameters. Sherman-Morrison keeps A inverse current; an exact solve every 200 updates bounds drift, and the measured maximum inverse error is about 5e-14 after 35,000 updates per bandit. float64 is required: the hybrid Schur-complement updates were ill-conditioned in float32. A learned mean (theta'x) was tested and hurt reward, so the default is the spec-literal score.");
  card(s, MX, 1.45, CW, 0.9, C.accent5, "formula band");
  txt(s, "Score_bandit(u, i) = U(u, i) + alpha * sqrt( x' A^-1 x )", { x: MX + 0.3, y: 1.45, w: CW - 0.6, h: 0.9, fontSize: 24, bold: true, color: C.background1, fontFace: "Courier New", valign: "middle" });
  bullets(s, [
    "Context x: p(click), p(buy), margin, novelty, price fit, cold flag, launch age, similarity, persona, embedding interaction (26 dims)",
    "15 category groups plus 6 shared features (hybrid LinUCB)",
    "Online update on GPU: rank-1 Sherman-Morrison, exact solve every 200 updates",
    "Warm start: 20,000 logged impressions; cold-start contexts have none, so their bonus stays high",
    "alpha 0.35 for LinUCB + MMR, 1.2 for high exploration",
    "Max |A A^-1 - I| = 5e-14 after 35,000 updates",
  ], { x: MX, y: 2.6, w: 6.3, h: 4.3, fontSize: 14 });
  const base = P.map((p) => 0), util = [0.2407, 0.2193, 0.2559, 0.2463], bonus = [0, 0, 0.0369, 0.1343];
  s.addChart(pres.charts.BAR, [
    { name: "Mean base utility", labels: L, values: util },
    { name: "Mean exploration bonus", labels: L, values: bonus },
  ], Object.assign(CHART_BASE(), { x: 7.2, y: 2.5, w: 5.5, h: 4.3, barDir: "col", barGrouping: "stacked", chartColors: [HEX.accent4, HEX.accent2], showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: "0.000", dataLabelColor: "FFFFFF", showLegend: true, legendPos: "b", showTitle: true, title: "Score components of displayed items", valAxisLabelFormatCode: "0.00" }));
}

// ================================================================ 12 MMR
{
  const s = content("Step 4: MMR assembles a diverse slate",
    "The similarity blends embedding cosine (0.6) with a same-subcategory indicator (0.4). Scores are min-max normalised to [0,1] before mixing. Lambda 0.75 for LinUCB+MMR and 0.6 for the high-exploration policy. The pure ranker and epsilon-greedy use lambda 1, which is plain top-K with the subcategory cap.");
  card(s, MX, 1.45, CW, 1.0, C.accent5, "formula band");
  txt(s, "next = argmax_{i in C\\S} [ lambda * Score(u,i) - (1 - lambda) * max_{j in S} Sim(i, j) ]", { x: MX + 0.3, y: 1.45, w: CW - 0.6, h: 1.0, fontSize: 21, bold: true, color: C.background1, fontFace: "Courier New", valign: "middle" });
  bullets(s, [
    "Sim(i, j) = 0.6 x embedding cosine + 0.4 x same-subcategory flag",
    "Scores min-max normalised so lambda is comparable across users",
    "Greedy K = 5 on GPU; subcategory cap (max 2) enforced inside the loop",
    "lambda 0.75 (LinUCB + MMR), 0.6 (high exploration), 1.0 (no diversity)",
    "Slot 1 has zero penalty; later slots pay for similarity to earlier picks",
  ], { x: MX, y: 2.8, w: 5.6, h: 4.0, fontSize: 15 });
  s.addChart(pres.charts.BAR, [{ name: "Intra-list diversity", labels: L, values: P.map((p) => +SUM[p].ild.toFixed(3)) }],
    Object.assign(CHART_BASE(), { x: 6.6, y: 2.7, w: 6.1, h: 4.2, barDir: "col", chartColors: POL_HEX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", showLegend: false, showTitle: true, title: "Intra-list diversity (mean pairwise cosine distance)", valAxisMinVal: 0.9, valAxisMaxVal: 1.02, valAxisLabelFormatCode: "0.00" }));
}

// ================================================================ 13 PROPENSITY
{
  const s = content("Step 5: Boltzmann propensities and structured logging",
    "Every displayed item carries P(i|u), the softmax of its bandit score over the guard-railed pool. This is what makes counterfactual evaluation possible. For off-policy evaluation each policy is treated as its stochastic Boltzmann variant (sequential sampling without replacement), because the deployed slates use deterministic greedy MMR.");
  card(s, MX, 1.45, CW, 0.9, C.accent5, "formula band");
  txt(s, "P(i | u) = exp( Score(u,i) / tau ) / sum_j exp( Score(u,j) / tau )        tau = 0.06", { x: MX + 0.3, y: 1.45, w: CW - 0.6, h: 0.9, fontSize: 21, bold: true, color: C.background1, fontFace: "Courier New", valign: "middle" });
  let rec = null;
  try {
    const fd = fs.openSync(path.join(__dirname, "..", "retail-discovery-bandit", "output", "logs", "propensity_log.jsonl"), "r");
    const buf = Buffer.alloc(6000); fs.readSync(fd, buf, 0, 6000, 0); fs.closeSync(fd);
    rec = JSON.parse(buf.toString("utf8").split("\n")[0]);
  } catch (e) { rec = null; }
  const ctxKeys = rec ? Object.keys(rec.context).slice(0, 3) : [];
  const logTxt = rec ? '{ "policy": "' + rec.policy + '", "request": ' + rec.request + ', "user_id": ' + rec.user_id + ',\n  "item_id": ' + rec.item_id + ', "slot": ' + rec.slot + ',\n  "propensity_score": ' + f(rec.propensity_score, 4) + ',\n  "context": { ' + ctxKeys.map((k) => '"' + k + '": ' + rec.context[k]).join(", ") + ', ... },\n  "reward": ' + rec.reward + ', "click": ' + rec.click + ', "buy": ' + rec.buy + ' }' : "(log not available)";
  txt(s, "Logged per slot: a real record", { x: MX, y: 2.7, w: 6, h: 0.4, fontSize: 18, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  card(s, MX, 3.15, 6.6, 3.5, "1B2A2F", "code panel");
  txt(s, logTxt, { x: MX + 0.25, y: 3.3, w: 6.1, h: 3.2, fontSize: 13, color: "E6F2EF", fontFace: "Courier New" });
  txt(s, "Why it matters", { x: 7.5, y: 2.7, w: 5.2, h: 0.4, fontSize: 18, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  bullets(s, [
    "Propensities make any logged slate reusable for counterfactual evaluation",
    "The inference and guardrail logs record stage latencies and drop counts per request",
    "The API writes an impression record at /discover and a feedback record at /feedback",
    "Slot-0 propensity is exact; later slots use sequential without-replacement conditionals",
  ], { x: 7.5, y: 3.15, w: 5.2, h: 3.6, fontSize: 15 });
}

// ================================================================ 14 WORKED EXAMPLE
{
  const ex = D.example;
  const s = content("Worked example: one slate, fully decomposed",
    "User 7, a gourmet persona, with alpha overridden to 0.8 and lambda to 0.6 through the API. Each row shows why the item was chosen: base utility from the ranker, the exploration bonus, and the MMR penalty it paid for similarity to earlier picks. The bandit score is utility plus bonus; the MMR penalty is applied during selection.");
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, align: "center", fontSize: 12 } });
  const c = (t, o = {}) => ({ text: String(t), options: Object.assign({ fontSize: 12, align: "center" }, o) });
  const rows = [[hdr("Slot"), hdr("SKU"), hdr("Novelty tier"), hdr("Cold"), hdr("p(click)"), hdr("p(buy)"), hdr("Base utility"), hdr("Explore bonus"), hdr("MMR penalty"), hdr("Bandit score"), hdr("Propensity")]];
  ex.slate.forEach((r) => rows.push([c(r.slot + 1), c(r.item_id), c(r.novelty_tier.replace("_", " "), { align: "left" }), c(r.is_cold_start ? "yes" : "no"), c(f(r.p_click)), c(f(r.p_buy)), c(f(r.base_utility)), c(f(r.exploration_bonus), { bold: true, color: HEX.accent2 }), c(f(r.mmr_penalty)), c(f(r.bandit_score)), c(f(r.propensity_score, 4))]));
  s.addTable(rows, { x: MX, y: 1.5, w: CW, colW: [0.6, 0.8, 2.0, 0.7, 1.0, 0.9, 1.2, 1.3, 1.2, 1.2, 1.2], rowH: 0.5, border: { type: "solid", color: "D9E2E0", pt: 0.75 }, fontFace: THEME.bodyFontFace, color: HEX.dk1 });
  const fd = ex.filter_drop_counts, gd = ex.guardrails;
  bullets(s, [
    "Pipeline: " + fd.retrieved + " retrieved, " + fd.out_of_stock + " out of stock, " + gd.dropped_price_band + " dropped by the price band, " + gd.n_out + " candidates scored",
    "Slate ILD " + f(ex.intra_list_diversity) + ", end-to-end latency " + f(ex.latency_ms.total, 1) + " ms (single request, includes first-call overhead)",
    "Request used alpha = 0.8 and lambda = 0.6 (API overrides), policy " + ex.policy,
  ], { x: MX, y: 4.9, w: CW, h: 1.9, fontSize: 15 });
}

// ================================================================ 15 SECTION
section("Measuring discovery", "Click-through is not enough: discovery needs its own metric families",
  "A discovery engine deliberately trades a little immediate reward for exploration. If we only look at CTR we will conclude it is worse. So we need metrics for exposure, conversion of the explored items, breadth, diversity, and counterfactual value.");

// ================================================================ 16 FRAMEWORK
{
  const s = content("Discovery metric families and what each answers",
    "Each family answers a different question. A healthy discovery launch moves the first four up while keeping the business guardrails close to the baseline, and uses off-policy evaluation to check it before any live experiment.");
  const fam = [
    ["Exploration", "Is new inventory getting seen?", "Cold-start impression share\nDiscovery impression share"],
    ["Conversion", "Does explored inventory convert?", "Cold-start conversion rate\nDiscovery conversion rate\nDiscovery purchase share"],
    ["Serendipity", "Relevant surprises the ranker would not show", "Serendipity score: converted items from unseen tiers outside the ranker's top-5"],
    ["Breadth", "How concentrated is exposure?", "Catalog coverage %, Gini, Shannon entropy"],
    ["Diversity", "Is each slate varied?", "Intra-list diversity (pairwise cosine distance)"],
    ["Guardrails and value", "What did exploration cost?", "CTR, reward, margin and profit per slate, 7-day repeat rate"],
  ];
  const cw = (CW - 0.4) / 3, ch = 2.5;
  fam.forEach((fm, i) => {
    const x = MX + (i % 3) * (cw + 0.2), y = 1.45 + Math.floor(i / 3) * (ch + 0.2);
    card(s, x, y, cw, ch, C.background2, "metric card");
    txt(s, fm[0], { x: x + 0.2, y: y + 0.15, w: cw - 0.4, h: 0.45, fontSize: 18, bold: true, color: i === 5 ? C.accent4 : C.accent1, fontFace: THEME.headFontFace });
    txt(s, fm[1], { x: x + 0.2, y: y + 0.65, w: cw - 0.4, h: 0.6, fontSize: 14, italic: true, color: C.text2 });
    txt(s, fm[2], { x: x + 0.2, y: y + 1.3, w: cw - 0.4, h: 1.1, fontSize: 13 });
  });
  txt(s, "Counterfactual layer: IPS, SNIPS, doubly robust and Switch-DR with bootstrap CIs, effective sample size and weight-clipping diagnostics (slide 22).", { x: MX, y: 6.55, w: CW, h: 0.4, fontSize: 13, color: C.accent4 });
}

// ================================================================ 17 COMPARISON TABLE
{
  const s = content("Four policies compared on discovery metrics",
    "Bold with a tint marks the best value in each row (lowest for Gini). Pure ranker wins immediate-value rows by a small margin. LinUCB+MMR leads discovery purchase share and serendipity; high-exploration leads cold-start exposure, cold conversion and diversity; epsilon-greedy has the highest coverage because its random picks spread exposure widely. All values from 3,000 requests (15,000 impressions) per policy.");
  const rows = [
    ["Overall CTR", "ctr", 1, (x) => pct(x)],
    ["Reward per impression", "avg_reward_per_impression", 1, (x) => f(x, 4)],
    ["Realised profit per slate", "realized_profit_per_slate", 1, (x) => f(x, 3)],
    ["7-day repeat purchase rate", "repeat_purchase_rate_7d", 1, (x) => pct(x)],
    ["Discovery impression share", "discovery_impression_share", 1, (x) => pct(x)],
    ["Discovery conversion rate", "discovery_conversion_rate", 1, (x) => pct(x, 2)],
    ["Discovery share of purchases", "discovery_purchase_share", 1, (x) => pct(x)],
    ["Cold-start impression share", "cold_start_impression_share", 1, (x) => pct(x)],
    ["Cold-start conversion rate", "cold_start_conversion_rate", 1, (x) => pct(x, 2)],
    ["Serendipity score", "serendipity_score", 1, (x) => f(x, 4)],
    ["Catalog coverage", "catalog_coverage_pct", 1, (x) => f(x, 1) + "%"],
    ["Gini of exposure (lower is better)", "gini_exposure", -1, (x) => f(x, 3)],
    ["Intra-list diversity", "ild", 1, (x) => f(x, 3)],
  ];
  const hdr = (t, i) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: i >= 0 ? POL_HEX[i] === HEX.accent3 ? "B8691F" : POL_HEX[i] : HEX.dk2 }, align: i >= 0 ? "center" : "left", fontSize: 12 } });
  const body = [[hdr("Metric", -1), ...L.map((l, i) => hdr(l, i))]];
  rows.forEach((r) => {
    const vals = P.map((p) => SUM[p][r[1]]);
    const best = r[2] === 1 ? Math.max(...vals) : Math.min(...vals);
    body.push([{ text: r[0], options: { fontSize: 12, align: "left" } }, ...vals.map((v) => ({ text: r[3](v), options: { fontSize: 12, align: "center", bold: v === best, fill: v === best ? { color: "D8EEE9" } : undefined } }))]);
  });
  s.addTable(body, { x: MX, y: 1.4, w: CW, colW: [4.1, 2.0, 2.0, 2.0, 2.0], rowH: 0.36, border: { type: "solid", color: "D9E2E0", pt: 0.75 }, fontFace: THEME.bodyFontFace, color: HEX.dk1 });
  txt(s, "3,000 requests x 5 slots per policy. Common random numbers: identical user sequence for every policy.", { x: MX, y: 6.55, w: CW, h: 0.3, fontSize: 11, color: C.accent4 });
}

// ================================================================ 18 PRICE OF EXPLORATION
{
  const s = content("The price of exploration is small",
    "Left: reward per impression. Right: share of impressions that are cold-start items. LinUCB+MMR gives up 3.4% of reward and shows 12 times more cold-start items; high exploration gives up 4.2% for 45 times more. Epsilon-greedy pays the largest price (7.5%) for less cold exposure than high-exploration LinUCB, because its exploration is random rather than directed.");
  s.addChart(pres.charts.BAR, [{ name: "Reward per impression", labels: L, values: P.map((p) => +SUM[p].avg_reward_per_impression.toFixed(4)) }],
    Object.assign(CHART_BASE(), { x: MX, y: 1.4, w: 6.0, h: 4.0, barDir: "col", chartColors: POL_HEX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", showLegend: false, showTitle: true, title: "Average reward per impression", valAxisMinVal: 0, valAxisMaxVal: 0.1, valAxisLabelFormatCode: "0.00" }));
  s.addChart(pres.charts.BAR, [{ name: "Cold-start impression share", labels: L, values: P.map((p) => +(100 * SUM[p].cold_start_impression_share).toFixed(2)) }],
    Object.assign(CHART_BASE(), { x: 6.8, y: 1.4, w: 5.9, h: 4.0, barDir: "col", chartColors: POL_HEX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showLegend: false, showTitle: true, title: "Cold-start share of impressions (%)", valAxisMinVal: 0, valAxisMaxVal: 7.5, valAxisLabelFormatCode: "0" }));
  const base = SUM.pure_ranker.avg_reward_per_impression;
  const call = P.slice(1).map((p, i) => L[i + 1] + ": " + pct(SUM[p].avg_reward_per_impression / base - 1) + " reward");
  txt(s, "Reward vs pure ranker  |  " + call.join("   |   "), { x: MX, y: 5.65, w: CW, h: 0.5, fontSize: 15, bold: true, color: C.text2 });
  txt(s, "Directed exploration (LinUCB) buys more cold-start exposure per unit of reward than random exploration (epsilon-greedy).", { x: MX, y: 6.2, w: CW, h: 0.6, fontSize: 15 });
}

// ================================================================ 19 TIER MIX
{
  const s = content("Where discovery traffic goes, and what it converts at",
    "Left: share of impressions by novelty tier. LinUCB+MMR more than doubles the share of unseen-subcategory items (16% to 35%). Right: purchase rate per impression by tier for the same policies. Unseen-subcategory items convert at about 2.5% vs 9.6% for previously bought items, so exploration is paid for in conversion per impression. Repeat items are not filtered after 14 days, which is realistic for grocery.");
  const tiers = ["Previously bought", "New SKU", "Unseen brand", "Unseen subcategory"];
  const mixSeries = tiers.map((t, k) => ({ name: t, labels: L, values: D.mix.map((r) => +(100 * r[k]).toFixed(1)) }));
  s.addChart(pres.charts.BAR, mixSeries, Object.assign(CHART_BASE(), { x: MX, y: 1.4, w: 6.5, h: 4.6, barDir: "col", barGrouping: "percentStacked", chartColors: ["B8C4C2", HEX.accent5, HEX.accent3, HEX.accent2], showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: "0", dataLabelColor: "FFFFFF", showLegend: true, legendPos: "b", showTitle: true, title: "Impression share by novelty tier (%)" }));
  const buySeries = [0, 2].map((pi) => ({ name: L[pi], labels: tiers, values: D.tier_buy[pi].map((v) => +(100 * v).toFixed(2)) }));
  s.addChart(pres.charts.BAR, buySeries, Object.assign(CHART_BASE(), { x: 7.2, y: 1.4, w: 5.5, h: 4.6, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent4, HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showLegend: true, legendPos: "b", showTitle: true, title: "Purchase rate per impression by tier (%)", catAxisLabelFontSize: 10, valAxisLabelFormatCode: "0" }));
  txt(s, "Discovery conversion is roughly 2.5-3% for unseen-subcategory and unseen-brand items vs 4% for new SKUs and 9.6% for repeat items.", { x: MX, y: 6.15, w: CW, h: 0.7, fontSize: 15, color: C.text2, bold: true });
}

// ================================================================ 20 BREADTH
{
  const s = content("Breadth and concentration: still a gap",
    "Exposure remains highly concentrated for all policies at this volume: a Gini near 0.98 and coverage of 5-11% of the catalog after 15,000 impressions. Epsilon-greedy has the best coverage and entropy because random swaps spread exposure; directed exploration (LinUCB) raises cold-start share but barely moves coverage. This is an honest weak spot and a target for a breadth-aware bonus or a longer simulation.");
  s.addChart(pres.charts.BAR, [{ name: "Coverage", labels: L, values: P.map((p) => +SUM[p].catalog_coverage_pct.toFixed(2)) }],
    Object.assign(CHART_BASE(), { x: MX, y: 1.4, w: 4.0, h: 3.9, barDir: "col", chartColors: POL_HEX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showLegend: false, showTitle: true, title: "Catalog coverage (%)", catAxisLabelFontSize: 10, valAxisMinVal: 0, valAxisMaxVal: 14, valAxisLabelFormatCode: "0" }));
  s.addChart(pres.charts.BAR, [{ name: "Gini", labels: L, values: P.map((p) => +SUM[p].gini_exposure.toFixed(3)) }],
    Object.assign(CHART_BASE(), { x: 4.7, y: 1.4, w: 4.0, h: 3.9, barDir: "col", chartColors: POL_HEX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", showLegend: false, showTitle: true, title: "Gini of exposure (lower is better)", catAxisLabelFontSize: 10, valAxisMinVal: 0.9, valAxisMaxVal: 1.0, valAxisLabelFormatCode: "0.00" }));
  s.addChart(pres.charts.BAR, [{ name: "Entropy", labels: L, values: P.map((p) => +SUM[p].shannon_entropy_normalised.toFixed(3)) }],
    Object.assign(CHART_BASE(), { x: 8.8, y: 1.4, w: 3.95, h: 3.9, barDir: "col", chartColors: POL_HEX, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", showLegend: false, showTitle: true, title: "Normalised Shannon entropy", catAxisLabelFontSize: 10, valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisLabelFormatCode: "0.0" }));
  bullets(s, [
    "Epsilon-greedy is the broadest (coverage 11.1%) because random swaps pick from anywhere in the pool",
    "LinUCB + MMR lifts cold-start share and serendipity but leaves coverage near the pure ranker (5.7% vs 5.3%)",
    "Next lever: a breadth-aware bonus or per-item exposure caps, and a longer simulation horizon",
  ], { x: MX, y: 5.5, w: CW, h: 1.4, fontSize: 15 });
}

// ================================================================ 21 COLD SLICE + PERSONA
{
  const s = content("Cold-start slice and persona sensitivity",
    "Left: cold-start impressions convert well when shown: 1.5-1.8% purchase rate against a cold-item average of about 0.6% when exposed to random users, because the bandit selects relevant users. Right: cold-start share by persona. Gourmet users receive two to three times the cold-start share of the other personas under both bandit policies; the cause was not investigated.");
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, align: "center", fontSize: 12 } });
  const c = (t, o = {}) => ({ text: String(t), options: Object.assign({ fontSize: 12, align: "center" }, o) });
  const cs = (p, k) => SUM[p][k];
  s.addTable([
    [hdr("Cold-start slice"), hdr("Impressions"), hdr("Click rate"), hdr("Purchase rate")],
    ...[1, 2, 3].map((i) => [c(L[i], { align: "left" }), c(Math.round(SUM[P[i]].cold_start_impression_share * 15000)), c(pct(SUM[P[i]].cold_start_click_rate)), c(pct(SUM[P[i]].cold_start_conversion_rate, 2))]),
    [c("Warm items, LinUCB + MMR", { align: "left" }), c(Math.round((1 - SUM.linucb_mmr.cold_start_impression_share) * 15000)), c("14.5%"), c(pct(SUM.linucb_mmr.warm_conversion_rate, 2))],
  ], { x: MX, y: 1.5, w: 5.9, colW: [2.5, 1.2, 1.0, 1.2], rowH: 0.5, border: { type: "solid", color: "D9E2E0", pt: 0.75 }, fontFace: THEME.bodyFontFace, color: HEX.dk1 });
  txt(s, "The pure ranker showed 21 cold-start impressions in 15,000, too few to report a rate.", { x: MX, y: 4.2, w: 5.9, h: 0.6, fontSize: 12, color: C.accent4 });
  bullets(s, [
    "Cold conversion (1.5-1.8%) is above the 0.6% cold-item baseline because the bandit places new SKUs with relevant users",
    "Cold-start CTR 3.5-6.4% vs 14.5% for warm items: new items are still less attractive on average",
  ], { x: MX, y: 4.85, w: 5.9, h: 2.0, fontSize: 14 });
  const pn = ["Organic", "Family bulk", "Gourmet", "Budget"];
  s.addChart(pres.charts.BAR, [1, 2, 3].map((i) => ({ name: L[i], labels: pn, values: D.persona_cold[P[i]].map((v) => +(100 * v).toFixed(2)) })),
    Object.assign(CHART_BASE(), { x: 6.8, y: 1.4, w: 5.9, h: 5.4, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent3, HEX.accent1, HEX.accent2], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showLegend: true, legendPos: "b", showTitle: true, title: "Cold-start impression share by persona (%)", valAxisLabelFormatCode: "0" }));
}

// ================================================================ 22 OPE
{
  const s = content("Off-policy evaluation recovers the truth within 1.6%",
    "Behaviour: Boltzmann sampling (tau 0.10) over the pure-ranker utility, 6,000 requests, 30,000 slot records. Every policy is scored on the identical candidate pool. IPS, SNIPS, DR and Switch-DR are all within 0.6-1.6% of the simulator truth and preserve the policy ordering; the direct method, which relies on the ranker as reward model, overestimates by 34-40% because the ranker is over-confident for the items that policies actually pick. Max weight was 6.4, so no weight reached the clip of 20 and Switch-DR equals DR.");
  const est = ["Ground truth", "DM", "IPS", "SNIPS", "DR"];
  const series = est.map((e, k) => ({ name: e === "Ground truth" ? e : e, labels: L, values: P.map((p) => +(k === 0 ? OPE[p].DM.true_value : OPE[p][e].estimate).toFixed(4)) }));
  s.addChart(pres.charts.BAR, series, Object.assign(CHART_BASE(), { x: MX, y: 1.4, w: 7.4, h: 5.4, barDir: "col", barGrouping: "clustered", chartColors: [HEX.dk1, "B8C4C2", HEX.accent3, HEX.accent1, HEX.accent2], showValue: false, showLegend: true, legendPos: "b", showTitle: true, title: "Expected reward per slot: estimate vs ground truth", valAxisMinVal: 0.05, valAxisMaxVal: 0.12, valAxisLabelFormatCode: "0.00" }));
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, align: "center", fontSize: 12 } });
  const c = (t, o = {}) => ({ text: String(t), options: Object.assign({ fontSize: 12, align: "center" }, o) });
  s.addTable([
    [hdr("Policy"), hdr("DM error"), hdr("DR error"), hdr("ESS")],
    ...P.map((p, i) => [c(L[i], { align: "left" }), c("+" + f(OPE[p].DM.rel_error_pct, 0) + "%"), c("+" + f(OPE[p].DR.rel_error_pct, 1) + "%"), c(pct(OPE[p].diagnostics.ess_fraction, 0))]),
  ], { x: 8.2, y: 1.5, w: 4.5, colW: [1.9, 0.9, 0.9, 0.8], rowH: 0.45, border: { type: "solid", color: "D9E2E0", pt: 0.75 }, fontFace: THEME.bodyFontFace, color: HEX.dk1 });
  bullets(s, [
    "95% bootstrap CIs (1,000 resamples over requests) cover the truth for IPS, SNIPS, DR and Switch-DR",
    "Effective sample size 74-90% of 30,000 records",
    "Zero weights clipped; max weight 6.4",
  ], { x: 8.2, y: 4.0, w: 4.5, h: 2.9, fontSize: 14 });
}

// ================================================================ 23 SYSTEM
{
  const s = content("System performance: about 19 ms median, 26 ms p99",
    "Stage latencies are for the LinUCB+MMR policy over 3,000 requests, with CUDA synchronised around each stage. Ranking (XGBoost inference plus feature building) and MMR (a sequential greedy loop with GPU-CPU syncs) dominate. The standalone retrieval benchmark is 0.17 ms p99; inside the pipeline it is 1.75 ms because it includes moving results to the CPU.");
  const stageNames = { "1_retrieval": "Retrieval", "2_filtering": "Filtering", "3_ranking": "Ranking", "4a_utility": "Utility", "4b_guardrails": "Guardrails", "4c_context": "Context build", "4d_bandit": "Bandit scoring", "4e_propensity": "Propensity", "4f_mmr": "MMR" };
  const keys = Object.keys(stageNames);
  s.addChart(pres.charts.BAR, [
    { name: "p50", labels: keys.map((k) => stageNames[k]), values: keys.map((k) => D.lat[k]) },
    { name: "p99", labels: keys.map((k) => stageNames[k]), values: keys.map((k) => D.lat99[k]) },
  ], Object.assign(CHART_BASE(), { x: MX, y: 1.4, w: 7.6, h: 5.4, barDir: "bar", barGrouping: "clustered", chartColors: [HEX.accent1, HEX.accent2], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showLegend: true, legendPos: "b", showTitle: true, title: "Latency by stage, LinUCB + MMR (ms)", catAxisOrientation: "maxMin", valAxisLabelFormatCode: "0" }));
  const sw = 2.2;
  stat(s, 8.5, 1.5, sw, f(D.lat.total, 1) + " ms", "median end-to-end", C.accent1, 30);
  stat(s, 8.5 + sw + 0.2, 1.5, sw, f(D.lat99.total, 1) + " ms", "p99 end-to-end", C.accent1, 30);
  stat(s, 8.5, 3.25, sw, f(FULL.stage1_retrieval.latency_ms.p99_ms, 2) + " ms", "retrieval p99, standalone", C.accent1, 30);
  stat(s, 8.5 + sw + 0.2, 3.25, sw, f(GPU.vram_used_mb / 1024, 1) + " GB", "VRAM in use of " + f(GPU.vram_total_mb / 1024, 0) + " GB", C.accent1, 30);
  txt(s, GPU.device + ", CUDA " + GPU.cuda_runtime + ", torch " + GPU.torch + ". Ranker and retrieval on GPU; the bandit state is float64 on GPU.", { x: 8.5, y: 5.05, w: 4.6, h: 1.0, fontSize: 13, color: C.accent4 });
}

// ================================================================ 24 TAKEAWAYS
{
  const s = content("Takeaways, caveats and next steps",
    "Close on what we can claim and what we cannot. The simulator is synthetic with a hidden ground truth, so the absolute numbers are not production predictions; the mechanism and the relative comparisons are the finding. The next step is to measure the oracle ceilings for the weak ranker and for cold-start retrieval before spending effort on them.");
  const cw = (CW - 0.4) / 3;
  const cols = [
    ["What we showed", C.accent1, [
      "Relevance rankers score new SKUs 38% lower at equal relevance",
      "LinUCB + MMR: 12x cold-start exposure for 3.4% less reward",
      "Discovery purchase share 29% vs 17%",
      "OPE matches the simulator within 1.6%; DM is 34-40% off",
    ]],
    ["Caveats", C.accent2, [
      "Synthetic data; ranker AUC only 0.63",
      "Learned bandit mean hurt reward, so the bonus-only score is the default",
      "Coverage barely improves; exposure remains concentrated",
      "OPE evaluates stochastic variants; deployed MMR is deterministic",
    ]],
    ["Next steps", C.accent4, [
      "Measure oracle ceilings for the ranker and for cold-start retrieval",
      "Tune the price band, which drops about 132 of 190 candidates",
      "Breadth-aware bonus and longer horizon simulation",
      "Validate with an online A/B test on discovery metrics",
    ]],
  ];
  cols.forEach((c, i) => {
    const x = MX + i * (cw + 0.2);
    card(s, x, 1.5, cw, 5.2, C.background2, "takeaway card");
    txt(s, c[0], { x: x + 0.25, y: 1.7, w: cw - 0.5, h: 0.5, fontSize: 20, bold: true, color: c[1], fontFace: THEME.headFontFace });
    bullets(s, c[2], { x: x + 0.25, y: 2.35, w: cw - 0.5, h: 4.2, fontSize: 15 });
  });
}

// ---------------------------------------------------------------- write
(async () => {
  const out = path.join(__dirname, "Discovery_Engine_Flow_and_Metrics.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
