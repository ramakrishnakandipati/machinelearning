// Builds Hospitality_Propensity_Engine.pptx from deck_data.json (+ screenshots under ../outputs).
// Run from this folder:  node build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const SKILL = process.env.PPTX_SKILL_DIR || "C:/Users/RamaKrishna/.claude/skills/synced/c4087e75-6e8c-4493-8472-1adcd2a2603e_2c02675d-fb05-4126-b416-349c05b8005e/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = path.join(__dirname, "Hospitality_Propensity_Engine.pptx");
const PROJ = path.join(__dirname, "..");

// ------------------------------------------------------------------ theme
const THEME = {
  name: "Hospitality Propensity",
  headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "14232B", lt1: "FFFFFF", dk2: "0B3C49", lt2: "EEF3F5", accent1: "1F6F8B", accent2: "C8963E", accent3: "5E9C76",
            accent4: "8E4162", accent5: "5B7C99", accent6: "C0504D", hlink: "1F6F8B", folHlink: "5B7C99" },
};
const H = THEME.colors; // hex values for charts/tables (hex-only options)
const MODEL_HEX = { xgboost: H.accent1, lightgbm: H.accent3, catboost: H.accent2, pytorch_nn: H.accent4 };
const MODEL_NAME = { xgboost: "XGBoost (CUDA)", lightgbm: "LightGBM (CPU)", catboost: "CatBoost (GPU)", pytorch_nn: "PyTorch NN (CUDA)" };
const MODEL_SHORT = { xgboost: "XGBoost", lightgbm: "LightGBM", catboost: "CatBoost", pytorch_nn: "PyTorch NN" };
const MODELS = D.models;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Hospitality Propensity Engine";
pres.author = "Hospitality Propensity Engine";
const C = pres.SchemeColor;
const W = 13.33, M = 0.6;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 7.2, h: 1.9, fontFace: "Cambria", fontSize: 46, bold: true, color: C.background1, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.8, y: 4.05, w: 6.9, h: 1.3, fontFace: "Calibri", fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.5, h: 1.2, fontFace: "Cambria", fontSize: 48, bold: true, color: C.background1, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.8, y: 3.9, w: 10.5, h: 1.2, fontFace: "Calibri", fontSize: 22, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.85, fontFace: "Cambria", fontSize: 34, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Hospitality Propensity Engine  |  synthetic data, results from the held-out temporal test snapshot", options: { x: M, y: 7.08, w: 9, h: 0.25, fontFace: "Calibri", fontSize: 10, color: "7A8A93", margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.08, w: 0.6, h: 0.25, fontFace: "Calibri", fontSize: 10, color: "7A8A93", align: "right" },
});

// ------------------------------------------------------------------ helpers
async function icon(Comp, color = "#FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}
const fa = require("react-icons/fa");
const ICON = {};
async function loadIcons() {
  const names = { data: "FaDatabase", feat: "FaLayerGroup", train: "FaMicrochip", cal: "FaBalanceScale", explain: "FaSearch", serve: "FaRocket",
                  hotel: "FaHotel", shield: "FaShieldAlt", chart: "FaChartBar", bolt: "FaBolt", check: "FaCheckCircle" };
  for (const [k, n] of Object.entries(names)) ICON[k] = fa[n] ? await icon(fa[n]) : null;
}
const f2 = (x) => x.toFixed(2), f3 = (x) => x.toFixed(3), f4 = (x) => x.toFixed(4);
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const flowBy = (k) => D.flows.find((f) => f.key === k);
let sectionOpen = null;
function slide(layout, section) {
  if (section && section !== sectionOpen) { pres.addSection({ title: section }); sectionOpen = section; }
  return pres.addSlide({ masterName: layout, sectionTitle: section });
}
function content(title, section, notes) {
  const s = slide("CONTENT", section);
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function tile(s, x, y, w, h, big, label, opts = {}) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: opts.dark ? C.text2 : C.background2 }, line: { color: opts.dark ? C.text2 : C.background2 }, objectName: "tile" });
  s.addText(big, { x: x + 0.12, y: y + 0.08, w: w - 0.24, h: h * 0.55, fontFace: "Cambria", fontSize: opts.bigSize || 30, bold: true, color: opts.dark ? C.background1 : C.text2, valign: "middle", margin: 0, isTextBox: true });
  s.addText(label, { x: x + 0.12, y: y + h * 0.6, w: w - 0.24, h: h * 0.36, fontFace: "Calibri", fontSize: opts.labelSize || 12, color: opts.dark ? C.background2 : C.text1, valign: "top", margin: 0, isTextBox: true });
}
function eyebrow(s, text, x, y, w) {
  s.addText(text, { x, y, w, h: 0.28, fontFace: "Calibri", fontSize: 11, bold: true, color: C.accent2, charSpacing: 2, margin: 0, isTextBox: true });
}
const CHART_BASE = {
  catAxisLabelFontFace: "Calibri", valAxisLabelFontFace: "Calibri", catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  catAxisLabelColor: "3A4A53", valAxisLabelColor: "6B7B84", valGridLine: { color: "E2E8EB", size: 0.75 }, catGridLine: { style: "none" },
  dataLabelFontFace: "Calibri", dataLabelFontSize: 10, dataLabelColor: "14232B", legendFontFace: "Calibri", legendFontSize: 11, legendColor: "3A4A53",
  titleFontFace: "Calibri", titleFontSize: 13, titleColor: "14232B",
};
function tblCell(text, o = {}) {
  return { text: String(text), options: { fontFace: "Calibri", fontSize: o.fontSize || 12, color: o.color || H.dk1, bold: !!o.bold, align: o.align || "left", valign: "middle",
    fill: o.fill ? { color: o.fill } : undefined, border: [{ type: "solid", color: "D5DEE2", pt: 0.75 }, { type: "none" }, { type: "solid", color: "D5DEE2", pt: 0.75 }, { type: "none" }], margin: [0.03, 0.08, 0.03, 0.08] } };
}
const hdr = (t, align = "left") => tblCell(t, { bold: true, color: "FFFFFF", fill: H.dk2, align });
const IMG = (p) => path.join(PROJ, p);
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function placeImage(s, file, x, y, maxW, maxH, o = {}) {
  const { w, h } = pngSize(file);
  let iw = Math.min(maxW, (maxH * w) / h), ih = (iw * h) / w;
  const px = o.center ? x + (maxW - iw) / 2 : x;
  s.addShape(pres.ShapeType.rect, { x: px - 0.03, y: y - 0.03, w: iw + 0.06, h: ih + 0.06, fill: { color: C.background1 }, line: { color: "CBD6DB", width: 0.75 }, shadow: { type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 }, objectName: "image frame" });
  s.addImage({ path: file, x: px, y, w: iw, h: ih, altText: o.alt || "screenshot" });
  return { w: iw, h: ih, x: px };
}

(async () => {
  await loadIcons();
  const A = D.agg;
  const bestRoc = D.flows.map((f) => ({ f, m: f.best_model, roc: f.models[f.best_model].roc_auc, pr: f.models[f.best_model].pr_auc, lift: f.models[f.best_model].top_decile_lift, ll: f.models[f.best_model].logloss }));
  const rocMin = Math.min(...bestRoc.map((b) => b.roc)), rocMax = Math.max(...bestRoc.map((b) => b.roc));
  const liftMax = bestRoc.reduce((a, b) => (b.lift > a.lift ? b : a));
  const nGuests = D.tables.guests, nBook = D.tables.bookings;

  // ============================================================ 1 title
  {
    const s = slide("TITLE_DARK", "Overview");
    s.addText("Hospitality Propensity Engine", { placeholder: "title" });
    s.addText("Eight guest-lifecycle propensity models: GPU-trained, calibrated, explained and served", { placeholder: "subtitle" });
    s.addText(`${nGuests.toLocaleString()} guests  |  ${nBook.toLocaleString()} bookings  |  ${D.n_models} models  |  ${D.cuda}`, { x: 0.8, y: 6.3, w: 7, h: 0.4, fontFace: "Calibri", fontSize: 14, color: "9FC3CE", margin: 0, isTextBox: true });
    // flow chips
    D.flows.forEach((f, i) => {
      const col = i % 2, row = Math.floor(i / 2);
      const x = 8.35 + col * 2.35, y = 1.35 + row * 1.32;
      s.addShape(pres.ShapeType.roundRect, { x, y, w: 2.2, h: 1.14, rectRadius: 0.1, fill: { color: C.background1, transparency: 90 }, line: { color: C.background1, transparency: 70, width: 0.75 }, objectName: "flow chip" });
      s.addText(String(f.n), { x: x + 0.12, y: y + 0.1, w: 0.5, h: 0.4, fontFace: "Cambria", fontSize: 20, bold: true, color: C.accent2, margin: 0, isTextBox: true });
      s.addText(f.short, { x: x + 0.12, y: y + 0.52, w: 1.98, h: 0.5, fontFace: "Calibri", fontSize: 14, bold: true, color: C.background1, valign: "top", margin: 0, isTextBox: true });
    });
    s.addNotes("Overview of the propensity engine: eight lifecycle flows, four model families each, calibrated probabilities, SHAP explanations, served via FastAPI and a Streamlit dashboard. All data is synthetic.");
  }

  // ============================================================ 2 executive summary
  {
    const s = content("What we built and what it shows", "Overview", "Key messages: breadth (8 flows x 4 models), signal varies by flow, tree models are tied, NN trails, calibration makes scores usable as probabilities.");
    tile(s, M, 1.5, 2.9, 1.5, "8", "propensity flows, from look-to-book to cancellation");
    tile(s, M + 3.1, 1.5, 2.9, 1.5, String(D.n_models), `models tuned (${D.n_trials} Optuna trials each) and calibrated`);
    tile(s, M, 3.2, 2.9, 1.5, `${f2(rocMin)}-${f2(rocMax)}`, "test ROC-AUC range of the best model per flow", { bigSize: 28 });
    tile(s, M + 3.1, 3.2, 2.9, 1.5, `${liftMax.lift.toFixed(1)}x`, `best top-decile lift (${liftMax.f.short})`);
    s.addText(`Data: ${nGuests.toLocaleString()} synthetic guests, ${nBook.toLocaleString()} bookings, ${D.tables.web_sessions.toLocaleString()} web sessions. ${D.tests} automated tests pass.`, { x: M, y: 4.95, w: 6.0, h: 0.8, fontFace: "Calibri", fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
    const pts = [
      ["Signal depends on the flow", `Look-to-book reaches ROC ${f2(flowBy("booking_abandonment").models[flowBy("booking_abandonment").best_model].roc_auc)}; room upgrade and OTA-to-direct stay near ${f2(rocMin)}.`],
      ["Tree models are effectively tied", `Mean ROC gap to the best model: CatBoost ${f2(A.catboost.roc_gap_pp)} pt, XGBoost ${f2(A.xgboost.roc_gap_pp)} pt, LightGBM ${f2(A.lightgbm.roc_gap_pp)} pt. The PyTorch NN trails by ${f2(A.pytorch_nn.roc_gap_pp)} pt.`],
      ["Scores behave like probabilities", `Calibration cut mean ECE from ${f4(D.cal.ece_raw_mean)} to ${f4(D.cal.ece_cal_mean)}; ${D.cal.improved_or_equal} of ${D.cal.n} models improved or matched log-loss.`],
      ["Built to be acted on", "Each flow maps a score decile to a next-best action, with SHAP reasons and an uplift view for the OTA offer."],
    ];
    pts.forEach((p, i) => {
      const y = 1.5 + i * 1.38;
      s.addShape(pres.ShapeType.ellipse, { x: 7.0, y: y + 0.05, w: 0.5, h: 0.5, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "number dot" });
      s.addText(String(i + 1), { x: 7.0, y: y + 0.05, w: 0.5, h: 0.5, fontFace: "Cambria", fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(p[0], { x: 7.7, y, w: 5.0, h: 0.35, fontFace: "Calibri", fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
      s.addText(p[1], { x: 7.7, y: y + 0.38, w: 5.0, h: 0.9, fontFace: "Calibri", fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true });
    });
  }

  // ============================================================ 3 flows at a glance
  {
    const s = content("The eight propensity flows", "Overview", "Each flow is one question, one unit of prediction, one label window, and one action.");
    D.flows.forEach((f, i) => {
      const col = i % 4, row = Math.floor(i / 4);
      const cw = 2.85, ch = 2.62, x = M + col * (cw + 0.23), y = 1.45 + row * (ch + 0.2);
      s.addShape(pres.ShapeType.roundRect, { x, y, w: cw, h: ch, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "flow card" });
      s.addShape(pres.ShapeType.ellipse, { x: x + 0.15, y: y + 0.15, w: 0.46, h: 0.46, fill: { color: C.text2 }, line: { color: C.text2 }, objectName: "flow number" });
      s.addText(String(f.n), { x: x + 0.15, y: y + 0.15, w: 0.46, h: 0.46, fontFace: "Cambria", fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(f.short, { x: x + 0.72, y: y + 0.12, w: cw - 0.85, h: 0.52, fontFace: "Calibri", fontSize: 16, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
      s.addText(f.action, { x: x + 0.15, y: y + 0.78, w: cw - 0.3, h: 0.8, fontFace: "Calibri", fontSize: 12, color: C.text1, valign: "top", margin: 0, isTextBox: true });
      s.addText([{ text: pct(f.base_rate, 0), options: { fontFace: "Cambria", fontSize: 24, bold: true, color: C.accent2 } }, { text: "  base rate", options: { fontFace: "Calibri", fontSize: 12, color: C.text1 } }], { x: x + 0.15, y: y + 1.68, w: cw - 0.3, h: 0.45, margin: 0, valign: "middle", isTextBox: true });
      s.addText(`${f.window_days}-day window  |  ${f.unit} level`, { x: x + 0.15, y: y + 2.18, w: cw - 0.3, h: 0.3, fontFace: "Calibri", fontSize: 11, color: "5A6A73", margin: 0, isTextBox: true });
    });
  }

  // ============================================================ 4 pipeline
  {
    const s = content("How the engine works", "Overview", "Pipeline: synthetic relational data, point-in-time features, GPU model training with Optuna, calibration, SHAP, serving.");
    const steps = [
      ["data", "Relational data", `${nGuests.toLocaleString()} guests, ${nBook.toLocaleString()} bookings, folios, web, email, surveys`],
      ["feat", "Point-in-time features", "RFM, inter-stay decay, seasonality, channel and price sensitivity"],
      ["train", "Train and tune", `XGBoost, LightGBM, CatBoost, PyTorch NN; ${D.n_trials} Optuna trials each`],
      ["cal", "Calibrate", "Sigmoid or isotonic, chosen on held-out log-loss"],
      ["explain", "Explain", "SHAP drivers, global and per guest"],
      ["serve", "Serve", "FastAPI endpoints and a Streamlit dashboard"],
    ];
    const bw = 1.85, gap = 0.2;
    steps.forEach((st, i) => {
      const x = M + i * (bw + gap), y = 1.7;
      s.addShape(pres.ShapeType.roundRect, { x, y, w: bw, h: 3.0, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "step card" });
      s.addShape(pres.ShapeType.ellipse, { x: x + (bw - 0.7) / 2, y: y + 0.2, w: 0.7, h: 0.7, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "step icon circle" });
      if (ICON[st[0]]) s.addImage({ data: ICON[st[0]], x: x + (bw - 0.36) / 2, y: y + 0.37, w: 0.36, h: 0.36, altText: st[1] });
      s.addText(st[1], { x: x + 0.1, y: y + 1.05, w: bw - 0.2, h: 0.55, fontFace: "Calibri", fontSize: 15, bold: true, color: C.text2, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(st[2], { x: x + 0.12, y: y + 1.65, w: bw - 0.24, h: 1.25, fontFace: "Calibri", fontSize: 12, color: C.text1, align: "center", valign: "top", margin: 0, isTextBox: true });
      if (i < steps.length - 1) s.addText("›", { x: x + bw - 0.02, y: y + 1.2, w: gap + 0.04, h: 0.5, fontFace: "Calibri", fontSize: 24, bold: true, color: C.accent2, align: "center", margin: 0, isTextBox: true });
    });
    s.addShape(pres.ShapeType.roundRect, { x: M, y: 5.1, w: W - 2 * M, h: 1.45, rectRadius: 0.08, fill: { color: C.text2 }, line: { color: C.text2 }, objectName: "gpu banner" });
    s.addText("GPU usage", { x: M + 0.3, y: 5.2, w: 3, h: 0.35, fontFace: "Calibri", fontSize: 14, bold: true, color: C.accent2, margin: 0, isTextBox: true });
    s.addText(`XGBoost (CUDA hist), CatBoost (GPU) and the PyTorch network (CUDA, mixed precision) train on the ${D.cuda}. LightGBM falls back to CPU automatically: the installed build has no GPU tree learner. Per-guest SHAP runs on CPU, which is faster for a single row.`, { x: M + 0.3, y: 5.55, w: W - 2 * M - 0.6, h: 0.9, fontFace: "Calibri", fontSize: 14, color: C.background1, valign: "top", margin: 0, isTextBox: true });
  }

  // ============================================================ 5 leakage-safe
  {
    const s = content("Leakage-safe by design", "Overview", "Every feature is computed at a snapshot date using only earlier events. Labels come from the window after the snapshot. Train, validation and test snapshots are separated in time.");
    // timeline 2021-01 .. 2025-12 (60 months)
    const x0 = 0.9, x1 = 12.4, y = 2.7;
    const mx = (yr, mo) => x0 + ((yr - 2021) * 12 + (mo - 1)) / 60 * (x1 - x0);
    s.addShape(pres.ShapeType.line, { x: x0, y, w: x1 - x0, h: 0, line: { color: "9FB0B8", width: 1.5 }, objectName: "timeline axis" });
    for (let yr = 2021; yr <= 2025; yr++) {
      s.addShape(pres.ShapeType.line, { x: mx(yr, 1), y: y - 0.08, w: 0, h: 0.16, line: { color: "9FB0B8", width: 1.25 }, objectName: "tick" });
      s.addText(String(yr), { x: mx(yr, 1) - 0.4, y: y + 0.12, w: 0.8, h: 0.28, fontFace: "Calibri", fontSize: 12, color: "5A6A73", align: "center", margin: 0, isTextBox: true });
    }
    const bar = (x, w, yy, color, label, tc) => {
      s.addShape(pres.ShapeType.roundRect, { x, y: yy, w, h: 0.5, rectRadius: 0.06, fill: { color }, line: { color }, objectName: label });
      s.addText(label, { x: x + 0.05, y: yy, w: w - 0.1, h: 0.5, fontFace: "Calibri", fontSize: 12, bold: true, color: tc || C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    };
    // train snapshots & their label windows
    bar(mx(2021, 7), mx(2022, 7) - mx(2021, 7), 1.45, C.accent1, "Train snapshots");
    bar(mx(2022, 7), mx(2023, 7) - mx(2022, 7), 1.45, C.accent5, "Train labels (365d)");
    bar(mx(2023, 7), mx(2024, 7) - mx(2023, 7), 3.2, C.accent3, "Validation labels");
    bar(mx(2024, 7), mx(2025, 7) - mx(2024, 7), 3.95, C.accent2, "Test labels");
    [[2023, 7, "Validation snapshot", C.accent3, 3.2], [2024, 7, "Test snapshot", C.accent2, 3.95]].forEach(([yr, mo, lab, col, yy]) => {
      s.addShape(pres.ShapeType.ellipse, { x: mx(yr, mo) - 0.1, y: y - 0.1, w: 0.2, h: 0.2, fill: { color: col }, line: { color: C.background1, width: 1.5 }, objectName: lab });
    });
    s.addShape(pres.ShapeType.ellipse, { x: mx(2022, 7) - 0.1, y: y - 0.1, w: 0.2, h: 0.2, fill: { color: C.accent1 }, line: { color: C.background1, width: 1.5 }, objectName: "last train snapshot" });
    s.addText("Features at a snapshot see only events before it. Labels come from the window after it.", { x: M, y: 4.65, w: W - 2 * M, h: 0.35, fontFace: "Calibri", fontSize: 14, italic: true, color: C.text1, margin: 0, isTextBox: true });
    const cards = [
      ["No future in features", "A cancellation counts only if it happened before the snapshot; a stay is complete only after check-out."],
      ["No window overlap", "Train label windows end before the validation snapshot; validation labels end before the test snapshot."],
      ["Proven by tests", "Deleting every post-cutoff event leaves all features unchanged; shuffled labels give an AUC near 0.5."],
    ];
    cards.forEach((c, i) => {
      const cw = (W - 2 * M - 0.4) / 3, x = M + i * (cw + 0.2), yy = 5.15;
      s.addShape(pres.ShapeType.roundRect, { x, y: yy, w: cw, h: 1.75, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "safeguard card" });
      if (ICON.shield) { s.addShape(pres.ShapeType.ellipse, { x: x + 0.15, y: yy + 0.15, w: 0.42, h: 0.42, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "icon circle" }); s.addImage({ data: ICON.shield, x: x + 0.26, y: yy + 0.26, w: 0.2, h: 0.2, altText: "shield" }); }
      s.addText(c[0], { x: x + 0.68, y: yy + 0.15, w: cw - 0.8, h: 0.42, fontFace: "Calibri", fontSize: 15, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
      s.addText(c[1], { x: x + 0.15, y: yy + 0.7, w: cw - 0.3, h: 1.0, fontFace: "Calibri", fontSize: 12, color: C.text1, valign: "top", margin: 0, isTextBox: true });
    });
  }

  // ============================================================ divider: results
  {
    const s = slide("SECTION_DARK", "Results");
    s.addText("Results across all flows", { placeholder: "title" });
    s.addText("Overall performance, model comparison, calibration and GPU efficiency", { placeholder: "subtitle" });
  }

  // ============================================================ 7 overall metrics
  {
    const s = content("Overall results: best model per flow", "Results", "Test-set metrics of the model chosen on validation PR-AUC. PR-AUC is not comparable across flows because base rates differ; use ROC-AUC and lift.");
    const sorted = [...bestRoc].sort((a, b) => a.roc - b.roc);
    s.addChart(pres.charts.BAR, [{ name: "Test ROC-AUC", labels: sorted.map((b) => b.f.short), values: sorted.map((b) => +b.roc.toFixed(3)) }], {
      x: M, y: 1.4, w: 5.5, h: 5.5, barDir: "bar", chartColors: [H.accent1], showTitle: true, title: "Test ROC-AUC, best model per flow", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
      valAxisMinVal: 0.5, valAxisMaxVal: 0.95, valAxisLabelFormatCode: "0.0", showLegend: false, barGapWidthPct: 45, ...CHART_BASE,
    });
    const rows = [[hdr("Flow"), hdr("Base", "right"), hdr("Best model"), hdr("PR-AUC", "right"), hdr("Lift@10%", "right"), hdr("LogLoss", "right")]];
    D.flows.forEach((f) => {
      const b = f.models[f.best_model];
      rows.push([tblCell(f.short, { bold: true }), tblCell(pct(f.base_rate, 0), { align: "right" }), tblCell(MODEL_SHORT[f.best_model]), tblCell(f3(b.pr_auc), { align: "right" }), tblCell(b.top_decile_lift.toFixed(2) + "x", { align: "right" }), tblCell(f3(b.logloss), { align: "right" })]);
    });
    s.addTable(rows, { x: 6.4, y: 1.45, w: 6.35, colW: [1.7, 0.7, 1.15, 0.9, 0.95, 0.95], rowH: 0.42, autoPage: false });
    s.addText("Read it as: top-decile lift is how many times more events the highest-scoring 10% of guests contain than average. Property repeat reaches 3.8x because its base rate is only 9%.", { x: 6.4, y: 5.55, w: 6.35, h: 1.2, fontFace: "Calibri", fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  }

  // ============================================================ 8 model comparison by flow
  {
    const s = content("Model comparison: ROC-AUC by flow", "Results", "The four model families land within a few thousandths of each other on most flows. The neural network is lower on churn, repeat and cancel flows.");
    const labels = D.flows.map((f) => f.short);
    const data = MODELS.map((m) => ({ name: MODEL_NAME[m], labels, values: D.flows.map((f) => +f.models[m].roc_auc.toFixed(3)) }));
    s.addChart(pres.charts.BAR, data, {
      x: M, y: 1.35, w: W - 2 * M, h: 4.6, barDir: "col", barGrouping: "clustered", chartColors: MODELS.map((m) => MODEL_HEX[m]), showLegend: true, legendPos: "t",
      valAxisMinVal: 0.5, valAxisMaxVal: 0.9, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", showValue: false, barGapWidthPct: 60, ...CHART_BASE, catAxisLabelFontSize: 12,
    });
    s.addText("Value axis starts at 0.5 (random ranking), so differences look larger than they are.", { x: M, y: 6.0, w: 6, h: 0.3, fontFace: "Calibri", fontSize: 11, italic: true, color: "5A6A73", margin: 0, isTextBox: true });
    s.addText(`Largest gap between best and worst model on any flow: ${(Math.max(...D.flows.map((f) => Math.max(...MODELS.map((m) => f.models[m].roc_auc)) - Math.min(...MODELS.map((m) => f.models[m].roc_auc)))) * 100).toFixed(1)} ROC points (always the neural network). Among the three tree models the largest gap is ${(Math.max(...D.flows.map((f) => { const v = ["xgboost", "lightgbm", "catboost"].map((m) => f.models[m].roc_auc); return Math.max(...v) - Math.min(...v); })) * 100).toFixed(1)} points.`, { x: M, y: 6.3, w: W - 2 * M, h: 0.6, fontFace: "Calibri", fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  }

  // ============================================================ 9 model comparison summary
  {
    const s = content("Model comparison: summary across flows", "Results", "Mean metrics across the eight flows, mean gap to the best model per flow, and how often each model was selected by validation PR-AUC.");
    const rows = [[hdr("Model"), hdr("ROC", "right"), hdr("PR-AUC", "right"), hdr("LogLoss", "right"), hdr("Lift@10%", "right"), hdr("Selected", "right")]];
    const bestOf = (k, hi) => MODELS.reduce((a, m) => ((hi ? A[m][k] > A[a][k] : A[m][k] < A[a][k]) ? m : a), MODELS[0]);
    const bR = bestOf("roc", true), bP = bestOf("pr", true), bL = bestOf("ll", false), bLi = bestOf("lift", true);
    MODELS.forEach((m) => {
      const a = A[m];
      const hl = (cond) => (cond ? { bold: true, fill: "F6EBD3" } : {});
      rows.push([tblCell(MODEL_NAME[m], { bold: true }), tblCell(f3(a.roc), { align: "right", ...hl(m === bR) }), tblCell(f3(a.pr), { align: "right", ...hl(m === bP) }), tblCell(f3(a.ll), { align: "right", ...hl(m === bL) }), tblCell(a.lift.toFixed(2) + "x", { align: "right", ...hl(m === bLi) }), tblCell(`${a.wins_val} of 8`, { align: "right" })]);
    });
    s.addTable(rows, { x: M, y: 1.45, w: 6.6, colW: [2.0, 0.8, 0.9, 0.95, 1.0, 0.95], rowH: 0.45, autoPage: false });
    s.addText("Selected = flows where the model had the best validation PR-AUC and so serves in production. Shaded = best mean value in the column.", { x: M, y: 3.95, w: 6.6, h: 0.7, fontFace: "Calibri", fontSize: 12, italic: true, color: "5A6A73", margin: 0, valign: "top", isTextBox: true });
    const take = [
      ["Pick on cost, not accuracy", "XGBoost, LightGBM and CatBoost are within noise of each other on a single test snapshot."],
      ["The NN earns its place only as a challenger", `It trails by ${f2(A.pytorch_nn.roc_gap_pp)} ROC points on average and never wins a flow.`],
    ];
    take.forEach((t, i) => {
      const y = 4.75 + i * 1.1;
      s.addText(t[0], { x: M, y, w: 6.6, h: 0.35, fontFace: "Calibri", fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
      s.addText(t[1], { x: M, y: y + 0.38, w: 6.6, h: 0.6, fontFace: "Calibri", fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
    });
    s.addChart(pres.charts.BAR, [{ name: "Mean ROC gap to best (pts)", labels: MODELS.map((m) => MODEL_SHORT[m]), values: MODELS.map((m) => +A[m].roc_gap_pp.toFixed(2)) }], {
      x: 7.6, y: 1.4, w: 5.15, h: 5.5, barDir: "col", chartColors: MODELS.map((m) => MODEL_HEX[m]), invertedColors: MODELS.map((m) => MODEL_HEX[m]), showTitle: true, title: "Mean ROC-AUC gap to the best model (points)",
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1.5, showLegend: false, barGapWidthPct: 50, ...CHART_BASE,
    });
  }

  // ============================================================ 10 calibration
  {
    const s = content("Calibration: scores you can read as probabilities", "Results", "Sigmoid or isotonic calibration is fit on a held-out half of the validation snapshot and chosen by log-loss. Calibration is not always better: eight models were neutral or marginally worse.");
    s.addChart(pres.charts.BAR, [
      { name: "Before calibration", labels: MODELS.map((m) => MODEL_SHORT[m]), values: MODELS.map((m) => +A[m].ece_raw.toFixed(4)) },
      { name: "After calibration", labels: MODELS.map((m) => MODEL_SHORT[m]), values: MODELS.map((m) => +A[m].ece.toFixed(4)) },
    ], { x: M, y: 1.4, w: 5.6, h: 3.6, barDir: "col", barGrouping: "clustered", chartColors: ["B7C4CA", H.accent1], showLegend: true, legendPos: "b", showTitle: true, title: "Mean expected calibration error (lower is better)", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 0.025, valAxisLabelFormatCode: "0.00", barGapWidthPct: 50, ...CHART_BASE });
    tile(s, M, 5.2, 2.7, 1.6, `${D.cal.improved_or_equal} / ${D.cal.n}`, "models improved or matched log-loss");
    tile(s, M + 2.9, 5.2, 2.7, 1.6, `${D.cal.sigmoid} / ${D.cal.isotonic}`, "sigmoid / isotonic chosen (isotonic only on 365-day flows)");
    const prop = flowBy("property_repeat");
    const file = IMG("outputs/plots/calibration_property_repeat.png");
    const p = placeImage(s, file, 6.5, 1.55, 6.25, 3.7);
    s.addText("Reliability diagrams for Same-Property Repeat, before (top) and after (bottom) calibration. Points on the diagonal mean the predicted probability matches the observed rate.", { x: 6.5, y: 1.55 + p.h + 0.2, w: 6.25, h: 0.9, fontFace: "Calibri", fontSize: 12, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  }

  // ============================================================ 11 GPU / efficiency
  {
    const s = content("GPU training cost per model", "Results", "Average Optuna tuning time per flow (15 trials) and final fit time. CatBoost is the slowest to tune; XGBoost is both fast and accurate. LightGBM ran on CPU in this environment.");
    s.addChart(pres.charts.BAR, [{ name: "Tuning time per flow (s)", labels: MODELS.map((m) => MODEL_SHORT[m]), values: MODELS.map((m) => +A[m].tune_s.toFixed(0)) }], {
      x: M, y: 1.4, w: 6.0, h: 4.0, barDir: "col", chartColors: MODELS.map((m) => MODEL_HEX[m]), invertedColors: MODELS.map((m) => MODEL_HEX[m]), showTitle: true, title: "Average tuning time per flow, 15 trials (seconds)",
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMinVal: 0, valAxisMaxVal: 130, showLegend: false, barGapWidthPct: 50, ...CHART_BASE,
    });
    const rows = [[hdr("Model"), hdr("Device"), hdr("Final fit", "right"), hdr("Peak GPU mem", "right")]];
    MODELS.forEach((m) => rows.push([tblCell(MODEL_SHORT[m], { bold: true }), tblCell(A[m].device === "cpu" ? "CPU (fallback)" : "CUDA"), tblCell(A[m].fit_s.toFixed(1) + " s", { align: "right" }), tblCell(A[m].device === "cpu" ? "n/a" : (A[m].peak_mb / 1000).toFixed(1) + " GB", { align: "right" })]));
    s.addTable(rows, { x: 7.0, y: 1.5, w: 5.75, colW: [1.5, 1.55, 1.2, 1.5], rowH: 0.45, autoPage: false });
    s.addText("Peak GPU memory is whole-device usage sampled from nvidia-smi, including about 1.3 GB of baseline use, so it overstates each model's own footprint. Timings are not a controlled CPU-versus-GPU comparison.", { x: 7.0, y: 4.0, w: 5.75, h: 1.4, fontFace: "Calibri", fontSize: 12, italic: true, color: "5A6A73", margin: 0, valign: "top", isTextBox: true });
    s.addShape(pres.ShapeType.roundRect, { x: M, y: 5.65, w: W - 2 * M, h: 1.2, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "callout" });
    s.addText([{ text: "LightGBM did not use the GPU. ", options: { bold: true, color: C.text2 } }, { text: "The installed build reports \"CUDA Tree Learner was not enabled\"; the trainer probes for CUDA, then OpenCL, then falls back to CPU and logs it. A CUDA-enabled build would be picked up automatically.", options: { color: C.text1 } }], { x: M + 0.25, y: 5.7, w: W - 2 * M - 0.5, h: 1.1, fontFace: "Calibri", fontSize: 14, valign: "middle", margin: 0, isTextBox: true });
  }

  // ============================================================ divider: flows
  {
    const s = slide("SECTION_DARK", "The eight flows");
    s.addText("The eight flows", { placeholder: "title" });
    s.addText("For each flow: the question, the metrics and model comparison, then a Streamlit walk-through from score to action", { placeholder: "subtitle" });
  }

  // ============================================================ per flow
  const fmtDrv = (d) => `${d.f} = ${d.v === null ? "missing" : d.v}`;
  for (const f of D.flows) {
    const b = f.models[f.best_model];
    // ---- overview
    {
      const s = content(`${f.n}. ${f.title}`, "The eight flows", `${f.question} Action: ${f.action}. Best model ${MODEL_NAME[f.best_model]} on validation; test ROC ${f3(b.roc_auc)}, PR-AUC ${f3(b.pr_auc)}, lift ${b.top_decile_lift.toFixed(2)}x.`);
      eyebrow(s, "THE QUESTION", M, 1.45, 4.3);
      s.addText(f.question, { x: M, y: 1.75, w: 4.3, h: 1.15, fontFace: "Calibri", fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true });
      eyebrow(s, "THE ACTION", M, 3.0, 4.3);
      s.addText(f.action, { x: M, y: 3.3, w: 4.3, h: 0.6, fontFace: "Calibri", fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true });
      s.addText(`${f.window_days}-day window  |  ${f.unit} level  |  base rate ${pct(f.base_rate)}`, { x: M, y: 3.95, w: 4.3, h: 0.3, fontFace: "Calibri", fontSize: 12, bold: true, color: C.accent1, margin: 0, isTextBox: true });
      eyebrow(s, "TOP DRIVERS (MEAN |SHAP|)", M, 4.45, 4.3);
      const mx = Math.max(...f.drivers.map((d) => d.v));
      f.drivers.forEach((d, i) => {
        const y = 4.8 + i * 0.44;
        s.addText(d.f, { x: M, y, w: 2.45, h: 0.34, fontFace: "Calibri", fontSize: 12, color: C.text1, valign: "middle", margin: 0, fit: "shrink", isTextBox: true });
        s.addShape(pres.ShapeType.rect, { x: M + 2.5, y: y + 0.07, w: Math.max(0.08, (d.v / mx) * 1.8), h: 0.2, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "driver bar" });
      });
      // KPI tiles
      const kp = [["ROC-AUC", f3(b.roc_auc)], ["PR-AUC", f3(b.pr_auc)], ["LogLoss", f3(b.logloss)], ["Brier", f3(b.brier)], ["Lift@10%", b.top_decile_lift.toFixed(2) + "x"]];
      const tw = 1.4, tg = 0.175;
      kp.forEach((k, i) => tile(s, 5.35 + i * (tw + tg), 1.45, tw, 1.15, k[1], k[0], { bigSize: 24, labelSize: 12 }));
      // model table
      const metr = [["roc_auc", "ROC-AUC", true, 3], ["pr_auc", "PR-AUC", true, 3], ["logloss", "LogLoss", false, 3], ["top_decile_lift", "Lift@10%", true, 2], ["train_time_s", "Fit (s)", false, 1]];
      const bestV = {};
      metr.forEach(([k, , hi]) => { const v = MODELS.map((m) => f.models[m][k]); bestV[k] = hi ? Math.max(...v) : Math.min(...v); });
      const rows = [[hdr("Model"), ...metr.map((m) => hdr(m[1], "right"))]];
      MODELS.forEach((m) => {
        const row = [tblCell(MODEL_SHORT[m] + (m === f.best_model ? "  (serving)" : ""), { bold: m === f.best_model })];
        metr.forEach(([k, , , d]) => { const v = f.models[m][k]; row.push(tblCell(v.toFixed(d), { align: "right", ...(Math.abs(v - bestV[k]) < 1e-9 ? { bold: true, fill: "F6EBD3" } : {}) })); });
        rows.push(row);
      });
      s.addTable(rows, { x: 5.35, y: 2.8, w: 7.4, colW: [2.0, 1.0, 1.0, 1.0, 1.2, 1.2], rowH: 0.36, autoPage: false });
      // decile lift chart
      s.addChart(pres.charts.BAR, [{ name: "Lift", labels: f.decile_lift.map((_, i) => String(i + 1)), values: f.decile_lift.map((v) => +v.toFixed(2)) }], {
        x: 5.35, y: 4.75, w: 7.4, h: 2.25, barDir: "col", chartColors: [H.accent2], showTitle: true, title: `Lift by score decile, ${MODEL_SHORT[f.best_model]} (1 = highest score; 1.0 = average)`, titleFontSize: 12,
        showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, showLegend: false, barGapWidthPct: 35, ...CHART_BASE, valAxisLabelFontSize: 10,
        valAxisMaxVal: Math.ceil(Math.max(...f.decile_lift) * 1.25),
      });
    }
    // ---- story
    {
      const st = f.story;
      const s = content(`${f.short} in action`, "The eight flows", `Streamlit walk-through for ${f.title}. Case ${st.case_no} (${st.case_name}): ${st.text} Score ${pct(st.score)} against a ${pct(st.base)} base rate. Next-best action: ${st.nba}.`);
      eyebrow(s, "CASE LADDER, FROM THE HELD-OUT TEST SET", M, 1.42, 4.8);
      const rows = [[hdr("Case"), hdr("Score", "right"), hdr("Actual")]];
      f.ladder.forEach((c, i) => {
        const on = i + 1 === st.case_no;
        rows.push([tblCell(`${i + 1}. ${c.name}`, { bold: on, fill: on ? "F6EBD3" : undefined, fontSize: 12 }), tblCell(pct(c.score), { align: "right", bold: on, fill: on ? "F6EBD3" : undefined, fontSize: 12 }), tblCell(c.actual ? "Event" : "No event", { bold: on, fill: on ? "F6EBD3" : undefined, fontSize: 12 })]);
      });
      const rh = 0.36;
      s.addTable(rows, { x: M, y: 1.75, w: 4.8, colW: [2.9, 0.8, 1.1], rowH: rh, autoPage: false });
      const ty = 1.75 + rh * rows.length + 0.25;
      const drv = st.drivers.slice(0, 3).map((d) => `${fmtDrv(d)} (${d.dir === "increases" ? "raises" : "lowers"} score)`).join(";  ");
      const steps = [
        ["1  Rank", `Case ${st.case_no} scores ${pct(st.score)} against a ${pct(st.base)} base rate (${(st.score / st.base).toFixed(1)}x), decile ${st.decile} of 10.` + (st.segment ? ` Uplift segment: ${st.segment}.` : "")],
        ["2  Explain", drv],
        ["3  Act", st.nba],
      ];
      steps.forEach((t, i) => {
        const y = ty + i * (i === 1 ? 1.05 : 0.95);
        s.addText(t[0], { x: M, y, w: 4.8, h: 0.3, fontFace: "Calibri", fontSize: 14, bold: true, color: C.accent1, margin: 0, isTextBox: true });
        s.addText(t[1], { x: M, y: y + 0.3, w: 4.8, h: i === 1 ? 0.75 : 0.62, fontFace: "Calibri", fontSize: 12, color: C.text1, valign: "top", margin: 0, isTextBox: true });
      });
      const img = IMG("presentation/" + st.image);
      const p = placeImage(s, img, 5.7, 1.45, 7.05, 4.9, { alt: `Streamlit scenario flows page for ${f.title}, case ${st.case_no}` });
      s.addText(`Streamlit, Scenario flows view, example ${st.case_no}`, { x: 5.7, y: 1.45 + p.h + 0.1, w: 7.05, h: 0.28, fontFace: "Calibri", fontSize: 11, italic: true, color: "5A6A73", margin: 0, isTextBox: true });
    }
    // ---- uplift slide after OTA flow
    if (f.key === "ota_to_direct") {
      const u = D.uplift;
      const s = content("OTA-to-Direct: who does the offer actually move?", "The eight flows", "Uplift T-Learner on GPU XGBoost, evaluated on a simulated randomized campaign. The ranking beats random and propensity targeting; the segment labels are only partly validated.");
      const p = placeImage(s, IMG("outputs/plots/uplift_ota_to_direct.png"), M, 1.45, 7.6, 3.4);
      tile(s, 8.6, 1.45, 4.15, 1.0, u.auuc_uplift.toFixed(0), "AUUC, uplift ranking", { bigSize: 26 });
      tile(s, 8.6, 2.6, 4.15, 1.0, u.auuc_random.toFixed(0), "AUUC, random targeting", { bigSize: 26 });
      tile(s, 8.6, 3.75, 4.15, 1.0, u.auuc_baseline_propensity.toFixed(0), "AUUC, ranking by propensity alone", { bigSize: 26 });
      const seg = u.segments;
      const rows = [[hdr("Segment"), hdr("Guests", "right"), hdr("Realised effect", "right")]];
      ["Persuadable", "Lost Cause", "Sure Thing", "Sleeping Dog"].forEach((k) => { if (seg[k]) rows.push([tblCell(k, { bold: true }), tblCell(seg[k].n, { align: "right" }), tblCell((seg[k].realised_uplift * 100 >= 0 ? "+" : "") + (seg[k].realised_uplift * 100).toFixed(1) + " pp", { align: "right" })]); });
      s.addTable(rows, { x: M, y: 5.2, w: 5.6, colW: [2.2, 1.4, 2.0], rowH: 0.34, autoPage: false });
      s.addText("Uplift ranks guests better than propensity does: targeting the most likely bookers is worse than random for incremental conversions. The four segment labels separate less cleanly: Persuadable and Lost Cause show similar realised effects, and Sleeping Dog is not negative. Use the continuous uplift score, not the buckets.", { x: 6.5, y: 5.2, w: 6.25, h: 1.75, fontFace: "Calibri", fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true });
    }
  }

  // ============================================================ serving
  {
    const s = slide("SECTION_DARK", "Serving and next steps");
    s.addText("Serving and next steps", { placeholder: "title" });
    s.addText("API, dashboard, and the limits of these results", { placeholder: "subtitle" });
  }
  {
    const s = content("Served through an API and a dashboard", "Serving and next steps", "FastAPI endpoints: /health, /predict/guest, /predict/batch, /leaderboard. Streamlit views: scenario flows, guest simulator, leaderboard, evaluation gallery. Inference runs on one dedicated thread because CUDA XGBoost models crashed across threads.");
    const a = placeImage(s, IMG("presentation/story_shots/api_predict_response_crop.png"), M, 1.45, 6.0, 3.9, { alt: "Swagger UI response for POST /predict/guest" });
    s.addText("POST /predict/guest in Swagger: calibrated scores, decile tiers and next-best actions for all 8 flows", { x: M, y: 1.45 + a.h + 0.1, w: 6.0, h: 0.5, fontFace: "Calibri", fontSize: 11, italic: true, color: "5A6A73", margin: 0, valign: "top", isTextBox: true });
    const d = placeImage(s, IMG("outputs/screenshots/streamlit/view_simulator.png"), 6.95, 1.45, 5.8, 3.9, { alt: "Streamlit guest simulator" });
    s.addText("Streamlit guest simulator: radar of all 8 propensities and recommended actions", { x: 6.95, y: 1.45 + d.h + 0.1, w: 5.8, h: 0.5, fontFace: "Calibri", fontSize: 11, italic: true, color: "5A6A73", margin: 0, valign: "top", isTextBox: true });
    const eps = [["GET /health", "CUDA status, GPU name, loaded models"], ["POST /predict/guest", "one guest, all 8 flows, optional SHAP"], ["POST /predict/batch", "vectorised scoring of many guests"], ["GET /leaderboard", "every flow by every model"]];
    eps.forEach((e, i) => {
      const x = M + i * 3.05;
      s.addText(e[0], { x, y: 5.95, w: 2.9, h: 0.3, fontFace: "Consolas", fontSize: 12, bold: true, color: C.accent1, margin: 0, isTextBox: true });
      s.addText(e[1], { x, y: 6.27, w: 2.85, h: 0.6, fontFace: "Calibri", fontSize: 12, color: C.text1, valign: "top", margin: 0, isTextBox: true });
    });
  }
  {
    const s = content("Limits and next steps", "Serving and next steps", "Be explicit that the data is synthetic, there is one test snapshot per flow, and the uplift effect is simulated.");
    const lim = [
      ["Synthetic data", "Absolute metrics and the strength of email-engagement signals are artifacts of the generator, not forecasts for a real portfolio."],
      ["One test snapshot per flow", "No confidence intervals; gaps of a few thousandths between tree models are within noise."],
      ["Uplift is simulated", "The campaign effect is generated, and the four segment labels are only partly validated."],
      ["Tuning budget", "15 Optuna trials on sub-sampled data; CatBoost and the neural network would gain most from more."],
    ];
    const nxt = [
      ["Run on real data", "Replace the generator with the property-management, CRS and web-analytics extracts; keep the snapshot logic."],
      ["Add rolling backtests", "Several test snapshots per flow to get intervals and detect drift."],
      ["Run a live holdout", "Randomised campaigns per flow to validate uplift and next-best actions."],
      ["Enable LightGBM on GPU", "Install a CUDA build so all four families use the GPU."],
    ];
    [["Limits", lim, C.accent6], ["Next steps", nxt, C.accent1]].forEach(([head, list, col], ci) => {
      const x = M + ci * 6.2, w = 5.9;
      s.addText(head, { x, y: 1.45, w, h: 0.4, fontFace: "Cambria", fontSize: 22, bold: true, color: C.text2, margin: 0, isTextBox: true });
      list.forEach((it, i) => {
        const y = 2.05 + i * 1.22;
        s.addShape(pres.ShapeType.ellipse, { x, y: y + 0.04, w: 0.36, h: 0.36, fill: { color: col }, line: { color: col }, objectName: "marker" });
        s.addText(String(i + 1), { x, y: y + 0.04, w: 0.36, h: 0.36, fontFace: "Calibri", fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
        s.addText(it[0], { x: x + 0.52, y, w: w - 0.52, h: 0.32, fontFace: "Calibri", fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true });
        s.addText(it[1], { x: x + 0.52, y: y + 0.34, w: w - 0.52, h: 0.8, fontFace: "Calibri", fontSize: 12, color: C.text1, valign: "top", margin: 0, isTextBox: true });
      });
    });
  }

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
