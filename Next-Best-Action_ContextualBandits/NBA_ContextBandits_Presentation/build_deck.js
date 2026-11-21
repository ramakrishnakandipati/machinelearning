// Builds NBA_ContextualBandits_Presentation.pptx from the project's real outputs (assets/data.json).
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("C:/Users/RamaKrishna/AppData/Local/Claude-Data/claude_pzs8sxrjxfjjc/local-agent-mode-sessions/skills-plugin/c4087e75-6e8c-4493-8472-1adcd2a2603e/2c02675d-fb05-4126-b416-349c05b8005e/skills/pptx/scripts/apply_theme.js");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "assets", "data.json"), "utf8"));
const OUT = path.join(__dirname, "NBA_ContextualBandits_Presentation.pptx");
const IMG1 = path.join(__dirname, "assets", "01_data_distributions.png");

// ---------- theme ----------
const THEME = {
  name: "Resort Teal",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1A2B33", lt1: "FFFFFF", dk2: "0B3C49", lt2: "EEF3F2",
    accent1: "2A9AA8", accent2: "E4572E", accent3: "E9B224",
    accent4: "9A8CCB", accent5: "5FA55F", accent6: "9AA5AA",
    hlink: "2A9AA8", folHlink: "9A8CCB",
  },
};
// hex mirrors (charts only accept hex)
const H = { ink: "1A2B33", deep: "0B3C49", mist: "EEF3F2", muted: "5B6B73", grid: "D5DEDD",
  teal: "2A9AA8", coral: "E4572E", amber: "E9B224", violet: "9A8CCB", green: "5FA55F", gray: "9AA5AA", softTeal: "8FBCC2" };
const ARM_NAMES = ["Control", "10% discount", "Room upgrade", "$50 dining/spa", "Late C/O + transfer", "2x points"];
const ARM_HEX = [H.gray, H.amber, H.teal, H.coral, H.violet, H.green];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Audience optimization with contextual bandits";
pres.author = "Data Science";
const C = pres.SchemeColor;
const ARM_C = [C.accent6, C.accent3, C.accent1, C.accent2, C.accent4, C.accent5];

// ---------- numbers (all from the logs) ----------
const T = (k) => D.ope[k].true;
const logv = T("Logging policy"), unif = T("Uniform random"), ctrl = T("Always control");
const orc = T("Oracle (true best)"), best = T("LinUCB-Disjoint");
const pct = (a, b) => ((a / b - 1) * 100);
const liftLog = pct(best, logv), liftUnif = pct(best, unif), liftCtrl = pct(best, ctrl);
const ofOracle = (best / orc) * 100;
const gap = orc - logv;
const perGuest = best - logv;
const rep = D.replay;
const r0 = rep["LinUCB-Disjoint"][0][1], rN = rep["LinUCB-Disjoint"].slice(-1)[0][1];
const regretDrop = (1 - rN / r0) * 100;
const explCost = 0.05 * (best - unif);
const f0 = (x) => Math.round(x).toLocaleString("en-US");
const f1 = (x) => x.toFixed(1);

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "CONTENT", masterName: "CONTENT", background: { color: "FFFFFF" },
  slideNumber: { x: 12.35, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: C.text1, align: "right" },
  objects: [
    { text: { text: "Next-best-offer · contextual bandits", options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontSize: 10, color: C.text1, margin: 0, isTextBox: true } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.13, h: 1.25, fontSize: 32, bold: true, color: C.text2, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "DARK", masterName: "DARK", background: { color: "FFFFFF" },
  objects: [
    { rect: { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.text2 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 0.9, w: 7.2, h: 2.6, fontSize: 44, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});

// ---------- helpers ----------
const bul = (items, o = {}) => items.map((t, i) => ({ text: t, options: Object.assign({ bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 }, o) }));
function content(section, title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  s.addNotes(notes);
  return s;
}
function card(s, name, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background2 }, line: { color: fill || C.background2, width: 0 }, objectName: name });
}
function circle(s, name, x, y, d, fill, label, size) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: name });
  s.addText(label, { x, y, w: d, h: d, align: "center", valign: "middle", bold: true, fontSize: size || 16, color: C.background1, margin: 0, isTextBox: true, objectName: name + " label" });
}
function big(s, name, x, y, w, text, color, size) {
  s.addText(text, { x, y, w, h: 0.95, fontFace: "+mj-lt", fontSize: size || 48, bold: true, color: color || C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: name });
}
function lbl(s, name, x, y, w, h, text, o = {}) {
  s.addText(text, Object.assign({ x, y, w, h, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: name }, o));
}
function chartText() {
  return { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mj-lt",
    catAxisLabelColor: H.ink, valAxisLabelColor: H.muted, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 11, legendFontSize: 12,
    titleFontSize: 14, titleColor: H.deep, showTitle: true, valGridLine: { color: H.grid, size: 0.75 }, catGridLine: { style: "none" } };
}
function foot(s, text) { lbl(s, "Footnote", 0.6, 6.55, 12.1, 0.35, text, { fontSize: 11, color: C.text1, italic: true, valign: "middle" }); }

// =====================================================================
// 1  Title
// =====================================================================
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
  s.addText("The right offer for every guest, learned as we go", { placeholder: "title" });
  lbl(s, "Subtitle", 0.8, 3.7, 6.8, 1.0, "Audience optimization with contextual bandits for hospitality & resort offers", { fontSize: 20, color: C.background1 });
  lbl(s, "Data note", 0.8, 6.3, 7, 0.4, "Built on a synthetic guest simulator · dollar values are illustrative", { fontSize: 12, color: C.background2, italic: true });
  card(s, "Stat panel", 8.5, 0.9, 4.03, 5.4, C.accent1); // tinted via transparency below
  s.addShape(pres.ShapeType.roundRect, { x: 8.5, y: 0.9, w: 4.03, h: 5.4, rectRadius: 0.12, fill: { color: C.text2 }, line: { color: C.accent1, width: 1.5 }, objectName: "Stat panel frame" });
  big(s, "Stat lift", 8.85, 1.2, 3.4, `+${f0(liftLog)}%`, C.accent3, 60);
  lbl(s, "Stat lift label", 8.85, 2.2, 3.4, 0.9, "margin per guest vs. today's offer mix", { fontSize: 16, color: C.background1 });
  big(s, "Stat oracle", 8.85, 3.5, 3.4, `${f0(ofOracle)}%`, C.accent3, 60);
  lbl(s, "Stat oracle label", 8.85, 4.5, 3.4, 0.9, "of the best achievable (oracle) margin", { fontSize: 16, color: C.background1 });
  lbl(s, "Stat note", 8.85, 5.45, 3.4, 0.6, "LinUCB, 9,000 held-out guests", { fontSize: 12, color: C.background2, italic: true });
  s.addNotes(`Welcome. This deck shows how a contextual bandit chooses the best offer for each hotel guest and keeps learning from every booking. On 9,000 held-out guests the best policy lifts margin about ${f0(liftLog)}% over today's offer mix and reaches ${f0(ofOracle)}% of the best achievable result. All numbers come from a simulator, so dollars are illustrative. Likely question: is this real guest data? Answer: no. It is a simulator with known ground truth, which lets us check every estimate before touching real guests.`);
}

// =====================================================================
// 2  Business problem
// =====================================================================
{
  const s = content("Opening", `One-size-fits-all offers leave $${f0(gap)} of margin on the table per guest`,
    `Today's offers come from a rule-of-thumb logging policy that earns about $${f0(logv)} margin per guest. A perfect, all-knowing selector would earn $${f0(orc)}. That gap, $${f0(gap)} per guest or ${f0((gap / orc) * 100)}% of the ceiling, is the prize for personalization. About ${f0(D.conv * 100)}% of guests book, so small per-guest gains compound across volume. Likely question: is the oracle realistic? Answer: no, it is a ceiling that knows hidden truths; the realistic target is to close most of the gap, which our best policy does.`);
  const xs = [0.6, 4.75, 8.9], w = 3.83;
  const stats = [["$" + f0(logv), "Today's offer mix (logging policy)", C.text2], ["$" + f0(orc), "Best achievable per guest (oracle)", C.accent1], ["$" + f0(gap), `Gap left on the table (${f0((gap / orc) * 100)}% of ceiling)`, C.accent2]];
  stats.forEach((st, i) => {
    card(s, `Stat card ${i + 1}`, xs[i], 1.85, w, 2.1);
    big(s, `Stat ${i + 1}`, xs[i] + 0.3, 2.0, w - 0.6, st[0], st[2], 54);
    lbl(s, `Stat ${i + 1} label`, xs[i] + 0.3, 3.0, w - 0.6, 0.8, st[1], { fontSize: 16 });
  });
  // proportional bar: filled = today, hatched coral = gap
  const bx = 0.6, bw = 12.13, by = 4.75;
  lbl(s, "Bar caption", bx, 4.25, bw, 0.4, "Expected margin per guest, scaled to the oracle ceiling", { fontSize: 14, color: C.text1 });
  s.addShape(pres.ShapeType.rect, { x: bx, y: by, w: bw * (logv / orc), h: 0.7, fill: { color: C.text2 }, line: { color: C.text2, width: 0 }, objectName: "Today bar" });
  s.addShape(pres.ShapeType.rect, { x: bx + bw * (logv / orc), y: by, w: bw * (1 - logv / orc), h: 0.7, fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 }, objectName: "Gap bar" });
  s.addText("Today", { x: bx + 0.2, y: by, w: 3, h: 0.7, fontSize: 16, bold: true, color: C.background1, valign: "middle", margin: 0, isTextBox: true, objectName: "Today bar label" });
  s.addText("Gap", { x: bx + bw * (logv / orc), y: by, w: bw * (1 - logv / orc), h: 0.7, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "Gap bar label" });
  lbl(s, "Takeaway", 0.6, 5.7, 12.1, 0.7, `${f0(D.conv * 100)}% of ${f0(D.n)} logged guest sessions booked, so even a small per-guest gain compounds across volume.`, { fontSize: 16 });
}

// =====================================================================
// 3  Why A/B tests & segments fall short
// =====================================================================
{
  const s = content("Opening", "No single offer wins everywhere, so static tests and segments fall short",
    `Classic A/B tests pick one winner for everyone and burn traffic on losing arms for the whole test. Fixed segments are slow to build and miss interactions. Our data shows the problem: a 10% discount lifts Bronze guests by $${f0(D.tier.Bronze.true_r_a1 - D.tier.Bronze.true_r_a0)} but costs Platinum guests $${f0(D.tier.Platinum.true_r_a0 - D.tier.Platinum.true_r_a1)} versus no offer. One winner is wrong for somebody. Likely question: can't we just run more tests? Answer: yes, but each test costs weeks and traffic, and context interactions explode the number of tests needed.`);
  const items = [["Slow and costly", "Every test arm burns traffic for weeks."], ["One winner for all", "The average hides who gains and who loses."], ["Static segments", "Hand-built groups miss interactions and go stale."]];
  items.forEach((it, i) => {
    const y = 1.85 + i * 1.5;
    card(s, `Reason card ${i + 1}`, 0.6, y, 5.3, 1.3);
    circle(s, `Reason badge ${i + 1}`, 0.85, y + 0.3, 0.7, C.accent2, String(i + 1), 20);
    lbl(s, `Reason ${i + 1} head`, 1.8, y + 0.2, 3.9, 0.4, it[0], { fontSize: 18, bold: true, color: C.text2 });
    lbl(s, `Reason ${i + 1} text`, 1.8, y + 0.65, 3.9, 0.55, it[1], { fontSize: 14 });
  });
  const tb = D.tier.Bronze, tp = D.tier.Platinum;
  s.addChart(pres.charts.BAR, [
    { name: "Bronze guests", labels: ARM_NAMES, values: ARM_NAMES.map((_, i) => tb["true_r_a" + i]) },
    { name: "Platinum guests", labels: ARM_NAMES, values: ARM_NAMES.map((_, i) => tp["true_r_a" + i]) },
  ], Object.assign(chartText(), {
    x: 6.2, y: 1.75, w: 6.55, h: 4.6, barDir: "col", barGrouping: "clustered", chartColors: ["C98B4B", "5E7A87"],
    title: "Expected margin per guest by offer ($)", valAxisMinVal: 0, showLegend: true, legendPos: "b", showValue: false,
    catAxisLabelFontSize: 10,
  }));
  foot(s, "Ground-truth expected margin from the simulator, averaged over all 30,000 sessions.");
}

// =====================================================================
// 4  Bandits in plain language
// =====================================================================
pres.addSection({ title: "Method" });
{
  const s = content("Method", "A contextual bandit works like a concierge who learns each guest's taste",
    `Imagine a concierge who, for every arriving guest, offers what usually works for people like them, and occasionally tries something new to learn more. Context means what we know about the guest, such as loyalty tier or trip type. Exploit means offering the best-known option. Explore means trying an uncertain one. After each stay the concierge remembers what happened. Likely question: why explore at all? Answer: without exploring we lock in early guesses and never discover better offers, especially for under-sampled guest types.`);
  const defs = [["Context", "What we know about the guest: tier, trip type, channel, lead time.", C.accent1, "C"],
    ["Exploit", "Offer the option we currently believe earns the most.", C.accent5, "E"],
    ["Explore", "Try an uncertain option now to learn more for later.", C.accent3, "X"]];
  defs.forEach((d, i) => {
    const y = 1.85 + i * 1.5;
    circle(s, `Def badge ${i + 1}`, 0.6, y + 0.1, 0.8, d[2], d[3], 22);
    lbl(s, `Def head ${i + 1}`, 1.65, y, 4.4, 0.4, d[0], { fontSize: 20, bold: true, color: C.text2 });
    lbl(s, `Def text ${i + 1}`, 1.65, y + 0.45, 4.4, 0.9, d[1], { fontSize: 15 });
  });
  card(s, "Loop panel", 6.5, 1.85, 6.23, 4.5);
  const steps = [["See the guest", "Context arrives"], ["Pick an offer", "Mostly best, sometimes test"], ["Observe the stay", "Booked? margin?"], ["Learn", "Update expectations"]];
  steps.forEach((st, i) => {
    const y = 2.05 + i * 1.0;
    circle(s, `Loop badge ${i + 1}`, 6.85, y, 0.65, C.text2, String(i + 1), 18);
    lbl(s, `Loop head ${i + 1}`, 7.7, y - 0.02, 4.7, 0.4, st[0], { fontSize: 18, bold: true, color: C.text2 });
    lbl(s, `Loop text ${i + 1}`, 7.7, y + 0.35, 4.7, 0.4, st[1], { fontSize: 14 });
  });
  lbl(s, "Loop footer", 6.85, 6.05, 5.6, 0.3, "Then it repeats for the next guest.", { fontSize: 12, italic: true });
}

// =====================================================================
// 5  Formulation
// =====================================================================
{
  const s = content("Method", "We frame it as: guest in, offer out, margin back, and the policy improves",
    `Four pieces define the problem. Context x is a guest described by 25 model inputs after encoding. Action a is one of six offers including no offer. Reward r is net contribution margin: booking revenue after discount plus ancillary spend minus the cost of the offer. The policy pi is the rule that maps context to an offer, and learning means improving pi. Likely question: why margin and not conversion? Answer: a discount can raise bookings yet lower profit, so we optimize what the business actually earns.`);
  const cards = [["x", "Context", "Guest profile", "25 encoded inputs: tier, channel, archetype, stay, spend", C.accent1],
    ["a", "Action", "6 offers", "Control, discount, upgrade, dining/spa, late C/O, points", C.accent3],
    ["r", "Reward", "Net margin", "Booking revenue + ancillary − offer cost", C.accent2],
    ["π", "Policy", "π(a | x)", "Rule that maps each guest to an offer", C.text2]];
  cards.forEach((c, i) => {
    const x = 0.6 + i * 3.08, w = 2.9;
    card(s, `Piece card ${i + 1}`, x, 1.85, w, 3.5);
    circle(s, `Piece symbol ${i + 1}`, x + 0.3, 2.1, 0.95, c[4], c[0], 34);
    lbl(s, `Piece name ${i + 1}`, x + 0.3, 3.2, w - 0.6, 0.4, c[1], { fontSize: 20, bold: true, color: C.text2 });
    lbl(s, `Piece key ${i + 1}`, x + 0.3, 3.65, w - 0.6, 0.4, c[2], { fontSize: 16, bold: true, color: C.text1 });
    lbl(s, `Piece text ${i + 1}`, x + 0.3, 4.1, w - 0.6, 1.1, c[3], { fontSize: 14 });
  });
  lbl(s, "Goal line", 0.6, 5.7, 12.13, 0.7, "Goal: a policy that picks, for every guest, the offer with the highest expected margin.", { fontSize: 20, bold: true, color: C.text2 });
}

// =====================================================================
// 6  Data & features
// =====================================================================
{
  const s = content("Method", "30,000 simulated guest sessions with realistic, biased past offers",
    `We generated ${f0(D.n)} guest sessions with a known ground truth. Past offers were chosen by a context-dependent logging policy, so Bronze and OTA guests got discounts far more often than others. That bias is deliberate, because real history is biased too. We kept at least 8% logging probability for every offer, so every action has overlap. Likely question: why synthetic? Answer: it lets us compare every estimate to the true value of each action, something impossible with real data.`);
  const st = [[f0(D.n), "guest sessions (21,000 train / 9,000 test)"], ["25", "model inputs after encoding"], [f0(D.conv * 100) + "%", "of sessions booked"]];
  st.forEach((x, i) => {
    const y = 1.8 + i * 1.3;
    card(s, `Data card ${i + 1}`, 0.6, y, 4.6, 1.15);
    big(s, `Data stat ${i + 1}`, 0.85, y + 0.1, 1.9, x[0], C.accent1, 34);
    lbl(s, `Data label ${i + 1}`, 2.75, y + 0.15, 2.3, 0.9, x[1], { fontSize: 13, valign: "middle" });
  });
  lbl(s, "Bias note", 0.6, 5.75, 4.6, 0.8, "Past offers depended on guest type, so naive averages mislead.", { fontSize: 14, bold: true, color: C.text2 });
  s.addImage({ path: IMG1, x: 5.5, y: 1.7, w: 7.0, h: 4.84, objectName: "Data distributions figure", altText: "Distributions of guest context features and the biased logging-policy propensities by archetype, tier and channel" });
}

// =====================================================================
// 7  Algorithm trade-offs
// =====================================================================
{
  const s = content("Method", "LinUCB won, but simple linear methods and trees were close behind",
    `We trained ten policies in eight families. Four are shown here. LinUCB adds an optimism bonus for uncertain offers. Thompson Sampling samples plausible reward models and acts on the sample. Epsilon-greedy explores at a fixed random rate. The linear methods train in well under a second and landed within one to two percent of each other. Likely question: why not the most advanced neural model? Answer: on this problem it did not beat linear and tree models, and it is slower and harder to explain, so we prefer the simpler winner.`);
  const hdr = ["Family", "How it explores", "Strength", "Watch-out", "Value / guest"].map((t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: H.deep }, fontSize: 14, valign: "middle" } }));
  const rows = [
    ["LinUCB", "Optimism bonus on uncertain offers", "Fast, explainable, exact", "Assumes near-linear effects", "$" + f0(T("LinUCB-Disjoint"))],
    ["Linear Thompson Sampling", "Samples from a posterior", "Natural, smooth exploration", "Sampling noise early", "$" + f0(T("LinearTS"))],
    ["XGBoost ε-greedy", "Random offer a fixed % of time", "Captures interactions", "Blind exploration", "$" + f0(T("XGBoost-EpsGreedy"))],
    ["Neural UCB", "Gradient-based uncertainty", "Flexible, non-linear", "Heavier, harder to explain", "$" + f0(T("NeuralUCB"))],
  ].map((r, ri) => r.map((t, ci) => ({ text: t, options: { fontSize: 14, color: H.ink, valign: "middle", bold: ci === 0 || ci === 4, fill: { color: ri === 0 ? "D6EBEE" : "FFFFFF" } } })));
  s.addTable([hdr, ...rows], { x: 0.6, y: 1.85, w: 12.13, colW: [2.5, 2.9, 2.5, 2.5, 1.73], rowH: [0.5, 0.85, 0.85, 0.85, 0.85], border: { type: "solid", pt: 0.75, color: H.grid }, margin: [0.05, 0.12, 0.05, 0.12], objectName: "Algorithm comparison table" });
  foot(s, "Value = ground-truth expected margin per guest on 9,000 held-out guests. Six more policies are in the appendix.");
}

// =====================================================================
// 8  Architecture
// =====================================================================
{
  const s = content("Method", "The system scores, picks, serves, logs and updates in a closed loop",
    `The service works in five steps. It scores every offer for the guest, selects the best while mixing in 5% random exploration, serves it through the predict endpoint with the serving probability, logs the booking and margin through the reward endpoint, and refreshes the model every 25 rewards. Logging the probability is what makes later offline evaluation valid. Likely question: what if the model is wrong? Answer: exploration plus logged probabilities means we keep measuring, and the best model by doubly robust value is the one served.`);
  const steps = [["Score", "Expected margin for all 6 offers"], ["Select", "Best offer, plus 5% random exploration"], ["Serve", "POST /predict_nba returns offer and propensity"], ["Log reward", "POST /update_reward records margin"], ["Update", "Model refreshed every 25 rewards"]];
  const w = 2.25, gapx = 0.22;
  steps.forEach((st, i) => {
    const x = 0.6 + i * (w + gapx);
    card(s, `Step card ${i + 1}`, x, 1.95, w, 2.6);
    circle(s, `Step badge ${i + 1}`, x + 0.25, 2.2, 0.65, i === 4 ? C.accent2 : C.text2, String(i + 1), 18);
    lbl(s, `Step head ${i + 1}`, x + 0.25, 3.0, w - 0.5, 0.45, st[0], { fontSize: 20, bold: true, color: C.text2 });
    lbl(s, `Step text ${i + 1}`, x + 0.25, 3.5, w - 0.5, 0.95, st[1], { fontSize: 14 });
    if (i < 4) s.addShape(pres.ShapeType.rightArrow, { x: x + w + 0.01, y: 3.05, w: gapx - 0.02, h: 0.4, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: `Arrow ${i + 1}` });
  });
  const facts = [["Best model served", "LinUCB-Disjoint, chosen by doubly robust value"], ["Propensity in every log", "Makes unbiased offline evaluation possible"], ["FastAPI + dashboard", "Same service for decisions, feedback and monitoring"]];
  facts.forEach((f, i) => {
    const x = 0.6 + i * 4.1;
    card(s, `Fact card ${i + 1}`, x, 4.9, 3.93, 1.45);
    lbl(s, `Fact head ${i + 1}`, x + 0.25, 5.05, 3.45, 0.4, f[0], { fontSize: 16, bold: true, color: C.accent1 });
    lbl(s, `Fact text ${i + 1}`, x + 0.25, 5.5, 3.45, 0.8, f[1], { fontSize: 14 });
  });
}

// =====================================================================
// 9  Experiment design
// =====================================================================
pres.addSection({ title: "Results" });
{
  const s = content("Results", "We tested ten policies offline, then replayed them online against ground truth",
    `The experiment has three phases. First, each policy trains on 21,000 logged sessions. Second, we evaluate on 9,000 held-out sessions using four estimators and the simulator truth. Third, an online replay runs 20 rounds in which the current policy acts on fresh guests, is scored against ground truth for regret, then learns from that round's feedback. Four baselines frame the results. Likely question: why an oracle baseline? Answer: it shows the ceiling, so we can say what share of the achievable value each policy captures.`);
  const ph = [["Offline training", "21,000 logged sessions", "Each of 10 policies learns from biased history"], ["Offline evaluation", "9,000 held-out sessions", "DM, IPS, SNIPS and doubly robust vs. ground truth"], ["Online replay", "20 rounds", "Act, score vs. truth (regret), learn, repeat"]];
  ph.forEach((p, i) => {
    const x = 0.6 + i * 4.1;
    card(s, `Phase card ${i + 1}`, x, 1.85, 3.93, 2.55);
    circle(s, `Phase badge ${i + 1}`, x + 0.25, 2.1, 0.65, C.accent1, String(i + 1), 18);
    lbl(s, `Phase head ${i + 1}`, x + 0.25, 2.9, 3.45, 0.4, p[0], { fontSize: 20, bold: true, color: C.text2 });
    lbl(s, `Phase key ${i + 1}`, x + 0.25, 3.3, 3.45, 0.35, p[1], { fontSize: 14, bold: true });
    lbl(s, `Phase text ${i + 1}`, x + 0.25, 3.65, 3.45, 0.7, p[2], { fontSize: 13 });
  });
  lbl(s, "Baseline header", 0.6, 4.7, 12.1, 0.4, "Baselines we compare against", { fontSize: 18, bold: true, color: C.text2 });
  const bl = [["Always control", "no offer", C.accent6], ["Uniform random", "any offer, equally likely", C.accent3], ["Logging policy", "today's offer mix", C.accent1], ["Oracle", "knows the true best offer", C.accent2]];
  bl.forEach((b, i) => {
    const x = 0.6 + i * 3.08;
    s.addShape(pres.ShapeType.roundRect, { x, y: 5.2, w: 2.9, h: 1.1, rectRadius: 0.12, fill: { color: b[2] }, line: { color: b[2], width: 0 }, objectName: `Baseline chip ${i + 1}` });
    s.addText([{ text: b[0], options: { bold: true, fontSize: 16, breakLine: true } }, { text: b[1], options: { fontSize: 13 } }], { x: x + 0.15, y: 5.2, w: 2.6, h: 1.1, color: (i >= 2 ? C.background1 : C.text1), valign: "middle", margin: 0, isTextBox: true, objectName: `Baseline text ${i + 1}` });
  });
}

// =====================================================================
// 10  Results: leaderboard + regret curves
// =====================================================================
{
  const s = content("Results", `LinUCB earns ${f0(liftLog)}% more margin per guest, and its regret fell ${f0(regretDrop)}% in 20 rounds`,
    `Left: ground-truth margin per guest on 9,000 held-out guests. LinUCB earns $${f0(best)}, ${f0(liftLog)}% above today's offer mix, ${f0(liftUnif)}% above random and ${f0(liftCtrl)}% above no offer, and reaches ${f0(ofOracle)}% of the oracle. Right: in online replay, per-guest regret for LinUCB drops from about $${f0(r0)} in round 1 to about $${f0(rN)} by round 20. Neural Thompson Sampling learns far more slowly. Likely question: is the gap between top policies meaningful? Answer: the top four are within about 1% of each other, so choose on speed and explainability.`);
  const cats = ["Oracle (ceiling)", "LinUCB-Disjoint", "XGBoost-UCB", "Linear TS", "Neural UCB", "Logging policy", "Uniform random", "Always control"];
  const vals = [orc, best, T("XGBoost-UCB"), T("LinearTS"), T("NeuralUCB"), logv, unif, ctrl];
  s.addChart(pres.charts.BAR, [{ name: "Margin per guest", labels: cats, values: vals.map((v) => Math.round(v)) }], Object.assign(chartText(), {
    x: 0.6, y: 1.7, w: 6.0, h: 4.75, barDir: "bar", chartColors: [H.muted, H.deep, H.softTeal, H.softTeal, H.softTeal, H.gray, H.gray, H.gray], valGridLine: { style: "none" },
    catAxisOrientation: "maxMin", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 760, showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "$#,##0", dataLabelColor: H.ink, showLegend: false, title: "Expected margin per guest, held-out guests ($)", barGapWidthPct: 40,
  }));
  const L = rep["LinUCB-Disjoint"], T2 = rep["LinearTS"], X = rep["XGBoost-UCB"], N = rep["NeuralTS"];
  const labels = L.map((r) => String(r[0]));
  const ser = (name, arr) => ({ name, labels, values: arr.map((r) => Math.round(r[1] * 10) / 10) });
  s.addChart(pres.charts.LINE, [ser("LinUCB", L), ser("Linear TS", T2), ser("XGBoost-UCB", X), ser("Neural TS", N)], Object.assign(chartText(), {
    x: 6.9, y: 1.7, w: 5.85, h: 4.75, chartColors: [H.deep, H.teal, H.amber, H.gray], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    showLegend: true, legendPos: "b", title: "Regret per guest by replay round ($)", showCatAxisTitle: true, catAxisTitle: "Replay round", catAxisTitleFontSize: 11, catAxisTitleColor: H.muted,
    valAxisMinVal: 0, catAxisLabelFontSize: 10,
  }));
  foot(s, "Regret = value of the best offer minus the value of the offer chosen. Synthetic simulator; dollars are illustrative.");
}

// =====================================================================
// 11  Which offer wins for whom
// =====================================================================
{
  const s = content("Results", "The winning offer differs by guest group, which is why personalization pays",
    `Each row is a guest group and each cell is the true expected margin of an offer. The dark cell marks the winner. Dining credit wins for Bronze, Silver, App, Web and Family guests; an upgrade wins for Gold and Couples; double points for Platinum; late checkout for Business; and the discount only for OTA and Solo guests. A single offer would be wrong for most rows. Likely question: does the model actually find these? Answer: yes. LinUCB sends only ${f1(D.ope["LinUCB-Disjoint"].share[0] * 100)}% of guests to no-offer versus 18% in the logged history.`);
  const groups = [["Tier", "Bronze", D.tier], ["Tier", "Silver", D.tier], ["Tier", "Gold", D.tier], ["Tier", "Platinum", D.tier],
    ["Channel", "App", D.chan], ["Channel", "Web", D.chan], ["Channel", "OTA", D.chan],
    ["Type", "Business", D.seg], ["Type", "Couple", D.seg], ["Type", "Family", D.seg], ["Type", "Solo", D.seg]];
  const blend = (t) => {
    const a = [0xF4, 0xF8, 0xF7], b = [0x8F, 0xCB, 0xD0];
    return a.map((v, i) => Math.round(v + (b[i] - v) * t).toString(16).padStart(2, "0")).join("").toUpperCase();
  };
  const head = [{ text: "Guest group", options: { bold: true, color: "FFFFFF", fill: { color: H.deep }, fontSize: 12, valign: "middle" } }]
    .concat(ARM_NAMES.map((n) => ({ text: n, options: { bold: true, color: "FFFFFF", fill: { color: H.deep }, fontSize: 12, align: "center", valign: "middle" } })));
  const body = groups.map(([kind, name, src]) => {
    const vals = ARM_NAMES.map((_, i) => src[name]["true_r_a" + i]);
    const mx = Math.max(...vals), mn = Math.min(...vals);
    return [{ text: `${kind} · ${name}`, options: { bold: true, fontSize: 12, color: H.ink, valign: "middle", fill: { color: "FFFFFF" } } }].concat(vals.map((v) => {
      const isBest = v === mx;
      return { text: "$" + f0(v), options: { fontSize: 12, align: "center", valign: "middle", bold: isBest, color: isBest ? "FFFFFF" : H.ink, fill: { color: isBest ? H.deep : blend((v - mn) / (mx - mn || 1)) } } };
    }));
  });
  s.addTable([head, ...body], { x: 0.6, y: 1.65, w: 12.13, colW: [2.3, 1.64, 1.64, 1.64, 1.64, 1.64, 1.63], rowH: [0.5].concat(Array(11).fill(0.37)), border: { type: "solid", pt: 0.75, color: "FFFFFF" }, margin: [0.02, 0.08, 0.02, 0.08], objectName: "Offer-by-group heatmap table" });
  foot(s, "True expected margin per guest by offer and group; dark cell = best offer for that group. Row shading is relative within each row.");
}

// =====================================================================
// 12  Exploration cost and risk
// =====================================================================
{
  const s = content("Results", `Exploration costs about $${f1(explCost)} per guest, and the policy stops over-offering "nothing"`,
    `Serving mixes in 5% uniform random exploration. Using the held-out values, random offers earn about $${f0(best - unif)} less than LinUCB's choice, so exploring 5% of the time costs roughly $${f1(explCost)} per guest, about ${f1((explCost / best) * 100)}% of value. In exchange the policy keeps learning. The chart shows how the offer mix changes: no-offer drops from 18% to about 1%. Guardrails such as frequency caps and offer-cost caps are proposals, not yet tested. Likely question: can we cut exploration to zero? Answer: not safely, because drift would go unnoticed and new guest types would never be learned.`);
  const lm = D.logshare.map((v) => Math.round(v * 1000) / 10);
  const um = D.ope["LinUCB-Disjoint"].share.map((v) => Math.round(v * 1000) / 10);
  s.addChart(pres.charts.BAR, ARM_NAMES.map((n, i) => ({ name: n, labels: ["Logging policy (today)", "LinUCB-Disjoint"], values: [lm[i], um[i]] })), Object.assign(chartText(), {
    x: 0.6, y: 1.7, w: 7.3, h: 4.75, barDir: "bar", barGrouping: "percentStacked", chartColors: ARM_HEX, showValue: true, dataLabelPosition: "ctr",
    dataLabelFormatCode: '[>=3]0"%";""', dataLabelColor: H.ink, valAxisHidden: true, showLegend: true, legendPos: "b", legendFontSize: 11,
    title: "Share of guests receiving each offer", catAxisOrientation: "maxMin", barGapWidthPct: 45,
  }));
  const cards = [[`≈ $${f1(explCost)}`, "cost per guest of 5% random exploration", C.accent2], [`${f0(lm[0])}% → ${f0(um[0])}%`, "guests receiving no offer", C.accent1], ["≥ 0.08", "minimum logged offer probability, so every offer stays testable", C.text2]];
  cards.forEach((c, i) => {
    const y = 1.75 + i * 1.62;
    card(s, `Risk card ${i + 1}`, 8.25, y, 4.48, 1.45);
    big(s, `Risk stat ${i + 1}`, 8.5, y + 0.08, 4.0, c[0], c[2], 32);
    lbl(s, `Risk text ${i + 1}`, 8.5, y + 0.85, 4.0, 0.55, c[1], { fontSize: 13 });
  });
  foot(s, "Proposed guardrails (to be validated): per-guest frequency caps, offer-cost budgets, a standing control holdout.");
}

// =====================================================================
// 13  OPE validity
// =====================================================================
{
  const keys = ["Always control", "Logging policy", "LinUCB-Disjoint", "LinearTS", "XGBoost-UCB", "NeuralUCB", "Oracle (true best)"];
  const map = { "Always control": "Always control", "Logging policy": "Logging policy", "LinUCB-Disjoint": "LinUCB", "LinearTS": "Linear TS", "XGBoost-UCB": "XGBoost-UCB", "NeuralUCB": "Neural UCB", "Oracle (true best)": "Oracle" };
  const o = D.ope["LinUCB-Disjoint"];
  const err = (k) => pct(D.ope[k].dr, D.ope[k].true);
  const s = content("Results", "Offline estimates match the simulator truth within about 2%, so we can trust them",
    `Offline evaluation estimates how a new policy would perform using only logged data. We use four estimators and clip the importance weights at 20. The doubly robust estimate for LinUCB is $${f0(o.dr)} against a true $${f0(best)}, a gap of ${f1(err("LinUCB-Disjoint"))}%, and its 95% interval covers the truth. Neural UCB is over-estimated by ${f1(err("NeuralUCB"))}%, so we show intervals not just points. Likely question: will this hold on real data? Answer: not automatically. Real data has no ground truth, so we rely on overlap checks, effective sample size and a live pilot.`);
  s.addChart(pres.charts.BAR, [
    { name: "Doubly robust estimate", labels: keys.map((k) => map[k]), values: keys.map((k) => Math.round(D.ope[k].dr)) },
    { name: "Ground truth", labels: keys.map((k) => map[k]), values: keys.map((k) => Math.round(D.ope[k].true)) },
  ], Object.assign(chartText(), {
    x: 0.6, y: 1.7, w: 7.6, h: 4.75, barDir: "col", barGrouping: "clustered", chartColors: [H.teal, H.deep], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10,
    dataLabelFormatCode: "#,##0", dataLabelColor: H.ink, valAxisMinVal: 0, valAxisMaxVal: 750, valAxisHidden: true, showLegend: true, legendPos: "b",
    title: "Estimated vs. true margin per guest ($)", catAxisLabelFontSize: 11,
  }));
  const cards = [[`${f1(err("LinUCB-Disjoint"))}%`, "LinUCB estimate vs. truth"], [`$${f0(o.ci[0])}–${f0(o.ci[1])}`, "95% interval, covers the true $" + f0(best)], [`${f0(o.ess)} / 9,000`, "effective sample size after importance weighting"]];
  cards.forEach((c, i) => {
    const y = 1.75 + i * 1.62;
    card(s, `OPE card ${i + 1}`, 8.5, y, 4.23, 1.45);
    big(s, `OPE stat ${i + 1}`, 8.75, y + 0.08, 3.8, c[0], C.text2, 32);
    lbl(s, `OPE text ${i + 1}`, 8.75, y + 0.85, 3.8, 0.55, c[1], { fontSize: 13 });
  });
  foot(s, "Estimators: DM, IPS, SNIPS, DR with 200-sample bootstrap intervals; importance weights clipped at 20.");
}

// =====================================================================
// 14  Rollout plan
// =====================================================================
pres.addSection({ title: "Rollout" });
{
  const s = content("Rollout", "Roll out in four guarded phases, from shadow mode to always-on monitoring",
    `This plan is a proposal, not something the simulator validated. Phase one runs the model in shadow: it scores guests and logs what it would do, without serving. Phase two serves a small share of traffic with a control holdout. Phase three widens traffic and adds frequency and cost caps. Phase four is steady state with drift monitoring and weekly re-evaluation. Timelines and traffic shares are placeholders to fill with your team. Likely question: what stops it going wrong? Answer: the control holdout, caps and automatic fallback to the current rules.`);
  const ph = [["Shadow", ["Score and log only", "Compare estimates to actual outcomes"]], ["Pilot", ["Serve a small share [PLACEHOLDER %]", "Keep a control holdout"]], ["Scale", ["Widen traffic gradually", "Add frequency and cost caps"]], ["Monitor", ["Track reward, propensity, drift", "Re-evaluate and retrain regularly"]]];
  ph.forEach((p, i) => {
    const x = 0.6 + i * 3.08;
    card(s, `Rollout card ${i + 1}`, x, 1.95, 2.9, 3.6);
    circle(s, `Rollout badge ${i + 1}`, x + 0.25, 2.2, 0.7, i === 1 ? C.accent2 : C.text2, String(i + 1), 20);
    lbl(s, `Rollout head ${i + 1}`, x + 0.25, 3.1, 2.4, 0.45, p[0], { fontSize: 22, bold: true, color: C.text2 });
    s.addText(bul(p[1], { fontSize: 14, color: C.text1 }), { x: x + 0.25, y: 3.7, w: 2.45, h: 1.7, margin: 0, valign: "top", isTextBox: true, objectName: `Rollout bullets ${i + 1}` });
  });
  lbl(s, "Rollout note", 0.6, 5.85, 12.1, 0.6, "Proposal for discussion: phase lengths and traffic shares are placeholders to set with the business owners.", { fontSize: 14, italic: true });
}

// =====================================================================
// 15  Decision (dark)
// =====================================================================
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Rollout" });
  s.addText("Approve a guarded pilot of LinUCB next-best-offer", { placeholder: "title" });
  s.addNotes(`The ask: approve a guarded pilot, starting in shadow mode. In simulation, LinUCB earns about ${f0(liftLog)}% more margin per guest than today's mix and captures ${f0(ofOracle)}% of the best achievable. Scaled illustratively, $${f0(perGuest)} per guest is about $${f1((perGuest * 100000) / 1e6)} million per 100,000 guests, but that depends on real response patterns. Next steps: replace the simulator with real logged data, define guardrails, and start shadow mode. Likely question: how sure are we of the lift? Answer: it is a simulation estimate; the pilot with a control holdout gives the real number.`);
  const st = [[`+${f0(liftLog)}%`, "margin per guest vs. today (simulated)"], [`$${f0(perGuest)}`, "more per guest than today's offers"], [`$${f1((perGuest * 100000) / 1e6)}M`, "per 100,000 guests (illustrative)"]];
  st.forEach((x, i) => {
    big(s, `Impact stat ${i + 1}`, 0.8, 3.7 + i * 1.05, 2.9, x[0], C.accent3, 40);
    lbl(s, `Impact label ${i + 1}`, 3.8, 3.7 + i * 1.05, 4.0, 0.95, x[1], { fontSize: 16, color: C.background1, valign: "middle" });
  });
  card(s, "Next steps panel", 8.5, 0.9, 4.03, 5.4, C.text2);
  s.addShape(pres.ShapeType.roundRect, { x: 8.5, y: 0.9, w: 4.03, h: 5.4, rectRadius: 0.12, fill: { color: C.text2 }, line: { color: C.accent1, width: 1.5 }, objectName: "Next steps frame" });
  lbl(s, "Next steps head", 8.85, 1.2, 3.4, 0.5, "Next steps", { fontSize: 22, bold: true, color: C.accent3 });
  const ns = ["Replace simulator with real logged data [PLACEHOLDER]", "Set guardrails: caps, budgets, holdout", "Start shadow mode and compare to outcomes"];
  ns.forEach((t, i) => {
    circle(s, `Next badge ${i + 1}`, 8.85, 2.0 + i * 1.4, 0.55, C.accent1, String(i + 1), 16);
    lbl(s, `Next text ${i + 1}`, 9.6, 1.95 + i * 1.4, 2.75, 1.2, t, { fontSize: 15, color: C.background1 });
  });
}

// =====================================================================
// Appendix
// =====================================================================
pres.addSection({ title: "Appendix" });
{
  const s = content("Appendix", "Appendix A: the math behind LinUCB, Thompson Sampling and doubly robust evaluation",
    `Reference slide for technical questions. LinUCB scores each offer by its estimated reward plus an uncertainty bonus. Thompson Sampling draws a plausible parameter vector and acts greedily on it. The doubly robust estimator combines a reward model with importance weights, so it is accurate if either part is right. Likely question: what is alpha? Answer: it controls how much optimism we add; larger means more exploration.`);
  const eq = [["LinUCB", "a = argmax_a [ xᵀ θ̂ₐ + α · sqrt( xᵀ Aₐ⁻¹ x ) ]", "Estimated reward plus an uncertainty bonus; α sets how optimistic we are."],
    ["Thompson Sampling", "θ̃ₐ ~ N( θ̂ₐ , v² Aₐ⁻¹ ),  a = argmax_a xᵀ θ̃ₐ", "Sample a plausible reward model for each offer and act on it."],
    ["Doubly robust value", "V̂ = (1/n) Σᵢ [ Σₐ π(a|xᵢ) q̂(xᵢ,a) + π(aᵢ|xᵢ)/pᵢ · ( rᵢ − q̂(xᵢ,aᵢ) ) ]", "Reward model q̂ plus an importance-weighted correction; pᵢ is the logged probability."]];
  eq.forEach((e, i) => {
    const y = 1.8 + i * 1.5;
    card(s, `Equation card ${i + 1}`, 0.6, y, 12.13, 1.3);
    lbl(s, `Equation name ${i + 1}`, 0.9, y + 0.15, 3.0, 0.4, e[0], { fontSize: 16, bold: true, color: C.accent1 });
    lbl(s, `Equation ${i + 1}`, 0.9, y + 0.55, 11.5, 0.4, e[1], { fontSize: 18, bold: true, color: C.text2 });
    lbl(s, `Equation gloss ${i + 1}`, 0.9, y + 0.93, 11.5, 0.3, e[2], { fontSize: 13 });
  });
}
{
  const s = content("Appendix", "Appendix B: all ten policies and four baselines, ranked by true value",
    `Full leaderboard. DR and SNIPS are offline estimates, True is simulator ground truth, ESS is effective sample size after importance weighting. Likely question: why do some estimates differ from truth by more than others? Answer: policies that differ more from the logging policy get less overlap, so estimates are noisier, which is why we show intervals.`);
  const names = Object.keys(D.ope).sort((a, b) => D.ope[b].true - D.ope[a].true);
  const hd = ["Policy", "DR est.", "SNIPS est.", "True value", "ESS"].map((t, i) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: H.deep }, fontSize: 12, align: i ? "right" : "left", valign: "middle" } }));
  const rows = names.map((n) => [n, D.ope[n].dr, D.ope[n].snips, D.ope[n].true, D.ope[n].ess].map((v, i) => ({ text: i === 0 ? v : (i === 4 ? f0(v) : "$" + f1(v)), options: { fontSize: 12, color: H.ink, bold: n === "LinUCB-Disjoint", align: i ? "right" : "left", valign: "middle", fill: { color: n === "LinUCB-Disjoint" ? "D6EBEE" : "FFFFFF" } } })));
  s.addTable([hd, ...rows], { x: 0.6, y: 1.75, w: 12.13, colW: [4.0, 2.03, 2.03, 2.04, 2.03], rowH: [0.38].concat(Array(names.length).fill(0.3)), border: { type: "solid", pt: 0.5, color: H.grid }, margin: [0.02, 0.12, 0.02, 0.12], objectName: "Full leaderboard table" });
}
{
  const s = content("Appendix", "Appendix C: five tough questions and short answers",
    `Use this slide to rehearse. Each answer is deliberately short. The most important point to land is that results are simulator-based and the pilot with a control holdout is what converts them into a real number.`);
  const qa = [["Is the lift real?", "It is a simulator estimate. A pilot with a control holdout produces the real figure."],
    ["Why not just A/B test offers?", "Tests find one winner for everyone and are slow; the bandit personalizes and learns during serving."],
    ["Does exploration hurt revenue?", `About $${f1(explCost)} per guest at 5% exploration, and it keeps the system learning.`],
    ["Can we trust offline estimates?", `Within about 2% of truth here; on real data we add overlap checks, intervals and a live pilot.`],
    ["Why a simple linear winner?", "It matched or beat neural models, trains in under a second and is easy to explain."]];
  qa.forEach((q, i) => {
    const y = 1.7 + i * 0.97;
    card(s, `QA card ${i + 1}`, 0.6, y, 12.13, 0.85);
    lbl(s, `QA question ${i + 1}`, 0.85, y, 3.7, 0.85, q[0], { fontSize: 15, bold: true, color: C.text2, valign: "middle" });
    lbl(s, `QA answer ${i + 1}`, 4.6, y, 8.0, 0.85, q[1], { fontSize: 14, valign: "middle" });
  });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
