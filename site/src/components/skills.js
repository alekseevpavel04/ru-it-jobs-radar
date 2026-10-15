// The skills of the main page in three blocks (rising, within noise, falling): a grey dot for the base
// year, a faint arrow to a dot for the recent years coloured by the change, green up and red down, the
// share written beside it and the change in points on the right. One x scale for all blocks, so dots
// compare across them.
import * as Plot from "npm:@observablehq/plot";
import {html} from "npm:htl";
import {COLOR, int, pct, points} from "./labels.js";

// Directions and grades of radar.site.skill_trends; a cell of the data is "<direction>_<grade>".
export const DIRECTION = {
  it: "Всё IT",
  backend: "Backend",
  frontend: "Frontend",
  qa: "Тестирование",
  analytics: "Аналитика",
  ml: "ML и Data Science",
  dataeng: "Инженерия данных",
  devops: "DevOps и администрирование",
  mobile: "Мобильная разработка"
};
export const GRADE_GROUP = {all: "Все грейды", junior: "Стажёры и джуны", middle: "Мидлы", senior: "Сеньоры и лиды"};

export const UP = "#1a7f37";
export const DOWN = "#cf222e";
const NOISE = "#9a9890";
const BASE = "#8c8a84";

// Change of two shares beyond the 95 % interval of their difference (normal approximation, both samples
// large here: hundreds to tens of thousands of posts).
export function change(base, recent) {
  const p1 = base.n / base.of, p2 = recent.n / recent.of;
  const se = Math.sqrt((p1 * (1 - p1)) / base.of + (p2 * (1 - p2)) / recent.of);
  return {diff: p2 - p1, significant: Math.abs(p2 - p1) > 1.96 * se};
}

export const changeColor = (c) => (!c.significant ? NOISE : c.diff > 0 ? UP : DOWN);
const changeText = (c) => (c.significant ? points(c.diff) : "≈");

const BLOCKS = [
  {title: "Растут", color: UP, test: (c) => c.significant && c.diff > 0},
  {title: "В пределах шума", color: NOISE, test: (c) => !c.significant},
  {title: "Падают", color: DOWN, test: (c) => c.significant && c.diff < 0}
];

function blockPlot(rows, {width, recentLabel, top, row, font}) {
  // Dots, arrow and label gap scale with the row height.
  const dot = Math.max(3.5, Math.min(7.5, row / 3.2));
  const gap = dot + 5;
  const narrow = width < 600;
  const base = (r) => r.base.share;
  const rises = (r) => r.share >= r.base.share;
  const right = narrow ? 72 : 120;
  const tip = (r) =>
    [`${r.skill}`, `${recentLabel}: ${int(r.n)} из ${int(r.of)} (${pct(r.share)})`, ...r.series.map((s) => `${s.year}: ${int(s.n)} из ${int(s.of)} (${pct(s.share)})`)].join("\n");
  return Plot.plot({
    width,
    height: rows.length * row + 16,
    marginLeft: narrow ? 96 : 160,
    marginRight: right,
    marginTop: 4,
    marginBottom: 8,
    x: {domain: [0, top * (narrow ? 1.5 : 1.12)], axis: null},
    y: {domain: rows.map((r) => r.skill), label: null, tickSize: 0},
    style: {fontSize: `${font}px`},
    marks: [
      Plot.arrow(rows, {y1: "skill", y2: "skill", x1: base, x2: "share", stroke: "#bdbab2", strokeWidth: 1.6, headLength: Math.min(6, row / 4), insetStart: 5, insetEnd: dot + 2}),
      Plot.dot(rows, {y: "skill", x: base, r: Math.max(2.5, dot / 1.9), fill: BASE}),
      Plot.dot(rows, {y: "skill", x: "share", r: dot, fill: (r) => changeColor(r.change), stroke: "white", strokeWidth: 1.5}),
      // The share beside the coloured dot, on the side away from the base dot.
      Plot.text(rows.filter(rises), {y: "skill", x: "share", text: (r) => pct(r.share), dx: gap, textAnchor: "start", fontWeight: 650, fill: COLOR.bar}),
      Plot.text(rows.filter((r) => !rises(r)), {y: "skill", x: "share", text: (r) => pct(r.share), dx: -gap, textAnchor: "end", fontWeight: 650, fill: COLOR.bar}),
      Plot.text(rows, {y: "skill", text: (r) => changeText(r.change), frameAnchor: "right", textAnchor: "end", dx: right - 6, fill: (r) => changeColor(r.change), fontWeight: 700}),
      Plot.tip(rows, Plot.pointerY({y: "skill", x: "share", title: tip}))
    ]
  });
}

// A tinted strip across the width with a bar of the block colour on the left.
const heading = (block) => html`<div class="block-head" style=${`--block: ${block.color}`}>${block.title}</div>`;

// Without a base year: the recent share alone, one dot per skill with the share beside it.
function nowPlot(rows, {width, recentLabel, row, font}) {
  const dot = Math.max(3.5, Math.min(7, row / 3.2));
  const narrow = width < 600;
  const top = Math.max(...rows.map((r) => r.share));
  return Plot.plot({
    width,
    height: rows.length * row + 16,
    marginLeft: narrow ? 96 : 160,
    marginRight: 16,
    marginTop: 4,
    marginBottom: 8,
    x: {domain: [0, top * 1.12], axis: null},
    y: {domain: rows.map((r) => r.skill), label: null, tickSize: 0},
    style: {fontSize: `${font}px`},
    marks: [
      Plot.ruleY(rows, {y: "skill", stroke: "#ebe9e4"}),
      Plot.dot(rows, {y: "skill", x: "share", r: dot, fill: COLOR.bar, stroke: "white", strokeWidth: 1.5}),
      Plot.text(rows, {y: "skill", x: "share", text: (r) => pct(r.share), dx: dot + 5, textAnchor: "start", fontWeight: 650, fill: COLOR.bar}),
      Plot.tip(rows, Plot.pointerY({y: "skill", x: "share", title: (r) => `${r.skill}
${recentLabel}: ${int(r.n)} из ${int(r.of)} (${pct(r.share)})`}))
    ]
  });
}

export function skillChart(rows, {width, recentLabel, baseYear, row = 30, font = 13}) {
  if (!baseYear) return nowPlot(rows, {width, recentLabel, row, font});
  const top = Math.max(...rows.map((r) => Math.max(r.share, r.base.share)));
  const legend = html`<div class="dot-legend"><span><i class="up-dot"></i><i class="down-dot"></i>сейчас, ${recentLabel}</span><span><i class="base"></i>${baseYear}</span><span>стрелка показывает, куда сдвинулась доля</span></div>`;
  const parts = BLOCKS.map((b) => [b, rows.filter((r) => b.test(r.change))]).filter(([, part]) => part.length);
  return html`<div>${legend}${parts.map(([b, part]) => html`${heading(b)}${blockPlot(part, {width, recentLabel, top, row, font})}`)}</div>`;
}
