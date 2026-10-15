// The skill rows of the main page in five looks (the author picks one): the share of the recent years
// against the share of the base year, and the change in percentage points, grey when sampling noise
// explains it. Rise is green, fall is red; the sign and the direction say the same for colour-blind readers.
import * as Plot from "npm:@observablehq/plot";
import {html} from "npm:htl";
import {COLOR, int, pct, points} from "./labels.js";

export const GROUP_LABEL = {
  it_all: "Всё IT",
  it_junior: "Стажёры и джуны, всё IT",
  ml_all: "ML и Data Science",
  ml_junior: "Стажёры и джуны в ML"
};

export const UP = "#1a7f37";
export const DOWN = "#cf222e";
const NOISE = "#9a9890";
const BASE = "#8c8a84";

// Change of two shares beyond the 95 % interval of their difference (normal approximation, both samples
// large here: hundreds to tens of thousands of posts).
export function change(base, recent, newBelow) {
  const p1 = base.n / base.of, p2 = recent.n / recent.of;
  const se = Math.sqrt((p1 * (1 - p1)) / base.of + (p2 * (1 - p2)) / recent.of);
  return {diff: p2 - p1, isNew: base.n < newBelow, significant: Math.abs(p2 - p1) > 1.96 * se};
}

export const changeColor = (c) => (!c.significant ? NOISE : c.diff > 0 ? UP : DOWN);
const changeText = (c, narrow) => (c.isNew ? (narrow ? "новый" : "новый навык") : !c.significant ? "≈" : points(c.diff));

export const VARIANTS = new Map([
  ["полоса и точка", "bar"],
  ["две точки", "dots"],
  ["гантель", "dumbbell"],
  ["стрелки", "arrows"],
  ["изменение", "delta"]
]);

const LEGEND = {
  bar: (recent, base) => html`<span><i class="bar"></i>${recent}</span><span><i class="base"></i>${base}</span>`,
  dots: (recent, base) => html`<span><i class="now"></i>${recent}</span><span><i class="hollow"></i>${base}</span>`,
  dumbbell: (recent, base) => html`<span><i class="now"></i>${recent}</span><span><i class="base"></i>${base}</span><span><i class="up-line"></i>рост</span><span><i class="down-line"></i>падение</span>`,
  arrows: (recent, base) => html`<span>стрелка от ${base} к ${recent}</span><span><i class="up-line"></i>рост</span><span><i class="down-line"></i>падение</span>`,
  delta: (recent, base) => html`<span>изменение доли с ${base} по ${recent}, п. п.</span><span><i class="up-line"></i>рост</span><span><i class="down-line"></i>падение</span>`
};

function marksFor(variant, rows) {
  const base = (r) => r.base.share;
  switch (variant) {
    case "bar":
      return [
        Plot.barX(rows, {y: "skill", x: "share", fill: COLOR.bar, rx: 5, insetTop: 8, insetBottom: 8}),
        Plot.dot(rows, {y: "skill", x: base, r: 5, fill: BASE, stroke: "white", strokeWidth: 1.5})
      ];
    case "dots":
      return [
        Plot.ruleY(rows, {y: "skill", stroke: "#ebe9e4"}),
        Plot.dot(rows, {y: "skill", x: base, r: 5, fill: "white", stroke: BASE, strokeWidth: 2}),
        Plot.dot(rows, {y: "skill", x: "share", r: 6.5, fill: COLOR.bar, stroke: "white", strokeWidth: 1.5})
      ];
    case "dumbbell":
      return [
        Plot.link(rows, {y1: "skill", y2: "skill", x1: base, x2: "share", stroke: (r) => changeColor(r.change), strokeWidth: 4, strokeLinecap: "round"}),
        Plot.dot(rows, {y: "skill", x: base, r: 5, fill: BASE, stroke: "white", strokeWidth: 1.5}),
        Plot.dot(rows, {y: "skill", x: "share", r: 6.5, fill: COLOR.bar, stroke: "white", strokeWidth: 1.5})
      ];
    case "arrows":
      return [
        Plot.ruleY(rows, {y: "skill", stroke: "#ebe9e4"}),
        Plot.arrow(rows, {y1: "skill", y2: "skill", x1: base, x2: "share", stroke: (r) => changeColor(r.change), strokeWidth: 2.6, headLength: 9, insetEnd: 1}),
        Plot.dot(rows, {y: "skill", x: base, r: 2.5, fill: BASE})
      ];
    case "delta":
      return [
        Plot.barX(rows, {y: "skill", x: (r) => r.change.diff * 100, fill: (r) => changeColor(r.change), rx: 4, insetTop: 7, insetBottom: 7}),
        Plot.ruleX([0], {stroke: "#6b6963"})
      ];
  }
}

export function skillChart(rows, {width, recentLabel, baseYear, variant}) {
  const narrow = width < 600;
  const delta = variant === "delta";
  const top = Math.max(...rows.map((r) => Math.max(r.share, r.base.share)));
  const reach = Math.max(...rows.map((r) => Math.abs(r.change.diff * 100)));
  const tip = (r) =>
    [`${r.skill}`, `${recentLabel}: ${int(r.n)} из ${int(r.of)} (${pct(r.share)})`, ...r.series.map((s) => `${s.year}: ${int(s.n)} из ${int(s.of)} (${pct(s.share)})`)].join("\n");
  const plot = Plot.plot({
    width,
    height: rows.length * 30 + 16,
    marginLeft: narrow ? 96 : 160,
    marginRight: narrow ? 120 : 200,
    marginTop: 4,
    marginBottom: 8,
    x: delta ? {domain: [-reach * 1.05, reach * 1.05], axis: null} : {domain: [0, top * 1.04], axis: null},
    y: {domain: rows.map((r) => r.skill), label: null, tickSize: 0},
    style: {fontSize: "13px"},
    marks: [
      ...marksFor(variant, rows),
      Plot.text(rows, {y: "skill", text: (r) => pct(r.share), frameAnchor: "right", textAnchor: "end", dx: narrow ? 46 : 76, fontWeight: 600, fill: COLOR.bar}),
      Plot.text(rows, {y: "skill", text: (r) => changeText(r.change, narrow), frameAnchor: "right", textAnchor: "end", dx: narrow ? 116 : 194, fill: (r) => changeColor(r.change), fontWeight: 700}),
      Plot.tip(rows, Plot.pointerY({y: "skill", x: delta ? (r) => r.change.diff * 100 : "share", title: tip}))
    ]
  });
  return html`<div><div class="dot-legend">${LEGEND[variant](recentLabel, baseYear)}</div>${plot}</div>`;
}
