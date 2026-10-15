// The skill rows of the main page as a dumbbell: the share of the base year and of the recent years on one
// line, green when it rose and red when it fell, grey when sampling noise explains the change. Four looks
// for the author to pick one; they differ in how the share is read and how "now" is marked.
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
  ["сетка", "grid"],
  ["доля у точки", "label"],
  ["цветная точка", "colored"],
  ["стрелка", "arrow"]
]);

const LEGEND = {
  grid: (now, base) => html`<span><i class="now"></i>сейчас, ${now}</span><span><i class="hollow"></i>${base}</span>`,
  label: (now, base) => html`<span><i class="now"></i>сейчас, ${now}, доля подписана</span><span><i class="hollow"></i>${base}</span>`,
  colored: (now, base) => html`<span><i class="up-dot"></i><i class="down-dot"></i>сейчас, ${now}</span><span><i class="base"></i>${base}</span>`,
  arrow: (now, base) => html`<span>от ${base} к «сейчас» (${now})</span>`
};

function marksFor(variant, rows) {
  const base = (r) => r.base.share;
  const line = Plot.link(rows, {y1: "skill", y2: "skill", x1: base, x2: "share", stroke: (r) => changeColor(r.change), strokeWidth: 4, strokeLinecap: "round"});
  const ring = Plot.dot(rows, {y: "skill", x: base, r: 5, fill: "white", stroke: BASE, strokeWidth: 2});
  const now = Plot.dot(rows, {y: "skill", x: "share", r: 7, fill: COLOR.bar, stroke: "white", strokeWidth: 1.5});
  switch (variant) {
    case "grid":
      return [line, ring, now];
    case "label":
      return [
        line,
        ring,
        now,
        Plot.text(rows.filter((r) => r.share >= r.base.share), {y: "skill", x: "share", text: (r) => pct(r.share), dx: 11, textAnchor: "start", fontWeight: 650, fill: COLOR.bar}),
        Plot.text(rows.filter((r) => r.share < r.base.share), {y: "skill", x: "share", text: (r) => pct(r.share), dx: -11, textAnchor: "end", fontWeight: 650, fill: COLOR.bar})
      ];
    case "colored":
      return [
        Plot.link(rows, {y1: "skill", y2: "skill", x1: base, x2: "share", stroke: "#d9d6cf", strokeWidth: 4, strokeLinecap: "round"}),
        Plot.dot(rows, {y: "skill", x: base, r: 4.5, fill: BASE}),
        Plot.dot(rows, {y: "skill", x: "share", r: 7.5, fill: (r) => changeColor(r.change), stroke: "white", strokeWidth: 1.5}),
        Plot.text(rows.slice(0, 1), {y: "skill", x: base, text: () => "2022", dy: -15, fill: BASE, fontWeight: 650}),
        Plot.text(rows.slice(0, 1), {y: "skill", x: "share", text: () => "сейчас", dy: -15, fill: COLOR.bar, fontWeight: 700})
      ];
    case "arrow":
      return [
        Plot.ruleY(rows, {y: "skill", stroke: "#ebe9e4"}),
        Plot.arrow(rows, {y1: "skill", y2: "skill", x1: base, x2: "share", stroke: (r) => changeColor(r.change), strokeWidth: 3, headLength: 10, insetEnd: 1}),
        Plot.dot(rows, {y: "skill", x: base, r: 3.5, fill: BASE}),
        Plot.text(rows.filter((r) => r.share >= r.base.share), {y: "skill", x: "share", text: (r) => pct(r.share), dx: 8, textAnchor: "start", fontWeight: 650, fill: COLOR.bar}),
        Plot.text(rows.filter((r) => r.share < r.base.share), {y: "skill", x: "share", text: (r) => pct(r.share), dx: -8, textAnchor: "end", fontWeight: 650, fill: COLOR.bar})
      ];
  }
}

export function skillChart(rows, {width, recentLabel, baseYear, variant}) {
  const narrow = width < 600;
  const top = Math.max(...rows.map((r) => Math.max(r.share, r.base.share)));
  // The grid look reads the share off the axis; the others keep a share column only when nothing on
  // the chart says the number.
  const shareColumn = variant === "colored";
  const axis = variant === "grid";
  const tip = (r) =>
    [`${r.skill}`, `${recentLabel}: ${int(r.n)} из ${int(r.of)} (${pct(r.share)})`, ...r.series.map((s) => `${s.year}: ${int(s.n)} из ${int(s.of)} (${pct(s.share)})`)].join("\n");
  const right = (shareColumn ? (narrow ? 50 : 80) : 0) + (narrow ? 72 : 120);
  const plot = Plot.plot({
    width,
    height: rows.length * 30 + (axis ? 56 : variant === "colored" ? 30 : 16),
    marginLeft: narrow ? 96 : 160,
    marginRight: right,
    marginTop: axis ? 28 : variant === "colored" ? 22 : 4,
    marginBottom: axis ? 28 : 8,
    x: axis
      ? {domain: [0, top * 1.04], grid: true, axis: "both", tickFormat: (x) => `${Math.round(x * 100)} %`, ticks: narrow ? 4 : 8, label: null}
      : {domain: [0, top * (variant === "colored" ? 1.04 : 1.12)], axis: null},
    y: {domain: rows.map((r) => r.skill), label: null, tickSize: 0},
    style: {fontSize: "13px"},
    marks: [
      ...marksFor(variant, rows),
      shareColumn ? Plot.text(rows, {y: "skill", text: (r) => pct(r.share), frameAnchor: "right", textAnchor: "end", dx: narrow ? 46 : 76, fontWeight: 600, fill: COLOR.bar}) : null,
      Plot.text(rows, {y: "skill", text: (r) => changeText(r.change, narrow), frameAnchor: "right", textAnchor: "end", dx: right - 6, fill: (r) => changeColor(r.change), fontWeight: 700}),
      Plot.tip(rows, Plot.pointerY({y: "skill", x: "share", title: tip}))
    ]
  });
  return html`<div><div class="dot-legend">${LEGEND[variant](recentLabel, baseYear)}</div>${plot}</div>`;
}
