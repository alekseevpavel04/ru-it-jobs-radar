// The skills of the main page in three blocks (rising, within noise, falling): a grey dot for the base
// year, a faint arrow to a dot for the recent years coloured by the change, green up and red down.
// One x scale for all blocks, so dots compare across them. Four ways to show the numbers, for the
// author to pick one.
import * as Plot from "npm:@observablehq/plot";
import * as d3 from "npm:d3";
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
// A skill is new when the base year barely had it: under newBelow posts and under this share of the
// group's posts (in a small group ten posts can be a fair share, that is not "new").
const NEW_MAX_SHARE = 0.005;
// Width of a digit at 13px in the site font, to set a second label after a first one.
const CHAR = 7.2;

// Change of two shares beyond the 95 % interval of their difference (normal approximation, both samples
// large here: hundreds to tens of thousands of posts).
export function change(base, recent, newBelow) {
  const p1 = base.n / base.of, p2 = recent.n / recent.of;
  const se = Math.sqrt((p1 * (1 - p1)) / base.of + (p2 * (1 - p2)) / recent.of);
  return {diff: p2 - p1, isNew: base.n < newBelow && p1 < NEW_MAX_SHARE, significant: Math.abs(p2 - p1) > 1.96 * se};
}

export const changeColor = (c) => (!c.significant ? NOISE : c.diff > 0 ? UP : DOWN);
const changeText = (c, narrow) => (c.isNew ? (narrow ? "новый" : "новый навык") : !c.significant ? "≈" : points(c.diff));

export const LOOKS = new Map([
  ["колонки", "columns"],
  ["сетка", "grid"],
  ["доля у точки", "label"],
  ["доля и изменение у точки", "both"]
]);

function blockPlot(rows, {width, recentLabel, look, top, first, last}) {
  const narrow = width < 600;
  const base = (r) => r.base.share;
  const rises = (r) => r.share >= r.base.share;
  const shareColumn = look === "columns";
  const changeColumn = look !== "both";
  const right = (shareColumn ? (narrow ? 50 : 80) : 0) + (changeColumn ? (narrow ? 72 : 120) : 8);
  const grid = look === "grid";
  const axisTop = grid && first;
  const axisBottom = grid && last;
  const tip = (r) =>
    [`${r.skill}`, `${recentLabel}: ${int(r.n)} из ${int(r.of)} (${pct(r.share)})`, ...r.series.map((s) => `${s.year}: ${int(s.n)} из ${int(s.of)} (${pct(s.share)})`)].join("\n");
  // Labels beside the coloured dot, on the side away from the base dot; the change follows the share.
  const beside = [];
  if (look === "label" || look === "both") {
    beside.push(
      Plot.text(rows.filter(rises), {y: "skill", x: "share", text: (r) => pct(r.share), dx: 12, textAnchor: "start", fontWeight: 650, fill: COLOR.bar}),
      Plot.text(rows.filter((r) => !rises(r)), {y: "skill", x: "share", text: (r) => pct(r.share), dx: -12, textAnchor: "end", fontWeight: 650, fill: COLOR.bar})
    );
  }
  if (look === "both") beside.push(...offsetLabels(rows, rises, narrow));
  return Plot.plot({
    width,
    height: rows.length * 30 + 16 + (axisTop ? 24 : 0) + (axisBottom ? 24 : 0),
    marginLeft: narrow ? 96 : 160,
    marginRight: right,
    marginTop: axisTop ? 28 : 4,
    marginBottom: axisBottom ? 28 : 8,
    x: {
      domain: [0, top * (look === "label" || look === "both" ? (look === "both" ? 1.32 : 1.12) : 1.04)],
      grid,
      axis: axisTop && axisBottom ? "both" : axisTop ? "top" : axisBottom ? "bottom" : null,
      tickFormat: (x) => `${Math.round(x * 100)} %`,
      ticks: narrow ? 4 : 8,
      label: null
    },
    y: {domain: rows.map((r) => r.skill), label: null, tickSize: 0},
    style: {fontSize: "13px"},
    marks: [
      Plot.arrow(rows, {y1: "skill", y2: "skill", x1: base, x2: "share", stroke: "#bdbab2", strokeWidth: 1.6, headLength: 6, insetStart: 5, insetEnd: 9}),
      Plot.dot(rows, {y: "skill", x: base, r: 4, fill: BASE}),
      Plot.dot(rows, {y: "skill", x: "share", r: 7.5, fill: (r) => changeColor(r.change), stroke: "white", strokeWidth: 1.5}),
      ...beside,
      shareColumn ? Plot.text(rows, {y: "skill", text: (r) => pct(r.share), frameAnchor: "right", textAnchor: "end", dx: narrow ? 46 : 76, fontWeight: 600, fill: COLOR.bar}) : null,
      changeColumn ? Plot.text(rows, {y: "skill", text: (r) => changeText(r.change, narrow), frameAnchor: "right", textAnchor: "end", dx: right - 6, fill: (r) => changeColor(r.change), fontWeight: 700}) : null,
      Plot.tip(rows, Plot.pointerY({y: "skill", x: "share", title: tip}))
    ]
  });
}

// The change label: after the share label for a rise (one text mark per offset, since dx is a constant);
// for a fall, to the right of the base dot, so nothing runs into the skill names on the left.
function offsetLabels(rows, rises, narrow) {
  const gap = (r) => pct(r.share).length * CHAR + 20;
  const text = (r) => changeText(r.change, narrow);
  const style = {fill: (r) => changeColor(r.change), fontWeight: 700, fontSize: 12};
  const up = [...d3.group(rows.filter(rises), (r) => Math.round(gap(r))).values()].map((part) =>
    Plot.text(part, {y: "skill", x: "share", text, dx: gap(part[0]), textAnchor: "start", ...style})
  );
  const down = Plot.text(rows.filter((r) => !rises(r)), {y: "skill", x: (r) => r.base.share, text, dx: 10, textAnchor: "start", ...style});
  return [...up, down];
}

export function skillChart(rows, {width, recentLabel, baseYear, look}) {
  const top = Math.max(...rows.map((r) => Math.max(r.share, r.base.share)));
  const legend = html`<div class="dot-legend"><span><i class="up-dot"></i><i class="down-dot"></i>сейчас, ${recentLabel}</span><span><i class="base"></i>${baseYear}</span><span>стрелка показывает, куда сдвинулась доля</span></div>`;
  const blocks = [
    ["Растут", rows.filter((r) => r.change.significant && r.change.diff > 0)],
    ["В пределах шума", rows.filter((r) => !r.change.significant)],
    ["Падают", rows.filter((r) => r.change.significant && r.change.diff < 0)]
  ].filter(([, part]) => part.length);
  return html`<div>${legend}${blocks.map(
    ([title, part], i) =>
      html`<h3 class="skill-block">${title} <span>${part.length}</span></h3>${blockPlot(part, {width, recentLabel, look, top, first: i === 0, last: i === blocks.length - 1})}`
  )}</div>`;
}
