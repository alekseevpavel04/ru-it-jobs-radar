// The skill rows of the main page: the share of the recent years as a bar, the share of the base year as
// a red dot on the same scale, and the relative change on the right, grey when sampling noise explains it.
import * as Plot from "npm:@observablehq/plot";
import {COLOR, int, pct} from "./labels.js";

export const GROUP_LABEL = {
  it_all: "Всё IT",
  it_junior: "Стажёры и джуны, всё IT",
  ml_all: "ML и Data Science",
  ml_junior: "Стажёры и джуны в ML"
};

export const BASE_DOT = "#e63946";
// Below this many base-year posts naming the skill a relative change is not worth a number.
const MIN_BASE_POSTS = 10;

// Change of two shares beyond the 95 % interval of their difference (normal approximation, both samples
// large here: hundreds to tens of thousands of posts).
export function change(base, recent) {
  const p1 = base.n / base.of, p2 = recent.n / recent.of;
  const se = Math.sqrt((p1 * (1 - p1)) / base.of + (p2 * (1 - p2)) / recent.of);
  return {relative: base.n >= MIN_BASE_POSTS ? p2 / p1 - 1 : null, significant: Math.abs(p2 - p1) > 1.96 * se, up: p2 > p1};
}

const changeText = (c) =>
  c.relative === null ? "с нуля" : !c.significant ? "≈" : `${c.up ? "+" : "−"}${int(Math.abs(Math.round(c.relative * 100)))} %`;
const changeColor = (c) => (!c.significant ? COLOR.muted : c.up ? COLOR.accent : COLOR.bar);

export function skillBars(rows, {width, recentLabel}) {
  const narrow = width < 600;
  const top = Math.max(...rows.map((r) => Math.max(r.share, r.base.share)));
  const tip = (r) =>
    [`${r.skill}`, `${recentLabel}: ${int(r.n)} из ${int(r.of)} (${pct(r.share)})`, ...r.series.map((s) => `${s.year}: ${int(s.n)} из ${int(s.of)} (${pct(s.share)})`)].join("\n");
  return Plot.plot({
    width,
    height: rows.length * 30 + 16,
    marginLeft: narrow ? 104 : 160,
    marginRight: narrow ? 116 : 170,
    marginTop: 4,
    marginBottom: 8,
    x: {domain: [0, top * 1.04], axis: null},
    y: {domain: rows.map((r) => r.skill), label: null, tickSize: 0},
    style: {fontSize: "13px"},
    marks: [
      Plot.barX(rows, {y: "skill", x: "share", fill: COLOR.bar, rx: 5, insetTop: 8, insetBottom: 8}),
      Plot.dot(rows, {y: "skill", x: (r) => r.base.share, r: 5, fill: BASE_DOT, stroke: "white", strokeWidth: 1.5}),
      Plot.text(rows, {y: "skill", text: (r) => pct(r.share), frameAnchor: "right", textAnchor: "end", dx: narrow ? 52 : 76, fontWeight: 600, fill: COLOR.bar}),
      Plot.text(rows, {y: "skill", text: (r) => changeText(r.change), frameAnchor: "right", textAnchor: "end", dx: narrow ? 112 : 164, fill: (r) => changeColor(r.change), fontWeight: 700}),
      Plot.tip(rows, Plot.pointerY({y: "skill", x: "share", title: tip}))
    ]
  });
}
