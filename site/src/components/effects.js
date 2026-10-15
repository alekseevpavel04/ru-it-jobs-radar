// What adding a skill does to the posted salary at equal role, grade, region, experience, format and other
// skills (results/models/skill_effects.json): the estimate as a dot with its 95 % interval, in three blocks
// by the Benjamini-Hochberg test over all skills, coloured as on the main page.
import * as Plot from "npm:@observablehq/plot";
import {html} from "npm:htl";
import {COLOR, int, signedPct} from "./labels.js";
import {DOWN, UP} from "./skills.js";

const NOISE = "#9a9890";
// The share of false findings the Benjamini-Hochberg test allows among the skills it marks.
export const FDR = 0.05;

const BLOCKS = [
  {title: "Повышают вилку", color: UP, test: (e) => e.q_bh < FDR && e.pct > 0},
  {title: "Влияние не доказано", color: NOISE, test: (e) => e.q_bh >= FDR},
  {title: "Понижают вилку", color: DOWN, test: (e) => e.q_bh < FDR && e.pct < 0}
];

function blockPlot(rows, {width, color, reach, name, row, font}) {
  // The dot and the interval band scale with the row height.
  const dot = Math.max(3, Math.min(6.5, row / 3.2));
  const band = Math.max(3, Math.min(8, row / 3.4));
  const narrow = width < 600;
  const right = narrow ? 112 : 190;
  return Plot.plot({
    width,
    height: rows.length * row + 16,
    marginLeft: narrow ? 118 : 200,
    marginRight: right,
    marginTop: 4,
    marginBottom: 8,
    x: {domain: [-reach, reach], axis: null},
    // On a phone a long skill name is cut short.
    y: {domain: rows.map(name), label: null, tickSize: 0, tickFormat: (t) => (narrow && t.length > 15 ? `${t.slice(0, 14)}…` : t)},
    style: {fontSize: `${narrow ? 12 : font}px`},
    marks: [
      Plot.ruleX([0], {stroke: "#9a9890"}),
      Plot.link(rows, {y1: name, y2: name, x1: "pct_ci_low", x2: "pct_ci_high", stroke: color, strokeOpacity: 0.35, strokeWidth: band, strokeLinecap: "round"}),
      Plot.dot(rows, {y: name, x: "pct", r: dot, fill: color, stroke: "white", strokeWidth: 1.5}),
      // An asterisk marks a found effect that the sensitivity run does not confirm (footnote below the chart).
      Plot.text(rows, {y: name, text: (e) => `${signedPct(e.pct)}${color !== NOISE && e.robust !== true ? "*" : ""}`, frameAnchor: "right", textAnchor: "end", dx: narrow ? 50 : 76, fill: color === NOISE ? COLOR.muted : color, fontWeight: 700}),
      Plot.text(rows, {y: name, text: (e) => int(e.rows_kept), frameAnchor: "right", textAnchor: "end", dx: right - 6, fill: COLOR.muted}),
    ]
  });
}

export function effectsChart(effects, {width, name, row = 16, font = 14}) {
  const rows = effects.slice().sort((a, b) => b.pct - a.pct);
  const reach = Math.max(...rows.map((e) => Math.max(Math.abs(e.pct_ci_low), Math.abs(e.pct_ci_high)))) * 1.05;
  const head = html`<div class="effects-head"><span>эффект</span><span>сравнимых<br>вакансий</span></div>`;
  const parts = BLOCKS.map((b) => [b, rows.filter(b.test)]).filter(([, part]) => part.length);
  const unconfirmed = rows.filter((e) => e.q_bh < FDR && e.robust !== true);
  const failed = unconfirmed.filter((e) => e.robust === false).map(name);
  const unchecked = unconfirmed.filter((e) => e.robust == null).map(name);
  const note = unconfirmed.length
    ? html`<p class="note-small">* Проверка без вакансий с одной границей вилки и с серединой от 500 тыс. руб.: ${[
        failed.length ? `эффект не доказан (${failed.join(", ")})` : "",
        unchecked.length ? `не оценивается, вакансий с навыком остаётся слишком мало (${unchecked.join(", ")})` : ""
      ].filter(Boolean).join("; ")}.</p>`
    : "";
  return html`<div>${head}${parts.map(
    ([b, part]) =>
      html`<div class="block-head" style=${`--block: ${b.color}`}>${b.title}</div>${blockPlot(part, {width, color: b.color, reach, name, row, font})}`
  )}${note}</div>`;
}
