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

function blockPlot(rows, {width, color, reach, name}) {
  const narrow = width < 600;
  const right = narrow ? 112 : 190;
  return Plot.plot({
    width,
    height: rows.length * 30 + 16,
    marginLeft: narrow ? 118 : 200,
    marginRight: right,
    marginTop: 4,
    marginBottom: 8,
    x: {domain: [-reach, reach], axis: null},
    // On a phone a long skill name is cut short; the tooltip has it whole.
    y: {domain: rows.map(name), label: null, tickSize: 0, tickFormat: (t) => (narrow && t.length > 15 ? `${t.slice(0, 14)}…` : t)},
    style: {fontSize: "13px"},
    marks: [
      Plot.ruleX([0], {stroke: "#9a9890"}),
      Plot.link(rows, {y1: name, y2: name, x1: "pct_ci_low", x2: "pct_ci_high", stroke: color, strokeOpacity: 0.35, strokeWidth: 8, strokeLinecap: "round"}),
      Plot.dot(rows, {y: name, x: "pct", r: 6.5, fill: color, stroke: "white", strokeWidth: 1.5}),
      Plot.text(rows, {y: name, text: (e) => signedPct(e.pct), frameAnchor: "right", textAnchor: "end", dx: narrow ? 50 : 76, fill: color === NOISE ? COLOR.muted : color, fontWeight: 700}),
      Plot.text(rows, {y: name, text: (e) => int(e.rows_kept), frameAnchor: "right", textAnchor: "end", dx: right - 6, fill: COLOR.muted}),
      Plot.tip(
        rows,
        Plot.pointerY({
          y: name,
          x: "pct",
          title: (e) =>
            `${name(e)}: ${signedPct(e.pct)}, интервал от ${signedPct(e.pct_ci_low)} до ${signedPct(e.pct_ci_high)}\nсравнимых вакансий ${int(e.rows_kept)}: ${int(e.rows_kept_with_skill)} с навыком и ${int(e.rows_kept - e.rows_kept_with_skill)} без` +
            (e.q_bh >= FDR && (e.pct_ci_low > 0 || e.pct_ci_high < 0)
              ? "\nинтервал не задевает ноль, но с поправкой на число проверенных навыков эффект не подтверждён"
              : "")
        })
      )
    ]
  });
}

export function effectsChart(effects, {width, name}) {
  const rows = effects.slice().sort((a, b) => b.pct - a.pct);
  const reach = Math.max(...rows.map((e) => Math.max(Math.abs(e.pct_ci_low), Math.abs(e.pct_ci_high)))) * 1.05;
  const head = html`<div class="effects-head"><span>эффект</span><span>сравнимых<br>вакансий</span></div>`;
  const parts = BLOCKS.map((b) => [b, rows.filter(b.test)]).filter(([, part]) => part.length);
  return html`<div>${head}${parts.map(
    ([b, part]) =>
      html`<div class="block-head" style=${`--block: ${b.color}`}>${b.title}</div>${blockPlot(part, {width, color: b.color, reach, name})}`
  )}</div>`;
}
