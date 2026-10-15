// The skill rows of the main page: share of the recent years as a bar, the yearly line and the change
// since the base year, one row per skill. Built as HTML so the four columns line up on every width.
import {html, svg} from "npm:htl";
import {COLOR, int, pct, points} from "./labels.js";

export const GROUP_LABEL = {
  it_all: "Всё IT",
  it_junior: "Стажёры и джуны, всё IT",
  ml_all: "ML и Data Science",
  ml_junior: "Стажёры и джуны в ML"
};

function sparkline(series, up) {
  const w = 104, h = 26, pad = 3;
  const max = Math.max(...series.map((s) => s.share)) || 1;
  const x = (i) => pad + (i / Math.max(1, series.length - 1)) * (w - 2 * pad);
  const y = (v) => h - pad - (v / max) * (h - 2 * pad);
  const d = series.map((s, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(s.share).toFixed(1)}`).join("");
  const last = series.length - 1;
  const stroke = up ? COLOR.accent : COLOR.bar;
  return svg`<svg width=${w} height=${h} class="spark" aria-hidden="true">
    <path d=${d} fill="none" stroke=${stroke} stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"></path>
    <circle cx=${x(last)} cy=${y(series[last].share)} r="2.6" fill=${stroke}></circle>
  </svg>`;
}

export function skillRows(rows, {years, recentLabel, baseYear, head = true}) {
  const max = Math.max(...rows.map((r) => r.share));
  const title = (r) =>
    [`${r.skill}, ${recentLabel}: ${int(r.n)} из ${int(r.of)} (${pct(r.share)})`, ...r.series.map((s) => `${s.year}: ${int(s.n)} из ${int(s.of)} (${pct(s.share)})`)].join("\n");
  return html`<div class="skill-rows">
    ${head ? html`<div class="skill-row skill-head">
      <span>Навык</span><span>Доля постов, ${recentLabel}</span><span>${years[0]}-${years.at(-1)}</span><span>С ${baseYear}</span>
    </div>` : ""}
    ${rows.map((r) => {
      const up = r.change !== null && r.change > 0;
      return html`<div class="skill-row" title=${title(r)}>
        <span class="skill-name">${r.skill}</span>
        <span class="skill-bar"><i style=${`width: ${((r.share / max) * 100).toFixed(1)}%`}></i><b>${pct(r.share)}</b></span>
        <span class="skill-spark">${sparkline(r.series, up)}</span>
        <span class=${`skill-delta ${r.change === null ? "" : up ? "up" : "down"}`}>${r.change === null ? "" : points(r.change)}</span>
      </div>`;
    })}
  </div>`;
}
