// Salary ranges of mart_salaries rows: a bound is drawn only when at least minN vacancies of the row
// state it (meta.salary_min_n, see radar.site.export.salary_min_n).
import * as Plot from "npm:@observablehq/plot";
import {int, rub} from "./labels.js";

export function shownBounds(rows, minN, label) {
  return rows
    .filter((d) => d.with_from >= minN || d.with_to >= minN)
    .map((d) => ({
      ...d,
      label: label(d),
      from: d.with_from >= minN ? d.salary_from_median : null,
      to: d.with_to >= minN ? d.salary_to_median : null
    }));
}

export function rangePlot(shown, {width, labelWidth = 330, sort = true} = {}) {
  const rows = sort ? shown.slice().sort((a, b) => (b.from ?? b.to) - (a.from ?? a.to)) : shown;
  const bounds = rows.flatMap((d) => [
    d.from === null ? null : {label: d.label, kind: "от", median: d.from, p25: d.salary_from_p25, p75: d.salary_from_p75, n: d.with_from},
    d.to === null ? null : {label: d.label, kind: "до", median: d.to, p25: d.salary_to_p25, p75: d.salary_to_p75, n: d.with_to}
  ]).filter(Boolean);
  const narrow = width < 600;
  return Plot.plot({
    width,
    height: rows.length * 30 + 60,
    marginLeft: narrow ? 10 : labelWidth,
    marginTop: narrow ? 14 : 30,
    x: {grid: true, label: "рублей в месяц", tickFormat: (x) => int(x / 1000) + " тыс.", ticks: narrow ? 4 : 8},
    y: {label: null, domain: rows.map((d) => d.label), axis: narrow ? null : "left"},
    color: {domain: ["от", "до"], range: ["#3ca951", "#4269d0"], legend: true},
    marks: [
      // "from" sits a little above the row line, "to" a little below, so both read on one row
      ...[["от", -4], ["до", 4]].flatMap(([kind, dy]) => {
        const part = bounds.filter((d) => d.kind === kind);
        return [
          Plot.link(part, {y1: "label", y2: "label", x1: "p25", x2: "p75", stroke: "kind", strokeWidth: 2, dy}),
          Plot.dot(part, {
            y: "label",
            x: "median",
            fill: "kind",
            r: 4,
            dy,
            title: (d) => `${d.label}\n${d.kind}: медиана ${rub(d.median)}, квартили ${rub(d.p25)} и ${rub(d.p75)}\nуказали ${int(d.n)} вакансий`,
            tip: true
          })
        ];
      }),
      narrow ? Plot.text(rows, {y: "label", text: "label", frameAnchor: "left", dy: -13, fontSize: 10}) : null
    ]
  });
}
