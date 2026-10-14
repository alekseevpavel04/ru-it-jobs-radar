// Salary ranges of mart_salaries rows: a bound is drawn only when at least minN vacancies of the row
// state it (meta.salary_min_n, see radar.site.export.salary_min_n).
import * as Plot from "npm:@observablehq/plot";
import * as Inputs from "npm:@observablehq/inputs";
import {COLOR, int, rub, thousands} from "./labels.js";

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

export function rangePlot(shown, {width, labelWidth = 230, sort = true, unit = "вакансий"} = {}) {
  const rows = sort ? shown.slice().sort((a, b) => (b.from ?? b.to) - (a.from ?? a.to)) : shown;
  const bounds = rows.flatMap((d) => [
    d.from === null ? null : {label: d.label, kind: "от", median: d.from, p25: d.salary_from_p25, p75: d.salary_from_p75, n: d.with_from},
    d.to === null ? null : {label: d.label, kind: "до", median: d.to, p25: d.salary_to_p25, p75: d.salary_to_p75, n: d.with_to}
  ]).filter(Boolean);
  const narrow = width < 600;
  // "from" and "to" are named once, next to the dots of the top row, instead of a legend
  const firstRow = rows[0]?.label;
  const named = ["от", "до"].map((kind) => bounds.find((b) => b.label === firstRow && b.kind === kind)).filter(Boolean);
  return Plot.plot({
    width,
    height: rows.length * (narrow ? 38 : 30) + 50,
    marginLeft: narrow ? 10 : labelWidth,
    marginRight: 20,
    marginTop: 26,
    x: {grid: true, label: "рублей в месяц", tickFormat: thousands, ticks: narrow ? 4 : 8},
    y: {label: null, domain: rows.map((d) => d.label), axis: narrow ? null : "left"},
    // on a phone the row names take the place above the rows, so "from" and "to" go to a legend
    color: {domain: ["от", "до"], range: [COLOR.from, COLOR.to], legend: narrow},
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
            title: (d) => `${d.label}\n${d.kind}: медиана ${rub(d.median)}, квартили ${rub(d.p25)} и ${rub(d.p75)}\nуказали ${int(d.n)} ${unit}`,
            tip: true
          })
        ];
      }),
      ...(narrow ? [] : [["от", -4], ["до", 5]]).map(([kind, dy]) =>
        Plot.text(named.filter((d) => d.kind === kind), {y: "label", x: (d) => d.p75 ?? d.median, text: "kind", dy, dx: 6, textAnchor: "start", fill: "kind", fontSize: 11, fontWeight: 600})
      ),
      narrow ? Plot.text(rows, {y: "label", text: "label", frameAnchor: "left", dy: -16, fontSize: 11}) : null
    ]
  });
}

// Exact numbers under a range plot; on a phone three columns, the count inside the cell.
export function salaryTable(rows, {width, header = "Роль", countHeader = "Вакансий"}) {
  const bound = (median, n) => (median === null ? "мало данных" : `${rub(median)} · n ${int(n)}`);
  const data = rows.map((d) => ({label: d.label, vacancies: d.vacancies, from: bound(d.from, d.with_from), to: bound(d.to, d.with_to)}));
  const narrow = width < 600;
  return Inputs.table(data, {
    columns: narrow ? ["label", "from", "to"] : ["label", "vacancies", "from", "to"],
    header: {label: header, vacancies: countHeader, from: "Медиана «от»", to: "Медиана «до»"},
    format: {vacancies: int},
    layout: narrow ? "fixed" : "auto",
    width: narrow ? {label: 130, from: 115, to: 115} : undefined,
    sort: null,
    rows: Math.min(data.length, 20) + 1
  });
}
