---
title: Зарплаты
toc: false
---

# Зарплаты в объявлениях

```js
import {ROLE, ROLE_GROUP, ROLE_GROUP_ORDER, SOURCE, GRADE, GRADE_ORDER, REGION, REGION_ORDER, int, pct, share, rub, dateLong} from "./components/labels.js";

const meta = FileAttachment("data/meta.json").json();
const cells = FileAttachment("data/salaries.json").json();
```

```js
const FAMILIES = ["trudvsem", "habr"];
const minN = meta.salary_min_n;
const sum = (rows, key) => d3.sum(rows, (d) => d[key]);
```

Зарплата здесь это вилка, которую работодатель написал в объявлении, а не то, сколько в итоге платят. Суммы взяты в рублях в месяц так, как их указал источник, вычет налога не учитывается. Вакансии с суммой в другой валюте в вилки не входят. На портале «Работа в России» верхняя граница часто совпадает с нижней: так устроена форма портала. Telegram на этой странице нет: зарплаты из постов пока не извлекаются.

<div class="grid grid-cols-2">
${FAMILIES.map((s) => {
  const rows = cells.filter((d) => d.source === s);
  return html`<div class="card">
    <h2>${SOURCE[s]}</h2>
    <div class="big">${pct(sum(rows, "with_salary_rub") / sum(rows, "vacancies"))}</div>
    вакансий указывают сумму в рублях: ${share(sum(rows, "with_salary_rub"), sum(rows, "vacancies"))}; в другой валюте ${int(sum(rows, "with_salary_other_currency"))}. Вакансии с ${dateLong(meta.periods[s].period_from)} по ${dateLong(meta.periods[s].period_to)}.
  </div>`;
})}
</div>

## Где зарплату указывают

```js
const shareSource = view(Inputs.radio(new Map(FAMILIES.map((s) => [SOURCE[s], s])), {value: "trudvsem", label: "Источник"}));
const shareBy = view(
  Inputs.radio(
    new Map([
      ["группа ролей", "group"],
      ["грейд", "grade"],
      ["регион", "region"]
    ]),
    {value: "group", label: "В разрезе"}
  )
);
```

```js
const keyOf = {group: (d) => ROLE_GROUP[d.role_bucket], grade: (d) => GRADE[d.grade], region: (d) => REGION[d.region_group]}[shareBy];
const orderOf = {group: ROLE_GROUP_ORDER, grade: GRADE_ORDER.map((g) => GRADE[g]), region: REGION_ORDER.map((r) => REGION[r])}[shareBy];
const stated = d3
  .rollups(
    cells.filter((d) => d.source === shareSource),
    (v) => ({n: sum(v, "with_salary_rub"), of: sum(v, "vacancies")}),
    keyOf
  )
  .map(([key, s]) => ({key, ...s, share: s.n / s.of}))
  .sort((a, b) => orderOf.indexOf(a.key) - orderOf.indexOf(b.key));
```

<div class="card">

${SOURCE[shareSource]}: доля вакансий, где указана сумма в рублях. Счётчики складываются, поэтому их можно показать в любом разрезе.

```js
Plot.plot({
  width,
  height: stated.length * 26 + 30,
  marginLeft: width < 600 ? 190 : 240,
  marginRight: width < 600 ? 80 : 140,
  x: {axis: null, domain: [0, 1]},
  y: {label: null, domain: stated.map((d) => d.key)},
  marks: [
    Plot.barX(stated, {y: "key", x: "share", fill: "#4269d0"}),
    Plot.text(stated, {
      y: "key",
      x: "share",
      text: (d) => (width < 600 ? `${int(d.n)} из ${int(d.of)}` : share(d.n, d.of)),
      textAnchor: "start",
      dx: 4
    })
  ]
})
```

</div>

## Вилки по ячейкам

Медианы и квартили считаются в витрине отдельно для каждой ячейки «источник, роль, грейд, регион» и между ячейками не складываются: медиану группы нельзя получить из медиан её частей. Поэтому вилки показаны по ячейкам, а фильтры только выбирают ячейки. Граница показана, если её указали хотя бы ${minN} вакансий ячейки: при меньшем числе 95-процентный интервал для медианы по порядковым статистикам совпадает со всем разбросом ячейки, от минимума до максимума (при ${minN} вакансиях интервал от второго до предпоследнего значения покрывает медиану с вероятностью ${pct(meta.salary_min_n_coverage)}).

```js
const pickSource = view(Inputs.radio(new Map(FAMILIES.map((s) => [SOURCE[s], s])), {value: "trudvsem", label: "Источник"}));
const pickGroup = view(Inputs.select(new Map([["все", null], ...ROLE_GROUP_ORDER.map((g) => [g, g])]), {label: "Группа ролей"}));
const pickGrade = view(Inputs.select(new Map([["все", null], ...GRADE_ORDER.map((g) => [GRADE[g], g])]), {label: "Грейд"}));
const pickRegion = view(Inputs.select(new Map([["все", null], ...REGION_ORDER.map((r) => [REGION[r], r])]), {label: "Регион"}));
```

```js
const selected = cells.filter(
  (d) =>
    d.source === pickSource &&
    (pickGroup === null || ROLE_GROUP[d.role_bucket] === pickGroup) &&
    (pickGrade === null || d.grade === pickGrade) &&
    (pickRegion === null || d.region_group === pickRegion)
);
const shown = selected
  .filter((d) => d.with_from >= minN || d.with_to >= minN)
  .map((d) => ({
    ...d,
    label: `${ROLE[d.role_bucket]} · ${GRADE[d.grade]} · ${REGION[d.region_group]}`,
    from: d.with_from >= minN ? d.salary_from_median : null,
    to: d.with_to >= minN ? d.salary_to_median : null
  }))
  .sort((a, b) => (b.from ?? b.to) - (a.from ?? a.to));
const hidden = selected.filter((d) => d.with_from < minN && d.with_to < minN);
const plotted = shown.slice().sort((a, b) => b.with_salary_rub - a.with_salary_rub).slice(0, 30).sort((a, b) => (b.from ?? b.to) - (a.from ?? a.to));
const bounds = plotted.flatMap((d) => [
  d.from === null ? null : {label: d.label, kind: "от", median: d.from, p25: d.salary_from_p25, p75: d.salary_from_p75, n: d.with_from},
  d.to === null ? null : {label: d.label, kind: "до", median: d.to, p25: d.salary_to_p25, p75: d.salary_to_p75, n: d.with_to}
]).filter(Boolean);
```

<div class="card">

${shown.length === 0 ? "В выбранном срезе нет ячейки, где границу вилки указали бы " + minN + " вакансий или больше." : `Ячеек с достаточным числом вакансий: ${int(shown.length)}${shown.length > plotted.length ? `, на графике ${plotted.length} самых крупных, остальные в таблице ниже` : ""}. Не показаны ${int(hidden.length)} ячеек, в них ${int(sum(hidden, "with_salary_rub"))} вакансий с суммой в рублях. Точка: медиана, отрезок: от 25-го до 75-го процентиля.`}

```js
shown.length === 0
  ? html``
  : Plot.plot({
      width,
      height: plotted.length * 30 + 60,
      marginLeft: width < 600 ? 10 : 360,
      marginTop: width < 600 ? 10 : 30,
      x: {grid: true, label: "рублей в месяц", tickFormat: (x) => int(x / 1000) + " тыс.", ticks: width < 600 ? 4 : 8},
      y: {label: null, domain: plotted.map((d) => d.label), axis: width < 600 ? null : "left"},
      color: {domain: ["от", "до"], range: ["#3ca951", "#4269d0"], legend: true},
      marks: [
        // "from" sits a little above the row line, "to" a little below, so both read on one row
        ...[["от", -4], ["до", 4]].flatMap(([kind, dy]) => {
          const rows = bounds.filter((d) => d.kind === kind);
          return [
            Plot.link(rows, {y1: "label", y2: "label", x1: "p25", x2: "p75", stroke: "kind", strokeWidth: 2, dy}),
            Plot.dot(rows, {
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
        width < 600 ? Plot.text(plotted, {y: "label", text: "label", frameAnchor: "left", dy: -13, fontSize: 10}) : null
      ]
    })
```

</div>

```js
Inputs.table(
  shown.map((d) => ({...d, role: ROLE[d.role_bucket], gradeLabel: GRADE[d.grade], region: REGION[d.region_group]})),
  {
    columns:
      width < 600
        ? ["role", "gradeLabel", "region", "from", "to"]
        : ["role", "gradeLabel", "region", "with_from", "from", "with_to", "to"],
    header: {
      role: "Роль",
      gradeLabel: "Грейд",
      region: "Регион",
      vacancies: "Вакансий",
      with_salary_rub: "С суммой",
      with_from: "n «от»",
      from: "Медиана «от»",
      with_to: "n «до»",
      to: "Медиана «до»"
    },
    format: {vacancies: int, with_salary_rub: int, with_from: int, with_to: int, from: rub, to: rub},
    // on a phone the columns keep their width and the table scrolls sideways instead of cutting names
    layout: width < 600 ? "fixed" : "auto",
    width: width < 600 ? {role: 170, gradeLabel: 120, region: 120, from: 100, to: 100} : undefined,
    sort: "with_from",
    reverse: true,
    rows: 12
  }
)
```
