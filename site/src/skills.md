---
title: Навыки
toc: true
---

# Навыки в вакансиях

```js
import {ROLE, SOURCE, CATEGORY, int, pct, share, dateLong} from "./components/labels.js";

const meta = FileAttachment("data/meta.json").json();
const totals = FileAttachment("data/skills.json").json();
const byRole = FileAttachment("data/skills_by_role.json").json();
const junior = FileAttachment("data/junior.json").json();
```

Навык засчитывается вакансии, если его название или синоним из словаря встречается в заголовке, тексте или тегах навыков источника. Доля считается от всех вакансий источника и роли, включая те, где не нашлось ни одного навыка. Словарь пока ручной (версия 0), поэтому широкие синонимы завышают отдельные строки: например, слово «интеграция» засчитывается навыку API integration.

```js
const FAMILIES = ["trudvsem", "habr", "telegram"];
const periodText = (s) => `${dateLong(meta.periods[s].period_from)} по ${dateLong(meta.periods[s].period_to)}`;
const unitOf = (s) => (s === "telegram" ? "постов" : "вакансий");
const source = view(
  Inputs.radio(new Map(FAMILIES.map((s) => [SOURCE[s], s])), {value: "trudvsem", label: "Источник"})
);
```

```js
const buckets = junior
  .filter((d) => d.source === source)
  .sort((a, b) => b.vacancies - a.vacancies);
const sourceTotal = d3.sum(buckets, (d) => d.vacancies);
const role = view(
  Inputs.select(new Map([[`Все роли (${int(sourceTotal)})`, null], ...buckets.map((d) => [`${ROLE[d.role_bucket]} (${int(d.vacancies)})`, d.role_bucket])]), {
    label: "Роль"
  })
);
```

```js
const rows = (role === null
  ? totals.filter((d) => d.source === source).map((d) => ({...d, of: d.vacancies_total}))
  : byRole.filter((d) => d.source === source && d.role_bucket === role).map((d) => ({...d, of: d.vacancies_in_bucket}))
).map((d) => ({skill: d.skill, category: CATEGORY[d.category] ?? d.category, n: d.vacancies_mentioning, of: d.of, share: d.vacancies_mentioning / d.of}));
const top = rows.slice().sort((a, b) => b.n - a.n).slice(0, 25);
const base = role === null ? sourceTotal : buckets.find((d) => d.role_bucket === role).vacancies;
```

## Самые частые навыки

<div class="card">

${SOURCE[source]}${role === null ? "" : `, ${ROLE[role]}`}: доли от ${int(base)} ${unitOf(source)} с ${periodText(source)}.${role !== null ? ` Навыки, упомянутые меньше ${meta.skill_min_mentions} раз, для отдельной роли не показаны.` : ""}

```js
top.length === 0
  ? html`<p>Для этой роли нет навыков, упомянутых хотя бы ${meta.skill_min_mentions} раза.</p>`
  : Plot.plot({
      width,
      height: top.length * 22 + 30,
      marginLeft: width < 600 ? 170 : 200,
      marginRight: 90,
      x: {axis: null},
      y: {label: null, domain: top.map((d) => d.skill)},
      marks: [
        Plot.barX(top, {y: "skill", x: "share", fill: "#4269d0", title: (d) => `${d.skill} (${d.category})\n${share(d.n, d.of)}`, tip: true}),
        Plot.text(top, {y: "skill", x: "share", text: (d) => `${int(d.n)} · ${pct(d.share)}`, textAnchor: "start", dx: 4})
      ]
    })
```

</div>

## Все навыки выбранного среза

```js
Inputs.table(rows, {
  columns: width < 600 ? ["skill", "n", "share"] : ["skill", "category", "n", "share"],
  header: {skill: "Навык", category: "Категория", n: `Упоминаний (из ${int(base)})`, share: "Доля"},
  format: {n: int, share: pct},
  sort: "n",
  reverse: true,
  rows: 12
})
```

## Сравнение источников

```js
const shareBy = new Map(totals.map((d) => [`${d.source}|${d.skill}`, d.vacancies_mentioning / d.vacancies_total]));
const compared = d3
  .groups(totals, (d) => d.skill)
  .map(([skill]) => ({skill, best: d3.max(FAMILIES, (s) => shareBy.get(`${s}|${skill}`) ?? 0)}))
  .sort((a, b) => b.best - a.best)
  .slice(0, 20)
  .map((d) => d.skill);
const points = compared.flatMap((skill) =>
  FAMILIES.map((s) => ({skill, source: SOURCE[s], share: shareBy.get(`${s}|${skill}`) ?? 0, row: totals.find((d) => d.source === s && d.skill === skill)}))
);
```

<div class="card">

Двадцать навыков с наибольшей долей хотя бы в одном источнике. Разница между источниками отражает прежде всего их состав (перекосы описаны в методике), а не разные рынки.

```js
Plot.plot({
  width,
  height: compared.length * 24 + 50,
  marginLeft: width < 600 ? 170 : 200,
  x: {grid: true, label: "доля вакансий (в Telegram постов) источника", tickFormat: pct, zero: true},
  y: {label: null, domain: compared},
  color: {domain: FAMILIES.map((s) => SOURCE[s]), range: ["#4269d0", "#efb118", "#3ca951"], legend: true},
  marks: [
    Plot.ruleY(compared, {y: (d) => d, stroke: "#e0e0e0"}),
    Plot.dot(points, {
      y: "skill",
      x: "share",
      fill: "source",
      r: 5,
      title: (d) => `${d.skill}\n${d.source}: ${d.row ? share(d.row.vacancies_mentioning, d.row.vacancies_total) : "не упоминается"}`,
      tip: true
    })
  ]
})
```

</div>
