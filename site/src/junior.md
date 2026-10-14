---
title: Начинающим
---

# Вакансии для начинающих

```js
import {ROLE, SOURCE_SHORT, SOURCE_COLOR, FAMILY_ORDER, CATEGORY, GRADE, COLOR, withSkillLabels, int, pct, share, dateLong, sourcePicker} from "./components/labels.js";
import {shownBounds, rangePlot} from "./components/salary.js";

const meta = FileAttachment("data/meta.json").json();
const junior = FileAttachment("data/junior.json").json();
const skillsJunior = FileAttachment("data/skills_junior.json").json().then(withSkillLabels);
const skillsAll = FileAttachment("data/skills.json").json().then(withSkillLabels);
const salaries = FileAttachment("data/salaries.json").json();
const salariesTg = FileAttachment("data/salaries_telegram.json").json();
const workFormat = FileAttachment("data/work_format.json").json();
const tgYears = FileAttachment("data/telegram_years.json").json();
```

<p class="lead">Начинающий здесь это стажёр или джуниор: грейд из заголовка вакансии или первой строки поста, у Хабр Карьеры квалификация, которую указал работодатель. Доли считаются среди вакансий, где грейд известен. <a href="./methodology#junior">Подробнее</a>.</p>

```js
const sum = (rows, key) => d3.sum(rows, (d) => d[key]);
const graded = (rows) => sum(rows, "graded_intern") + sum(rows, "graded_junior");
const unitOf = (s) => (s === "telegram" ? "постов" : "вакансий");
const ml = (s) => junior.filter((d) => d.source === s && d.role_bucket === "ml_ds");
```

<div class="grid grid-cols-4 facts">
${FAMILY_ORDER.map((s) => {
  const rows = junior.filter((d) => d.source === s);
  return html`<div class="card" style=${`border-top: 3px solid ${SOURCE_COLOR[s]}`}>
    <h2>${SOURCE_SHORT[s]}</h2>
    <div class="big">${pct(graded(rows) / sum(rows, "grade_known"))}</div>
    <p>${int(graded(rows))} из ${int(sum(rows, "grade_known"))} ${unitOf(s)} с грейдом</p>
    ${s === "trudvsem" ? html`<p class="sub">«без опыта» по полю портала: ${share(sum(rows, "no_experience_required"), sum(rows, "vacancies"))} вакансий, это другая величина</p>` : html`<p class="sub">грейд указан у ${share(sum(rows, "grade_known"), sum(rows, "vacancies"))}</p>`}
  </div>`;
})}
<div class="card accent-junior">
  <h2>ML и Data Science</h2>
  <div class="big">${int(graded(ml("habr")))} из ${int(sum(ml("habr"), "grade_known"))}</div>
  <p>ML/DS-вакансий Хабр Карьеры с грейдом</p>
  <p class="sub">Telegram: ${share(graded(ml("telegram")), sum(ml("telegram"), "grade_known"))} постов</p>
</div>
</div>

```js
const sourceInput = sourcePicker(Inputs, FAMILY_ORDER, "telegram");
const source = Generators.input(sourceInput);
const softInput = Inputs.toggle({label: "Без гибких навыков", value: true});
const noSoft = Generators.input(softInput);
```

<div class="filters">${sourceInput}${softInput}</div>

## В какой роли проще войти

```js
const MIN_GRADED = 30;
const all = junior.filter((d) => d.source === source);
const big = all.filter((d) => d.grade_known >= MIN_GRADED);
const small = all.filter((d) => d.grade_known < MIN_GRADED && d.grade_known > 0);
// a post with several vacancies is not a role: it is shown, grey, under the roles
const roleRows = [
  ...big
    .map((d) => ({role: ROLE[d.role_bucket], n: graded([d]), of: d.grade_known, rest: d.role_bucket === "multi_post"}))
    .sort((a, b) => a.rest - b.rest || b.n / b.of - a.n / a.of),
  ...(small.length ? [{role: `остальные (${small.length})`, n: graded(small), of: sum(small, "grade_known"), rest: true, names: small.map((d) => ROLE[d.role_bucket]).join(", ")}] : [])
];
const average = graded(all) / sum(all, "grade_known");
```

<div class="card">

```js
Plot.plot({
  width,
  height: roleRows.length * 24 + 40,
  marginLeft: width < 600 ? 150 : 230,
  marginRight: width < 600 ? 70 : 130,
  x: {axis: null, domain: [0, Math.max(average, d3.max(roleRows, (d) => d.n / d.of)) * 1.05]},
  y: {label: null, domain: roleRows.map((d) => d.role), tickFormat: (r) => (width < 600 && r.length > 24 ? r.slice(0, 23) + "…" : r)},
  marks: [
    Plot.barX(roleRows, {y: "role", x: (d) => d.n / d.of, fill: (d) => (d.rest ? COLOR.all : COLOR.junior), title: (d) => `${d.role}\n${share(d.n, d.of)} ${unitOf(source)} с грейдом${d.names ? `\n${d.names}` : ""}`, tip: true}),
    Plot.ruleX([average], {stroke: "#555", strokeDasharray: "3,3"}),
    Plot.text([average], {x: (d) => d, frameAnchor: "top", dy: -12, text: (d) => `в среднем ${pct(d)}`, fontSize: 11, fill: "#555"}),
    Plot.text(roleRows, {y: "role", x: (d) => d.n / d.of, text: (d) => (width < 600 ? pct(d.n / d.of) : share(d.n, d.of)), textAnchor: "start", dx: 4})
  ]
})
```

<p class="note-small">${SOURCE_SHORT[source]}: доля стажёров и джуниоров среди ${unitOf(source)} роли с известным грейдом. Роли, где грейд известен меньше чем у ${MIN_GRADED} ${unitOf(source)}, собраны в «остальные».</p>

</div>

## Что спрашивают у начинающих

```js
const SOFT = new Set(["soft", "language_human"]);
const juniorRows = skillsJunior.filter((d) => d.source === source && !(noSoft && SOFT.has(d.category)));
const base = juniorRows.length ? juniorRows[0].vacancies_graded_junior : 0;
const allShare = new Map(skillsAll.filter((d) => d.source === source).map((d) => [d.skill, d]));
const topJunior = juniorRows
  .slice(0, 20)
  .map((d) => {
    const a = allShare.get(d.skill);
    return {skill: d.skill, j: d.vacancies_mentioning / d.vacancies_graded_junior, jn: d.vacancies_mentioning, jof: d.vacancies_graded_junior, a: a ? a.vacancies_mentioning / a.vacancies_total : 0, an: a?.vacancies_mentioning ?? 0, aof: a?.vacancies_total ?? 0};
  })
  .sort((x, y) => y.j - y.a - (x.j - x.a));
const skillPoints = topJunior.flatMap((d) => [
  {skill: d.skill, who: "стажёр и джуниор", share: d.j, text: share(d.jn, d.jof)},
  {skill: d.skill, who: "все вакансии", share: d.a, text: share(d.an, d.aof)}
]);
```

<div class="card">

<p class="note-small">${base === 0 ? "В этом источнике нет вакансий с грейдом стажёр или джуниор." : `Двадцать самых частых навыков в ${int(base)} ${source === "telegram" ? "постах" : "вакансиях"} стажёров и джуниоров. Сверху то, что у начинающих спрашивают чаще, чем у всех.`}</p>

```js
base === 0
  ? html``
  : Plot.plot({
      width,
      height: topJunior.length * 24 + 50,
      marginLeft: width < 600 ? 130 : 200,
      x: {grid: true, label: source === "telegram" ? "доля постов" : "доля вакансий", tickFormat: pct, zero: true},
      y: {label: null, domain: topJunior.map((d) => d.skill)},
      color: {domain: ["стажёр и джуниор", "все вакансии"], range: [COLOR.junior, COLOR.all], legend: true},
      marks: [
        Plot.link(topJunior, {y1: "skill", y2: "skill", x1: "a", x2: "j", stroke: "#ddd", strokeWidth: 3}),
        Plot.dot(skillPoints, {y: "skill", x: "share", fill: "who", r: 5, title: (d) => `${d.skill}, ${d.who}\n${d.text}`, tip: true})
      ]
    })
```

</div>

## Сколько платят на старте

```js
const STEPS = ["intern", "junior", "middle"];
const ladder = [
  ...["trudvsem", "habr"].flatMap((s) => salaries.filter((d) => d.level === "grade" && d.source === s && STEPS.includes(d.grade)).map((d) => ({...d, src: s}))),
  ...salariesTg.filter((d) => d.level === "grade" && STEPS.includes(d.grade)).map((d) => ({...d, src: "telegram"}))
].sort((a, b) => FAMILY_ORDER.indexOf(a.src) - FAMILY_ORDER.indexOf(b.src) || STEPS.indexOf(a.grade) - STEPS.indexOf(b.grade));
const ladderShown = shownBounds(ladder, meta.salary_min_n, (d) => `${SOURCE_SHORT[d.src]} · ${GRADE[d.grade]}`);
const ladderHidden = ladder.filter((d) => d.with_from < meta.salary_min_n && d.with_to < meta.salary_min_n);
```

<div class="card">

<p class="note-small">Медиана вилки стажёра, джуниора и следующего шага, мидла, по источникам. Точка: медиана, отрезок: квартили.${ladderHidden.length ? ` Мало данных: ${ladderHidden.map((d) => `${SOURCE_SHORT[d.src]}, ${GRADE[d.grade]}`).join("; ")}.` : ""}</p>

```js
rangePlot(ladderShown, {width, labelWidth: 210, sort: false})
```

<p class="note-small">Вилка из объявления. В Telegram сумму, роль и грейд из поста читает модель извлечения. <a href="./salaries">Все зарплаты</a>.</p>

</div>

## Есть ли удалёнка

```js
// Juniors against the other grades. Levels region and grade_region of mart_salaries each cover every
// vacancy once, so the other grades are the source total minus the juniors.
const JUNIOR_GRADES = ["intern", "junior"];
const portalRemote = ["trudvsem", "habr"].map((s) => {
  const gr = salaries.filter((d) => d.level === "grade_region" && d.source === s && JUNIOR_GRADES.includes(d.grade));
  const reg = salaries.filter((d) => d.level === "region" && d.source === s);
  const jn = sum(gr.filter((d) => d.region_group === "remote"), "vacancies");
  const jof = sum(gr, "vacancies");
  const an = sum(reg.filter((d) => d.region_group === "remote"), "vacancies");
  const aof = sum(reg, "vacancies");
  return {label: SOURCE_SHORT[s], jn, jof, on: an - jn, oof: aof - jof, unit: "вакансий"};
});
const lastYear = d3.max(tgYears.work_format, (d) => d.year);
const tgRemote = (scope, year) => {
  const rows = tgYears.work_format.filter((d) => d.scope === scope && d.year === year);
  const j = rows.find((d) => d.group === "junior" && d.work_format === "remote");
  const o = rows.find((d) => d.group === "other" && d.work_format === "remote");
  return {jn: j?.posts ?? 0, jof: rows.find((d) => d.group === "junior")?.posts_with_format ?? 0, on: o?.posts ?? 0, oof: rows.find((d) => d.group === "other")?.posts_with_format ?? 0};
};
const remoteRows = [
  ...portalRemote,
  {label: `Telegram, ${lastYear}`, ...tgRemote("it", lastYear), unit: "постов"},
  {label: `Telegram, ML-роли, ${lastYear}`, ...tgRemote("ml", lastYear), unit: "постов"}
];
const remotePoints = remoteRows.flatMap((d) => [
  {label: d.label, who: "стажёр и джуниор", share: d.jn / d.jof, text: `${share(d.jn, d.jof)} ${d.unit}`},
  {label: d.label, who: "остальные грейды", share: d.on / d.oof, text: `${share(d.on, d.oof)} ${d.unit}`}
]);
const tgYearsRemote = [...new Set(tgYears.work_format.filter((d) => d.scope === "it").map((d) => d.year))].sort().flatMap((year) => {
  const r = tgRemote("it", year);
  return [
    {year, who: "стажёр и джуниор", share: r.jn / r.jof, text: share(r.jn, r.jof)},
    {year, who: "остальные грейды", share: r.on / r.oof, text: share(r.on, r.oof)}
  ];
});
```

<div class="grid grid-cols-2">
<div class="card">
<h3>Доля удалённых</h3>

```js
resize((width) => Plot.plot({
  width,
  height: remoteRows.length * 34 + 60,
  marginLeft: width < 400 ? 130 : 170,
  marginRight: 20,
  x: {grid: true, label: "доля удалённых", tickFormat: pct, zero: true},
  y: {label: null, domain: remoteRows.map((d) => d.label)},
  color: {domain: ["стажёр и джуниор", "остальные грейды"], range: [COLOR.junior, COLOR.all], legend: true},
  marks: [
    Plot.ruleY(remoteRows, {y: "label", stroke: "#eee"}),
    Plot.dot(remotePoints, {y: "label", x: "share", fill: "who", r: 6, title: (d) => `${d.label}, ${d.who}\n${d.text}`, tip: true})
  ]
}))
```

<p class="note-small">Портал и Хабр Карьера: вакансия удалённая, если так указан формат работы. Telegram: формат из поста, среди постов с грейдом и форматом.</p>

</div>
<div class="card">
<h3>Telegram по годам</h3>

```js
resize((width) => Plot.plot({
  width,
  height: 230,
  marginLeft: 44,
  x: {label: null, tickFormat: (y) => `${y}`, ticks: [...new Set(tgYearsRemote.map((d) => d.year))]},
  y: {grid: true, label: "доля удалённых", tickFormat: pct, zero: true},
  color: {domain: ["стажёр и джуниор", "остальные грейды"], range: [COLOR.junior, COLOR.all], legend: true},
  marks: [
    Plot.lineY(tgYearsRemote, {x: "year", y: "share", stroke: "who", strokeWidth: 2}),
    Plot.dot(tgYearsRemote, {x: "year", y: "share", fill: "who", r: 3, title: (d) => `${d.year}, ${d.who}\n${d.text} постов`, tip: true})
  ]
}))
```

<p class="note-small">${lastYear} по неделю с ${dateLong(tgYears.last_week)}.</p>

</div>
</div>

<details>
<summary>Удалёнка по ролям, ${SOURCE_SHORT[source]}</summary>

```js
const wf = d3
  .groups(workFormat.filter((d) => d.source === source), (d) => d.role_bucket)
  .map(([bucket, v]) => {
    const remote = v.find((d) => d.work_format === "remote")?.vacancies ?? 0;
    const known = d3.sum(v.filter((d) => d.work_format !== "unknown"), (d) => d.vacancies);
    return {role: ROLE[bucket], n: remote, of: known};
  })
  .filter((d) => d.of >= 30)
  .sort((a, b) => b.n / b.of - a.n / a.of);
display(
  wf.length === 0
    ? html`<p class="note-small">Формат работы в этом источнике почти не указан.</p>`
    : Plot.plot({
        width,
        height: wf.length * 22 + 20,
        marginLeft: width < 600 ? 150 : 230,
        marginRight: width < 600 ? 70 : 130,
        x: {axis: null, domain: [0, 1]},
        y: {label: null, domain: wf.map((d) => d.role), tickFormat: (r) => (width < 600 && r.length > 24 ? r.slice(0, 23) + "…" : r)},
        marks: [
          Plot.barX(wf, {y: "role", x: (d) => d.n / d.of, fill: SOURCE_COLOR[source]}),
          Plot.text(wf, {y: "role", x: (d) => d.n / d.of, text: (d) => (width < 600 ? pct(d.n / d.of) : share(d.n, d.of)), textAnchor: "start", dx: 4})
        ]
      })
);
```

<p class="note-small">Доля «удалённо» среди ${unitOf(source)} роли с указанным форматом, все грейды; роли, где формат указан меньше чем у 30, не показаны.</p>

</details>
