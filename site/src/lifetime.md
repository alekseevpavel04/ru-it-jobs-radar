---
title: Время жизни вакансии
---

# Сколько вакансия остаётся открытой

```js
import {LIFETIME, COLOR, int, pct, share, days, dateLong} from "./components/labels.js";

const life = FileAttachment("data/lifetime.json").json();
const hist = FileAttachment("data/vacancy_history.json").json();
```

<p class="lead">Сколько дней IT-вакансия портала «Работа в России» остаётся открытой, с учётом тех, что ещё не закрылись. <a href="./methodology#survival">Как считается</a>.</p>

```js
const median = (group, level) => life.medians.find((m) => m.group === group && m.level === level && m.cohort === "prevalent");
const overall = median("overall", "all");
const mlRow = median("role", "ml_ds_data");
const interval = (m) => (m.low === null ? "" : m.high === null ? `[${int(m.low)}; не оценивается]` : `[${int(m.low)}; ${int(m.high)}]`);
```

<div class="grid grid-cols-3">
  <div class="card">
    <h2>Медиана, все IT-вакансии</h2>
    <div class="big">${days(overall.median_days)}</div>
    <p class="sub">95 % интервал ${interval(overall)}; вакансий ${int(overall.subjects)}, закрылись ${int(overall.events)}</p>
  </div>
  <div class="card accent-ml">
    <h2>ML, DS и данные</h2>
    <div class="big">${days(mlRow.median_days)}</div>
    <p class="sub">95 % интервал ${interval(mlRow)}; вакансий ${int(mlRow.subjects)}, закрылись ${int(mlRow.events)}</p>
  </div>
  <div class="card">
    <h2>Закрытие не всегда навсегда</h2>
    <div class="big">${int(hist.reopened)} из ${int(hist.vacancies_tracked)}</div>
    <p class="sub">вакансий вернулись после закрытия; ещё ${int(hist.reposted)} из ${int(hist.closed)} закрытых перевыложены под новым номером. Пропавшую вакансию стоит проверить позже.</p>
  </div>
</div>

## Сколько вакансий ещё открыты

```js
const ROLE_ORDER = ["dev", "ml_ds_data", "analyst", "qa_devops_sys", "other_it"];
// The curve starts at 1 on day 0; the model reports the share still open on the listed days.
const withStart = (points) => [{day: 0, survival: 1}, ...points.filter((p) => p.day > 0)];
const curves = [
  {key: "all", label: LIFETIME.overall, points: withStart(life.overall_curve)},
  ...life.group_curves.filter((c) => c.group === "role").sort((a, b) => ROLE_ORDER.indexOf(a.level) - ROLE_ORDER.indexOf(b.level)).map((c) => ({key: c.level, label: LIFETIME[c.level], points: withStart(c.points)}))
];
const curvePoints = curves.flatMap((c) => c.points.map((p) => ({...p, key: c.key, label: c.label})));
const colorOf = (key) => (key === "all" ? COLOR.to : key === "ml_ds_data" ? COLOR.accent : "#c8c8c8");
const maxDay = d3.max(curvePoints, (p) => p.day);
```

<div class="card">

```js
Plot.plot({
  width,
  height: 320,
  marginLeft: 44,
  marginRight: width < 600 ? 20 : 170,
  x: {label: "дней с публикации", domain: [0, maxDay], ticks: curves[0].points.map((p) => p.day)},
  y: {grid: true, label: "ещё открыто", tickFormat: pct, domain: [0, 1]},
  marks: [
    ...curves
      .slice()
      .sort((a, b) => (a.key === "ml_ds_data" || a.key === "all") - (b.key === "ml_ds_data" || b.key === "all"))
      .map((c) => Plot.line(c.points, {x: "day", y: "survival", stroke: colorOf(c.key), strokeWidth: c.key === "all" || c.key === "ml_ds_data" ? 2 : 1.2})),
    Plot.dot(curvePoints, {x: "day", y: "survival", r: (p) => (p.key === "all" || p.key === "ml_ds_data" ? 4 : 2.5), fill: (p) => colorOf(p.key), title: (p) => `${p.label}\nчерез ${days(p.day)} открыто ${pct(p.survival)}`, tip: true}),
    width < 600
      ? null
      : Plot.text(curves.filter((c) => c.key === "all" || c.key === "ml_ds_data"), {x: maxDay, y: (c) => c.points.at(-1).survival, text: "label", textAnchor: "start", dx: 8, fill: (c) => colorOf(c.key), fontWeight: 600})
  ]
})
```

```js
const at = (c, d) => c.points.find((p) => p.day === d)?.survival;
const monthDrop = (c) => 1 - at(c, 60) / at(c, 30);
```

<p class="note-small">Точки: доля ещё открытых через ${curves[0].points.filter((p) => p.day > 0).map((p) => p.day).join(", ")} дней (оценка Каплана-Мейера), линия между точками только соединяет их. Тёмная: все IT-вакансии, красная: ML, DS и данные, серые: остальные группы ролей. Из открытых на 30-й день к 60-му закрываются ${pct(monthDrop(curves[0]))} (ML, DS и данные ${pct(monthDrop(curves.find((c) => c.key === "ml_ds_data")))}): портал снимает вакансии примерно через месяц после публикации.</p>

</div>

## От чего зависит срок

```js
// group and level order of the model (src/radar/models/survival.py, COVARIATES)
const ORDER = {
  role: ROLE_ORDER,
  experience: ["none", "1-2", "3-5", "6+"],
  salary: ["given", "not_given"],
  region: ["moscow", "spb", "other", "remote"]
};
const panels = Object.entries(ORDER).map(([group, levels]) => {
  const rows = life.medians.filter((m) => m.cohort === "prevalent" && m.group === group).sort((a, b) => levels.indexOf(a.level) - levels.indexOf(b.level));
  return {group, rows: rows.filter((m) => m.median_days !== null), missing: rows.filter((m) => m.median_days === null)};
});
const xMax = Math.ceil(d3.max(life.medians.filter((m) => m.cohort === "prevalent"), (m) => m.high ?? m.median_days) / 10) * 10 + 10;
```

<div class="grid grid-cols-2">
${panels.map(({group, rows, missing}) => html`<div class="card"><h3>${LIFETIME[group]}</h3>${resize((w) =>
  Plot.plot({
    width: w,
    height: rows.length * 30 + 40,
    marginLeft: 150,
    marginRight: 10,
    x: {grid: true, label: "дней", domain: [0, xMax], ticks: d3.range(0, xMax + 1, w < 360 ? 20 : 10)},
    y: {label: null, domain: rows.map((m) => LIFETIME[m.level] ?? m.level)},
    marks: [
      Plot.link(rows, {y1: (m) => LIFETIME[m.level], y2: (m) => LIFETIME[m.level], x1: "low", x2: (m) => m.high ?? xMax, stroke: (m) => (m.level === "ml_ds_data" ? COLOR.accent : COLOR.to), strokeWidth: 2}),
      Plot.dot(rows, {
        y: (m) => LIFETIME[m.level],
        x: "median_days",
        r: 4.5,
        fill: (m) => (m.level === "ml_ds_data" ? COLOR.accent : COLOR.to),
        title: (m) => `${LIFETIME[m.level]}\nмедиана ${days(m.median_days)}, 95 % интервал ${interval(m)}\nвакансий ${int(m.subjects)}, закрылись ${int(m.events)}`,
        tip: true
      })
    ]
  })
)}${missing.length ? html`<p class="note-small">Медиана не оценивается: ${missing.map((m) => `${LIFETIME[m.level]} (вакансий ${int(m.subjects)})`).join(", ")}.</p>` : ""}</div>`)}
</div>

<p class="note-small">Точка: медиана, отрезок: 95 % интервал; если верхняя граница не оценивается, отрезок доходит до края. Окно наблюдения с ${dateLong(life.window_from)} по ${dateLong(life.window_to)}. Расчёт <code>${life.source_file}</code>, ${life.source_script}, коммит ${life.commit}; медианы перенесены без изменений. <a href="./methodology#survival">Ограничения расчёта</a>.</p>
