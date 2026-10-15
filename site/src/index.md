---
title: Навыки
toc: false
---

# Какие навыки нужны в IT

```js
import {SOURCE_SHORT, SOURCE_COLOR, FAMILY_ORDER, SKILL, COLOR, withSkillLabels, int, pct, share, points, day, dateLong, monthTick, plural} from "./components/labels.js";
import {GROUP_LABEL, skillRows} from "./components/skills.js";

const trends = FileAttachment("data/skill_trends.json").json();
const totals = FileAttachment("data/skills.json").json().then(withSkillLabels);
const fc = FileAttachment("data/forecast.json").json();
const history = FileAttachment("data/forecast_history.json").json();
```

<p class="lead">Что просят вакансии из публичных Telegram-каналов с 2019 года: всё IT, ML и стажёры с джунами. Навыки из текста поста читает модель проекта. Показаны только доли, под которыми достаточно постов. <a href="./methodology#skills">Как считается</a>.</p>

```js
const groupInput = Inputs.radio(new Map([["Всё IT", "it_all"], ["ML", "ml_all"], ["Стажёры и джуны", "it_junior"]]), {label: "Вакансии", value: "it_all"});
const group = Generators.input(groupInput);
const sortInput = Inputs.radio(new Map([["по доле", "share"], ["по росту", "growth"]]), {label: "Порядок", value: "share"});
const sortBy = Generators.input(sortInput);
```

<div class="filters">${groupInput}${sortInput}</div>

```js
const BASE_YEAR = 2022;
const TOP = 15;
const posts = new Map(trends.groups.filter((g) => g.group === group).map((g) => [g.year, g.posts]));
const years = [...posts.keys()].filter((y) => y !== null).sort();
const recentLabel = `${trends.recent_years[0]}-${trends.recent_years.at(-1)}`;
const bySkill = d3.group(trends.shares.filter((s) => s.group === group), (s) => s.skill);
const rows = [...bySkill].map(([skill, list]) => {
  const recent = list.find((s) => s.year === null);
  const series = years.map((y) => list.find((s) => s.year === y)).map((s, i) => ({year: years[i], share: s.posts_mentioning / s.posts, n: s.posts_mentioning, of: s.posts}));
  const base = series.find((s) => s.year === BASE_YEAR);
  return {skill: SKILL[skill] ?? skill, share: recent.posts_mentioning / recent.posts, n: recent.posts_mentioning, of: recent.posts, series, change: base ? recent.posts_mentioning / recent.posts - base.share : null};
});
const withChange = rows.filter((r) => r.change !== null).sort((a, b) => b.change - a.change);
const risers = withChange.slice(0, 3);
const fallers = withChange.slice(-3).reverse();
const ordered = rows.slice().sort(sortBy === "share" ? (a, b) => b.share - a.share : (a, b) => (b.change ?? -1) - (a.change ?? -1));
```

<div class="grid grid-cols-2 movers">
  <div class="card">
    <h2>Растут с ${BASE_YEAR} года</h2>
    ${risers.map((r) => html`<div class="mover"><span class="mover-skill">${r.skill}</span><span class="mover-delta up">${points(r.change)}</span><span class="mover-path">${pct(r.series.find((s) => s.year === BASE_YEAR).share)} → ${pct(r.share)}</span></div>`)}
  </div>
  <div class="card">
    <h2>Теряют с ${BASE_YEAR} года</h2>
    ${fallers.map((r) => html`<div class="mover"><span class="mover-skill">${r.skill}</span><span class="mover-delta down">${points(r.change)}</span><span class="mover-path">${pct(r.series.find((s) => s.year === BASE_YEAR).share)} → ${pct(r.share)}</span></div>`)}
  </div>
</div>

<div class="card">

<p class="note-small">${GROUP_LABEL[group]}: доля из ${int(posts.get(null))} постов за ${recentLabel}, линия по годам с ${years[0]}, справа изменение доли с ${BASE_YEAR} года. ${years[0] > 2019 ? `Годы, где постов меньше ${int(trends.min_year_posts)}, не показаны. ` : ""}Наведите на строку, чтобы увидеть числа по годам.</p>

```js
display(skillRows(ordered.slice(0, TOP), {years, recentLabel, baseYear: BASE_YEAR}));
if (ordered.length > TOP) {
  const rest = ordered.length - TOP;
  display(html`<details class="more"><summary>Ещё ${rest} ${plural(rest, ["навык", "навыка", "навыков"])}</summary>${skillRows(ordered.slice(TOP), {years, recentLabel, baseYear: BASE_YEAR, head: false})}</details>`);
}
```

</div>

## Все источники

```js
// Portal and Habr only for all IT: their ML samples (55 and 54 vacancies) are too small for skill shares.
const compared = d3
  .groups(totals.filter((d) => d.category !== "soft" && d.category !== "language_human"), (d) => d.skill)
  .map(([skill, v]) => ({skill, best: d3.max(v, (d) => d.vacancies_mentioning / d.vacancies_total), v}))
  .sort((a, b) => b.best - a.best)
  .slice(0, 15);
const comparePoints = compared.flatMap(({skill, v}) =>
  FAMILY_ORDER.map((s) => {
    const row = v.find((d) => d.source === s);
    return {skill, source: SOURCE_SHORT[s], share: row ? row.vacancies_mentioning / row.vacancies_total : 0, row};
  })
);
```

<div class="card">

<p class="note-small">Всё IT, пятнадцать навыков с наибольшей долей хотя бы в одном источнике, по ручному словарю навыков, одинаковому для трёх источников. Портал и Хабр собираются недавно, поэтому здесь срез без истории. Источники не складываются: в Telegram считаются посты, на сайтах вакансии.</p>

```js
Plot.plot({
  width,
  height: compared.length * 28 + 60,
  marginLeft: width < 600 ? 110 : 170,
  x: {grid: true, label: null, tickFormat: pct, zero: true},
  y: {label: null, domain: compared.map((d) => d.skill), tickSize: 0},
  color: {domain: FAMILY_ORDER.map((s) => SOURCE_SHORT[s]), range: FAMILY_ORDER.map((s) => SOURCE_COLOR[s]), legend: true},
  style: {fontSize: "13px"},
  marks: [
    Plot.ruleY(compared.map((d) => d.skill), {stroke: "#ebe9e4", strokeWidth: 1}),
    Plot.link(compared, {y1: "skill", y2: "skill", x1: (d) => d3.min(FAMILY_ORDER, (s) => comparePoints.find((p) => p.skill === d.skill && p.source === SOURCE_SHORT[s]).share), x2: "best", stroke: "#d9d6cf", strokeWidth: 3}),
    Plot.dot(comparePoints, {y: "skill", x: "share", fill: "source", r: 6, stroke: "white", strokeWidth: 1.5, title: (d) => `${d.skill}\n${d.source}: ${d.row ? share(d.row.vacancies_mentioning, d.row.vacancies_total) : "не упоминается"}`, tip: true})
  ]
})
```

</div>

## Прогноз на месяц

```js
const label = (skill) => SKILL[skill] ?? skill;
const skillsFc = [...new Set(fc.points.map((p) => p.skill))].sort((a, b) => label(a).localeCompare(label(b)));
const skillPick = view(Inputs.select(new Map(skillsFc.map((s) => [label(s), s])), {label: "Навык", value: skillsFc.includes("Python") ? "Python" : skillsFc[0]}));
```

```js
const SHOWN_WEEKS = 52;
const observed = history
  .filter((d) => d.skill === skillPick)
  .slice(-SHOWN_WEEKS)
  .map((d) => ({week: day(d.week), share: d.posts_mentioning / d.posts_it, n: d.posts_mentioning, of: d.posts_it}));
const ahead = fc.points.filter((p) => p.skill === skillPick).map((p) => ({...p, week: day(p.week)}));
const lastObs = observed.at(-1);
// The band and the forecast line start at the last observed week so they read as its continuation.
const band = [{week: lastObs.week, low: lastObs.share, high: lastObs.share, forecast: lastObs.share}, ...ahead];
const level = Math.round(fc.interval_quantile * 100);
```

<div class="card">

<p class="note-small">Доля IT-постов Telegram за неделю, последние ${SHOWN_WEEKS} недели, и прогноз ETS на ${fc.horizon} недели с полосой ${level} %. Прогноз читается как сглаженный текущий уровень, а не как тренд. <a href="./methodology#forecast">Как выбран метод</a>.</p>

```js
Plot.plot({
  width,
  height: 280,
  marginLeft: 44,
  x: {type: "utc", label: null, tickFormat: monthTick},
  y: {grid: true, label: null, tickFormat: pct, zero: true},
  style: {fontSize: "12px"},
  marks: [
    Plot.areaY(band, {x: "week", y1: "low", y2: "high", fill: COLOR.accent, fillOpacity: 0.16}),
    Plot.lineY(observed, {x: "week", y: "share", stroke: COLOR.bar, strokeWidth: 1.6}),
    Plot.tip(observed, Plot.pointerX({x: "week", y: "share", title: (d) => `неделя с ${dateLong(d.week)}\n${share(d.n, d.of)}`})),
    Plot.lineY(band, {x: "week", y: "forecast", stroke: COLOR.accent, strokeWidth: 2.2}),
    Plot.dot(ahead, {x: "week", y: "forecast", r: 3.5, fill: COLOR.accent, title: (d) => `неделя с ${dateLong(d.week)}, прогноз на ${d.horizon} нед.\n${pct(d.forecast)}, полоса от ${pct(d.low)} до ${pct(d.high)}`, tip: true}),
    Plot.ruleY([0])
  ]
})
```

<p class="note-small">ETS точнее среднего за 4 недели у <!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.skills_better.n:d-->24<!--/n--> из <!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.skills_better.denominator:d-->24<!--/n--> навыков, но выигрыш мал: MASE <!--n:models/forecast.json:metrics.telegram.share.ets.mase.value:.3f|ru-->0,926<!--/n--> против <!--n:models/forecast.json:metrics.telegram.share.moving_average.mase.value:.3f|ru-->0,960<!--/n-->.</p>

</div>
