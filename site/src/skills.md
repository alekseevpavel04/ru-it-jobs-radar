---
title: Навыки
---

# Навыки в вакансиях

```js
import {ROLE, ROLE_GROUP, ROLE_GROUP_ORDER, SOURCE_SHORT, SOURCE_COLOR, FAMILY_ORDER, CATEGORY, SKILL, COLOR, withSkillLabels, int, pct, share, points, day, dateLong, dateShort, monthTick, sourcePicker, plural} from "./components/labels.js";

const meta = FileAttachment("data/meta.json").json();
const totals = FileAttachment("data/skills.json").json().then(withSkillLabels);
const byRole = FileAttachment("data/skills_by_role.json").json().then(withSkillLabels);
const junior = FileAttachment("data/junior.json").json();
const fc = FileAttachment("data/forecast.json").json();
const history = FileAttachment("data/forecast_history.json").json();
```

<p class="lead">Доля вакансий, где назван навык. Словарь навыков ручной, <a href="./methodology#skills">подробнее в методике</a>.</p>

```js
const SOFT = new Set(["soft", "language_human"]);
const sourceInput = sourcePicker(Inputs, FAMILY_ORDER, "habr");
const source = Generators.input(sourceInput);
const softInput = Inputs.toggle({label: "Без гибких навыков", value: true});
const noSoft = Generators.input(softInput);
```

```js
const buckets = junior.filter((d) => d.source === source && d.vacancies > 0);
const sourceTotal = d3.sum(buckets, (d) => d.vacancies);
// A native select: the roles are grouped as on the main page, which Inputs.select cannot do.
const roleSelect = html`<select>
  <option value="">Все роли (${int(sourceTotal)})</option>
  ${[...ROLE_GROUP_ORDER, "Несколько вакансий в посте"].map((group) => {
    const items = buckets.filter((d) => ROLE_GROUP[d.role_bucket] === group).sort((a, b) => b.vacancies - a.vacancies);
    return items.length ? html`<optgroup label=${group}>${items.map((d) => html`<option value=${d.role_bucket}>${ROLE[d.role_bucket]} (${int(d.vacancies)})</option>`)}</optgroup>` : null;
  })}
</select>`;
const roleInput = html`<label class="select-native">Роль ${roleSelect}</label>`;
Object.defineProperty(roleInput, "value", {get: () => roleSelect.value || null});
const role = Generators.input(roleInput);
```

<div class="filters">${sourceInput}${roleInput}${softInput}</div>

```js
const unit = source === "telegram" ? "постов" : "вакансий";
const keep = (d) => !(noSoft && SOFT.has(d.category));
const allShare = new Map(totals.filter((d) => d.source === source).map((d) => [d.skill, d.vacancies_mentioning / d.vacancies_total]));
const rows = (role === null
  ? totals.filter((d) => d.source === source).map((d) => ({...d, of: d.vacancies_total}))
  : byRole.filter((d) => d.source === source && d.role_bucket === role).map((d) => ({...d, of: d.vacancies_in_bucket}))
)
  .filter(keep)
  .map((d) => ({skill: d.skill, category: CATEGORY[d.category] ?? d.category, n: d.vacancies_mentioning, of: d.of, share: d.vacancies_mentioning / d.of, lift: d.vacancies_mentioning / d.of / allShare.get(d.skill)}));
const top = rows.slice().sort((a, b) => b.n - a.n).slice(0, 15);
const base = role === null ? sourceTotal : buckets.find((d) => d.role_bucket === role).vacancies;
const period = meta.periods[source];
const times = new Intl.NumberFormat("ru-RU", {maximumFractionDigits: 1});
```

## Самые частые навыки

<div class="card">

<p class="note-small">${SOURCE_SHORT[source]}${role === null ? "" : `, ${ROLE[role]}`}: доля от ${int(base)} ${unit} с ${dateLong(period.period_from)} по ${dateLong(period.period_to)}.${role === null ? "" : " Справа: во сколько раз чаще, чем во всех IT-вакансиях источника."}</p>

```js
top.length === 0
  ? html`<p>Для этой роли нет навыков, названных хотя бы ${meta.skill_min_mentions} раза.</p>`
  : Plot.plot({
      width,
      height: top.length * 24 + 20,
      marginLeft: width < 600 ? 130 : 200,
      marginRight: role === null ? 70 : width < 600 ? 110 : 150,
      x: {axis: null},
      y: {label: null, domain: top.map((d) => d.skill)},
      marks: [
        Plot.barX(top, {y: "skill", x: "share", fill: SOURCE_COLOR[source], title: (d) => `${d.skill} (${d.category})\n${share(d.n, d.of)}${role === null ? "" : `\nво всех IT-вакансиях источника ${pct(allShare.get(d.skill))}`}`, tip: true}),
        Plot.text(top, {y: "skill", x: "share", text: (d) => pct(d.share), textAnchor: "start", dx: 4}),
        role === null
          ? null
          : Plot.text(top, {y: "skill", text: (d) => `×${times.format(d.lift)}`, frameAnchor: "right", textAnchor: "end", dx: width < 600 ? 105 : 140, fill: (d) => (d.lift >= 2 ? COLOR.accent : "#888"), fontWeight: (d) => (d.lift >= 2 ? 600 : 400)})
      ]
    })
```

<details>
<summary>Все навыки (${int(rows.length)})</summary>

```js
Inputs.table(rows, {
  columns: width < 600 ? ["skill", "share"] : role === null ? ["skill", "category", "n", "share"] : ["skill", "category", "n", "share", "lift"],
  header: {skill: "Навык", category: "Категория", n: `Упоминаний (из ${int(base)})`, share: "Доля", lift: "Чаще, чем во всех IT"},
  format: {n: int, share: (x) => pct(x), lift: (x) => `×${times.format(x)}`},
  sort: "n",
  reverse: true,
  rows: 12
})
```

</details>
</div>

## Сравнение источников

```js
// With a role picked, each source is compared on that role's vacancies, not on all of its own.
const compareRows = (role === null
  ? totals.map((d) => ({...d, of: d.vacancies_total}))
  : byRole.filter((d) => d.role_bucket === role).map((d) => ({...d, of: d.vacancies_in_bucket}))
).filter(keep);
const shareBy = new Map(compareRows.map((d) => [`${d.source}|${d.skill}`, d]));
const compared = d3
  .groups(compareRows, (d) => d.skill)
  .map(([skill, v]) => ({skill, best: d3.max(v, (d) => d.vacancies_mentioning / d.of)}))
  .sort((a, b) => b.best - a.best)
  .slice(0, 20)
  .map((d) => d.skill);
const comparePoints = compared.flatMap((skill) =>
  FAMILY_ORDER.map((s) => {
    const row = shareBy.get(`${s}|${skill}`);
    return {skill, source: SOURCE_SHORT[s], share: row ? row.vacancies_mentioning / row.of : 0, row};
  })
);
```

<div class="card">

<p class="note-small">Двадцать навыков с наибольшей долей хотя бы в одном источнике${role === null ? "" : `, роль «${ROLE[role]}»`}.</p>

```js
Plot.plot({
  width,
  height: compared.length * 24 + 50,
  marginLeft: width < 600 ? 130 : 200,
  x: {grid: true, label: "доля вакансий (в Telegram постов)", tickFormat: pct, zero: true},
  y: {label: null, domain: compared},
  color: {domain: FAMILY_ORDER.map((s) => SOURCE_SHORT[s]), range: FAMILY_ORDER.map((s) => SOURCE_COLOR[s]), legend: true},
  marks: [
    Plot.ruleY(compared, {y: (d) => d, stroke: "#e0e0e0"}),
    Plot.dot(comparePoints, {
      y: "skill",
      x: "share",
      fill: "source",
      r: 5,
      title: (d) => `${d.skill}\n${d.source}: ${d.row ? share(d.row.vacancies_mentioning, d.row.of) : role === null ? "не упоминается" : `меньше ${meta.skill_min_mentions} упоминаний`}`,
      tip: true
    })
  ]
})
```

<p class="note-small">Разница между источниками отражает прежде всего их состав, а не разные рынки: <a href="./methodology#sources">перекосы источников</a>.</p>

</div>

<h2 id="demand">Как меняется спрос</h2>

```js
const label = (skill) => SKILL[skill] ?? skill;
const weeks = [...new Set(history.map((d) => d.week))].sort();
const skillsFc = [...new Set(fc.points.map((p) => p.skill))];
const RECENT = 13;
const recentWeeks = new Set(weeks.slice(-RECENT));
// the same calendar weeks one year (52 weeks) earlier
const yearAgoWeeks = new Set(weeks.slice(-RECENT - 52, -52));
const lastHorizon = fc.points.filter((p) => p.horizon === fc.horizon);
const trend = skillsFc
  .map((skill) => {
    const rows = history.filter((d) => d.skill === skill);
    const sumOf = (set) => ({n: d3.sum(rows.filter((d) => set.has(d.week)), (d) => d.posts_mentioning), of: d3.sum(rows.filter((d) => set.has(d.week)), (d) => d.posts_it)});
    const now = sumOf(recentWeeks);
    const before = sumOf(yearAgoWeeks);
    // four-week sums smooth the line of the sparkline; the table numbers are 13-week sums
    const spark = rows.map((d, i) => {
      const w = rows.slice(Math.max(0, i - 3), i + 1);
      return d3.sum(w, (r) => r.posts_mentioning) / d3.sum(w, (r) => r.posts_it);
    });
    const p = lastHorizon.find((q) => q.skill === skill);
    return {skill, label: label(skill), now, before, nowShare: now.n / now.of, beforeShare: before.n / before.of, change: now.n / now.of - before.n / before.of, spark, forecast: p.forecast, low: p.low, high: p.high};
  })
  .sort((a, b) => b.change - a.change);
const sparkline = (values, change) => {
  const w = 120, h = 24, max = d3.max(values), min = d3.min(values);
  const x = (i) => (i / (values.length - 1)) * w;
  const y = (v) => h - 2 - ((v - min) / (max - min || 1)) * (h - 4);
  return svg`<svg width=${w} height=${h} style="vertical-align: middle"><path d=${values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("")} fill="none" stroke=${change > 0 ? COLOR.accent : "#4a4a4a"} stroke-width="1.3"/></svg>`;
};
```

<p class="note-small">Telegram, ${skillsFc.length} ${plural(skillsFc.length, ["навык", "навыка", "навыков"])}, ${weeks.length} недель, прогноз на ${fc.horizon} недели. Навыки в постах читает модель извлечения проекта, поэтому доли здесь не совпадают с графиками выше. <a href="./methodology#forecast">Как выбран метод</a>.</p>

<div class="card">
<h3>Растут и падают</h3>

<p class="note-small">Доля IT-постов, где назван навык: последние ${RECENT} недель против тех же недель год назад. Линия: два года, скользящие 4 недели.</p>

```js
Inputs.table(trend, {
  columns: width < 600 ? ["label", "change", "spark"] : ["label", "now", "before", "change", "spark"],
  header: {label: "Навык", now: `Последние ${RECENT} недель`, before: "Год назад", change: "Изменение", spark: "Два года"},
  format: {
    label: (x, i, data) => html`<span title=${`прогноз на неделю с ${dateLong(lastHorizon[0].week)}: ${pct(data[i].forecast)}, полоса от ${pct(data[i].low)} до ${pct(data[i].high)}`}>${x}</span>`,
    now: (x) => share(x.n, x.of),
    before: (x) => share(x.n, x.of),
    change: (x) => html`<span style=${`color: ${x > 0 ? COLOR.accent : "inherit"}; font-weight: 600`}>${points(x)}</span>`,
    spark: (x, i, data) => sparkline(x, data[i].change)
  },
  layout: "auto",
  sort: null,
  rows: trend.length + 1
})
```

</div>

```js
const skillPick = view(Inputs.select(new Map(trend.map((d) => [d.label, d.skill])), {label: "Навык", value: trend[0].skill}));
```

```js
const observed = history
  .filter((d) => d.skill === skillPick)
  .map((d) => ({week: day(d.week), share: d.posts_mentioning / d.posts_it, n: d.posts_mentioning, of: d.posts_it}));
const ahead = fc.points.filter((p) => p.skill === skillPick).map((p) => ({...p, week: day(p.week)}));
const lastObs = observed.at(-1);
// The band and the forecast line start at the last observed week so they read as its continuation.
const band = [{week: lastObs.week, low: lastObs.share, high: lastObs.share, forecast: lastObs.share}, ...ahead];
const level = Math.round(fc.interval_quantile * 100);
```

<div class="card">

```js
Plot.plot({
  width,
  height: 300,
  marginLeft: 50,
  x: {type: "utc", label: null, tickFormat: monthTick},
  y: {grid: true, label: "доля IT-постов недели", tickFormat: pct, zero: true},
  marks: [
    Plot.areaY(band, {x: "week", y1: "low", y2: "high", fill: COLOR.accent, fillOpacity: 0.18}),
    Plot.lineY(observed, {x: "week", y: "share", stroke: "#4a4a4a"}),
    Plot.tip(observed, Plot.pointerX({x: "week", y: "share", title: (d) => `неделя с ${dateLong(d.week)}\n${share(d.n, d.of)}`})),
    Plot.lineY(band, {x: "week", y: "forecast", stroke: COLOR.accent, strokeWidth: 2, strokeDasharray: "4,3"}),
    Plot.dot(ahead, {x: "week", y: "forecast", r: 3, fill: COLOR.accent, title: (d) => `неделя с ${dateLong(d.week)}, прогноз на ${d.horizon} нед.\n${pct(d.forecast)}, полоса от ${pct(d.low)} до ${pct(d.high)}`, tip: true}),
    Plot.ruleY([0])
  ]
})
```

<p class="note-small">Серая линия: доля за неделю. Красный пунктир: прогноз ETS, закрашено: полоса ${level} %. ETS точнее среднего за 4 недели у <!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.skills_better.n:d-->24<!--/n--> из <!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.skills_better.denominator:d-->24<!--/n--> навыков, но выигрыш мал: MASE <!--n:models/forecast.json:metrics.telegram.share.ets.mase.value:.3f|ru-->0,926<!--/n--> против <!--n:models/forecast.json:metrics.telegram.share.moving_average.mase.value:.3f|ru-->0,960<!--/n-->. Прогноз читается как сглаженный текущий уровень, а не как тренд.</p>

</div>
