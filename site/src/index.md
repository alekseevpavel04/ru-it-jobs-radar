---
title: Рынок
toc: true
---

# IT-вакансии в открытых источниках

```js
import {ROLE, ROLE_GROUP, ROLE_GROUP_ORDER, SOURCE, int, pct, share, day, dateLong, dateShort, monthTick, ofChannels} from "./components/labels.js";

const meta = FileAttachment("data/meta.json").json();
const life = FileAttachment("data/lifetime.json").json();
const weeklyRows = FileAttachment("data/weekly_vacancies.json").json();
const telegramRows = FileAttachment("data/weekly_telegram.json").json();
```

```js
const WEEK_MS = 7 * 864e5;

// Week totals of one vacancy source; a week after a week with no data is flagged, because
// closures of the missing weeks land in it.
function weekTotals(source) {
  const rows = weeklyRows.filter((d) => d.source === source);
  const weeks = d3
    .rollups(
      rows,
      (v) => ({
        active: d3.sum(v, (d) => d.active),
        new: d3.sum(v, (d) => d.new),
        closed: v[0].closed === null ? null : d3.sum(v, (d) => d.closed),
        days: v[0].days_collected,
        first: v[0].first_collected_week
      }),
      (d) => d.week
    )
    .map(([week, s]) => ({week, date: day(week), ...s}))
    .sort((a, b) => a.date - b.date);
  weeks.forEach((w, i) => (w.afterGap = i > 0 && w.date - weeks[i - 1].date > WEEK_MS));
  return weeks;
}

const snapshotWeeks = weekTotals("trudvsem_snapshot");
const lastFull = snapshotWeeks.findLast((w) => w.days === 7);
const habrWeeks = weekTotals("habr");
const habrLast = habrWeeks.at(-1);

// A Telegram week is complete when its Sunday is not after the newest collected post.
const telegramEnd = day(meta.periods.telegram.period_to);
const telegramWeeks = d3
  .rollups(
    telegramRows,
    (v) => ({
      posts: d3.sum(v, (d) => d.posts),
      before: d3.sum(v, (d) => d.posts_before_dedup),
      // channels that carried any vacancy post of the week, every bucket together (the same on each row)
      channels: v[0].channels_week
    }),
    (d) => d.week
  )
  .map(([week, s]) => ({week, date: day(week), ...s, complete: +day(week) + 6 * 864e5 <= +telegramEnd}))
  .sort((a, b) => a.date - b.date);
const telegramComplete = telegramWeeks.filter((w) => w.complete);
const telegramLast4 = telegramComplete.slice(-4);
const telegramLast13 = telegramComplete.slice(-13);
// Channel coverage of the last weeks against the half year before them: a rise in posts that comes
// with more channels is the sample growing, and the page says so.
const telegramEarlier = telegramComplete.slice(-30, -4);
const earlierChannels = d3.max(telegramEarlier, (w) => w.channels);
const widerWeeks = telegramLast4.filter((w) => w.channels > earlierChannels);
const coverageGrew = widerWeeks.length > 0;
```

Три источника считаются раздельно и не складываются: у них разные единицы счёта и разный охват. Портал «Работа в России» виден целиком по ежедневным снимкам, Хабр Карьера собирается с ${dateLong(meta.periods.habr.period_from)}, в Telegram считаются посты, а не вакансии.

<div class="grid grid-cols-3">
  <div class="card">
    <h2>${SOURCE.trudvsem_snapshot}</h2>
    <div class="big">${int(lastFull.active)}</div>
    IT-вакансий были открыты хотя бы в один день недели с ${dateLong(lastFull.week)} (собраны все 7 дней)
  </div>
  <div class="card">
    <h2>${SOURCE.habr}</h2>
    <div class="big">${int(habrLast.active)}</div>
    IT-вакансий были открыты на неделе с ${dateLong(habrLast.week)}; собрано дней: ${habrLast.days} из 7
  </div>
  <div class="card">
    <h2>${SOURCE.telegram}</h2>
    <div class="big">${int(d3.sum(telegramLast4, (w) => w.posts))}</div>
    постов с вакансиями за 4 недели с ${dateLong(telegramLast4[0].week)} по ${dateLong(new Date(+telegramLast4.at(-1).date + 6 * 864e5))}, повторы одного текста за неделю убраны${coverageGrew ? "; в части этих недель каналов больше, чем раньше (график Telegram ниже)" : ""}
  </div>
</div>

## Открытые вакансии по неделям

```js
const groupWeeks = d3
  .rollups(
    weeklyRows.filter((d) => d.source === "trudvsem_snapshot"),
    (v) => ({active: d3.sum(v, (d) => d.active), days: v[0].days_collected}),
    (d) => d.week,
    (d) => ROLE_GROUP[d.role_bucket]
  )
  .flatMap(([week, groups]) => groups.map(([group, s]) => ({date: day(week), week, group, ...s})));
const partialWeeks = snapshotWeeks.filter((w) => w.days < 7);
```

<div class="card">

```js
Plot.plot({
  width,
  height: 360,
  marginLeft: 50,
  x: {type: "utc", label: null, tickFormat: dateShort, ticks: d3.utcMonth.every(1)},
  y: {grid: true, label: "вакансий", tickFormat: int},
  color: {domain: ROLE_GROUP_ORDER, scheme: "tableau10", legend: true},
  marks: [
    Plot.rectY(groupWeeks, {
      x: "date",
      interval: d3.utcMonday,
      y: "active",
      fill: "group",
      fillOpacity: (d) => (d.days === 7 ? 1 : 0.4),
      title: (d) => `${d.group}\nнеделя с ${dateLong(d.week)}\n${int(d.active)} вакансий\nсобрано дней: ${d.days} из 7`,
      tip: true
    }),
    Plot.ruleY([0])
  ]
})
```

<div class="note-small">

${SOURCE.trudvsem_snapshot}: вакансия считается в неделе, если была открыта хотя бы в один собранный день. Бледные столбцы: собраны не все дни недели (${partialWeeks.map((w) => `с ${dateShort(w.week)}: ${w.days} из 7`).join("; ")}), такая неделя занижена. Пустое место между столбцами: снимков за эти недели в выборке пока нет. Какие роли входят в группы, описано в методике.

</div>
</div>

## Новые и закрытые

```js
const flows = snapshotWeeks.flatMap((w) => [
  {date: w.date, week: w.week, kind: "новые", n: w.first ? null : w.new, flagged: w.days < 7},
  {date: w.date, week: w.week, kind: "закрытые", n: w.closed, flagged: w.days < 7 || w.afterGap}
]).filter((d) => d.n !== null);
const afterGap = snapshotWeeks.filter((w) => w.afterGap);
// A flagged week far above the rest would squash the scale; it is listed under the chart instead.
const flowCap = 1.5 * d3.max(flows.filter((d) => !d.flagged), (d) => d.n);
const offScale = flows.filter((d) => d.n > flowCap);
const flowsShown = flows.filter((d) => d.n <= flowCap);
const flowLines = ["новые", "закрытые"].flatMap((kind) =>
  d3.utcMondays(snapshotWeeks[0].date, +snapshotWeeks.at(-1).date + 1).map((date) => {
    const point = flows.find((d) => d.kind === kind && +d.date === +date);
    return {date, kind, n: point && !point.flagged ? point.n : NaN};
  })
);
```

<div class="card">

```js
Plot.plot({
  width,
  height: 280,
  marginLeft: 50,
  y: {grid: true, label: "вакансий за неделю", tickFormat: int},
  x: {type: "utc", label: null, tickFormat: dateShort, ticks: d3.utcMonth.every(1), domain: d3.extent(snapshotWeeks, (w) => w.date)},
  color: {domain: ["новые", "закрытые"], range: ["#3ca951", "#ff725c"], legend: true},
  marks: [
    // every week of the series is a point; a flagged or missing week is NaN, which breaks the line
    Plot.lineY(flowLines, {x: "date", y: "n", stroke: "kind"}),
    Plot.dot(flowsShown, {
      x: "date",
      y: "n",
      stroke: "kind",
      fill: "kind",
      fillOpacity: (d) => (d.flagged ? 0 : 1),
      strokeWidth: 1.5,
      r: 4,
      title: (d) => `${d.kind}: ${int(d.n)}\nнеделя с ${dateLong(d.week)}${d.flagged ? "\nнеделя неполная или после пропуска" : ""}`,
      tip: true
    }),
    Plot.ruleY([0])
  ]
})
```

<div class="note-small">

Новая вакансия: впервые появилась в снимке этой недели (первая неделя сбора не считается, в ней всё уже открыто). Закрытая: её нет в ${life.grace_days} полных снимках портала подряд. Пустые точки: неделя собрана не полностью${afterGap.length ? ` или идёт после пропуска (${afterGap.map((w) => `с ${dateShort(w.week)}`).join(", ")}), и тогда в неё попадают закрытия пропущенных недель` : ""}; линия через такие точки не проводится.${offScale.length ? ` За пределами шкалы: ${offScale.map((d) => `${d.kind} на неделе с ${dateShort(d.week)}: ${int(d.n)}`).join(", ")}.` : ""}

</div>
</div>

## Какие роли

```js
const roleSource = view(
  Inputs.radio(
    new Map([
      [`${SOURCE.trudvsem_snapshot}, неделя с ${dateShort(lastFull.week)}`, "trudvsem_snapshot"],
      [`${SOURCE.habr}, неделя с ${dateShort(habrLast.week)}`, "habr"],
      [`${SOURCE.telegram}, последние ${telegramLast13.length} недель`, "telegram"]
    ]),
    {value: "trudvsem_snapshot"}
  )
);
```

```js
const telegramWeekSet = new Set(telegramLast13.map((w) => w.week));
const roleCounts =
  roleSource === "telegram"
    ? d3.rollups(telegramRows.filter((d) => telegramWeekSet.has(d.week)), (v) => d3.sum(v, (d) => d.posts), (d) => d.role_bucket)
    : d3.rollups(
        weeklyRows.filter((d) => d.source === roleSource && d.week === (roleSource === "habr" ? habrLast.week : lastFull.week)),
        (v) => d3.sum(v, (d) => d.active),
        (d) => d.role_bucket
      );
const roleTotal = d3.sum(roleCounts, ([, n]) => n);
const roles = roleCounts.map(([bucket, n]) => ({role: ROLE[bucket], n, share: n / roleTotal})).sort((a, b) => b.n - a.n);
const unit = roleSource === "telegram" ? "постов" : "вакансий";
```

<div class="card">

Доли от ${int(roleTotal)} ${unit}. Роль определена правилом по заголовку (в Telegram по первой строке поста).

```js
Plot.plot({
  width,
  height: roles.length * 24 + 30,
  marginLeft: width < 600 ? 180 : 230,
  marginRight: 90,
  x: {axis: null},
  y: {label: null, domain: roles.map((d) => d.role)},
  marks: [
    Plot.barX(roles, {y: "role", x: "share", fill: "#4269d0", title: (d) => `${d.role}\n${share(d.n, roleTotal)}`, tip: true}),
    Plot.text(roles, {y: "role", x: "share", text: (d) => `${int(d.n)} · ${pct(d.share)}`, textAnchor: "start", dx: 4})
  ]
})
```

</div>

## Telegram по неделям

<div class="card">

```js
Plot.plot({
  width,
  height: 260,
  marginLeft: 50,
  x: {type: "utc", label: null, tickFormat: monthTick},
  y: {grid: true, label: "постов за неделю", tickFormat: int},
  color: {domain: ["после удаления повторов", "все копии"], range: ["#4269d0", "#a0a0a0"], legend: true},
  marks: [
    Plot.lineY(telegramComplete, {x: "date", y: "before", stroke: () => "все копии"}),
    Plot.lineY(telegramComplete, {x: "date", y: "posts", stroke: () => "после удаления повторов"}),
    Plot.dot(widerWeeks, {x: "date", y: "posts", r: 5, stroke: "#4269d0", fill: "white"}),
    Plot.tip(telegramComplete, Plot.pointerX({x: "date", y: "posts", title: (d) => `неделя с ${dateLong(d.week)}\n${int(d.posts)} постов (всех копий ${int(d.before)})\nканалов ${d.channels}`})),
    Plot.ruleY([0])
  ]
})
```

```js
Plot.plot({
  width,
  height: 120,
  marginLeft: 50,
  x: {type: "utc", label: null, tickFormat: monthTick},
  y: {grid: true, label: "каналов с постами", tickFormat: int, ticks: 3},
  marks: [
    Plot.lineY(telegramComplete, {x: "date", y: "channels", stroke: "#606060"}),
    Plot.dot(widerWeeks, {x: "date", y: "channels", r: 4, stroke: "#606060", fill: "white"}),
    Plot.ruleY([0])
  ]
})
```

<div class="note-small">

Посты с вакансиями из публичных каналов за ${meta.telegram_weeks_shown} недель с ${dateLong(meta.telegram_first_week_shown)}; неполная последняя неделя не показана. Пост не равен вакансии: один пост может содержать несколько вакансий, одна вакансия выходит в нескольких каналах (повтор текста в ту же неделю считается один раз). Ряд зависит от того, сколько каналов попадает в неделю. ${coverageGrew ? `Отмеченные точки: ${widerWeeks.map((w) => `на неделе с ${dateShort(w.week)} посты пришли из ${ofChannels(w.channels)}`).join(", ")}, а в ${telegramEarlier.length} неделях до этого не больше чем из ${ofChannels(earlierChannels)}. Рост в такие недели говорит об охвате выборки, а не о росте рынка.` : "Это сигнал активности сообществ, а не объём рынка."}

</div>
</div>
