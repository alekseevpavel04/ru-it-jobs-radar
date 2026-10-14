---
title: Главное
---

# Рынок IT-вакансий по открытым источникам

```js
import {ROLE, ROLE_GROUP, ROLE_GROUP_ORDER, SOURCE, SOURCE_SHORT, SOURCE_COLOR, FAMILY_ORDER, COLOR, int, pct, share, rub, days, day, dateLong, dateShort, plural} from "./components/labels.js";

const meta = FileAttachment("data/meta.json").json();
const life = FileAttachment("data/lifetime.json").json();
const weeklyRows = FileAttachment("data/weekly_vacancies.json").json();
const telegramRows = FileAttachment("data/weekly_telegram.json").json();
const junior = FileAttachment("data/junior.json").json();
const salaries = FileAttachment("data/salaries.json").json();
const tgYears = FileAttachment("data/telegram_years.json").json();
```

```js
const DAY_MS = 864e5;
const WEEK_MS = 7 * DAY_MS;

// Week totals of one vacancy source, optionally of one role group; a week after a week with no data
// is flagged, because closures of the missing weeks land in it.
function weekTotals(source, group = null) {
  const rows = weeklyRows.filter((d) => d.source === source && (group === null || ROLE_GROUP[d.role_bucket] === group));
  const weeks = d3
    .rollups(
      rows,
      (v) => ({
        active: d3.sum(v, (d) => d.active),
        new: d3.sum(v, (d) => d.new),
        closed: v[0].closed === null ? null : d3.sum(v, (d) => d.closed),
        days: v[0].days_collected,
        complete: v[0].complete_days,
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
// A closure is dated to the first day the vacancy is missing and known only after grace_days complete
// snapshots without it, so the closures of the last days of the data are still partial. The last
// collected day counts as complete only when every collected day of its week was.
const snapshotTail = snapshotWeeks.at(-1);
const lastCompleteDay = snapshotTail.complete === snapshotTail.days ? new Date(+snapshotTail.date + (snapshotTail.days - 1) * DAY_MS) : null;
const closuresKnownThrough = lastCompleteDay && new Date(+lastCompleteDay - (life.grace_days - 1) * DAY_MS);
const closuresPending = (w) => !closuresKnownThrough || +w.date + 6 * DAY_MS > +closuresKnownThrough;
const lastFull = snapshotWeeks.findLast((w) => w.days === 7);
const habrLast = weekTotals("habr").at(-1);

// A Telegram week is complete when its Sunday is not after the newest collected post.
const telegramEnd = day(meta.periods.telegram.period_to);
const telegramComplete = [...new Set(telegramRows.map((d) => d.week))].sort().filter((w) => +day(w) + 6 * DAY_MS <= +telegramEnd);
const telegramLast13 = telegramComplete.slice(-13);
const telegramLast13Set = new Set(telegramLast13);
```

```js
// The four facts: each number is read from the export here, nothing is typed in by hand.
const bucketWeek = (source, week, bucket) =>
  d3.sum(weeklyRows.filter((d) => d.source === source && d.week === week && (bucket === null || d.role_bucket === bucket)), (d) => d.active);
const mlHabr = bucketWeek("habr", habrLast.week, "ml_ds");
const mlPortal = bucketWeek("trudvsem_snapshot", lastFull.week, "ml_ds");
const juniorMl = (s) => junior.find((d) => d.source === s && d.role_bucket === "ml_ds");
const [salaryHabr] = salaries.filter((d) => d.level === "source" && d.source === "habr");
const [salaryPortal] = salaries.filter((d) => d.level === "source" && d.source === "trudvsem");
const lifeAll = life.medians.find((m) => m.group === "overall" && m.cohort === "prevalent");
const lifeMl = life.medians.find((m) => m.group === "role" && m.level === "ml_ds_data" && m.cohort === "prevalent");
const telegramPosts13 = d3.sum(telegramRows.filter((d) => telegramLast13Set.has(d.week)), (d) => d.posts);
```

<p class="lead">Сколько IT-вакансий открыто, кого ищут, сколько платят и как долго вакансия висит. Портал «Работа в России» виден целиком, Хабр Карьера собирается с ${dateLong(meta.periods.habr.period_from)}, Telegram с ${tgYears.years[0].year} года. Источники считаются раздельно.</p>

<div class="grid grid-cols-4 facts">
  <a class="card fact" href="#roles">
    <h2>ML и Data Science это ниша</h2>
    <div class="big">${int(mlHabr)} из ${int(habrLast.active)}</div>
    <p>IT-вакансий Хабр Карьеры на неделе с ${dateLong(habrLast.week)} (${pct(mlHabr / habrLast.active)}). На портале ${share(mlPortal, lastFull.active)}.</p>
  </a>
  <a class="card fact" href="./junior">
    <h2>Начинающему в ML</h2>
    <div class="big">${int(juniorMl("habr").open_to_junior)} из ${int(juniorMl("habr").vacancies)}</div>
    <p>ML/DS-вакансий Хабр Карьеры открыты стажёру, джуниору или без опыта. В Telegram ${share(juniorMl("telegram").open_to_junior, juniorMl("telegram").vacancies)} постов.</p>
  </a>
  <a class="card fact" href="./salaries">
    <h2>Вилка зависит от источника</h2>
    <div class="big">${int(Math.round(salaryHabr.salary_from_median / 1000))} и ${int(Math.round(salaryPortal.salary_from_median / 1000))} тыс. ₽</div>
    <p>Медиана нижней границы вилки: Хабр Карьера ${rub(salaryHabr.salary_from_median)} (n ${int(salaryHabr.with_from)}), портал ${rub(salaryPortal.salary_from_median)} (n ${int(salaryPortal.with_from)}).</p>
  </a>
  <a class="card fact" href="./lifetime">
    <h2>Вакансия живёт месяц</h2>
    <div class="big">${days(lifeAll.median_days)}</div>
    <p>Медиана для IT-вакансии портала. ML, DS и данные быстрее: ${days(lifeMl.median_days)} [${int(lifeMl.low)}; ${int(lifeMl.high)}].</p>
  </a>
</div>

<p class="note-small">Объём: портал ${int(lastFull.active)} IT-вакансий на неделе с ${dateLong(lastFull.week)}, Хабр Карьера ${int(habrLast.active)} на неделе с ${dateLong(habrLast.week)} (собрано ${habrLast.days} ${plural(habrLast.days, ["день", "дня", "дней"])} из 7), Telegram ${int(telegramPosts13)} постов с вакансиями за ${telegramLast13.length} недель.</p>

## Что изменилось за неделю

```js
const prevFull = snapshotWeeks.find((w) => +w.date === +lastFull.date - WEEK_MS && w.days === 7);
const pendingLast = closuresPending(lastFull);
const groupWeek = (week) =>
  new Map(
    d3.rollups(
      weeklyRows.filter((d) => d.source === "trudvsem_snapshot" && d.week === week),
      (v) => ({active: d3.sum(v, (d) => d.active), new: d3.sum(v, (d) => d.new), closed: d3.sum(v, (d) => d.closed ?? 0)}),
      (d) => ROLE_GROUP[d.role_bucket]
    )
  );
const nowByGroup = groupWeek(lastFull.week);
const prevByGroup = prevFull ? groupWeek(prevFull.week) : new Map();
const changeRows = [
  {group: "Все IT", ...lastFull, prev: prevFull?.active, total: true},
  ...ROLE_GROUP_ORDER.map((group) => ({group, ...nowByGroup.get(group), prev: prevByGroup.get(group)?.active})).sort((a, b) => b.active - a.active)
];
const delta = (d) => {
  if (d.prev === undefined) return "";
  const x = d.active - d.prev;
  const sign = x > 0 ? "+" : x < 0 ? "-" : "";
  return html`${sign}${int(Math.abs(x))}<span class="wide-only"> (${sign}${pct(Math.abs(x) / d.prev)})</span>`;
};
```

<p class="note-small">${SOURCE_SHORT.trudvsem}, неделя с ${dateLong(lastFull.week)}${prevFull ? ` против недели с ${dateLong(prevFull.week)}` : ""}. Открытые: вакансия была видна хотя бы в один день недели.</p>

<div class="table-wrap">
<table class="change">
  <thead><tr><th>Группа ролей</th><th>Открыто</th><th>За неделю</th><th>Новые</th><th>Закрыты${pendingLast ? "*" : ""}</th></tr></thead>
  <tbody>${changeRows.map((d) => html`<tr class=${d.total ? "total" : ""}><td>${d.group}</td><td>${int(d.active)}</td><td>${delta(d)}</td><td>${int(d.new)}</td><td>${int(d.closed ?? 0)}</td></tr>`)}</tbody>
</table>
</div>

${pendingLast ? html`<p class="note-small">* Закрытия этой недели ещё подтверждаются, число вырастет. <a href="./methodology#closures">Как считается закрытие</a>.</p>` : ""}

## Открытые вакансии по неделям

```js
const groupPick = view(Inputs.select(new Map([["Все IT", null], ...ROLE_GROUP_ORDER.map((g) => [g, g])]), {label: "Группа ролей"}));
```

```js
const series = weekTotals("trudvsem_snapshot", groupPick);
const byDate = new Map(series.map((w) => [+w.date, w]));
const mondays = d3.utcMondays(series[0].date, +series.at(-1).date + 1);
// A week without snapshots is NaN and breaks the line; an incomplete week is drawn dashed.
const activeLine = mondays.map((date) => {
  const w = byDate.get(+date);
  return {date, all: w ? w.active : NaN, full: w && w.days === 7 ? w.active : NaN};
});
const flows = series.flatMap((w) => {
  const flagged = w.days < 7 || w.afterGap;
  return [
    w.first ? null : {date: w.date, week: w.week, kind: "новые", n: w.new, y: w.new, flagged, days: w.days},
    w.closed === null ? null : {date: w.date, week: w.week, kind: "закрытые", n: w.closed, y: -w.closed, flagged, pending: closuresPending(w), afterGap: w.afterGap, days: w.days}
  ].filter(Boolean);
});
const flowCap = 1.5 * d3.max(flows.filter((d) => !d.flagged), (d) => d.n);
const offScale = flows.filter((d) => d.n > flowCap);
const xDomain = [series[0].date, new Date(+series.at(-1).date + WEEK_MS)];
const weekTitle = (w) => `неделя с ${dateLong(w.week)}${w.days < 7 ? `\nсобрано дней ${w.days} из 7, неделя занижена` : ""}`;
```

<div class="card">

```js
Plot.plot({
  width,
  height: 200,
  marginLeft: 50,
  x: {type: "utc", label: null, domain: xDomain, tickFormat: dateShort, ticks: d3.utcMonth.every(1)},
  y: {grid: true, label: "открыто вакансий", tickFormat: int, zero: true},
  marks: [
    Plot.lineY(activeLine, {x: (d) => +d.date + WEEK_MS / 2, y: "all", stroke: COLOR.bar, strokeDasharray: "4,4"}),
    Plot.lineY(activeLine, {x: (d) => +d.date + WEEK_MS / 2, y: "full", stroke: COLOR.bar, strokeWidth: 2}),
    Plot.dot(series, {x: (d) => +d.date + WEEK_MS / 2, y: "active", r: 2.5, fill: COLOR.bar}),
    Plot.tip(series, Plot.pointerX({x: (d) => +d.date + WEEK_MS / 2, y: "active", title: (w) => `${int(w.active)} открытых\n${weekTitle(w)}`})),
    Plot.ruleY([0])
  ]
})
```

```js
Plot.plot({
  width,
  height: 190,
  marginLeft: 50,
  x: {type: "utc", label: null, domain: xDomain, tickFormat: dateShort, ticks: d3.utcMonth.every(1)},
  y: {grid: true, label: "за неделю", tickFormat: (y) => int(Math.abs(y)), domain: [-flowCap, flowCap]},
  clip: true,
  marks: [
    () => svg`<defs><pattern id="hatch" patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)"><rect width="5" height="5" fill="white"/><line x1="0" y1="0" x2="0" y2="5" stroke="${COLOR.accent}" stroke-width="2.5"/></pattern></defs>`,
    Plot.rectY(flows, {
      x1: "date",
      x2: (d) => +d.date + WEEK_MS,
      y: "y",
      inset: 2,
      fill: (d) => (d.pending ? "url(#hatch)" : d.kind === "новые" ? "#3a3a3a" : COLOR.accent),
      fillOpacity: (d) => (d.flagged ? 0.35 : 1),
      title: (d) => `${d.kind}: ${int(d.n)}\n${weekTitle(d)}${d.pending ? "\nзакрытия ещё подтверждаются" : ""}${d.afterGap ? "\nпосле недели без снимков" : ""}`,
      tip: true
    }),
    Plot.text(["новые", "закрытые"], {x: () => xDomain[0], y: (k) => (k === "новые" ? flowCap * 0.85 : -flowCap * 0.85), text: (k) => k, textAnchor: "start", dx: 4, fill: (k) => (k === "новые" ? "#3a3a3a" : COLOR.accent), fontWeight: 600}),
    Plot.ruleY([0])
  ]
})
```

<p class="note-small">${SOURCE_SHORT.trudvsem}, ежедневные снимки. Пунктир и бледные столбцы: неделя собрана не целиком. Штриховка: закрытия последних двух недель ещё подтверждаются. Пустое место: снимков за неделю нет.${offScale.length ? ` За шкалой: ${offScale.map((d) => `${d.kind} на неделе с ${dateShort(d.week)}, ${int(d.n)}`).join("; ")}.` : ""} <a href="./methodology#closures">Как считаются новые и закрытые</a>.</p>

</div>

<h2 id="roles">Какие роли нужны</h2>

```js
// The weekly groups of ROLE_GROUP, with ML split from the other data roles and the unspecified roles
// set apart, so the comparison shows where each source leans.
const SHARE_GROUP = {
  ...ROLE_GROUP,
  ml_ds: "ML и Data Science",
  data_engineering: "Инженерия и аналитика данных",
  data_analytics: "Инженерия и аналитика данных",
  design: "Дизайн",
  software_engineer: "Без уточнения и прочее IT",
  other_it: "Без уточнения и прочее IT",
  multi_post: "Несколько вакансий в посте"
};
const GREY = ["Без уточнения и прочее IT", "Несколько вакансий в посте"];
const roleCounts = {
  trudvsem: weeklyRows.filter((d) => d.source === "trudvsem_snapshot" && d.week === lastFull.week).map((d) => ({bucket: d.role_bucket, n: d.active})),
  habr: weeklyRows.filter((d) => d.source === "habr" && d.week === habrLast.week).map((d) => ({bucket: d.role_bucket, n: d.active})),
  telegram: d3
    .rollups(telegramRows.filter((d) => telegramLast13Set.has(d.week)), (v) => d3.sum(v, (d) => d.posts), (d) => d.role_bucket)
    .map(([bucket, n]) => ({bucket, n}))
};
const roleTotals = Object.fromEntries(FAMILY_ORDER.map((s) => [s, d3.sum(roleCounts[s], (d) => d.n)]));
const groupPoints = FAMILY_ORDER.flatMap((s) =>
  d3.rollups(roleCounts[s], (v) => d3.sum(v, (d) => d.n), (d) => SHARE_GROUP[d.bucket]).map(([group, n]) => ({source: s, group, n, of: roleTotals[s], share: n / roleTotals[s]}))
);
const groupOrder = d3
  .groups(groupPoints, (d) => d.group)
  .map(([group, v]) => ({group, best: d3.max(v, (d) => d.share)}))
  .sort((a, b) => GREY.includes(a.group) - GREY.includes(b.group) || b.best - a.best)
  .map((d) => d.group);
const sourceLabel = {
  trudvsem: `${SOURCE_SHORT.trudvsem}, неделя с ${dateShort(lastFull.week)}`,
  habr: `${SOURCE_SHORT.habr}, неделя с ${dateShort(habrLast.week)}`,
  telegram: `Telegram, ${telegramLast13.length} недель`
};
```

<div class="card">

```js
Plot.plot({
  width,
  height: groupOrder.length * 28 + 60,
  marginLeft: width < 600 ? 150 : 240,
  marginRight: 20,
  x: {grid: true, label: "доля IT-вакансий источника (в Telegram постов)", tickFormat: pct, zero: true},
  y: {label: null, domain: groupOrder, tickFormat: (g) => (width < 600 && g.length > 22 ? g.slice(0, 21) + "…" : g)},
  color: {domain: FAMILY_ORDER.map((s) => sourceLabel[s]), range: FAMILY_ORDER.map((s) => SOURCE_COLOR[s]), legend: true},
  marks: [
    Plot.ruleY(groupOrder, {y: (d) => d, stroke: (g) => (GREY.includes(g) ? "#f0f0f0" : "#e0e0e0")}),
    Plot.dot(groupPoints, {
      y: "group",
      x: "share",
      fill: (d) => sourceLabel[d.source],
      fillOpacity: (d) => (GREY.includes(d.group) ? 0.45 : 1),
      r: 5.5,
      title: (d) => `${d.group}\n${sourceLabel[d.source]}: ${share(d.n, d.of)}`,
      tip: true
    })
  ]
})
```

<p class="note-small">Роль определена правилом по заголовку (в Telegram по первой строке поста). Серым роли без уточнения: на портале много заголовков вида «программист». <a href="./methodology#roles">Какие роли в группах</a>.</p>

<details>
<summary>Все роли по источникам</summary>

```js
const roleTable = Object.keys(ROLE).map((bucket) => {
  const row = {role: ROLE[bucket]};
  for (const s of FAMILY_ORDER) {
    const n = d3.sum(roleCounts[s].filter((d) => d.bucket === bucket), (d) => d.n);
    row[s] = n;
    row[`${s}_share`] = n / roleTotals[s];
  }
  return row;
}).filter((r) => FAMILY_ORDER.some((s) => r[s] > 0));
display(
  Inputs.table(roleTable, {
    columns: ["role", ...FAMILY_ORDER],
    header: {role: "Роль", trudvsem: SOURCE_SHORT.trudvsem, habr: SOURCE_SHORT.habr, telegram: "Telegram"},
    format: Object.fromEntries(FAMILY_ORDER.map((s) => [s, (n) => (width < 600 ? pct(n / roleTotals[s]) : share(n, roleTotals[s]))])),
    sort: "habr",
    reverse: true,
    rows: roleTable.length + 1,
    layout: width < 600 ? "fixed" : "auto",
    width: width < 600 ? {role: 130} : undefined
  })
);
```

</details>
</div>

## Telegram по годам

```js
const scopePick = view(
  Inputs.radio(new Map([["все каналы", "all"], [`${tgYears.panel_channels} каналов, которые писали каждый год ${tgYears.panel_years[0]}-${tgYears.panel_years.at(-1)}`, "panel"]]), {value: "all", label: "Каналы"})
);
```

```js
const yearRows = tgYears.years.filter((d) => d.scope === scopePick);
const lastYear = d3.max(yearRows, (d) => d.year);
const SERIES = [
  {key: "ml", label: "ML-посты среди IT-постов", n: "ml_posts", of: "it_posts", color: COLOR.accent},
  {key: "junior", label: "стажёры и джуны среди IT-постов с грейдом", n: "junior_posts", of: "graded_posts", color: COLOR.junior},
  {key: "mljunior", label: "стажёры и джуны среди ML-постов с грейдом", n: "ml_junior_posts", of: "ml_graded_posts", color: "#c39bd3"}
];
const yearPoints = SERIES.flatMap((s) => yearRows.map((d) => ({series: s.label, year: d.year, n: d[s.n], of: d[s.of], share: d[s.n] / d[s.of]})));
const SKILLS_SHOWN = ["RAG", "FastAPI", "Docker", "PyTorch", "pandas", "TensorFlow"];
const skillCols = width < 600 ? 2 : 3;
const skillYears = tgYears.skills
  .filter((d) => d.scope === scopePick && SKILLS_SHOWN.includes(d.skill))
  .map((d) => ({...d, share: d.posts_mentioning / d.ml_posts}));
const y0 = tgYears.panel_years[0];
const first = yearRows.find((d) => d.year === y0);
const last = yearRows.find((d) => d.year === lastYear);
```

<div class="grid grid-cols-2">
<div class="card">
<h3>Доля ML и доля начинающих</h3>

```js
resize((width) => Plot.plot({
  width,
  height: 280,
  marginLeft: 44,
  marginRight: 10,
  x: {label: null, tickFormat: (y) => `${y}`, ticks: yearRows.map((d) => d.year)},
  y: {grid: true, label: "доля", tickFormat: pct, zero: true},
  color: {domain: SERIES.map((s) => s.label), range: SERIES.map((s) => s.color), legend: true},
  marks: [
    Plot.lineY(yearPoints, {x: "year", y: "share", stroke: "series", strokeWidth: 2}),
    Plot.dot(yearPoints, {x: "year", y: "share", fill: "series", r: 3, title: (d) => `${d.year}, ${d.series}\n${share(d.n, d.of)}${d.year === lastYear ? `\n${lastYear} по неделю с ${dateLong(tgYears.last_week)}` : ""}`, tip: true}),
    Plot.ruleY([0])
  ]
}))
```

<p class="note-small">ML-посты: ${share(first.ml_posts, first.it_posts)} в ${y0} и ${share(last.ml_posts, last.it_posts)} в ${lastYear}. Стажёры и джуны среди постов с грейдом: ${share(first.junior_posts, first.graded_posts)} и ${share(last.junior_posts, last.graded_posts)}.</p>

</div>
<div class="card">
<h3>Навыки в ML-постах</h3>

```js
// One small chart per skill on a shared y scale, so the heights compare across skills.
const skillMax = d3.max(skillYears, (d) => d.share);
display(
  html`<div class="multiples" style="grid-template-columns: repeat(${skillCols}, 1fr)">${SKILLS_SHOWN.map((skill) => {
    const rows = skillYears.filter((d) => d.skill === skill);
    const lastRow = rows.at(-1);
    return html`<div><div class="multiple-title"><b>${skill}</b> ${pct(lastRow.share)}</div>${resize((w) =>
      Plot.plot({
        width: w,
        height: 110,
        marginLeft: 34,
        marginRight: 8,
        marginTop: 6,
        x: {label: null, tickFormat: (y) => `${y}`, ticks: [rows[0].year, lastYear]},
        y: {grid: true, label: null, tickFormat: (y) => `${Math.round(y * 100)} %`, domain: [0, skillMax], ticks: 3},
        marks: [
          Plot.lineY(rows, {x: "year", y: "share", stroke: COLOR.accent, strokeWidth: 2}),
          Plot.dot(rows, {x: "year", y: "share", r: 2, fill: COLOR.accent, title: (d) => `${d.skill}, ${d.year}\n${share(d.posts_mentioning, d.ml_posts)} ML-постов`, tip: true})
        ]
      })
    )}</div>`;
  })}</div>`
);
```

<p class="note-small">Доля ML-постов года, где назван навык. Навыки, роль и грейд из постов читает модель извлечения проекта.</p>

</div>
</div>

<p class="note-small">Посты с одной вакансией из публичных каналов, повторы за неделю убраны; ML-посты это роли ${tgYears.ml_roles.join(", ")}. Состав каналов меняется по годам, поэтому есть переключатель на каналы, которые писали все годы. ${lastYear} по неделю с ${dateLong(tgYears.last_week)}. <a href="./methodology#telegram">Охват Telegram</a>.</p>
