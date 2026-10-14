---
title: Прогноз спроса на навыки
toc: true
---

# Прогноз спроса на навыки

```js
import {SKILL, int, pct, share, day, dateLong, dateShort, monthTick} from "./components/labels.js";

const fc = FileAttachment("data/forecast.json").json();
const history = FileAttachment("data/forecast_history.json").json();
```

Что прогнозируется: доля IT-постов недели в публичных Telegram-каналах, где назван навык. Навыки в постах читает модель извлечения проекта, а не словарь, поэтому доли здесь не совпадают со страницей «Навыки». Вакансии порталов в прогнозе не участвуют: они собираются <!--n:models/forecast.json:metrics.data.all_portals.weeks.value:d-->15<!--/n--> недель, а для проверки прогноза на прошлых неделях нужно больше полугода.

```js
const label = (skill) => SKILL[skill] ?? skill;
const lastHorizon = fc.points.filter((p) => p.horizon === fc.horizon);
const summary = history
  .filter((d) => d.week === fc.last_week)
  .map((d) => {
    const p = lastHorizon.find((q) => q.skill === d.skill);
    return {skill: d.skill, label: label(d.skill), n: d.posts_mentioning, of: d.posts_it, last: d.posts_mentioning / d.posts_it, forecast: p.forecast, low: p.low, high: p.high};
  })
  .sort((a, b) => b.last - a.last);
const forecastWeeks = [...new Set(fc.points.map((p) => p.week))].sort();
```

Ряд идёт с ${dateLong(fc.series_first_week)}, в нём ${int(fc.weeks)} недель, последняя полная неделя начинается ${dateLong(fc.last_week)}. Прогноз на ${fc.horizon} недели вперёд, с недели ${dateLong(forecastWeeks[0])} по неделю ${dateLong(forecastWeeks.at(-1))}. Он пересчитывается при каждой выгрузке сайта: закончившаяся неделя входит в ряд, и прогноз сдвигается.

```js
const skill = view(Inputs.select(new Map(summary.map((d) => [d.label, d.skill])), {label: "Навык", value: "Python"}));
const span = view(
  Inputs.radio(
    new Map([
      ["полгода", 26],
      ["год", 52],
      ["два года", 104]
    ]),
    {value: 26, label: "История"}
  )
);
```

```js
const observed = history
  .filter((d) => d.skill === skill)
  .map((d) => ({week: day(d.week), share: d.posts_mentioning / d.posts_it, n: d.posts_mentioning, of: d.posts_it}))
  .slice(-span);
const ahead = fc.points.filter((p) => p.skill === skill).map((p) => ({...p, week: day(p.week)}));
const last = observed.at(-1);
// The band and the forecast line start at the last observed week so they read as its continuation.
const band = [{week: last.week, low: last.share, high: last.share, forecast: last.share}, ...ahead];
const row = summary.find((d) => d.skill === skill);
```

<div class="card">

**${label(skill)}**: за неделю с ${dateLong(fc.last_week)} ${share(row.n, row.of)} IT-постов. Прогноз на неделю с ${dateLong(forecastWeeks.at(-1))}: ${pct(row.forecast)}, полоса ${Math.round(fc.interval_quantile * 100)}% от ${pct(row.low)} до ${pct(row.high)}.

```js
Plot.plot({
  width,
  height: 320,
  marginLeft: 50,
  x: {type: "utc", label: null, tickFormat: span > 52 ? monthTick : dateShort},
  y: {grid: true, label: "доля IT-постов недели", tickFormat: pct, zero: true},
  marks: [
    Plot.areaY(band, {x: "week", y1: "low", y2: "high", fill: "#4269d0", fillOpacity: 0.18}),
    Plot.lineY(observed, {x: "week", y: "share", stroke: "#333"}),
    Plot.dot(observed, {x: "week", y: "share", r: 2, fill: "#333", title: (d) => `неделя с ${dateLong(d.week)}\n${share(d.n, d.of)}`, tip: true}),
    Plot.lineY(band, {x: "week", y: "forecast", stroke: "#4269d0", strokeDasharray: "4,3"}),
    Plot.dot(ahead, {
      x: "week",
      y: "forecast",
      r: 3,
      fill: "#4269d0",
      title: (d) => `неделя с ${dateLong(d.week)}, прогноз на ${d.horizon} нед.\n${pct(d.forecast)}, полоса от ${pct(d.low)} до ${pct(d.high)}`,
      tip: true
    }),
    Plot.ruleY([0])
  ]
})
```

Чёрная линия: доля за неделю. Синяя пунктирная: прогноз, закрашено: полоса ${Math.round(fc.interval_quantile * 100)}%. Прогноз одинаковый на все ${fc.horizon} недели, потому что экспоненциальное сглаживание без тренда даёт уровень, а не рост или падение; полоса расширяется с горизонтом.

</div>

## Насколько прогноз лучше простого среднего

Метод выбран бэктестом со скользящей точкой старта (расчёт от ${dateLong(fc.created_at.slice(0, 10))}, файл <code>${fc.source_file}</code>): на каждой из <!--n:models/forecast.json:metrics.telegram.share.origins.value:d-->220<!--/n--> точек старта метод видит только недели до неё и прогнозирует следующие четыре, по <!--n:models/forecast.json:metrics.data.telegram.skills.value:d-->24<!--/n--> навыкам. Экспоненциальное сглаживание (ETS) точнее среднего за последние 4 недели у <!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.skills_better.n:d-->24<!--/n--> из <!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.skills_better.denominator:d-->24<!--/n--> навыков, но выигрыш маленький: MASE <!--n:models/forecast.json:metrics.telegram.share.ets.mase.value:.3f|ru-->0,926<!--/n--> против <!--n:models/forecast.json:metrics.telegram.share.moving_average.mase.value:.3f|ru-->0,960<!--/n-->, разность <!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.mase.value:+.3f|ru-->-0,034<!--/n-->, 95% интервал [<!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.mase.ci_low:+.3f|ru-->-0,057<!--/n-->; <!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.mase.ci_high:+.3f|ru-->-0,015<!--/n-->]. MASE это средняя ошибка прогноза, делённая на среднюю смену доли от недели к неделе в истории навыка; ниже 1 значит точнее, чем повторить прошлую неделю. Бустинг по лагам среднего не обошёл: разность <!--n:models/forecast.json:metrics.telegram.share.boosting_minus_moving_average.mase.value:+.3f|ru-->+0,011<!--/n--> [<!--n:models/forecast.json:metrics.telegram.share.boosting_minus_moving_average.mase.ci_low:+.3f|ru-->-0,029<!--/n-->; <!--n:models/forecast.json:metrics.telegram.share.boosting_minus_moving_average.mase.ci_high:+.3f|ru-->+0,049<!--/n-->]. Поэтому прогноз стоит читать как сглаженный текущий уровень доли с разбросом, а не как предсказание тренда.

## Что значит полоса

```js
const covered = fc.interval_check_covered / fc.interval_check_cells;
```

Для каждого навыка и каждого горизонта (1-${fc.horizon} недели) берутся ошибки ETS на ${int(fc.origins)} точках старта бэктеста, посчитанного на тех же неделях, что и прогноз. Полуширина полосы равна ${Math.round(fc.interval_quantile * 100)}-му процентилю модуля этих ошибок, внизу полоса обрезана нулём. На самих точках бэктеста факт попадает в полосу в ${Math.round(fc.interval_quantile * 100)}% случаев по построению, поэтому полоса проверена и честнее: на каждой точке старта её строили только по ошибкам, уже известным к этому моменту (не меньше ${fc.interval_check_min_origins} прошлых точек). Тогда факт попал в полосу в ${share(fc.interval_check_covered, fc.interval_check_cells)} случаев по всем навыкам и горизонтам. Полоса описывает обычный разброс доли от недели к неделе и не учитывает сдвиги, которых не было в истории.

## Все навыки

<div class="note-small">

Доля за неделю с ${dateLong(fc.last_week)} и прогноз на неделю с ${dateLong(forecastWeeks.at(-1))} (горизонт ${fc.horizon} недели) с полосой ${Math.round(fc.interval_quantile * 100)}%.

</div>

```js
Inputs.table(summary, {
  columns: width < 600 ? ["label", "last", "forecast"] : ["label", "n", "last", "forecast", "low", "high"],
  header: {
    label: "Навык",
    n: "Постов",
    last: "Доля",
    forecast: "Прогноз",
    low: "Полоса, от",
    high: "Полоса, до"
  },
  format: {n: (n) => `${int(n)} из ${int(summary[0].of)}`, last: pct, forecast: pct, low: pct, high: pct},
  sort: "last",
  reverse: true,
  rows: summary.length + 1
})
```

## Ограничения

```js
const limits = ["Доля считается только по постам Telegram, а пост не равен вакансии. Рынок за пределами этих каналов прогноз не описывает.", ...fc.limits];
display(html`<ul>${limits.map((text) => html`<li>${text}</li>`)}</ul>`);
```
