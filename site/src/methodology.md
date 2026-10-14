---
title: Методика и ограничения
toc:
  show: true
  label: На странице
---

# Методика и ограничения

```js
import {ROLE, ROLE_GROUP, ROLE_GROUP_ORDER, MART_GROUP, MART_GROUP_ORDER, SOURCE, SOURCE_SHORT, COLOR, int, pct, share, day, dateLong, dateShort, monthTick, ofChannels, days} from "./components/labels.js";

const meta = FileAttachment("data/meta.json").json();
const life = FileAttachment("data/lifetime.json").json();
const observed = FileAttachment("data/lifetime_observed.json").json();
const telegramRows = FileAttachment("data/weekly_telegram.json").json();
const fc = FileAttachment("data/forecast.json").json();
const effects = FileAttachment("data/skill_effects.json").json();
```

<h2 id="sources">Чей это срез рынка</h2>

Сайт показывает IT-вакансии из открытых источников. hh.ru в выборку не входит: у него нет открытого API для такой задачи. Поэтому цифры описывают эти источники, а не весь рынок IT-вакансий России. Источники считаются раздельно и не складываются: у них разные единицы счёта и разный охват.

| Источник | Что берётся | Перекос |
|---|---|---|
| «Работа в России» (trudvsem.ru) | открытые данные портала: ежедневные снимки всего набора вакансий и ежедневная выгрузка через API | много госсектора, бюджетных организаций и регионов; IT-роли часто названы общо: «программист», «инженер» |
| Хабр Карьера | вакансии IT-компаний | вакансий меньше, чем на портале, сбор начат недавно |
| Публичные Telegram-каналы | посты с вакансиями из веб-превью каналов | уклон в ML и Python; считаются посты, а не вакансии |
| SuperJob | вакансии через открытый API | сбор начат недавно; на сайт выйдет, когда накопится история |

```js
display(
  html`<ul>${Object.entries(meta.periods).map(
    ([source, p]) => html`<li><b>${SOURCE[source]}</b>: данные с ${dateLong(p.period_from)} по ${dateLong(p.period_to)}</li>`
  )}</ul>`
);
```

<p class="note-small">Для Telegram указано окно, по которому считаются навыки, роли и зарплаты. Ряд постов по неделям идёт с ${dateLong(meta.telegram_first_week_shown)}, годовые доли на главной охватывают всю историю каналов. Снимки портала есть с недели ${dateLong(meta.snapshot_first_week)}. Данные выгружены ${dateLong(meta.exported_at.slice(0, 10))}, коммит ${meta.commit}.</p>

## Что публикуется

Только агрегаты: число вакансий или постов по неделям, ролям, навыкам, формату работы и группам зарплат, каждое вместе со своим знаменателем, и оценки моделей проекта из файлов `results/` с их провенансом. Тексты вакансий Хабр Карьеры и постов Telegram на сайт не попадают, они хранятся только в локальном сыром слое. Персональные данные (имена контактов, телефоны, почты, ники) вычищаются ещё до записи в сырой слой.

## Как считается

**Вакансия.** Одна вакансия внутри источника считается один раз: если она пришла и через API, и в снимке портала, или выложена повторно, берутся поля её последней записи.

<h2 id="closures">Открытые, новые, закрытые</h2>

Вакансия открыта на неделе, если была видна хотя бы в один собранный день. Новая: впервые появилась на этой неделе; в первый день сбора новых нет, там всё уже открыто, поэтому первая неделя новых не показывает. Закрытая: её нет в ${life.grace_days} полных снимках портала подряд, потому что портал временами прячет открытые вакансии на несколько дней (для Хабр Карьеры: нет в трёх следующих полных днях сбора). Закрытие датируется первым днём, когда вакансии нет, и становится известно только после ${life.grace_days} полных снимков без неё, поэтому закрытия последних двух недель ещё подтверждаются и потом вырастут. Через API вакансия не закрывается, потому что API отдаёт только изменённые записи.

Неделя, собранная не целиком, занижена и на графиках отмечена пунктиром или бледным цветом. Неделя после пропуска снимков получает закрытия пропущенных недель. Недели без снимков остаются пустыми, без интерполяции.

<h2 id="roles">Роли</h2>

Правило по словам заголовка (в Telegram по первой строке поста, затем по хэштегам): первая подходящая роль выигрывает. Заголовки без IT-роли (на портале в IT-раздел попадают почтальоны и операторы, на Хабр Карьере продажи и тексты) из всех цифр исключены. Правило используется, пока для «Работы в России» и Хабр Карьеры нет полей, извлечённых моделью проекта; годовые доли Telegram, зарплаты Telegram и прогноз считаются по полям модели извлечения.

Группы ролей для недельной динамики:

```js
display(
  html`<ul>${ROLE_GROUP_ORDER.map(
    (group) =>
      html`<li><b>${group}</b>: ${Object.entries(ROLE_GROUP)
        .filter(([, g]) => g === group)
        .map(([bucket]) => ROLE[bucket])
        .join(", ")}</li>`
  )}</ul>`
);
```

На графике ролей главной страницы ML и Data Science вынесены из «Данных и ML» отдельной строкой, а «Программист без уточнения» и «Прочее IT» стоят внизу серым.

<h2 id="skills">Навыки</h2>

Навык засчитывается вакансии, если его название или синоним из словаря встречается в заголовке, тексте или тегах навыков источника. Одна вакансия засчитывается навыку один раз. Доля считается от всех вакансий источника и роли, включая те, где не нашлось ни одного навыка. Словарь пока ручной, поэтому широкие синонимы завышают отдельные строки: например, слово «интеграция» засчитывается навыку «Интеграции по API». Для отдельной роли не показаны навыки, названные меньше ${meta.skill_min_mentions} раз. «Без гибких навыков» убирает категории «гибкие навыки» и «иностранный язык».

<h2 id="telegram">Telegram</h2>

Считаются посты с вакансиями. Повтор того же текста в ту же неделю (в любом канале, без учёта ссылок, упоминаний и хэштегов) считается один раз. Пост с несколькими вакансиями не делится и идёт отдельной строкой «${ROLE.multi_post}». Пост не равен вакансии, а число постов зависит от того, какие каналы в выборке. Годовые доли на главной считаются по постам с одной вакансией; состав каналов по годам меняется, поэтому рядом дан тот же ряд по каналам, которые писали каждый год.

```js
const DAY_MS = 864e5;
const telegramEnd = day(meta.periods.telegram.period_to);
const telegramComplete = d3
  .rollups(telegramRows, (v) => ({posts: d3.sum(v, (d) => d.posts), before: d3.sum(v, (d) => d.posts_before_dedup), channels: v[0].channels_week}), (d) => d.week)
  .map(([week, s]) => ({week, date: day(week), ...s}))
  .filter((w) => +w.date + 6 * DAY_MS <= +telegramEnd)
  .sort((a, b) => a.date - b.date);
// Channel coverage of the last weeks against the half year before them: a rise in posts that comes
// with more channels is the sample growing.
const last4 = telegramComplete.slice(-4);
const earlier = telegramComplete.slice(-30, -4);
const earlierChannels = d3.max(earlier, (w) => w.channels);
const widerWeeks = last4.filter((w) => w.channels > earlierChannels);
```

<div class="card">
<h3>Охват Telegram по неделям</h3>

```js
Plot.plot({
  width,
  height: 240,
  marginLeft: 50,
  x: {type: "utc", label: null, tickFormat: monthTick},
  y: {grid: true, label: "постов за неделю", tickFormat: int},
  color: {domain: ["после удаления повторов", "все копии"], range: [COLOR.to, COLOR.all], legend: true},
  marks: [
    Plot.lineY(telegramComplete, {x: "date", y: "before", stroke: () => "все копии"}),
    Plot.lineY(telegramComplete, {x: "date", y: "posts", stroke: () => "после удаления повторов"}),
    Plot.dot(widerWeeks, {x: "date", y: "posts", r: 5, stroke: COLOR.to, fill: "white"}),
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

<p class="note-small">${meta.telegram_weeks_shown} недель с ${dateLong(meta.telegram_first_week_shown)}, неполная последняя неделя не показана. ${widerWeeks.length ? `Отмеченные точки: ${widerWeeks.map((w) => `на неделе с ${dateShort(w.week)} посты пришли из ${ofChannels(w.channels)}`).join(", ")}, а в ${earlier.length} неделях до этого не больше чем из ${ofChannels(earlierChannels)}. Рост в такие недели говорит об охвате выборки, а не о росте рынка.` : "Ряд это сигнал активности сообществ, а не объём рынка."}</p>

</div>

<h2 id="salaries">Зарплаты</h2>

Зарплата это вилка, которую работодатель написал в объявлении, а не то, сколько в итоге платят. Суммы взяты в рублях в месяц так, как их указал источник, вычет налога не учитывается. Вакансии с суммой в другой валюте в вилки не входят. На портале «Работа в России» верхняя граница часто совпадает с нижней: так устроена форма портала. Зарплаты Telegram, роль и грейд в них читает модель извлечения проекта; регион из постов в витрину зарплат пока не входит. Медианы и квартили считаются в витрине по вакансиям каждой группы, а не собираются из медиан её частей.

<p id="salary-threshold">Граница вилки показана, если её указали хотя бы ${meta.salary_min_n} вакансий группы: при меньшем числе 95-процентный интервал для медианы по порядковым статистикам совпадает со всем разбросом группы, от минимума до максимума. При ${meta.salary_min_n} вакансиях интервал от второго до предпоследнего значения покрывает медиану с вероятностью ${pct(meta.salary_min_n_coverage)}.</p>

<h2 id="skill-effects">Навык и вилка</h2>

Для ${effects.estimated} навыков, которые часто встречаются и часто отсутствуют в вакансиях с вилкой, оценена разница середины вилки при равных роли, грейде, регионе, формате работы, опыте, источнике и остальных навыках (частично линейная модель, double ML), на ${int(effects.rows)} вакансиях порталов. Поправку Холма на множественные проверки прошли ${effects.significant_holm} из ${effects.estimated}. Расчёт <code>${effects.source_file}</code>, коммит ${effects.commit}. Ограничения, как их записал расчёт:

${html`<ul>${effects.limits.map((t) => html`<li>${t}</li>`)}</ul>`}

<h2 id="junior">Начинающие</h2>

Вакансия открыта начинающему, если её грейд стажёр или джуниор, или если в ней сказано, что опыт не нужен. На портале «Работа в России» это поле требуемого опыта, равное нулю лет («без опыта»); у Хабр Карьеры и в Telegram такого поля нет, там ищется фраза в тексте. Поэтому «без опыта» у портала и у остальных источников это разные величины, и сравнимая доля на странице «Начинающим» считается только по грейду, среди вакансий, где грейд известен. Опыт «до года» отдельно не выделяется: поле портала считает опыт в целых годах. Навыки и зарплаты начинающих считаются по грейду, потому что витрины навыков и зарплат разбиты по грейду, а не по требуемому опыту.

<h2 id="survival">Время жизни вакансии</h2>

```js
const incident = life.medians.find((m) => m.group === "overall" && m.cohort === "incident");
const obsSnap = observed.filter((d) => d.source === "trudvsem_snapshot");
const obsTotal = obsSnap.find((d) => d.level === "source");
const obsRows = [
  {...obsTotal, label: "все IT-вакансии"},
  ...obsSnap
    .filter((d) => d.level === "group")
    .sort((a, b) => MART_GROUP_ORDER.indexOf(a.role_group) - MART_GROUP_ORDER.indexOf(b.role_group))
    .map((d) => ({...d, label: MART_GROUP[d.role_group]}))
].filter((d) => d.duration_days_median !== null);
const habrObs = observed.find((d) => d.source === "habr" && d.level === "source");
```

Срок жизни считается от даты создания вакансии на портале до её закрытия (правило закрытия выше). Окно наблюдения с ${dateLong(life.window_from)} по ${dateLong(life.window_to)}. Вакансии, которые к концу окна ещё открыты, учитываются как незакрытые (оценка Каплана-Мейера), а не выбрасываются. Проверка метода: если считать только вакансии, впервые появившиеся в окне, от первого появления, медиана ${days(incident.median_days)} [${int(incident.low)}; ${int(incident.high)}], вакансий ${int(incident.subjects)}. Расчёт <code>${life.source_file}</code>, коммит ${life.commit}.

Наблюдаемые закрытия считаются иначе: это сроки только тех вакансий, закрытие которых попало в окно сбора, от первого появления в снимке. Долгие вакансии успевают закрыться реже, поэтому эти сроки короче настоящих.

<div class="card">

```js
Plot.plot({
  width,
  height: obsRows.length * 30 + 50,
  marginLeft: width < 600 ? 150 : 230,
  marginRight: width < 600 ? 10 : 110,
  x: {grid: true, label: "дней до закрытия", domain: [0, d3.max(obsRows, (d) => d.duration_days_p90) * 1.1]},
  y: {label: null, domain: obsRows.map((d) => d.label)},
  marks: [
    Plot.link(obsRows, {y1: "label", y2: "label", x1: "duration_days_p25", x2: "duration_days_p75", stroke: COLOR.from, strokeWidth: 3}),
    Plot.tickX(obsRows, {y: "label", x: "duration_days_p90", stroke: COLOR.from}),
    Plot.dot(obsRows, {y: "label", x: "duration_days_median", fill: COLOR.to, r: 4, title: (d) => `${d.label}\nмедиана ${days(d.duration_days_median)}, квартили ${int(d.duration_days_p25)} и ${int(d.duration_days_p75)}, 90-й процентиль ${int(d.duration_days_p90)}\nзакрытий в расчёте ${int(d.closed_in_quantiles)} из ${int(d.vacancies)}`, tip: true}),
    width < 600 ? null : Plot.text(obsRows, {y: "label", text: (d) => `n ${int(d.closed_in_quantiles)} из ${int(d.vacancies)}`, frameAnchor: "right", dx: 105})
  ]
})
```

<p class="note-small">${SOURCE.trudvsem_snapshot}, сбор с ${dateLong(obsTotal.period_from)} по ${dateLong(obsTotal.period_to)}. В расчёт попали ${share(obsTotal.closed_in_quantiles, obsTotal.vacancies)} вакансий: не входят ещё открытые (${int(obsTotal.still_open)}), открытые уже в первый день сбора (${int(obsTotal.left_censored)}, пересекаются с остальными) и закрытые перепубликацией под новым номером (${int(obsTotal.closed_by_repost)}). Точка: медиана, отрезок: квартили, риска: 90-й процентиль.${habrObs ? ` ${SOURCE.habr}: собрано ${days(habrObs.window_days)}, из ${int(habrObs.vacancies)} вакансий закрылись ${int(habrObs.closed)}, сроки пока не считаются.` : ""}</p>

</div>

Ограничения модели выживаемости, как их записал расчёт:

${html`<ul>${life.limits.map((t) => html`<li>${t}</li>`)}</ul>`}

<h2 id="forecast">Прогноз спроса на навыки</h2>

```js
const forecastWeeks = [...new Set(fc.points.map((p) => p.week))].sort();
const level = Math.round(fc.interval_quantile * 100);
```

Прогнозируется доля IT-постов недели в публичных Telegram-каналах, где назван навык. Навыки в постах читает модель извлечения проекта, а не словарь. Вакансии порталов в прогнозе не участвуют: они собираются <!--n:models/forecast.json:metrics.data.all_portals.weeks.value:d-->15<!--/n--> недель, а для проверки прогноза на прошлых неделях нужно больше полугода. Ряд идёт с ${dateLong(fc.series_first_week)}, в нём ${int(fc.weeks)} недель; прогноз на ${fc.horizon} недели, с недели ${dateLong(forecastWeeks[0])} по неделю ${dateLong(forecastWeeks.at(-1))}, пересчитывается при каждой выгрузке сайта.

Метод выбран бэктестом со скользящей точкой старта (расчёт от ${dateLong(fc.created_at.slice(0, 10))}, файл <code>${fc.source_file}</code>): на каждой из <!--n:models/forecast.json:metrics.telegram.share.origins.value:d-->220<!--/n--> точек старта метод видит только недели до неё и прогнозирует следующие четыре, по <!--n:models/forecast.json:metrics.data.telegram.skills.value:d-->24<!--/n--> навыкам. Экспоненциальное сглаживание (ETS) точнее среднего за последние 4 недели у <!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.skills_better.n:d-->24<!--/n--> из <!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.skills_better.denominator:d-->24<!--/n--> навыков, но выигрыш маленький: MASE <!--n:models/forecast.json:metrics.telegram.share.ets.mase.value:.3f|ru-->0,926<!--/n--> против <!--n:models/forecast.json:metrics.telegram.share.moving_average.mase.value:.3f|ru-->0,960<!--/n-->, разность <!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.mase.value:+.3f|ru-->-0,034<!--/n-->, 95 % интервал [<!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.mase.ci_low:+.3f|ru-->-0,057<!--/n-->; <!--n:models/forecast.json:metrics.telegram.share.ets_minus_moving_average.mase.ci_high:+.3f|ru-->-0,015<!--/n-->]. MASE это средняя ошибка прогноза, делённая на среднюю смену доли от недели к неделе в истории навыка; ниже 1 значит точнее, чем повторить прошлую неделю. Бустинг по лагам среднего не обошёл: разность <!--n:models/forecast.json:metrics.telegram.share.boosting_minus_moving_average.mase.value:+.3f|ru-->+0,011<!--/n--> [<!--n:models/forecast.json:metrics.telegram.share.boosting_minus_moving_average.mase.ci_low:+.3f|ru-->-0,029<!--/n-->; <!--n:models/forecast.json:metrics.telegram.share.boosting_minus_moving_average.mase.ci_high:+.3f|ru-->+0,049<!--/n-->]. Поэтому прогноз стоит читать как сглаженный текущий уровень доли с разбросом, а не как предсказание тренда. Прогноз одинаковый на все ${fc.horizon} недели: экспоненциальное сглаживание без тренда даёт уровень.

**Что значит полоса.** Для каждого навыка и горизонта (1-${fc.horizon} недели) берутся ошибки ETS на ${int(fc.origins)} точках старта бэктеста, посчитанного на тех же неделях, что и прогноз. Полуширина полосы равна ${level}-му процентилю модуля этих ошибок, внизу полоса обрезана нулём. На самих точках бэктеста факт попадает в полосу в ${level} % случаев по построению, поэтому полоса проверена честнее: на каждой точке старта её строили только по ошибкам, уже известным к этому моменту (не меньше ${fc.interval_check_min_origins} прошлых точек). Тогда факт попал в полосу в ${share(fc.interval_check_covered, fc.interval_check_cells)} случаев по всем навыкам и горизонтам. Полоса описывает обычный разброс доли от недели к неделе и не учитывает сдвиги, которых не было в истории.

Ограничения прогноза:

${html`<ul>${["Доля считается только по постам Telegram, а пост не равен вакансии. Рынок за пределами этих каналов прогноз не описывает.", ...fc.limits].map((t) => html`<li>${t}</li>`)}</ul>`}

## Недельные выпуски

Сводка из витрин за прошедшую неделю, которую собирает сценарий n8n: [архив выпусков](./digest/). Свежая неделя всегда видна в блоке «Что изменилось за неделю» на [главной](./).

## Ограничения

- Нет hh.ru, поэтому доли ролей и навыков описывают открытые источники, а не весь рынок.
- История короткая: снимки портала с ${dateLong(meta.snapshot_first_week)}, Хабр Карьера с ${dateLong(meta.periods.habr.period_from)}. Недели, за которые снимков нет, на графиках пустые, неполные недели отмечены.
- Роли и навыки порталов определены правилами и словарём, а не разметкой. Правило ошибается на общих заголовках («инженер», «специалист»), словарь завышает навыки с широкими синонимами.
- Пост в Telegram не равен вакансии, а число постов зависит от того, какие каналы в выборке.
- Цифры пересчитываются при каждой выгрузке и могут меняться задним числом, когда догружаются пропущенные дни.
