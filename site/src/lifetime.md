---
title: Время жизни вакансии
toc: true
---

# Сколько вакансия остаётся открытой

```js
import {LIFETIME, MART_GROUP, MART_GROUP_ORDER, SOURCE, int, pct, share, dateLong} from "./components/labels.js";

const life = FileAttachment("data/lifetime.json").json();
const observed = FileAttachment("data/lifetime_observed.json").json();
```

На странице два разных числа, и они отвечают на разные вопросы.

- **Оценка выживаемости** (модель проекта, файл <code>${life.source_file}</code>, расчёт от ${dateLong(life.created_at.slice(0, 10))}): сколько вакансия живёт, с учётом тех, что к концу наблюдения ещё открыты. Это ответ на вопрос «сколько обычно висит вакансия».
- **Наблюдаемые закрытия** (витрина mart_lifetime): сроки только тех вакансий, закрытие которых попало в окно сбора. Долгие вакансии успевают закрыться реже, поэтому эти сроки короче настоящих; они показывают, как распределены уже случившиеся закрытия.

Оба расчёта сейчас стоят на ежедневных снимках портала «Работа в России»: Хабр Карьера собирается слишком недолго.

## Оценка выживаемости

```js
const overall = life.medians.find((m) => m.group === "overall" && m.cohort === "prevalent");
const days = (x) => (x === null ? "не оценивается" : `${int(x)} дн.`);
const interval = (m) =>
  m.low === null ? "" : m.high === null ? `95% интервал от ${int(m.low)} дн., верхняя граница не оценивается` : `95% интервал от ${int(m.low)} до ${int(m.high)} дн.`;
```

Срок жизни считается от даты создания вакансии на портале до её закрытия, а закрытой вакансия считается, когда её ${life.grace_days} собранных дней подряд нет среди открытых в снимке портала. Окно наблюдения с ${dateLong(life.window_from)} по ${dateLong(life.window_to)}. Вакансии, которые к концу окна ещё открыты, учитываются как незакрытые (оценка Каплана-Мейера), а не выбрасываются.

<div class="grid grid-cols-2">
  <div class="card">
    <h2>Медиана, все IT-вакансии портала</h2>
    <div class="big">${days(overall.median_days)}</div>
    ${interval(overall)}; вакансий ${int(overall.subjects)}, из них закрылись в окне ${int(overall.events)}
  </div>
  <div class="card">
    <h2>Доля ещё открытых, по возрасту вакансии</h2>
    ${life.overall_curve.map((p) => html`<div>${p.day} дн.: ${pct(p.survival)} (под наблюдением ${int(p.at_risk)})</div>`)}
  </div>
</div>

### Медианы по группам

```js
const cohort = view(
  Inputs.radio(
    new Map([
      ["от даты создания, все вакансии", "prevalent"],
      ["от первого появления, только новые", "incident"]
    ]),
    {value: "prevalent", label: "Отсчёт"}
  )
);
```

```js
// group and level order of the model (src/radar/models/survival.py, COVARIATES)
const ORDER = {
  role: ["dev", "ml_ds_data", "analyst", "qa_devops_sys", "other_it"],
  experience: ["none", "1-2", "3-5", "6+"],
  salary: ["given", "not_given"],
  region: ["moscow", "spb", "other", "remote"]
};
const GROUPS = Object.keys(ORDER);
const rank = (m) => GROUPS.indexOf(m.group) * 100 + (ORDER[m.group].indexOf(m.level) + 1 || 99);
const rows = life.medians
  .filter((m) => m.cohort === cohort && m.group !== "overall")
  .sort((a, b) => rank(a) - rank(b))
  .map((m) => ({...m, groupLabel: LIFETIME[m.group], label: LIFETIME[m.level] ?? m.level}));
const estimable = rows.filter((m) => m.median_days !== null);
const notEstimable = rows.filter((m) => m.median_days === null);
const xMax = d3.max(estimable, (m) => m.high ?? m.median_days) * 1.1;
```

<div class="card">

Точка: медиана в днях, отрезок: 95-процентный интервал. Если верхняя граница не оценивается, отрезок доходит до края графика. Подпись: сколько вакансий в группе и сколько из них закрылись в окне.${notEstimable.length ? ` Медиана не оценивается: ${notEstimable.map((m) => `${m.groupLabel.toLowerCase()}, ${m.label} (вакансий ${int(m.subjects)})`).join("; ")}.` : ""}

```js
Plot.plot({
  width,
  height: estimable.length * 26 + 80,
  marginLeft: width < 600 ? 200 : 240,
  marginRight: width < 600 ? 10 : 150,
  x: {grid: true, label: "дней", domain: [0, xMax]},
  y: {label: null, domain: estimable.map((m) => `${m.groupLabel}: ${m.label}`)},
  marks: [
    Plot.link(estimable, {y1: (m) => `${m.groupLabel}: ${m.label}`, y2: (m) => `${m.groupLabel}: ${m.label}`, x1: "low", x2: (m) => m.high ?? xMax, stroke: "#4269d0", strokeWidth: 2}),
    Plot.dot(estimable, {
      y: (m) => `${m.groupLabel}: ${m.label}`,
      x: "median_days",
      fill: "#4269d0",
      r: 4,
      title: (m) => `${m.groupLabel}: ${m.label}\nмедиана ${days(m.median_days)}\n${interval(m)}\nвакансий ${int(m.subjects)}, закрылись ${int(m.events)}`,
      tip: true
    }),
    width < 600
      ? null
      : Plot.text(estimable, {y: (m) => `${m.groupLabel}: ${m.label}`, text: (m) => `${int(m.events)} из ${int(m.subjects)}`, frameAnchor: "right", dx: 140})
  ]
})
```

</div>

## Наблюдаемые закрытия

```js
const obsSources = [...new Set(observed.map((d) => d.source))];
const obsSnap = observed.filter((d) => d.source === "trudvsem_snapshot");
const obsTotal = obsSnap.find((d) => d.level === "source");
const obsGroups = obsSnap
  .filter((d) => d.level === "group")
  .sort((a, b) => MART_GROUP_ORDER.indexOf(a.role_group) - MART_GROUP_ORDER.indexOf(b.role_group))
  .map((d) => ({...d, label: MART_GROUP[d.role_group]}));
const obsRows = [{...obsTotal, label: "все IT-вакансии"}, ...obsGroups];
const obsShown = obsRows.filter((d) => d.duration_days_median !== null);
const habrObs = observed.find((d) => d.source === "habr" && d.level === "source");
```

<div class="card">

${SOURCE.trudvsem_snapshot}, сбор с ${dateLong(obsTotal.period_from)} по ${dateLong(obsTotal.period_to)} (${int(obsTotal.window_days)} дн.). Сроки от первого появления в снимке до закрытия. В расчёт попали ${share(obsTotal.closed_in_quantiles, obsTotal.vacancies)} вакансий: не входят ещё открытые (${int(obsTotal.still_open)}), открытые уже в первый день сбора, когда начало неизвестно (${int(obsTotal.left_censored)}, пересекаются с остальными), и закрытые перепубликацией под новым номером (${int(obsTotal.closed_by_repost)}). Точка: медиана, отрезок: от 25-го до 75-го процентиля, риска: 90-й процентиль. Вакансия здесь это номер объявления у источника, а не сущность после дедупликации, как в модели.

```js
Plot.plot({
  width,
  height: obsShown.length * 30 + 50,
  marginLeft: width < 600 ? 170 : 230,
  marginRight: width < 600 ? 10 : 110,
  x: {grid: true, label: "дней до закрытия", domain: [0, d3.max(obsShown, (d) => d.duration_days_p90) * 1.1]},
  y: {label: null, domain: obsShown.map((d) => d.label)},
  marks: [
    Plot.link(obsShown, {y1: "label", y2: "label", x1: "duration_days_p25", x2: "duration_days_p75", stroke: "#efb118", strokeWidth: 3}),
    Plot.tickX(obsShown, {y: "label", x: "duration_days_p90", stroke: "#efb118"}),
    Plot.dot(obsShown, {
      y: "label",
      x: "duration_days_median",
      fill: "#efb118",
      r: 4,
      title: (d) =>
        `${d.label}
медиана ${int(d.duration_days_median)} дн., квартили ${int(d.duration_days_p25)} и ${int(d.duration_days_p75)}, 90-й процентиль ${int(d.duration_days_p90)}
закрытий в расчёте ${int(d.closed_in_quantiles)} из ${int(d.vacancies)} вакансий`,
      tip: true
    }),
    width < 600
      ? null
      : Plot.text(obsShown, {y: "label", text: (d) => `n ${int(d.closed_in_quantiles)} из ${int(d.vacancies)}`, frameAnchor: "right", dx: 105})
  ]
})
```

${habrObs ? `${SOURCE.habr}: собрано ${int(habrObs.window_days)} дн., из ${int(habrObs.vacancies)} вакансий закрылись ${int(habrObs.closed)}, сроки пока не считаются.` : ""}

</div>

## Ограничения расчёта

Ограничения модели выживаемости, как их записал расчёт:

${html`<ul>${life.limits.map((t) => html`<li>${t}</li>`)}</ul>`}

<div class="note-small">

Расчёт ${life.source_script}, коммит ${life.commit}. Медианы и интервалы перенесены в данные сайта без изменений.

</div>
