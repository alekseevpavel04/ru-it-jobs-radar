---
title: Время жизни вакансии
toc: true
---

# Сколько вакансия остаётся открытой

```js
import {LIFETIME, SOURCE, int, pct, dateLong} from "./components/labels.js";

const life = FileAttachment("data/lifetime.json").json();
```

<div class="card">

**Страница в работе.** Витрины со сроками жизни по всем источникам пока нет; здесь медианы из модели выживаемости, посчитанной по ежедневным снимкам портала «Работа в России». Файл расчёта: <code>${life.source_file}</code>, расчёт от ${dateLong(life.created_at.slice(0, 10))}.

</div>

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

## Медианы по группам

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

## Ограничения расчёта

${html`<ul>${life.limits.map((t) => html`<li>${t}</li>`)}</ul>`}

<div class="note-small">

Расчёт ${life.source_script}, коммит ${life.commit}. Медианы и интервалы перенесены в данные сайта без изменений.

</div>
