---
title: Навыки и деньги
toc: false
---

# Навыки и деньги

```js
import {SKILL, GRADE, COLOR, int, share, signedPct} from "./components/labels.js";
import {FDR, effectsChart} from "./components/effects.js";

const effects = FileAttachment("data/skill_effects.json").json();
const years = FileAttachment("data/telegram_years.json").json();
```

<p class="lead">Сколько добавляет к вилке конкретный навык и сколько обещают по грейдам. Вилка из объявления, рубли в месяц до вычета налога. Это не выплаченная зарплата. <a href="./methodology#salaries">Подробнее в методике</a>.</p>

## Сколько добавляет навык к вилке

```js
const name = (e) => SKILL[e.skill] ?? e.skill;
const level = Math.round(effects.ci_level * 100);
const k8s = effects.effects.find((e) => e.skill === "Kubernetes");
// Temporary: row density for the author to pick one.
const compact = view(Inputs.radio(new Map([["обычная", false], ["компактная", true]]), {label: "Плотность", value: false}));
```

<div class="card">

<p class="note-small">Сравниваются похожие вакансии: та же роль, грейд, регион, опыт, формат работы и остальной стек, одни с навыком, другие без. Точка показывает, на сколько процентов выше или ниже середина вилки у вакансий с навыком, полоса показывает ${level} % интервал. Чем меньше сравнимых вакансий, тем шире полоса: у Kubernetes их ${int(k8s.rows_kept)}, поэтому его ${signedPct(k8s.pct)} неотличимы от нуля. Навык попадает в «повышают» или «понижают», если эффект подтверждён с поправкой на ${effects.estimated} проверок (доля ложных находок до ${Math.round(FDR * 100)} %, поправка Бенджамини-Хохберга): таких ${effects.significant_bh} из ${effects.estimated}.</p>

```js
effectsChart(effects.effects, {width, name, compact})
```

<p class="note-small">${int(effects.rows)} вакансий с вилкой в рублях, почти все с портала «Работа в России». Это связь, а не обещание: выучив навык, этих процентов к офферу не получишь. <a href="./methodology#skill-effects">Ограничения оценки</a>.</p>

</div>

## Сколько обещают по грейдам

```js
const LADDER = ["junior", "middle", "senior"];
const k = (x) => Math.round(x / 1000);
const cells = years.salary_grades.filter((g) => LADDER.includes(g.grade) && g.from_median !== null && g.to_median !== null);
const grades = LADDER.filter((g) => cells.some((c) => c.grade === g));
const SCOPE = {ml: {label: "ML", color: COLOR.accent, dy: -9}, it: {label: "всё IT", color: COLOR.bar, dy: 9}};
const ladder = cells.map((c) => ({...c, gradeName: GRADE[c.grade], ...SCOPE[c.scope]}));
const narrow = width < 600;
```

<div class="card">

<p class="note-small">Медианы нижней и верхней границы вилки в постах Telegram за ${years.salary_years[0]}-${years.salary_years.at(-1)}, тысячи рублей в месяц. Медиана показана, если границу указали хотя бы 9 постов.</p>

```js
Plot.plot({
  width,
  height: grades.length * 78 + 40,
  marginLeft: narrow ? 64 : 90,
  marginRight: narrow ? 8 : 20,
  x: {domain: [0, d3.max(ladder, (d) => d.to_median) * (narrow ? 1.9 : 1.55)], grid: true, tickFormat: k, label: null, ticks: narrow ? 4 : 7},
  y: {domain: grades.map((g) => GRADE[g]), label: null, tickSize: 0},
  style: {fontSize: "13px"},
  marks: ["ml", "it"].flatMap((scope) => {
    const part = ladder.filter((d) => d.scope === scope);
    const {color, dy} = SCOPE[scope];
    return [
      Plot.link(part, {y1: "gradeName", y2: "gradeName", x1: "from_median", x2: "to_median", stroke: color, strokeWidth: 12, strokeLinecap: "round", dy}),
      Plot.text(part, {y: "gradeName", x: "to_median", dy, dx: 12, textAnchor: "start", text: (d) => `${d.label}: от ${k(d.from_median)} до ${k(d.to_median)} тыс.`, fill: color, fontWeight: 650}),
      Plot.tip(part, Plot.pointer({y: "gradeName", x: "to_median", dy, title: (d) => `${d.gradeName}, ${d.label}
вилку указали ${share(d.stated, d.posts)}
медиана «от» по ${int(d.from_rows)}, «до» по ${int(d.to_rows)}`}))
    ];
  })
})
```

</div>
