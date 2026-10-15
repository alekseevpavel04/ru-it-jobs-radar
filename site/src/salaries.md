---
title: Навыки и деньги
toc: false
---

# Навыки и деньги

```js
import {SKILL, GRADE, COLOR, int, share, signedPct} from "./components/labels.js";

const effects = FileAttachment("data/skill_effects.json").json();
const years = FileAttachment("data/telegram_years.json").json();
```

<p class="lead">Сколько обещают в вакансиях и платят ли больше за конкретный навык. Вилка из объявления, рубли в месяц до вычета налога. Это не выплаченная зарплата. <a href="./methodology#salaries">Подробнее в методике</a>.</p>

## За какой навык платят больше

```js
const level = Math.round(effects.ci_level * 100);
const holm = (e) => e.p_holm < 1 - effects.ci_level;
const k8s = effects.effects.find((e) => e.skill === "Kubernetes");
// The skills with the largest plain gap among those named in 100+ vacancies, and every skill that
// passed the Holm correction; the full list is in the table below.
const slopeSkills = effects.effects
  .filter((e) => e.rows_with_skill >= 100)
  .sort((a, b) => b.naive_pct - a.naive_pct)
  .slice(0, 9)
  .concat(effects.effects.filter((e) => holm(e)))
  .filter((e, i, all) => all.findIndex((x) => x.skill === e.skill) === i);
const hot = (e) => holm(e) || e.skill === "Kubernetes";
const name = (e) => SKILL[e.skill] ?? e.skill;
// Labels of close values are pushed apart symmetrically around their points until no two are closer
// than the gap; the points stay where they are.
function spreadLabels(rows, key, gap) {
  const out = rows.map((d) => ({...d, labelY: d[key]})).sort((a, b) => a[key] - b[key]);
  for (let pass = 0; pass < 200; pass++) {
    let moved = false;
    for (let i = 1; i < out.length; i++) {
      const overlap = gap - (out[i].labelY - out[i - 1].labelY);
      if (overlap > 1e-6) {
        out[i].labelY += overlap / 2;
        out[i - 1].labelY -= overlap / 2;
        moved = true;
      }
    }
    if (!moved) break;
  }
  return out;
}
const span = d3.max(slopeSkills, (e) => e.naive_pct) - Math.min(0, d3.min(slopeSkills, (e) => e.pct));
const leftLabels = spreadLabels(slopeSkills, "naive_pct", span / 19);
const rightLabels = spreadLabels(slopeSkills, "pct", span / 19);
const slopeRows = slopeSkills.flatMap((e) => [
  {skill: e.skill, side: "в лоб", value: e.naive_pct, e},
  {skill: e.skill, side: "при прочих равных", value: e.pct, e}
]);
const allEffects = effects.effects.slice().sort((a, b) => b.naive_pct - a.naive_pct);
```

<div class="card">

<p class="note-small">Слева вакансии с навыком и без сравниваются в лоб. Справа те же вакансии, но с одинаковыми ролью, грейдом, регионом, опытом и остальным стеком. Так Kubernetes из ${signedPct(k8s.naive_pct)} превращается в ${signedPct(k8s.pct)}, а интервал от ${signedPct(k8s.pct_ci_low)} до ${signedPct(k8s.pct_ci_high)} захватывает ноль.</p>

```js
{
  const narrow = width < 600;
  const L = 0, R = 1, off = narrow ? 0.07 : 0.04;
  const gutter = narrow ? 2.1 : 0.5;
  const label = (d, side) => (side === L ? `${name(d)} ${signedPct(d.naive_pct)}` : `${signedPct(d.pct)} ${name(d)}${holm(d) ? " *" : ""}`);
  const ink = (d) => (hot(d) ? COLOR.accent : COLOR.muted);
  display(Plot.plot({
    width,
    height: narrow ? 520 : 500,
    marginLeft: 4,
    marginRight: 4,
    marginTop: 40,
    x: {domain: [L - gutter, R + gutter], axis: null},
    y: {axis: null, inset: 10},
    style: {fontSize: narrow ? "11px" : "13px"},
    marks: [
      Plot.ruleY([0], {x1: L, x2: R, stroke: "#9a9a9a", strokeDasharray: "3,3"}),
      Plot.text([["в лоб", L], [narrow ? "при равных" : "при прочих равных", R]], {x: (d) => d[1], text: (d) => d[0], frameAnchor: "top", dy: -28, fontWeight: 700, fill: COLOR.bar}),
      Plot.link(slopeSkills, {x1: L, x2: R, y1: "naive_pct", y2: "pct", stroke: (d) => (hot(d) ? COLOR.accent : "#d9d6cf"), strokeWidth: (d) => (hot(d) ? 3.2 : 1.6)}),
      Plot.link(leftLabels, {x1: L, y1: "naive_pct", x2: L - off, y2: "labelY", stroke: "#cfccc4", strokeWidth: 0.8}),
      Plot.link(rightLabels, {x1: R, y1: "pct", x2: R + off, y2: "labelY", stroke: "#cfccc4", strokeWidth: 0.8}),
      Plot.dot(slopeSkills, {x: L, y: "naive_pct", r: (d) => (hot(d) ? 5 : 3.5), fill: (d) => (hot(d) ? COLOR.accent : "#b8b5ad")}),
      Plot.dot(slopeSkills, {x: R, y: "pct", r: (d) => (hot(d) ? 5 : 3.5), fill: (d) => (hot(d) ? COLOR.accent : "#b8b5ad")}),
      Plot.text(leftLabels, {x: L - off, y: "labelY", text: (d) => label(d, L), textAnchor: "end", dx: -4, fill: ink, fontWeight: (d) => (hot(d) ? 700 : 400)}),
      Plot.text(rightLabels, {x: R + off, y: "labelY", text: (d) => label(d, R), textAnchor: "start", dx: 4, fill: ink, fontWeight: (d) => (hot(d) ? 700 : 400)}),
      Plot.tip(slopeRows, Plot.pointer({x: (d) => (d.side === "в лоб" ? L : R), y: "value", title: (d) => `${name(d.e)}, вакансий с навыком ${int(d.e.rows_with_skill)}
в лоб ${signedPct(d.e.naive_pct)}
при прочих равных ${signedPct(d.e.pct)} [${signedPct(d.e.pct_ci_low)}; ${signedPct(d.e.pct_ci_high)}]`}))
    ]
  }));
}
```

<p class="note-small">Разница середины вилки у вакансий с навыком и без, ${int(effects.rows)} вакансий порталов с вилкой в рублях. * единственный навык, прошедший строгую поправку на ${effects.estimated} проверок. Это связь, а не причина: выучив навык, этих процентов к офферу не получишь. <a href="./methodology#skill-effects">Ограничения оценки</a>.</p>

<details class="more"><summary>Все ${effects.estimated} навыков</summary>

```js
Inputs.table(allEffects.map((e) => ({...e, skill: name(e)})), {
  columns: ["skill", "naive_pct", "pct", "pct_ci_low", "pct_ci_high", "rows_with_skill"],
  header: {skill: "Навык", naive_pct: "в лоб, %", pct: "при прочих равных, %", pct_ci_low: `${level} % от`, pct_ci_high: `${level} % до`, rows_with_skill: "вакансий"},
  format: {naive_pct: signedPct, pct: signedPct, pct_ci_low: signedPct, pct_ci_high: signedPct, rows_with_skill: int},
  rows: 14
})
```

</details>

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
