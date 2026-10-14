---
title: Зарплаты
---

# Зарплаты в объявлениях

```js
import {ROLE, XROLE, SOURCE_SHORT, SOURCE_COLOR, FAMILY_ORDER, GRADE, GRADE_ORDER, REGION, REGION_ORDER, COLOR, int, pct, share, rub, signedPct, dateLong, sourcePicker} from "./components/labels.js";
import {shownBounds, rangePlot, salaryTable} from "./components/salary.js";

const meta = FileAttachment("data/meta.json").json();
const salaries = FileAttachment("data/salaries.json").json();
const telegram = FileAttachment("data/salaries_telegram.json").json();
const effects = FileAttachment("data/skill_effects.json").json();
```

<p class="lead">Вилка из объявления, рубли в месяц до вычета налога. Это не выплаченная зарплата. <a href="./methodology#salaries">Подробнее в методике</a>.</p>

```js
const minN = meta.salary_min_n;
const sourceInput = sourcePicker(Inputs, FAMILY_ORDER, "telegram");
const source = Generators.input(sourceInput);
const regionInput = Inputs.select(new Map([["вся выборка", null], ...REGION_ORDER.map((r) => [REGION[r], r])]), {label: "Регион"});
const region = Generators.input(regionInput);
const gradeInput = Inputs.select(new Map([["все", null], ...GRADE_ORDER.map((g) => [GRADE[g], g])]), {label: "Грейд"});
const grade = Generators.input(gradeInput);
```

<div class="filters">${sourceInput}${regionInput}${gradeInput}</div>

```js
const isTg = source === "telegram";
// Rows of one level cover every vacancy of the source once; nothing is ever summed across levels.
const at = (level) => (isTg ? telegram.filter((d) => d.level === level) : salaries.filter((d) => d.level === level && d.source === source));
const unit = isTg ? "постов" : "вакансий";
const regionUsed = isTg ? null : region;
const sourceRow = (s) => (s === "telegram" ? telegram.find((d) => d.level === "source") : salaries.find((d) => d.level === "source" && d.source === s));
const withAmount = (d) => d.with_salary_rub ?? d.with_salary_rub_month;
const period = (s) => (s === "telegram" ? telegram[0] : meta.periods[s]);
```

<div class="grid grid-cols-3">
${FAMILY_ORDER.map((s) => {
  const row = sourceRow(s);
  return html`<div class="card" style=${`border-top: 3px solid ${SOURCE_COLOR[s]}`}>
    <h2>${SOURCE_SHORT[s]}</h2>
    <div class="big">${row.with_from >= minN ? rub(row.salary_from_median) : "мало данных"}</div>
    <p>медиана «от», n ${int(row.with_from)}; «до» ${row.with_to >= minN ? rub(row.salary_to_median) : "мало данных"}</p>
    <p class="sub">вилку пишут ${share(withAmount(row), row.vacancies)} ${s === "telegram" ? "постов" : "вакансий"} с ${dateLong(period(s).period_from)} по ${dateLong(period(s).period_to)}</p>
  </div>`;
})}
</div>

${isTg && region !== null ? html`<p class="note-small">Регион из постов Telegram пока не извлекается в витрину зарплат, фильтр региона для Telegram не действует.</p>` : ""}

## Лестница грейдов

```js
const LADDER = ["intern", "junior", "middle", "senior", "lead"];
const ladderRows = (regionUsed === null ? at("grade") : at("grade_region").filter((d) => d.region_group === regionUsed)).filter((d) => LADDER.includes(d.grade));
const ladderShown = shownBounds(ladderRows, minN, (d) => GRADE[d.grade]).sort((a, b) => LADDER.indexOf(a.grade) - LADDER.indexOf(b.grade));
const ladderHidden = ladderRows.filter((d) => d.with_from < minN && d.with_to < minN);
```

<div class="card">

<p class="note-small">${SOURCE_SHORT[source]}${regionUsed ? `, ${REGION[regionUsed]}` : ""}. Точка: медиана, отрезок: от 25-го до 75-го процентиля.${ladderHidden.length ? ` Мало данных: ${ladderHidden.map((d) => GRADE[d.grade]).join(", ")}.` : ""}</p>

```js
ladderShown.length === 0 ? html`<p>Ни один грейд не набрал ${minN} ${unit} с указанной границей.</p>` : rangePlot(ladderShown, {width, labelWidth: 110, sort: false, unit})
```

${source === "trudvsem" ? html`<p class="note-small">На портале грейд берётся из заголовка: «ведущий специалист» в госсекторе попадает в сеньоры, поэтому сеньор здесь может стоять ниже мидла.</p>` : ""}

</div>

## По ролям

```js
// The level of mart_salaries (or of the extracted mart for Telegram) that splits by exactly the filters.
const roleLevel = isTg ? (grade === null ? "role" : "cell") : grade === null ? (regionUsed === null ? "bucket" : "bucket_region") : regionUsed === null ? "bucket_grade" : "cell";
const roleRows = at(roleLevel).filter((d) => (grade === null || d.grade === grade) && (regionUsed === null || d.region_group === regionUsed));
const roleName = (d) => (isTg ? XROLE[d.role] ?? d.role : ROLE[d.role_bucket]);
const roleShown = shownBounds(roleRows, minN, roleName);
const roleHidden = roleRows.filter((d) => d.with_from < minN && d.with_to < minN);
const roleSorted = roleShown.slice().sort((a, b) => (b.from ?? b.to) - (a.from ?? a.to));
```

<div class="card">

<p class="note-small">${SOURCE_SHORT[source]}${grade ? `, ${GRADE[grade]}` : ""}${regionUsed ? `, ${REGION[regionUsed]}` : ""}: ролей с достаточным числом ${unit} ${int(roleShown.length)} из ${int(roleRows.length)}.${isTg ? " Роль в Telegram определяет модель извлечения." : ""}</p>

```js
roleShown.length === 0 ? html`<p>Ни одна роль не набрала ${minN} ${unit} с указанной границей.</p>` : rangePlot(roleSorted, {width, labelWidth: 230, sort: false, unit})
```

<p class="note-small">Граница показана, если её указали ${minN} ${unit} и больше (<a href="./methodology#salary-threshold">почему</a>).${roleHidden.length ? ` Мало данных: ${roleHidden.map(roleName).join(", ")}.` : ""}</p>

<details>
<summary>Таблица</summary>

```js
salaryTable(roleSorted, {width, header: "Роль", countHeader: isTg ? "Постов" : "Вакансий"})
```

</details>
</div>

## Навык и вилка

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
const slopeRows = slopeSkills.flatMap((e) => [
  {skill: e.skill, side: "в лоб", value: e.naive_pct, e},
  {skill: e.skill, side: "при прочих равных", value: e.pct, e}
]);
const allEffects = effects.effects.slice().sort((a, b) => b.naive_pct - a.naive_pct);
// Labels of close values are nudged apart, the points stay where they are.
function spreadLabels(rows, gap) {
  const sorted = rows.slice().sort((a, b) => a.value - b.value);
  let last = -Infinity;
  return sorted.map((d) => ({...d, labelY: (last = Math.max(d.value, last + gap))}));
}
const naiveSpan = d3.extent(slopeSkills, (e) => e.naive_pct);
const leftLabels = spreadLabels(slopeRows.filter((d) => d.side === "в лоб"), (naiveSpan[1] - Math.min(0, naiveSpan[0])) / 18);
```

<div class="card">

<p class="note-small">Слева вакансии с навыком и без сравниваются в лоб, справа те же вакансии при одинаковых роли, грейде, регионе, опыте и остальном стеке. Kubernetes: ${signedPct(k8s.naive_pct)} в лоб и ${signedPct(k8s.pct)} [${signedPct(k8s.pct_ci_low)}; ${signedPct(k8s.pct_ci_high)}] при прочих равных.</p>

```js
Plot.plot({
  width,
  height: 460,
  marginLeft: width < 600 ? 120 : 170,
  marginRight: width < 600 ? 120 : 170,
  x: {domain: ["в лоб", "при прочих равных"], label: null, axis: "top", padding: 0},
  y: {axis: null, inset: 16},
  marks: [
    Plot.ruleY([0], {stroke: "#888", strokeDasharray: "3,3"}),
    Plot.line(slopeRows, {x: "side", y: "value", z: "skill", stroke: (d) => (hot(d.e) ? COLOR.accent : "#cfccc4"), strokeWidth: (d) => (hot(d.e) ? 3 : 1.5)}),
    Plot.dot(slopeRows, {x: "side", y: "value", r: 3.5, fill: (d) => (hot(d.e) ? COLOR.accent : "#b8b5ad"), tip: true,
      title: (d) => `${d.skill}, вакансий с навыком ${int(d.e.rows_with_skill)}\nв лоб ${signedPct(d.e.naive_pct)}\nпри прочих равных ${signedPct(d.e.pct)} [${signedPct(d.e.pct_ci_low)}; ${signedPct(d.e.pct_ci_high)}]`}),
    Plot.text(leftLabels, {x: "side", y: "labelY", text: (d) => `${d.skill} ${signedPct(d.value)}`, textAnchor: "end", dx: -8, fill: (d) => (hot(d.e) ? COLOR.accent : "#6b6963"), fontWeight: (d) => (hot(d.e) ? "bold" : "normal")}),
    Plot.text(slopeRows.filter((d) => d.side !== "в лоб" && hot(d.e)), {x: "side", y: "value", text: (d) => `${signedPct(d.value)} ${d.skill}${holm(d.e) ? " *" : ""}`, textAnchor: "start", dx: 8, fill: COLOR.accent, fontWeight: "bold"})
  ]
})
```

<p class="note-small">* прошёл поправку Холма на ${effects.estimated} проверок, таких ${effects.significant_holm} из ${effects.estimated}. Вакансии порталов с вилкой в рублях, ${int(effects.rows)} строк; навыки извлечены моделью проекта. <a href="./methodology#skill-effects">Ограничения оценки</a>.</p>

<details><summary>Все ${effects.estimated} навыков</summary>

```js
Inputs.table(allEffects, {
  columns: ["skill", "naive_pct", "pct", "pct_ci_low", "pct_ci_high", "rows_with_skill"],
  header: {skill: "Навык", naive_pct: "в лоб, %", pct: "при прочих равных, %", pct_ci_low: `${level} % от`, pct_ci_high: `${level} % до`, rows_with_skill: "вакансий"},
  format: {naive_pct: signedPct, pct: signedPct, pct_ci_low: signedPct, pct_ci_high: signedPct, rows_with_skill: int},
  rows: 14
})
```

</details>

</div>
