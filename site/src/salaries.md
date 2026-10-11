---
title: Зарплаты
toc: false
---

# Зарплаты в объявлениях

```js
import {ROLE, SOURCE, GRADE, GRADE_ORDER, REGION, REGION_ORDER, MART_GROUP, MART_GROUP_ORDER, int, pct, share, rub, dateLong} from "./components/labels.js";
import {shownBounds, rangePlot} from "./components/salary.js";

const meta = FileAttachment("data/meta.json").json();
const salaries = FileAttachment("data/salaries.json").json();
```

```js
const FAMILIES = ["trudvsem", "habr"];
const minN = meta.salary_min_n;
// Rows of one level cover every vacancy of the source once; nothing is ever summed across levels.
const at = (level, source) => salaries.filter((d) => d.level === level && d.source === source);
const DIMENSIONS = {
  group: {name: "группа ролей", key: (d) => MART_GROUP[d.role_group], order: MART_GROUP_ORDER.map((g) => MART_GROUP[g])},
  grade: {name: "грейд", key: (d) => GRADE[d.grade], order: GRADE_ORDER.map((g) => GRADE[g])},
  region: {name: "регион", key: (d) => REGION[d.region_group], order: REGION_ORDER.map((r) => REGION[r])},
  group_grade: {name: "группа ролей и грейд", key: (d) => `${MART_GROUP[d.role_group]} · ${GRADE[d.grade]}`, order: null},
  bucket: {name: "роль", key: (d) => ROLE[d.role_bucket], order: null}
};
const dimensionInput = (levels) =>
  Inputs.radio(new Map(levels.map((l) => [DIMENSIONS[l].name, l])), {value: levels[0], label: "В разрезе"});
const sourceInput = () => Inputs.radio(new Map(FAMILIES.map((s) => [SOURCE[s], s])), {value: "trudvsem", label: "Источник"});
```

Зарплата здесь это вилка, которую работодатель написал в объявлении, а не то, сколько в итоге платят. Суммы взяты в рублях в месяц так, как их указал источник, вычет налога не учитывается. Вакансии с суммой в другой валюте в вилки не входят. На портале «Работа в России» верхняя граница часто совпадает с нижней: так устроена форма портала. Telegram на этой странице нет: зарплаты из постов пока не извлекаются.

<div class="grid grid-cols-2">
${FAMILIES.map((s) => {
  const [row] = at("source", s);
  return html`<div class="card">
    <h2>${SOURCE[s]}</h2>
    <div class="big">${pct(row.with_salary_rub / row.vacancies)}</div>
    вакансий указывают сумму в рублях: ${share(row.with_salary_rub, row.vacancies)}; в другой валюте ${int(row.with_salary_other_currency)}. Медиана «от» ${row.with_from >= minN ? `${rub(row.salary_from_median)} (n ${int(row.with_from)})` : "не показана, мало вакансий"}, медиана «до» ${row.with_to >= minN ? `${rub(row.salary_to_median)} (n ${int(row.with_to)})` : "не показана, мало вакансий"}. Вакансии с ${dateLong(meta.periods[s].period_from)} по ${dateLong(meta.periods[s].period_to)}.
  </div>`;
})}
</div>

Медианы и квартили считаются в витрине по вакансиям каждой группы, а не собираются из медиан её частей. Граница вилки показана, если её указали хотя бы ${minN} вакансий группы: при меньшем числе 95-процентный интервал для медианы по порядковым статистикам совпадает со всем разбросом группы, от минимума до максимума (при ${minN} вакансиях интервал от второго до предпоследнего значения покрывает медиану с вероятностью ${pct(meta.salary_min_n_coverage)}).

## Медианы по группам

```js
const medSource = view(sourceInput());
const medBy = view(dimensionInput(["group", "grade", "region", "group_grade", "bucket"]));
```

```js
const medRows = at(medBy, medSource);
const dim = DIMENSIONS[medBy];
const medShown = shownBounds(medRows, minN, dim.key);
const medOrdered = dim.order
  ? medShown.slice().sort((a, b) => dim.order.indexOf(a.label) - dim.order.indexOf(b.label))
  : medShown.slice().sort((a, b) => (b.from ?? b.to) - (a.from ?? a.to));
const medHidden = medRows.filter((d) => d.with_from < minN && d.with_to < minN);
```

<div class="card">

${SOURCE[medSource]}, в разрезе «${dim.name}»: групп с достаточным числом вакансий ${int(medShown.length)} из ${int(medRows.length)}${medHidden.length ? `; не показаны ${medHidden.map((d) => dim.key(d)).join(", ")} (в них ${int(d3.sum(medHidden, (d) => d.with_salary_rub))} вакансий с суммой)` : ""}. Точка: медиана, отрезок: от 25-го до 75-го процентиля.

```js
medShown.length === 0 ? html`<p>Ни одна группа не набрала ${minN} вакансий с указанной границей.</p>` : rangePlot(medOrdered, {width, labelWidth: medBy === "group_grade" ? 330 : 230, sort: false})
```

</div>

```js
Inputs.table(
  medOrdered.map((d) => ({...d, share: d.with_salary_rub / d.vacancies})),
  {
    columns: width < 600 ? ["label", "from", "to"] : ["label", "vacancies", "with_salary_rub", "with_from", "from", "with_to", "to"],
    header: {
      label: dim.name[0].toUpperCase() + dim.name.slice(1),
      vacancies: "Вакансий",
      with_salary_rub: "С суммой",
      with_from: "n «от»",
      from: "Медиана «от»",
      with_to: "n «до»",
      to: "Медиана «до»"
    },
    format: {vacancies: int, with_salary_rub: int, with_from: int, with_to: int, from: rub, to: rub},
    layout: width < 600 ? "fixed" : "auto",
    width: width < 600 ? {label: 150, from: 85, to: 85} : undefined,
    sort: null,
    rows: 12
  }
)
```

## Где зарплату указывают

```js
const shareSource = view(sourceInput());
const shareBy = view(dimensionInput(["group", "grade", "region"]));
```

```js
const shareDim = DIMENSIONS[shareBy];
const stated = at(shareBy, shareSource)
  .map((d) => ({key: shareDim.key(d), n: d.with_salary_rub, of: d.vacancies, share: d.with_salary_rub / d.vacancies}))
  .sort((a, b) => shareDim.order.indexOf(a.key) - shareDim.order.indexOf(b.key));
```

<div class="card">

${SOURCE[shareSource]}: доля вакансий, где указана сумма в рублях.

```js
Plot.plot({
  width,
  height: stated.length * 26 + 30,
  marginLeft: width < 600 ? 190 : 240,
  marginRight: width < 600 ? 80 : 140,
  x: {axis: null, domain: [0, 1]},
  y: {label: null, domain: stated.map((d) => d.key)},
  marks: [
    Plot.barX(stated, {y: "key", x: "share", fill: "#4269d0"}),
    Plot.text(stated, {
      y: "key",
      x: "share",
      text: (d) => (width < 600 ? `${int(d.n)} из ${int(d.of)}` : share(d.n, d.of)),
      textAnchor: "start",
      dx: 4
    })
  ]
})
```

</div>

## Вилки по ячейкам

Самый мелкий срез: роль, грейд и регион вместе. Фильтры выбирают ячейки, сами ячейки не складываются.

```js
const pickSource = view(sourceInput());
const pickGroup = view(Inputs.select(new Map([["все", null], ...MART_GROUP_ORDER.map((g) => [MART_GROUP[g], g])]), {label: "Группа ролей"}));
const pickGrade = view(Inputs.select(new Map([["все", null], ...GRADE_ORDER.map((g) => [GRADE[g], g])]), {label: "Грейд"}));
const pickRegion = view(Inputs.select(new Map([["все", null], ...REGION_ORDER.map((r) => [REGION[r], r])]), {label: "Регион"}));
```

```js
const selected = at("cell", pickSource).filter(
  (d) =>
    (pickGroup === null || d.role_group === pickGroup) &&
    (pickGrade === null || d.grade === pickGrade) &&
    (pickRegion === null || d.region_group === pickRegion)
);
const shown = shownBounds(selected, minN, (d) => `${ROLE[d.role_bucket]} · ${GRADE[d.grade]} · ${REGION[d.region_group]}`);
const hidden = selected.filter((d) => d.with_from < minN && d.with_to < minN);
const plotted = shown.slice().sort((a, b) => b.with_salary_rub - a.with_salary_rub).slice(0, 30);
```

<div class="card">

${shown.length === 0 ? "В выбранном срезе нет ячейки, где границу вилки указали бы " + minN + " вакансий или больше." : `Ячеек с достаточным числом вакансий: ${int(shown.length)}${shown.length > plotted.length ? `, на графике ${plotted.length} самых крупных, остальные в таблице ниже` : ""}. Не показаны ${int(hidden.length)} ячеек, в них ${int(d3.sum(hidden, (d) => d.with_salary_rub))} вакансий с суммой в рублях.`}

```js
shown.length === 0 ? html`` : rangePlot(plotted, {width, labelWidth: 360})
```

</div>

```js
Inputs.table(
  shown.map((d) => ({...d, role: ROLE[d.role_bucket], gradeLabel: GRADE[d.grade], region: REGION[d.region_group]})),
  {
    columns: width < 600 ? ["role", "gradeLabel", "region", "from", "to"] : ["role", "gradeLabel", "region", "with_from", "from", "with_to", "to"],
    header: {
      role: "Роль",
      gradeLabel: "Грейд",
      region: "Регион",
      with_from: "n «от»",
      from: "Медиана «от»",
      with_to: "n «до»",
      to: "Медиана «до»"
    },
    format: {with_from: int, with_to: int, from: rub, to: rub},
    // on a phone the columns keep their width and the table scrolls sideways instead of cutting names
    layout: width < 600 ? "fixed" : "auto",
    width: width < 600 ? {role: 170, gradeLabel: 120, region: 120, from: 100, to: 100} : undefined,
    sort: "with_from",
    reverse: true,
    rows: 12
  }
)
```
