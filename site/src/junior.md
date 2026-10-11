---
title: Джуниорский срез
toc: true
---

# Вакансии для начинающих

```js
import {ROLE, SOURCE, CATEGORY, GRADE, REGION, REGION_ORDER, int, pct, share, rub, dateLong} from "./components/labels.js";

const meta = FileAttachment("data/meta.json").json();
const junior = FileAttachment("data/junior.json").json();
const skillsJunior = FileAttachment("data/skills_junior.json").json();
const skillsAll = FileAttachment("data/skills.json").json();
const cells = FileAttachment("data/salaries.json").json();
```

```js
const FAMILIES = ["trudvsem", "habr", "telegram"];
const JUNIOR_GRADES = ["intern", "junior"];
const sum = (rows, key) => d3.sum(rows, (d) => d[key]);
const minN = meta.salary_min_n;
```

Вакансия открыта начинающему, если её грейд стажёр или джуниор, или если в ней сказано, что опыт не нужен. На портале «Работа в России» это поле требуемого опыта, равное нулю лет («без опыта»); у Хабр Карьеры и в Telegram такого поля нет, там ищется фраза в тексте. Опыт «до года» отдельно не выделяется: поле портала считает опыт в целых годах, у остальных источников поля нет. Грейд определяется правилом по заголовку (на Хабр Карьере по квалификации, которую указал работодатель).

<div class="grid grid-cols-3">
${FAMILIES.map((s) => {
  const rows = junior.filter((d) => d.source === s);
  const unit = s === "telegram" ? "постов" : "вакансий";
  return html`<div class="card">
    <h2>${SOURCE[s]}</h2>
    <div class="big">${pct(sum(rows, "open_to_junior") / sum(rows, "vacancies"))}</div>
    ${unit} открыты начинающим: ${share(sum(rows, "open_to_junior"), sum(rows, "vacancies"))}. Из них стажёр ${int(sum(rows, "graded_intern"))}, джуниор ${int(sum(rows, "graded_junior"))}, «без опыта» ${int(sum(rows, "no_experience_required"))} (признаки пересекаются).
  </div>`;
})}
</div>

## Роли

```js
const roleSource = view(Inputs.radio(new Map(FAMILIES.map((s) => [SOURCE[s], s])), {value: "trudvsem", label: "Источник"}));
```

```js
const roles = junior
  .filter((d) => d.source === roleSource && d.vacancies > 0)
  .map((d) => ({...d, role: ROLE[d.role_bucket], share: d.open_to_junior / d.vacancies}))
  .sort((a, b) => b.open_to_junior - a.open_to_junior);
```

<div class="card">

Сколько вакансий каждой роли открыты начинающим. Роли отсортированы по числу таких вакансий, подпись: сколько из всех вакансий роли.

```js
Plot.plot({
  width,
  height: roles.length * 24 + 30,
  marginLeft: width < 600 ? 185 : 230,
  marginRight: width < 600 ? 70 : 150,
  x: {axis: null, domain: [0, 1]},
  y: {label: null, domain: roles.map((d) => d.role)},
  marks: [
    Plot.barX(roles, {
      y: "role",
      x: "share",
      fill: "#3ca951",
      title: (d) =>
        `${d.role}\n${share(d.open_to_junior, d.vacancies)}\nстажёр ${int(d.graded_intern)}, джуниор ${int(d.graded_junior)}, без опыта ${int(d.no_experience_required)}`,
      tip: true
    }),
    Plot.text(roles, {
      y: "role",
      x: "share",
      text: (d) => (width < 600 ? `${int(d.open_to_junior)} из ${int(d.vacancies)}` : share(d.open_to_junior, d.vacancies)),
      textAnchor: "start",
      dx: 4
    })
  ]
})
```

</div>

## Навыки, зарплаты и города

```js
const graded = FAMILIES.map((s) => {
  const rows = junior.filter((d) => d.source === s);
  return {s, graded: sum(rows, "graded_intern") + sum(rows, "graded_junior"), open: sum(rows, "open_to_junior")};
});
```

Навыки, зарплаты и города ниже считаются только по вакансиям с грейдом стажёр или джуниор: витрины навыков и зарплат разбиты по грейду, а не по требуемому опыту. Это заметная часть джуниорского среза не везде: ${graded.map((g) => `${SOURCE[g.s]} ${share(g.graded, g.open)}`).join("; ")} вакансий, открытых начинающим (грейд и «без опыта» пересекаются, поэтому доля не дополняет «без опыта» до целого).

### Навыки

```js
const skillSource = view(Inputs.radio(new Map(FAMILIES.map((s) => [SOURCE[s], s])), {value: "trudvsem", label: "Источник"}));
```

```js
const juniorRows = skillsJunior.filter((d) => d.source === skillSource);
const base = juniorRows.length ? juniorRows[0].vacancies_graded_junior : 0;
const allShare = new Map(skillsAll.filter((d) => d.source === skillSource).map((d) => [d.skill, d.vacancies_mentioning / d.vacancies_total]));
const allBase = skillsAll.find((d) => d.source === skillSource)?.vacancies_total ?? 0;
const topJunior = juniorRows.slice(0, 20).flatMap((d) => [
  {skill: d.skill, who: "стажёр и джуниор", share: d.vacancies_mentioning / d.vacancies_graded_junior, text: share(d.vacancies_mentioning, d.vacancies_graded_junior)},
  {skill: d.skill, who: "все вакансии", share: allShare.get(d.skill) ?? 0, text: ""}
]);
```

<div class="card">

${base === 0 ? "В этом источнике нет вакансий с грейдом стажёр или джуниор." : `Двадцать самых частых навыков в ${int(base)} ${skillSource === "telegram" ? "постах" : "вакансиях"} с грейдом стажёр или джуниор; серым та же доля среди всех ${int(allBase)}. Вакансии «без опыта» без грейда сюда не входят: навыки в витрине разбиты по грейду, а не по опыту.`}

```js
base === 0
  ? html``
  : Plot.plot({
      width,
      height: 20 * 26 + 50,
      marginLeft: width < 600 ? 150 : 200,
      x: {grid: true, label: "доля вакансий", tickFormat: pct, zero: true},
      y: {label: null, domain: juniorRows.slice(0, 20).map((d) => d.skill)},
      color: {domain: ["стажёр и джуниор", "все вакансии"], range: ["#3ca951", "#a0a0a0"], legend: true},
      marks: [
        Plot.ruleY(juniorRows.slice(0, 20), {y: "skill", stroke: "#e0e0e0"}),
        Plot.dot(topJunior, {y: "skill", x: "share", fill: "who", r: 5, title: (d) => `${d.skill}, ${d.who}\n${d.text || pct(d.share)}`, tip: true})
      ]
    })
```

</div>

### Зарплаты

```js
const juniorCells = cells.filter((d) => JUNIOR_GRADES.includes(d.grade));
const salaryRows = ["trudvsem", "habr"].map((s) => {
  const j = juniorCells.filter((d) => d.source === s);
  const all = cells.filter((d) => d.source === s);
  return {s, jn: sum(j, "with_salary_rub"), jof: sum(j, "vacancies"), an: sum(all, "with_salary_rub"), aof: sum(all, "vacancies")};
});
const juniorShown = juniorCells
  .filter((d) => d.with_from >= minN || d.with_to >= minN)
  .map((d) => ({
    ...d,
    label: `${SOURCE[d.source]} · ${ROLE[d.role_bucket]} · ${GRADE[d.grade]} · ${REGION[d.region_group]}`,
    from: d.with_from >= minN ? d.salary_from_median : null,
    to: d.with_to >= minN ? d.salary_to_median : null
  }))
  .sort((a, b) => b.with_salary_rub - a.with_salary_rub);
const juniorHidden = juniorCells.filter((d) => d.with_from < minN && d.with_to < minN);
```

Доля вакансий с суммой в рублях. ${salaryRows.map((r) => `${SOURCE[r.s]}: ${share(r.jn, r.jof)} у стажёров и джуниоров, ${share(r.an, r.aof)} у всех вакансий`).join(". ")}.

Вилки показаны только для ячеек, где границу указали хотя бы ${minN} вакансий (почему так, на странице «Зарплаты»). Таких ячеек стажёров и джуниоров ${int(juniorShown.length)}; ещё ${int(juniorHidden.length)} ячеек с ${int(sum(juniorHidden, "with_salary_rub"))} вакансиями с суммой слишком малы. Это вилка из объявления, а не выплаченная зарплата.

```js
Inputs.table(
  juniorShown.map((d) => ({...d, src: SOURCE[d.source], role: ROLE[d.role_bucket], gradeLabel: GRADE[d.grade], region: REGION[d.region_group]})),
  {
    columns: width < 600 ? ["role", "gradeLabel", "region", "from", "to"] : ["src", "role", "gradeLabel", "region", "with_from", "from", "with_to", "to"],
    header: {
      src: "Источник",
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
    rows: 10
  }
)
```

### Города и удалённая работа

```js
const regionRows = ["trudvsem", "habr"].flatMap((s) => {
  const j = juniorCells.filter((d) => d.source === s);
  const all = cells.filter((d) => d.source === s);
  return REGION_ORDER.flatMap((r) => [
    {source: SOURCE[s], who: "стажёр и джуниор", region: REGION[r], n: sum(j.filter((d) => d.region_group === r), "vacancies"), of: sum(j, "vacancies")},
    {source: SOURCE[s], who: "все вакансии", region: REGION[r], n: sum(all.filter((d) => d.region_group === r), "vacancies"), of: sum(all, "vacancies")}
  ]);
});
```

<div class="card">

Где вакансии стажёров и джуниоров (зелёные полосы) по сравнению со всеми вакансиями источника (серые риски). «Удалённо» стоит выше города: вакансия с удалённым форматом считается удалённой, даже если город указан. Доли от ${regionRows.filter((d) => d.who === "стажёр и джуниор" && d.region === REGION.moscow).map((d) => `${int(d.of)} вакансий стажёров и джуниоров (${d.source})`).join(" и ")}. Telegram не входит: место работы из постов пока не извлекается.

```js
html`<div class="grid grid-cols-2">${[SOURCE.trudvsem, SOURCE.habr].map((source) => {
  const rows = regionRows.filter((d) => d.source === source);
  const jun = rows.filter((d) => d.who === "стажёр и джуниор");
  const all = rows.filter((d) => d.who === "все вакансии");
  return html`<div><h3>${source}</h3>${Plot.plot({
    width: width < 600 ? width : width / 2 - 20,
    height: REGION_ORDER.length * 28 + 30,
    marginLeft: 130,
    marginRight: 60,
    x: {axis: null, domain: [0, 1]},
    y: {label: null, domain: REGION_ORDER.map((r) => REGION[r])},
    marks: [
      Plot.barX(jun, {y: "region", x: (d) => d.n / d.of, fill: "#3ca951", title: (d) => `${d.region}, стажёр и джуниор
${share(d.n, d.of)}`, tip: true}),
      Plot.tickX(all, {y: "region", x: (d) => d.n / d.of, stroke: "#606060", strokeWidth: 2, title: (d) => `${d.region}, все вакансии
${share(d.n, d.of)}`}),
      Plot.text(jun, {
        y: "region",
        x: (d) => {
          const a = all.find((r) => r.region === d.region);
          return Math.max(d.n / d.of, a.n / a.of);
        },
        text: (d) => pct(d.n / d.of),
        textAnchor: "start",
        dx: 6
      })
    ]
  })}</div>`;
})}</div>`
```

</div>
