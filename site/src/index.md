---
title: Навыки
toc: false
---

# Какие навыки нужны в IT

```js
import {SKILL, int, pct, plural} from "./components/labels.js";
import {DIRECTION, GRADE_GROUP, MIX_LABEL, change, directionTiles, gradeChart, skillChart} from "./components/skills.js";
import {effectsChart} from "./components/effects.js";
import {parseStack, stackReport} from "./components/stack.js";
import {csvLink} from "./components/csv.js";

const trends = FileAttachment("data/skill_trends.json").json();
const effects = FileAttachment("data/skill_effects.json").json();
```

<p class="lead">Какие навыки просят, за какие платят больше, какие направления растут и кого ищут, по вакансиям из публичных Telegram-каналов. Роль, грейд и навыки из текста читает модель проекта. <a href="./methodology">Как считается</a>.</p>

```js
// Headings say what the data shows, so they are computed: a claim is made only while the data holds it.
const lowerFirst = (s) => (/^[А-ЯЁ][а-яё]/.test(s) ? s[0].toLowerCase() + s.slice(1) : s);
const listRu = (xs) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} и ${xs.at(-1)}`);
const lastYear = trends.recent_years.at(-1);
const gradeShare = (year, grade) => {
  const g = trends.grades.find((r) => r.year === year && r.grade === grade);
  return g.posts / g.graded;
};
const midDrop = gradeShare(trends.base_year, "middle") - gradeShare(lastYear, "middle");
const juniorRise = gradeShare(lastYear, "junior") - gradeShare(trends.base_year, "junior");
const gradeTitle = midDrop > 0.05 ? "Мидлов ищут всё реже" : juniorRise > 0.05 ? "Джунов ищут всё чаще" : "Кого ищут";
const graded = [...new Set(trends.grades.map((g) => g.year))].map((y) => {
  const g = trends.grades.find((r) => r.year === y);
  return g.graded / g.total;
});
```

```js
// The skills of a cell (direction and grade) as the chart and the stack check read them.
function cellData(direction, grade) {
  const cell = `${direction}_${grade}`;
  const posts = new Map(trends.groups.filter((g) => g.group === cell).map((g) => [g.year, g.posts]));
  // The base year of the result file (2022): from then on the channel set is steady. A cell without
  // enough posts that year shows the recent shares only.
  const baseYear = posts.has(trends.base_year) ? trends.base_year : null;
  const rows = [...d3.group(trends.shares.filter((s) => s.group === cell), (s) => s.skill)].map(([skill, list]) => {
    const recent = list.find((s) => s.year === null);
    const b = baseYear ? list.find((s) => s.year === baseYear) : null;
    const base = b ? {share: b.posts_mentioning / b.posts, n: b.posts_mentioning, of: b.posts} : null;
    const now = {n: recent.posts_mentioning, of: recent.posts};
    return {key: skill, skill: SKILL[skill] ?? skill, share: now.n / now.of, ...now, base, change: base ? change(base, now) : null};
  });
  return {cell, posts, baseYear, rows, label: `${DIRECTION[direction]}, ${GRADE_GROUP[grade].toLowerCase()}`};
}
```

```js
const directionInput = Inputs.select(new Map(Object.entries(DIRECTION).map(([k, v]) => [v, k])), {label: "Направление", value: "it"});
const direction = Generators.input(directionInput);
const gradeInput = Inputs.radio(new Map(Object.entries(GRADE_GROUP).map(([k, v]) => [v, k])), {label: "Грейд", value: "all"});
const grade = Generators.input(gradeInput);
const sortInput = Inputs.radio(new Map([["по доле", "share"], ["по росту", "growth"]]), {label: "Порядок", value: "share"});
const sortBy = Generators.input(sortInput);
```

```js
const {cell, posts, baseYear, rows, label} = cellData(direction, grade);
const recentLabel = `${trends.recent_years[0]}-${trends.recent_years.at(-1)}`;
// Growth order: significant rises by size, noise in the middle, falls last.
const growthKey = (r) => (r.change?.significant ? r.change.diff : 0);
const ordered = rows.slice().sort(sortBy === "share" || !baseYear ? (a, b) => b.share - a.share : (a, b) => growthKey(b) - growthKey(a));
const round4 = (x) => (x == null ? null : Math.round(x * 10000) / 10000);
const skillsCsv = () =>
  csvLink(
    ordered,
    [
      ["skill", (r) => r.skill],
      [`share_${recentLabel}`, (r) => round4(r.share)],
      ["posts_with_skill", (r) => r.n],
      ["posts", (r) => r.of],
      [`share_${baseYear ?? "base"}`, (r) => round4(r.base?.share)],
      ["change_points", (r) => (r.change ? round4(r.change.diff * 100) : null)],
      ["beyond_noise", (r) => (r.change ? r.change.significant : null)]
    ],
    `skills_${cell}.csv`
  );
const top = rows.slice().sort((a, b) => b.share - a.share)[0];
const riser = rows.filter((r) => r.change?.significant && r.change.diff > 0).sort((a, b) => b.change.diff - a.change.diff)[0];
const skillTitle = !posts.size
  ? "Что просят"
  : !riser
    ? `Чаще всего просят ${top.skill}`
    : riser === top
      ? `${top.skill} просят чаще всех, и его доля растёт быстрее всех`
      : `Чаще всего просят ${top.skill}, быстрее всех растёт ${riser.skill}`;
```

<h2>${skillTitle}</h2>

<div class="filters stacked">${directionInput}${gradeInput}${sortInput}</div>

<div class="card">

${!posts.size
  ? html`<p>Для сочетания «${label}» данных недостаточно: за ${recentLabel} меньше ${int(trends.min_cell_posts)} постов. Выберите все грейды или другое направление.</p>`
  : html`<p class="note-small">${baseYear ? html`Доля постов с навыком за ${recentLabel} (${int(posts.get(null))} ${plural(posts.get(null), ["пост", "поста", "постов"])}) против ${baseYear} года (${int(posts.get(baseYear))}), справа изменение в п. п. ${skillsCsv()}` : html`Доля постов с навыком за ${recentLabel} (${int(posts.get(null))} ${plural(posts.get(null), ["пост", "поста", "постов"])}); за ${trends.base_year} год данных мало, поэтому без сравнения. ${skillsCsv()}`}</p>`}

```js
// Always display a node: a block that displays nothing keeps its previous output.
display(posts.size ? skillChart(ordered, {width, recentLabel, baseYear}) : document.createElement("span"));
```

</div>

## Проверьте свой стек

```js
// The stack and its cell live in the address (#stack=Python,SQL&dir=ml&grade=junior), so a reader can
// share the result.
const fromHash = (key) => {
  const m = location.hash.match(new RegExp(`${key}=([^&]*)`));
  return m ? decodeURIComponent(m[1]) : "";
};
const stackInput = Inputs.text({label: "Ваши навыки", placeholder: "Python, SQL, Docker", value: fromHash("stack"), width: Math.min(width, 520)});
const stackText = Generators.input(stackInput);
const stackDirectionInput = Inputs.select(new Map(Object.entries(DIRECTION).map(([k, v]) => [v, k])), {label: "Направление", value: DIRECTION[fromHash("dir")] ? fromHash("dir") : "it"});
const stackDirection = Generators.input(stackDirectionInput);
const stackGradeInput = Inputs.radio(new Map(Object.entries(GRADE_GROUP).map(([k, v]) => [v, k])), {label: "Грейд", value: GRADE_GROUP[fromHash("grade")] ? fromHash("grade") : "all"});
const stackGrade = Generators.input(stackGradeInput);
```

<div class="filters stacked">${stackDirectionInput}${stackGradeInput}${stackInput}</div>

```js
const mine = parseStack(stackText);
const stackCell = cellData(stackDirection, stackGrade);
const addSkill = (skill) => {
  stackInput.value = [...mine, skill].join(", ");
  stackInput.dispatchEvent(new Event("input", {bubbles: true}));
};
const stackHash = [mine.length ? `stack=${encodeURIComponent(mine.join(","))}` : "", stackDirection !== "it" ? `dir=${stackDirection}` : "", stackGrade !== "all" ? `grade=${stackGrade}` : ""].filter(Boolean).join("&");
history.replaceState(null, "", stackHash ? `#${stackHash}` : location.pathname + location.search);
```

<div class="card">

<p class="note-small">Навыки через запятую, сравнение с долями «${stackCell.label}». Вилка по всем IT-вакансиям из раздела ниже.</p>

${stackCell.posts.size ? stackReport(stackCell.rows, effects.effects, {mine, name: (e) => SKILL[e.skill] ?? e.skill, fdr: effects.fdr, label: stackCell.label, baseYear: stackCell.baseYear, onAdd: addSkill}) : html`<p>Для «${stackCell.label}» данных недостаточно, выберите все грейды или другое направление.</p>`}

</div>

```js
const name = (e) => SKILL[e.skill] ?? e.skill;
const confirmed = (sign) => effects.effects.filter((e) => e.q_bh < effects.fdr && Math.sign(e.pct) === sign).sort((a, b) => sign * (b.pct - a.pct));
const ups = confirmed(1).slice(0, 2).map((e) => lowerFirst(name(e)));
const downs = confirmed(-1).slice(0, 2).map((e) => lowerFirst(name(e)));
const moneyTitle =
  ups.length && downs.length ? `Больше платят за ${listRu(ups)}, меньше за ${listRu(downs)}` : ups.length ? `Больше платят за ${listRu(ups)}` : downs.length ? `Меньше платят за ${listRu(downs)}` : "Сколько добавляет навык к вилке";
```

<h2>${moneyTitle}</h2>

<div class="card">

<p class="note-small">Разница середины вилки у вакансий с навыком и у похожих без него; полоса это интервал с поправкой на ${effects.estimated} проверок, и навык с полосой через ноль не доказан; ${int(effects.rows)} вакансий, почти все с портала «Работа в России». <a href="./methodology#skill-effects">Как считается</a>. ${csvLink(effects.effects, [["skill", name], ["effect_pct", (e) => e.pct], ["band_low_pct", (e) => e.pct_ci_low], ["band_high_pct", (e) => e.pct_ci_high], ["q_bh", (e) => e.q_bh], ["comparable_vacancies", (e) => e.rows_kept]], "skill_salary_effects.csv")}</p>

```js
effectsChart(effects.effects, {width, name})
```

</div>


```js
// The direction that gained most and the one that lost most since the base year, if the change is real.
const dirYears = [...new Set(trends.directions.map((d) => d.year))].sort();
const dirChanges = [...new Set(trends.directions.map((d) => d.direction))]
  .filter((k) => k !== "other" && DIRECTION[k])
  .map((k) => {
    const at = (y) => trends.directions.find((d) => d.direction === k && d.year === y);
    const a = at(dirYears[0]), b = at(dirYears.at(-1));
    return {name: DIRECTION[k], c: change({n: a.posts, of: a.total}, {n: b.posts, of: b.total})};
  })
  .filter((d) => d.c.significant)
  .sort((x, y) => y.c.diff - x.c.diff);
const gainer = dirChanges.find((d) => d.c.diff > 0);
const loser = dirChanges.at(-1)?.c.diff < 0 ? dirChanges.at(-1) : null;
const otherNow = trends.directions.find((d) => d.direction === "other" && d.year === dirYears.at(-1));
const dirTitle = gainer && loser ? `${gainer.name} растёт, ${loser.name} теряет долю` : "Какие направления растут";
```

<h2>${dirTitle}</h2>

<div class="card">

<p class="note-small">Доля среди всех IT-постов ${int(trends.direction_panel_channels)} каналов, которые писали вакансии каждый год с ${dirYears[0]}-го (так новые каналы не сдвигают доли); остальные ${pct(otherNow.posts / otherNow.total)} это прочие роли; справа изменение с ${dirYears[0]} года в п. п. ${csvLink(trends.directions, [["year", (d) => d.year], ["direction", (d) => MIX_LABEL[d.direction] ?? "Прочие роли"], ["share", (d) => Math.round((d.posts / d.total) * 10000) / 10000], ["posts", (d) => d.posts], ["panel_posts", (d) => d.total]], "directions.csv")}</p>

```js
directionTiles(trends.directions, {width, partialYear: lastYear})
```

</div>

<h2>${gradeTitle}</h2>

<div class="card">

<p class="note-small">Доля среди IT-постов, где указан грейд (а он указан в ${Math.round(d3.min(graded) * 100)}-${Math.round(d3.max(graded) * 100)} % постов); ${lastYear} год неполный. ${csvLink(trends.grades, [["year", (g) => g.year], ["grade", (g) => g.grade], ["share", (g) => Math.round((g.posts / g.graded) * 10000) / 10000], ["posts", (g) => g.posts], ["graded_posts", (g) => g.graded]], "grades.csv")}</p>

```js
gradeChart(trends.grades, {width, partialYear: lastYear})
```

</div>
