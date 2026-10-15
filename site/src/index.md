---
title: Навыки
toc: false
---

# Какие навыки нужны в IT

```js
import {SKILL, int, pct, plural} from "./components/labels.js";
import {DIRECTION, GRADE_GROUP, MIX_LABEL, directionTiles, gradeChart, skillChart} from "./components/skills.js";
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
// Grade and direction shares are at fixed channel weights (skill_trends.json); a heading claims a change only
// when it is beyond noise.
const gradeChange = Object.fromEntries(trends.grades.changes.map((c) => [c.key, c]));
const moved = (c, sign) => c && c.q_bh < 0.05 && Math.sign(c.diff) === sign;
const seniorsUp = moved(gradeChange.senior, 1), middlesDown = moved(gradeChange.middle, -1), juniorsUp = moved(gradeChange.junior, 1);
const gradeTitle = seniorsUp && middlesDown ? "Сеньоров ищут всё чаще, мидлов реже" : seniorsUp ? "Сеньоров ищут всё чаще" : middlesDown ? "Мидлов ищут всё реже" : juniorsUp ? "Джунов ищут всё чаще" : "Кого ищут";
const graded = trends.graded.map((g) => g.graded / g.total);
```

```js
// The skills of a cell (direction and grade) as the chart and the stack check read them.
function cellData(direction, grade) {
  const cell = `${direction}_${grade}`;
  const posts = new Map(trends.groups.filter((g) => g.group === cell).map((g) => [g.year, g.posts]));
  // The change since the base year is measured within channels (a channel that started posting since
  // then would move a pooled share by itself), so the grey dot is today's share minus that change: the
  // base year in the same channels. A cell without enough posts that year shows the recent shares only.
  const changes = new Map(trends.changes.filter((c) => c.group === cell).map((c) => [c.skill, c]));
  const baseYear = changes.size ? trends.base_year : null;
  const rows = trends.shares
    .filter((s) => s.group === cell && s.year === null)
    .map(({skill, posts_mentioning: n, posts: of}) => {
      const c = baseYear ? changes.get(skill) : null;
      const share = n / of;
      return {key: skill, skill: SKILL[skill] ?? skill, share, n, of, base: c ? {share: share - c.diff} : null, change: c ? {diff: c.diff, significant: c.q_bh < 0.05} : null};
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
      [`change_points_since_${baseYear ?? "base"}_within_channels`, (r) => (r.change ? round4(r.change.diff * 100) : null)],
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
  : html`<p class="note-small">${baseYear ? html`Доля постов с навыком за ${recentLabel} (${int(posts.get(null))} ${plural(posts.get(null), ["пост", "поста", "постов"])}); справа изменение с ${baseYear} года в п. п., посчитанное внутри одних и тех же каналов. ${skillsCsv()}` : html`Доля постов с навыком за ${recentLabel} (${int(posts.get(null))} ${plural(posts.get(null), ["пост", "поста", "постов"])}); за ${trends.base_year} год данных мало, поэтому без сравнения. ${skillsCsv()}`}</p>`}

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

<p class="note-small">Навыки через запятую, сравнение с долями «${stackCell.label}» в Telegram. Вилка из раздела ниже: она посчитана по вакансиям портала «Работа в России» сразу для всех направлений.</p>

${stackCell.posts.size ? stackReport(stackCell.rows, effects.effects, {mine, name: (e) => SKILL[e.skill] ?? e.skill, fdr: effects.fdr, label: stackCell.label, baseYear: stackCell.baseYear, onAdd: addSkill}) : html`<p>Для «${stackCell.label}» данных недостаточно, выберите все грейды или другое направление.</p>`}

</div>

```js
const name = (e) => SKILL[e.skill] ?? e.skill;
// The heading names only effects the sensitivity run confirms as well.
const confirmed = (sign) => effects.effects.filter((e) => e.q_bh < effects.fdr && e.robust === true && Math.sign(e.pct) === sign).sort((a, b) => sign * (b.pct - a.pct));
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
const dirYears = [...new Set(trends.directions.years.map((d) => d.year))].sort();
const dirChanges = trends.directions.changes
  .filter((c) => DIRECTION[c.key] && c.q_bh < 0.05)
  .map((c) => ({name: DIRECTION[c.key], diff: c.diff}))
  .sort((x, y) => y.diff - x.diff);
const gainer = dirChanges.find((d) => d.diff > 0);
const loser = dirChanges.at(-1)?.diff < 0 ? dirChanges.at(-1) : null;
const otherNow = trends.directions.years.find((d) => d.key === "other" && d.year === dirYears.at(-1));
const mixCsv = (mix, labels, name) =>
  csvLink(mix.years, [["year", (d) => d.year], ["key", (d) => labels[d.key] ?? d.key], ["share_fixed_channel_weights", (d) => d.share]], name);
const dirTitle = gainer && loser ? `${gainer.name} растёт, ${loser.name} теряет долю` : "Какие направления растут";
```

<h2>${dirTitle}</h2>

<div class="card">

<p class="note-small">Доля среди всех IT-постов ${int(trends.directions.panel_channels)} каналов, которые писали вакансии каждый год с ${dirYears[0]}-го, при постоянном весе каждого канала: так ни новые, ни выросшие каналы не сдвигают доли; остальные ${pct(otherNow.share)} это прочие роли; справа изменение с ${dirYears[0]} года в п. п. ${mixCsv(trends.directions, {...MIX_LABEL, other: "Прочие роли"}, "directions.csv")}</p>

```js
directionTiles(trends.directions, {width, partialYear: lastYear})
```

</div>

<h2>${gradeTitle}</h2>

<div class="card">

<p class="note-small">Доля среди IT-постов с грейдом (он указан в ${Math.round(d3.min(graded) * 100)}-${Math.round(d3.max(graded) * 100)} % постов) в ${int(trends.grades.panel_channels)} каналах, которые писали такие посты каждый год с ${trends.base_year}-го, при постоянном весе каждого канала; ${lastYear} год неполный. ${mixCsv(trends.grades, {junior: "стажёры и джуны", middle: "мидлы", senior: "сеньоры и лиды"}, "grades.csv")}</p>

```js
gradeChart(trends.grades, {width, partialYear: lastYear})
```

</div>
