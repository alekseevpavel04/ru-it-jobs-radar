---
title: Навыки
toc: false
---

# Какие навыки нужны в IT

```js
import {SKILL, int, pct, plural} from "./components/labels.js";
import {DIRECTION, GRADE_GROUP, change, gradeChart, skillChart} from "./components/skills.js";
import {FDR, effectsChart} from "./components/effects.js";

const trends = FileAttachment("data/skill_trends.json").json();
const effects = FileAttachment("data/skill_effects.json").json();
```

<p class="lead">Кого ищут, какие навыки просят и за какие платят больше, по вакансиям из публичных Telegram-каналов. Роль, грейд и навыки из текста читает модель проекта. <a href="./methodology">Как считается</a>.</p>

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

<h2>${gradeTitle}</h2>

<div class="card">

<p class="note-small">Доля среди IT-постов, где указан грейд (а он указан в ${Math.round(d3.min(graded) * 100)}-${Math.round(d3.max(graded) * 100)} % постов); ${lastYear} год неполный.</p>

```js
gradeChart(trends.grades, {width, partialYear: lastYear})
```

</div>

```js
const directionInput = Inputs.select(new Map(Object.entries(DIRECTION).map(([k, v]) => [v, k])), {label: "Направление", value: "it"});
const direction = Generators.input(directionInput);
const gradeInput = Inputs.radio(new Map(Object.entries(GRADE_GROUP).map(([k, v]) => [v, k])), {label: "Грейд", value: "all"});
const grade = Generators.input(gradeInput);
const sortInput = Inputs.radio(new Map([["по доле", "share"], ["по росту", "growth"]]), {label: "Порядок", value: "share"});
const sortBy = Generators.input(sortInput);
```

```js
const cell = `${direction}_${grade}`;
const posts = new Map(trends.groups.filter((g) => g.group === cell).map((g) => [g.year, g.posts]));
const years = [...posts.keys()].filter((y) => y !== null).sort();
// The base year of the result file (2022): from then on the channel set is steady. A cell without
// enough posts that year shows the recent shares only.
const baseYear = posts.has(trends.base_year) ? trends.base_year : null;
const recentLabel = `${trends.recent_years[0]}-${trends.recent_years.at(-1)}`;
const rows = [...d3.group(trends.shares.filter((s) => s.group === cell), (s) => s.skill)].map(([skill, list]) => {
  const recent = list.find((s) => s.year === null);
  const series = years.map((y) => list.find((s) => s.year === y)).map((s, i) => ({year: years[i], share: s.posts_mentioning / s.posts, n: s.posts_mentioning, of: s.posts}));
  const base = baseYear ? series.find((x) => x.year === baseYear) : null;
  const now = {n: recent.posts_mentioning, of: recent.posts};
  return {skill: SKILL[skill] ?? skill, share: now.n / now.of, ...now, base, series, change: base ? change(base, now) : null};
});
// Growth order: significant rises by size, noise in the middle, falls last.
const growthKey = (r) => (r.change?.significant ? r.change.diff : 0);
const ordered = rows.slice().sort(sortBy === "share" || !baseYear ? (a, b) => b.share - a.share : (a, b) => growthKey(b) - growthKey(a));
const label = `${DIRECTION[direction]}, ${GRADE_GROUP[grade].toLowerCase()}`;
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
  : html`<p class="note-small">${baseYear ? html`Доля постов с навыком за ${recentLabel} (${int(posts.get(null))} ${plural(posts.get(null), ["пост", "поста", "постов"])}) против ${baseYear} года (${int(posts.get(baseYear))}), справа изменение в п. п.` : html`Доля постов с навыком за ${recentLabel} (${int(posts.get(null))} ${plural(posts.get(null), ["пост", "поста", "постов"])}); за ${trends.base_year} год данных мало, поэтому без сравнения.`}</p>`}

```js
// Always display a node: a block that displays nothing keeps its previous output.
display(posts.size ? skillChart(ordered, {width, recentLabel, baseYear}) : document.createElement("span"));
```

</div>

```js
const name = (e) => SKILL[e.skill] ?? e.skill;
const level = Math.round(effects.ci_level * 100);
const confirmed = (sign) => effects.effects.filter((e) => e.q_bh < FDR && Math.sign(e.pct) === sign).sort((a, b) => sign * (b.pct - a.pct));
const ups = confirmed(1).slice(0, 2).map((e) => lowerFirst(name(e)));
const downs = confirmed(-1).slice(0, 2).map((e) => lowerFirst(name(e)));
const moneyTitle =
  ups.length && downs.length ? `Больше платят за ${listRu(ups)}, меньше за ${listRu(downs)}` : ups.length ? `Больше платят за ${listRu(ups)}` : downs.length ? `Меньше платят за ${listRu(downs)}` : "Сколько добавляет навык к вилке";
```

<h2>${moneyTitle}</h2>

<div class="card">

<p class="note-small">Разница середины вилки у вакансий с навыком и у похожих без него, полоса ${level} % интервал; ${int(effects.rows)} вакансий, почти все с портала «Работа в России». <a href="./methodology#skill-effects">Как считается</a>.</p>

```js
effectsChart(effects.effects, {width, name})
```

</div>
