---
title: Навыки
toc: false
---

# Какие навыки нужны в IT

```js
import {SKILL, int, plural} from "./components/labels.js";
import {DIRECTION, GRADE_GROUP, change, skillChart} from "./components/skills.js";

const trends = FileAttachment("data/skill_trends.json").json();
```

<p class="lead">Что просят вакансии из публичных Telegram-каналов, по направлениям IT и грейдам. Навыки, роль и грейд из текста поста читает модель проекта. Показаны только доли, под которыми достаточно постов. <a href="./methodology#skills">Как считается</a>.</p>

```js
const directionInput = Inputs.select(new Map(Object.entries(DIRECTION).map(([k, v]) => [v, k])), {label: "Направление", value: "it"});
const direction = Generators.input(directionInput);
const gradeInput = Inputs.radio(new Map(Object.entries(GRADE_GROUP).map(([k, v]) => [v, k])), {label: "Грейд", value: "all"});
const grade = Generators.input(gradeInput);
const sortInput = Inputs.radio(new Map([["по доле", "share"], ["по росту", "growth"]]), {label: "Порядок", value: "share"});
const sortBy = Generators.input(sortInput);
```

<div class="filters stacked">${directionInput}${gradeInput}${sortInput}</div>

```js
const cell = `${direction}_${grade}`;
const posts = new Map(trends.groups.filter((g) => g.group === cell).map((g) => [g.year, g.posts]));
const years = [...posts.keys()].filter((y) => y !== null).sort();
// The base year of the result file (2022): from then on the channel set is steady. A cell without
// enough posts that year shows the recent shares only.
const baseYear = posts.has(trends.base_year) ? trends.base_year : null;
const recentLabel = `${trends.recent_years[0]}-${trends.recent_years.at(-1)}`;
// Only the cell's own skills: its top and its fastest growers, not the companions of its juniors.
const own = new Set(trends.shares.filter((s) => s.group === cell && s.year === null && s.own).map((s) => s.skill));
const rows = [...d3.group(trends.shares.filter((s) => s.group === cell && own.has(s.skill)), (s) => s.skill)].map(([skill, list]) => {
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
```

<div class="card">

${!posts.size
  ? html`<p>Для сочетания «${label}» данных недостаточно: за ${recentLabel} меньше ${int(trends.min_cell_posts)} постов. Выберите все грейды или другое направление.</p>`
  : html`<p class="note-small">${label}: доля постов, где назван навык, за ${recentLabel} (${int(posts.get(null))} постов)${baseYear ? html` и в ${baseYear} году (${int(posts.get(baseYear))}). Справа изменение доли с ${baseYear} года в процентных пунктах; «≈» значит, что изменение не выходит за случайный разброс. База ${baseYear} год: с него набор каналов устойчивый, <a href="./methodology#telegram">почему</a>` : html`. За ${trends.base_year} год постов этой группы меньше ${int(trends.min_year_posts)}, поэтому без сравнения, только доля сейчас`}. В списке ${rows.length} ${plural(rows.length, ["навык", "навыка", "навыков"])}${baseYear ? ": самые частые и самые быстрорастущие" : ""}. Наведите на строку, чтобы увидеть числа.</p>`}

```js
// Always display a node: a block that displays nothing keeps its previous output.
display(posts.size ? skillChart(ordered, {width, recentLabel, baseYear}) : document.createElement("span"));
```

</div>
