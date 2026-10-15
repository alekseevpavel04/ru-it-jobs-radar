---
title: Навыки
toc: false
---

# Какие навыки нужны в IT

```js
import {SKILL, int, plural} from "./components/labels.js";
import {GROUP_LABEL, VARIANTS, change, skillChart} from "./components/skills.js";

const trends = FileAttachment("data/skill_trends.json").json();
```

<p class="lead">Что просят вакансии из публичных Telegram-каналов: всё IT, ML и стажёры с джунами. Навыки из текста поста читает модель проекта. Показаны только доли, под которыми достаточно постов. <a href="./methodology#skills">Как считается</a>.</p>

```js
const groupInput = Inputs.radio(new Map([["Всё IT", "it_all"], ["ML", "ml_all"], ["Стажёры и джуны", "it_junior"]]), {label: "Вакансии", value: "it_all"});
const group = Generators.input(groupInput);
const sortInput = Inputs.radio(new Map([["по доле", "share"], ["по росту", "growth"]]), {label: "Порядок", value: "share"});
const sortBy = Generators.input(sortInput);
// Temporary: five looks of the same chart for the author to pick one.
const variantInput = Inputs.radio(VARIANTS, {label: "Вид", value: "bar"});
const variant = Generators.input(variantInput);
```

<div class="filters">${groupInput}${sortInput}${variantInput}</div>

```js
const posts = new Map(trends.groups.filter((g) => g.group === group).map((g) => [g.year, g.posts]));
const years = [...posts.keys()].filter((y) => y !== null).sort();
// The base year of the result file (2022): from then on the channel set is steady.
const baseYear = trends.base_year;
const recentLabel = `${trends.recent_years[0]}-${trends.recent_years.at(-1)}`;
// Only the group's own skills: its top and its fastest growers, not the companions of its juniors.
const own = new Set(trends.shares.filter((s) => s.group === group && s.year === null && s.own).map((s) => s.skill));
const rows = [...d3.group(trends.shares.filter((s) => s.group === group && own.has(s.skill)), (s) => s.skill)].map(([skill, list]) => {
  const recent = list.find((s) => s.year === null);
  const series = years.map((y) => list.find((s) => s.year === y)).map((s, i) => ({year: years[i], share: s.posts_mentioning / s.posts, n: s.posts_mentioning, of: s.posts}));
  const base = series.find((x) => x.year === baseYear);
  const now = {n: recent.posts_mentioning, of: recent.posts};
  return {skill: SKILL[skill] ?? skill, share: now.n / now.of, ...now, base, series, change: change(base, now, trends.new_below)};
});
// Growth order: significant rises by size, noise in the middle, falls last.
const growthKey = (r) => (r.change.significant ? r.change.diff : 0);
const ordered = rows.slice().sort(sortBy === "share" ? (a, b) => b.share - a.share : (a, b) => growthKey(b) - growthKey(a));
```

<div class="card">

<p class="note-small">${GROUP_LABEL[group]}: доля постов, где назван навык, за ${recentLabel} (${int(posts.get(null))} постов) и в ${baseYear} году (${int(posts.get(baseYear))}). Справа изменение доли с ${baseYear} года в процентных пунктах; «≈» значит, что изменение не выходит за случайный разброс, «новый навык» значит, что в ${baseYear} году его назвали меньше ${trends.new_below} постов. В списке ${rows.length} ${plural(rows.length, ["навык", "навыка", "навыков"])}: самые частые и самые быстрорастущие. База ${baseYear} год: с него набор каналов устойчивый, <a href="./methodology#telegram">почему</a>. Наведите на строку, чтобы увидеть долю по каждому году.</p>

```js
skillChart(ordered, {width, recentLabel, baseYear, variant})
```

</div>
