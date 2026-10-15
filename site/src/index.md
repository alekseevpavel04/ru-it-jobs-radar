---
title: Навыки
toc: false
---

# Какие навыки нужны в IT

```js
import {SKILL, int, plural} from "./components/labels.js";
import {GROUP_LABEL, change, skillBars} from "./components/skills.js";

const trends = FileAttachment("data/skill_trends.json").json();
```

<p class="lead">Что просят вакансии из публичных Telegram-каналов: всё IT, ML и стажёры с джунами. Навыки из текста поста читает модель проекта. Показаны только доли, под которыми достаточно постов. <a href="./methodology#skills">Как считается</a>.</p>

```js
const groupInput = Inputs.radio(new Map([["Всё IT", "it_all"], ["ML", "ml_all"], ["Стажёры и джуны", "it_junior"]]), {label: "Вакансии", value: "it_all"});
const group = Generators.input(groupInput);
const sortInput = Inputs.radio(new Map([["по доле", "share"], ["по росту", "growth"]]), {label: "Порядок", value: "share"});
const sortBy = Generators.input(sortInput);
```

<div class="filters">${groupInput}${sortInput}</div>

```js
const TOP = 20;
const posts = new Map(trends.groups.filter((g) => g.group === group).map((g) => [g.year, g.posts]));
const years = [...posts.keys()].filter((y) => y !== null).sort();
// 2022: from then on the channel set is steady (39 channels wrote every year); in 2019 a few frontend
// channels made up much of the sample, so a change from 2019 would describe the channels, not demand.
const BASE_YEAR = 2022;
const baseYear = years.includes(BASE_YEAR) ? BASE_YEAR : years[0];
const recentLabel = `${trends.recent_years[0]}-${trends.recent_years.at(-1)}`;
const rows = [...d3.group(trends.shares.filter((s) => s.group === group), (s) => s.skill)].map(([skill, list]) => {
  const recent = list.find((s) => s.year === null);
  const series = years.map((y) => list.find((s) => s.year === y)).map((s, i) => ({year: years[i], share: s.posts_mentioning / s.posts, n: s.posts_mentioning, of: s.posts}));
  const base = series.find((x) => x.year === baseYear);
  const now = {n: recent.posts_mentioning, of: recent.posts};
  return {skill: SKILL[skill] ?? skill, share: now.n / now.of, ...now, base, series, change: change(base, now)};
});
// Growth order: a new skill first, then significant rises by size, noise in the middle, falls last.
const growthKey = (r) => (r.change.relative === null ? Infinity : r.change.significant ? r.change.relative : 0);
const ordered = rows.slice().sort(sortBy === "share" ? (a, b) => b.share - a.share : (a, b) => growthKey(b) - growthKey(a));
```

<div class="card">

<p class="note-small">${GROUP_LABEL[group]}: доля постов, где назван навык, за ${recentLabel} (${int(posts.get(null))} постов) и в ${baseYear} году (${int(posts.get(baseYear))}). Справа рост доли с ${baseYear} года в процентах; «≈» значит, что изменение не выходит за случайный разброс. База ${baseYear} год: с него набор каналов устойчивый, <a href="./methodology#telegram">почему</a>. Наведите на строку, чтобы увидеть долю по каждому году.</p>

<div class="dot-legend"><span><i class="bar"></i>${recentLabel}</span><span><i class="base"></i>${baseYear}</span></div>

```js
display(skillBars(ordered.slice(0, TOP), {width, recentLabel}));
if (ordered.length > TOP) {
  const rest = ordered.length - TOP;
  display(html`<details class="more"><summary>Ещё ${rest} ${plural(rest, ["навык", "навыка", "навыков"])}</summary>${skillBars(ordered.slice(TOP), {width, recentLabel})}</details>`);
}
```

</div>
