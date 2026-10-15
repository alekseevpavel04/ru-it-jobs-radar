---
title: Навыки джуна
toc: false
---

# Что спрашивают у джуна

```js
import {SKILL, COLOR, int, pct, share, points} from "./components/labels.js";
import {GROUP_LABEL} from "./components/skills.js";

const trends = FileAttachment("data/skill_trends.json").json();
```

<p class="lead">Навыки в постах для стажёров и джунов против всех постов той же группы, публичные Telegram-каналы за ${trends.recent_years[0]}-${trends.recent_years.at(-1)}. Сверху то, что у джуна просят чаще, снизу то, что реже. <a href="./methodology#skills">Как считается</a>.</p>

```js
const scopeInput = Inputs.radio(new Map([["Всё IT", "it"], ["ML", "ml"]]), {label: "Вакансии", value: "it"});
const scope = Generators.input(scopeInput);
```

<div class="filters">${scopeInput}</div>

```js
const juniorGroup = `${scope}_junior`;
const allGroup = `${scope}_all`;
const recent = trends.shares.filter((s) => s.year === null);
const pick = (group) => new Map(recent.filter((s) => s.group === group).map((s) => [s.skill, s]));
const jun = pick(juniorGroup);
const all = pick(allGroup);
const postsOf = (group) => trends.groups.find((g) => g.group === group && g.year === null).posts;
const rows = [...jun.values()]
  .filter((j) => all.has(j.skill))
  .map((j) => {
    const a = all.get(j.skill);
    return {skill: SKILL[j.skill] ?? j.skill, jun: j.posts_mentioning / j.posts, all: a.posts_mentioning / a.posts, j, a};
  })
  .map((d) => ({...d, diff: d.jun - d.all}))
  .sort((a, b) => b.diff - a.diff);
const more = rows.filter((d) => d.diff > 0).length;
// Blue when juniors are asked the skill more often, red when less; a gap under half a point stays grey.
const tone = (d) => (Math.abs(d.diff) < 0.005 ? "#9a9890" : d.diff > 0 ? "#2f6fde" : "#cf222e");
```

<div class="card">

<p class="note-small">${GROUP_LABEL[juniorGroup]}: ${int(postsOf(juniorGroup))} постов против всех ${int(postsOf(allGroup))} постов группы «${GROUP_LABEL[allGroup]}». Навык показан, если его назвали хотя бы ${int(trends.min_mentions)} постов для джунов. У джуна чаще: ${more} из ${rows.length}.</p>

<div class="dot-legend"><span><i class="dark"></i>стажёры и джуны</span><span><i class="ring"></i>все посты группы</span><span><i class="more-line"></i>у джуна чаще</span><span><i class="less-line"></i>у джуна реже</span></div>

```js
Plot.plot({
  width,
  height: rows.length * 30 + 50,
  marginLeft: width < 600 ? 100 : 170,
  marginRight: width < 600 ? 70 : 100,
  marginTop: 10,
  x: {grid: true, tickFormat: pct, label: null, zero: true, ticks: width < 600 ? 3 : 6},
  y: {domain: rows.map((d) => d.skill), label: null, tickSize: 0},
  style: {fontSize: "13px"},
  marks: [
    Plot.link(rows, {y1: "skill", y2: "skill", x1: "all", x2: "jun", stroke: tone, strokeWidth: 4, strokeLinecap: "round"}),
    Plot.dot(rows, {y: "skill", x: "all", r: 5.5, fill: "white", stroke: "#8c8a84", strokeWidth: 2}),
    Plot.dot(rows, {y: "skill", x: "jun", r: 6.5, fill: "#2b2b2b", stroke: "white", strokeWidth: 1.5}),
    Plot.text(rows, {y: "skill", text: (d) => points(d.diff), frameAnchor: "right", textAnchor: "end", dx: width < 600 ? 66 : 92, fill: tone, fontWeight: 650}),
    Plot.tip(rows, Plot.pointerY({y: "skill", x: "jun", title: (d) => `${d.skill}\nу джунов ${share(d.j.posts_mentioning, d.j.posts)}\nу всех ${share(d.a.posts_mentioning, d.a.posts)}`}))
  ]
})
```

</div>
