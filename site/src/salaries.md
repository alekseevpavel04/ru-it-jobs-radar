---
title: Навыки и деньги
toc: false
---

# Навыки и деньги

```js
import {SKILL, int} from "./components/labels.js";
import {effectsChart} from "./components/effects.js";

const effects = FileAttachment("data/skill_effects.json").json();
```

<p class="lead">Сколько добавляет к вилке конкретный навык. Вилка из объявления, рубли в месяц до вычета налога. Это не выплаченная зарплата. <a href="./methodology#salaries">Подробнее в методике</a>.</p>

```js
const name = (e) => SKILL[e.skill] ?? e.skill;
const level = Math.round(effects.ci_level * 100);
```

<div class="card">

<p class="note-small">На сколько процентов выше или ниже середина вилки у вакансий с навыком, чем у похожих без него (${int(effects.rows)} вакансий, почти все с портала «Работа в России»); полоса показывает ${level} % интервал. <a href="./methodology#skill-effects">Как считается</a>.</p>

```js
effectsChart(effects.effects, {width, name})
```

</div>
