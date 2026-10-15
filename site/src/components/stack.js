// "Check your stack": the reader's skills against the shares of the chosen direction and grade and the
// salary effects, plus what the cell asks for most and what grows fastest that the reader lacks.
import {html} from "npm:htl";
import {pct, signedPct} from "./labels.js";
import {DOWN, UP, changeColor, changeText} from "./skills.js";

// Spellings people type for skills whose names in the data differ.
const ALIAS = {
  js: "javascript",
  ts: "typescript",
  k8s: "kubernetes",
  postgres: "postgresql",
  golang: "go",
  py: "python",
  питон: "python",
  докер: "docker",
  ci: "ci/cd",
  cicd: "ci/cd",
  "1c": "1с"
};
const norm = (s) => s.toLowerCase().replace(/ё/g, "е").replace(/\s+/g, " ").trim();
const canon = (s) => ALIAS[norm(s)] ?? norm(s);

export const parseStack = (text) => [...new Set(text.split(/[,;\n]/).map((s) => s.trim()).filter(Boolean))];

// rows: the cell's skills as on the main chart ({key, skill, share, change}); effects: skill_effects.json
// effects; name: the display name of an effect; fdr: the false discovery rate of the effect blocks.
// onAdd(skill) puts a suggested skill into the reader's list.
export function stackReport(rows, effects, {mine, name, fdr, label, baseYear, onAdd}) {
  const byName = new Map();
  for (const r of rows) for (const n of [r.key, r.skill]) byName.set(canon(n), r);
  const effectOf = new Map();
  for (const e of effects) for (const n of [e.skill, name(e)]) effectOf.set(canon(n), e);
  const found = [], missing = [];
  for (const s of mine) {
    const r = byName.get(canon(s));
    if (r) found.push({r, e: effectOf.get(canon(r.key)) ?? effectOf.get(canon(r.skill))});
    else missing.push(s);
  }
  const have = new Set(found.map((f) => f.r));
  const asked = rows.filter((r) => !have.has(r)).sort((a, b) => b.share - a.share).slice(0, 5);
  const growing = rows
    .filter((r) => !have.has(r) && r.change?.significant && r.change.diff > 0)
    .sort((a, b) => b.change.diff - a.change.diff)
    .slice(0, 5);
  const effectText = (e) =>
    !e ? html`<span class="muted">не оценивали</span>` : e.q_bh < fdr ? html`<b style=${`color: ${e.pct > 0 ? UP : DOWN}`}>${signedPct(e.pct)}</b>` : html`<span class="muted">не доказано</span>`;
  const list = (xs, text) => (xs.length ? html`<ul class="stack-list">${xs.map((r) => html`<li><button class="stack-add" title=${`Добавить ${r.skill} в мои навыки`} onclick=${() => onAdd(r.skill)}>+</button><b>${r.skill}</b> <span class="muted">${text(r)}</span></li>`)}</ul>` : html`<p class="muted">Таких нет.</p>`);
  return html`<div class="stack">
    ${found.length
      ? html`<table class="stack-table">
          <thead><tr><th>Навык</th><th>Доля постов</th>${baseYear ? html`<th>С ${baseYear}</th>` : ""}<th>Вилка</th></tr></thead>
          <tbody>${found.map(
            ({r, e}) => html`<tr><td>${r.skill}</td><td>${pct(r.share)}</td>${baseYear ? html`<td style=${`color: ${changeColor(r.change)}`}>${changeText(r.change)}</td>` : ""}<td>${effectText(e)}</td></tr>`
          )}</tbody>
        </table>`
      : mine.length
        ? ""
        : html`<p class="muted">Впишите свои навыки через запятую, например «Python, SQL, Docker».</p>`}
    ${missing.length ? html`<p class="muted">Нет среди показанных навыков «${label}» (30 самых частых и 10 быстрее всех растущих): ${missing.join(", ")}.</p>` : ""}
    <div class="stack-columns">
      <div><h3>Чаще всего просят, а у вас нет</h3>${list(asked, (r) => pct(r.share))}</div>
      ${baseYear ? html`<div><h3>Быстрее всего растут, а у вас нет</h3>${list(growing, (r) => changeText(r.change))}</div>` : ""}
    </div>
  </div>`;
}
