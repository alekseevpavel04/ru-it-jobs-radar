// The skills of the main page in three blocks (rising, within noise, falling): a grey dot for the base
// year, a faint arrow to a dot for the recent years coloured by the change, green up and red down, the
// share written beside it and the change in points on the right. One x scale for all blocks, so dots
// compare across them.
import * as Plot from "npm:@observablehq/plot";
import {html} from "npm:htl";
import {COLOR, int, pct, points} from "./labels.js";

// Directions and grades of radar.site.skill_trends; a cell of the data is "<direction>_<grade>".
export const DIRECTION = {
  it: "Всё IT",
  backend: "Backend",
  frontend: "Frontend",
  qa: "Тестирование",
  analytics: "Аналитика",
  ml: "ML и Data Science",
  dataeng: "Инженерия данных",
  devops: "DevOps и администрирование",
  mobile: "Мобильная разработка",
  fullstack: "Fullstack",
  security: "Инфобезопасность"
};
// Tiles of the direction mix: the eight skill directions and the roles counted in the mix only.
const MIX_LABEL = {
  ...Object.fromEntries(Object.entries(DIRECTION).filter(([k]) => k !== "it")),
  management: "Управление",
  design: "Дизайн"
};
export const GRADE_GROUP = {all: "Все грейды", junior: "Стажёры и джуны", middle: "Мидлы", senior: "Сеньоры и лиды"};

export const UP = "#1a7f37";
export const DOWN = "#cf222e";
const NOISE = "#9a9890";
const BASE = "#8c8a84";

// Change of two shares beyond the 95 % interval of their difference (normal approximation, both samples
// large here: hundreds to tens of thousands of posts).
export function change(base, recent) {
  const p1 = base.n / base.of, p2 = recent.n / recent.of;
  const se = Math.sqrt((p1 * (1 - p1)) / base.of + (p2 * (1 - p2)) / recent.of);
  return {diff: p2 - p1, significant: Math.abs(p2 - p1) > 1.96 * se};
}

export const changeColor = (c) => (!c.significant ? NOISE : c.diff > 0 ? UP : DOWN);
const changeText = (c) => (c.significant ? points(c.diff) : "≈");

const BLOCKS = [
  {title: "Растут", color: UP, test: (c) => c.significant && c.diff > 0},
  {title: "В пределах шума", color: NOISE, test: (c) => !c.significant},
  {title: "Падают", color: DOWN, test: (c) => c.significant && c.diff < 0}
];

function blockPlot(rows, {width, top, row, font}) {
  // Dots, arrow and label gap scale with the row height.
  const dot = Math.max(3.5, Math.min(7.5, row / 3.2));
  const gap = dot + 5;
  const narrow = width < 600;
  const base = (r) => r.base.share;
  const rises = (r) => r.share >= r.base.share;
  const right = narrow ? 72 : 120;
  return Plot.plot({
    width,
    height: rows.length * row + 16,
    marginLeft: narrow ? 96 : 160,
    marginRight: right,
    marginTop: 4,
    marginBottom: 8,
    x: {domain: [0, top * (narrow ? 1.5 : 1.12)], axis: null},
    y: {domain: rows.map((r) => r.skill), label: null, tickSize: 0, tickFormat: (t) => (narrow && t.length > 13 ? `${t.slice(0, 12)}…` : t)},
    // A phone keeps 12px whatever the desktop size, and cuts long names short.
    style: {fontSize: `${narrow ? 12 : font}px`},
    marks: [
      Plot.arrow(rows, {y1: "skill", y2: "skill", x1: base, x2: "share", stroke: "#bdbab2", strokeWidth: 1.6, headLength: Math.min(6, row / 4), insetStart: 5, insetEnd: dot + 2}),
      Plot.dot(rows, {y: "skill", x: base, r: Math.max(2.5, dot / 1.9), fill: BASE}),
      Plot.dot(rows, {y: "skill", x: "share", r: dot, fill: (r) => changeColor(r.change), stroke: "white", strokeWidth: 1.5}),
      // The share beside the coloured dot, on the side away from the base dot.
      Plot.text(rows.filter(rises), {y: "skill", x: "share", text: (r) => pct(r.share), dx: gap, textAnchor: "start", fontWeight: 650, fill: COLOR.bar}),
      Plot.text(rows.filter((r) => !rises(r)), {y: "skill", x: "share", text: (r) => pct(r.share), dx: -gap, textAnchor: "end", fontWeight: 650, fill: COLOR.bar}),
      Plot.text(rows, {y: "skill", text: (r) => changeText(r.change), frameAnchor: "right", textAnchor: "end", dx: right - 6, fill: (r) => changeColor(r.change), fontWeight: 700}),
    ]
  });
}

// A tinted strip across the width with a bar of the block colour on the left.
const heading = (block) => html`<div class="block-head" style=${`--block: ${block.color}`}>${block.title}</div>`;

// Without a base year: the recent share alone, one dot per skill with the share beside it.
function nowPlot(rows, {width, row, font}) {
  const dot = Math.max(3.5, Math.min(7, row / 3.2));
  const narrow = width < 600;
  const top = Math.max(...rows.map((r) => r.share));
  return Plot.plot({
    width,
    height: rows.length * row + 16,
    marginLeft: narrow ? 96 : 160,
    marginRight: 16,
    marginTop: 4,
    marginBottom: 8,
    x: {domain: [0, top * 1.12], axis: null},
    y: {domain: rows.map((r) => r.skill), label: null, tickSize: 0, tickFormat: (t) => (narrow && t.length > 13 ? `${t.slice(0, 12)}…` : t)},
    // A phone keeps 12px whatever the desktop size, and cuts long names short.
    style: {fontSize: `${narrow ? 12 : font}px`},
    marks: [
      Plot.dot(rows, {y: "skill", x: "share", r: dot, fill: COLOR.bar, stroke: "white", strokeWidth: 1.5}),
      Plot.text(rows, {y: "skill", x: "share", text: (r) => pct(r.share), dx: dot + 5, textAnchor: "start", fontWeight: 650, fill: COLOR.bar}),
    ]
  });
}

export function skillChart(rows, {width, recentLabel, baseYear, row = 16, font = 14}) {
  if (!baseYear) return nowPlot(rows, {width, row, font});
  const top = Math.max(...rows.map((r) => Math.max(r.share, r.base.share)));
  const legend = html`<div class="dot-legend"><span><i class="up-dot"></i><i class="down-dot"></i>сейчас, ${recentLabel}</span><span><i class="base"></i>${baseYear}</span><span>стрелка показывает, куда сдвинулась доля</span></div>`;
  const parts = BLOCKS.map((b) => [b, rows.filter((r) => b.test(r.change))]).filter(([, part]) => part.length);
  return html`<div>${legend}${parts.map(([b, part]) => html`${heading(b)}${blockPlot(part, {width, top, row, font})}`)}</div>`;
}

// The grade mix of all IT by year: a line per grade, labelled at its end, the unfinished year dashed.
const GRADE_LINE = [
  {grade: "junior", label: "стажёры и джуны", short: "джуны", color: "#3987e5"},
  {grade: "middle", label: "мидлы", short: "мидлы", color: "#d4a017"},
  {grade: "senior", label: "сеньоры и лиды", short: "сеньоры", color: "#7a3fb0"}
];

export function gradeChart(grades, {width, partialYear}) {
  const narrow = width < 600;
  const rows = grades.map((g) => ({...g, share: g.posts / g.graded}));
  const years = [...new Set(rows.map((r) => r.year))].sort();
  const last = years.at(-1);
  // End labels of close lines are pushed apart symmetrically; the dots stay where they are.
  const GAP = 0.045;
  const ends = GRADE_LINE.map((g) => ({...g, y: rows.find((r) => r.grade === g.grade && r.year === last).share})).sort((a, b) => a.y - b.y);
  for (let pass = 0; pass < 50; pass++) {
    for (let i = 1; i < ends.length; i++) {
      const overlap = GAP - (ends[i].y - ends[i - 1].y);
      if (overlap > 0) {
        ends[i].y += overlap / 2;
        ends[i - 1].y -= overlap / 2;
      }
    }
  }
  const labelY = new Map(ends.map((e) => [e.grade, e.y]));
  const marks = [];
  for (const {grade, label, short, color} of GRADE_LINE) {
    const line = rows.filter((r) => r.grade === grade);
    const done = line.filter((r) => r.year !== partialYear);
    const tail = line.filter((r) => r.year >= last - 1);
    marks.push(
      Plot.line(done, {x: "year", y: "share", stroke: color, strokeWidth: 2.6}),
      partialYear ? Plot.line(tail, {x: "year", y: "share", stroke: color, strokeWidth: 2.6, strokeDasharray: "4,3"}) : null,
      Plot.dot(line, {x: "year", y: "share", r: 3.5, fill: (r) => (r.year === partialYear ? "white" : color), stroke: color, strokeWidth: 1.8}),
      Plot.text(line.filter((r) => r.year === last), {x: "year", y: () => labelY.get(grade), text: (r) => `${narrow ? short : label} ${pct(r.share)}`, dx: 10, textAnchor: "start", fill: color, fontWeight: 700})
    );
  }
  return Plot.plot({
    width,
    height: narrow ? 260 : 300,
    marginLeft: 44,
    marginRight: narrow ? 104 : 190,
    marginTop: 12,
    x: {domain: years, type: "point", label: null, tickFormat: (y) => (y === partialYear ? `${y}*` : `${y}`), ticks: narrow ? years.filter((y) => y % 2 === 0) : years},
    y: {domain: [0, 0.6], grid: true, label: null, tickFormat: (x) => `${Math.round(x * 100)} %`, ticks: 4},
    style: {fontSize: narrow ? "12px" : "14px"},
    marks: [
      ...marks,
    ]
  });
}

// Each direction's share of all IT posts of the panel channels by year, one small chart per direction:
// the share of the last year in big type, the change since the base year coloured as on the rest of the page.
export function directionTiles(directions, {width, partialYear}) {
  const narrow = width < 600;
  const years = [...new Set(directions.map((d) => d.year))].sort();
  const first = years[0], last = years.at(-1);
  const tiles = Object.keys(MIX_LABEL)
    .map((key) => {
      const line = directions.filter((d) => d.direction === key).map((d) => ({...d, share: d.posts / d.total}));
      const a = line.find((d) => d.year === first), b = line.find((d) => d.year === last);
      return {key, line, a, b, c: change({n: a.posts, of: a.total}, {n: b.posts, of: b.total})};
    })
    .sort((x, y) => y.c.diff - x.c.diff);
  const cols = narrow ? 2 : 4;
  // width is the page column; the card takes 2 x 16px of padding, each tile 2 x 12px plus its border.
  const tileWidth = Math.floor((width - 32 - (cols - 1) * 12) / cols) - 26;
  return html`<div class="direction-tiles" style=${`grid-template-columns: repeat(${cols}, 1fr)`}>${tiles.map(({key, line, b, c}) => {
    const color = changeColor(c);
    const done = line.filter((d) => d.year !== partialYear);
    const plot = Plot.plot({
      width: tileWidth,
      height: 72,
      marginTop: 6,
      marginLeft: 14,
      marginRight: 18,
      marginBottom: 20,
      x: {domain: years, type: "point", axis: "bottom", ticks: [first, last], tickFormat: (y) => (y === partialYear ? `${y}*` : `${y}`), tickSize: 0, label: null},
      y: {domain: [0, Math.max(...line.map((d) => d.share)) * 1.15], axis: null},
      style: {fontSize: "11px", color: "#8c8a84"},
      marks: [
        Plot.areaY(line, {x: "year", y: "share", fill: color, fillOpacity: 0.08}),
        Plot.line(done, {x: "year", y: "share", stroke: color, strokeWidth: 2}),
        partialYear ? Plot.line(line.filter((d) => d.year >= last - 1), {x: "year", y: "share", stroke: color, strokeWidth: 2, strokeDasharray: "3,3"}) : null,
        Plot.dot(line.filter((d) => d.year === last), {x: "year", y: "share", r: 3, fill: color}),
      ]
    });
    return html`<div class="direction-tile">
      <div class="direction-name" title=${MIX_LABEL[key]}>${MIX_LABEL[key]}</div>
      <div class="direction-numbers"><b>${pct(b.share)}</b><span style=${`color: ${color}`}>${c.significant ? points(c.diff) : "≈"}</span></div>
      ${plot}
    </div>`;
  })}</div>`;
}
