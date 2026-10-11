import {readFileSync} from "node:fs";

// The header of every page says whose slice of the market this is and how fresh it is; both come
// from the export (python -m radar.site.export), read at build time.
const meta = JSON.parse(readFileSync(new URL("./src/data/meta.json", import.meta.url), "utf-8"));
const longDate = (iso) =>
  new Date(iso.length === 10 ? `${iso}T00:00:00Z` : iso).toLocaleDateString("ru-RU", {
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC"
  }).replace(/\s*г\.$/, "");

const REPO = "https://github.com/alekseevpavel04/ru-it-jobs-radar";

export default {
  title: "Радар IT-вакансий",
  root: "src",
  lang: "ru",
  // GitHub Pages serves the project at /ru-it-jobs-radar/; pages link relatively, the 404 page needs the base
  base: "/ru-it-jobs-radar/",
  pages: [
    {name: "Навыки", path: "/skills"},
    {name: "Зарплаты", path: "/salaries"},
    {name: "Джуниорский срез", path: "/junior"},
    {name: "Время жизни вакансии", path: "/lifetime"},
    {name: "Недельные выпуски", path: "/digest/"},
    {name: "Методика и ограничения", path: "/methodology"}
  ],
  header: `<div class="radar-header">Срез рынка по открытым источникам: «Работа в России», Хабр Карьера, публичные
Telegram-каналы. hh.ru в выборку не входит. Данные по ${longDate(meta.data_through)}, выгрузка
${longDate(meta.exported_at)}. <a href="/methodology">Как считается</a></div>`,
  footer: `Код, витрины и методика: <a href="${REPO}">${REPO.replace("https://", "")}</a>`,
  toc: {label: "На странице"},
  search: false,
  // The default stylesheet links Google Fonts; the site loads nothing from other hosts.
  globalStylesheets: [],
  style: "style.css"
};
