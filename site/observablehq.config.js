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
  // The order is the order of the "Назад" and "Дальше" links: the questions of a reader come first.
  pages: [
    {name: "Навыки", path: "/index"},
    {name: "О данных", path: "/methodology"}
  ],
  header: `<div class="radar-header">Открытые источники, без hh.ru · данные по ${longDate(meta.data_through)} · <a href="/methodology">О данных</a></div>`,
  footer: `Выгрузка данных ${longDate(meta.exported_at)} · <a href="/digest/">Недельные выпуски</a> · Код, витрины и методика: <a href="${REPO}">${REPO.replace("https://", "")}</a>`,
  toc: false,
  search: false,
  // The default stylesheet links Google Fonts; the site loads nothing from other hosts.
  globalStylesheets: [],
  style: "style.css"
};
