// Names and number formats of the site. Bucket and source codes come from the marts
// (dbt/models/marts/_marts.yml); every label here is the Russian name of one code.

export const ROLE = {
  ml_ds: "ML и Data Science",
  data_engineering: "Инженерия данных",
  data_analytics: "Аналитика данных",
  developer_1c: "1С",
  business_system_analysis: "Бизнес- и системный анализ",
  security: "Информационная безопасность",
  devops: "DevOps",
  qa: "Тестирование",
  mobile: "Мобильная разработка",
  fullstack: "Fullstack",
  backend: "Backend",
  frontend: "Frontend",
  embedded: "Встраиваемые системы",
  sysadmin_support: "Администрирование и поддержка",
  design: "Дизайн",
  management: "Управление",
  software_engineer: "Программист без уточнения",
  other_it: "Прочее IT",
  multi_post: "Несколько вакансий в посте"
};

// Eight colours read on a stacked chart, eighteen do not: buckets are grouped for the weekly chart.
export const ROLE_GROUP = {
  backend: "Разработка",
  frontend: "Разработка",
  fullstack: "Разработка",
  mobile: "Разработка",
  embedded: "Разработка",
  software_engineer: "Разработка",
  developer_1c: "1С",
  ml_ds: "Данные и ML",
  data_engineering: "Данные и ML",
  data_analytics: "Данные и ML",
  sysadmin_support: "Администрирование, поддержка, DevOps",
  devops: "Администрирование, поддержка, DevOps",
  security: "Информационная безопасность",
  qa: "Тестирование",
  business_system_analysis: "Анализ и управление",
  management: "Анализ и управление",
  design: "Дизайн и прочее IT",
  other_it: "Дизайн и прочее IT",
  multi_post: "Несколько вакансий в посте"
};

export const ROLE_GROUP_ORDER = [
  "Разработка",
  "1С",
  "Данные и ML",
  "Администрирование, поддержка, DevOps",
  "Информационная безопасность",
  "Тестирование",
  "Анализ и управление",
  "Дизайн и прочее IT"
];

export const SOURCE = {
  trudvsem_snapshot: "Работа в России, снимки портала",
  trudvsem: "Работа в России",
  habr: "Хабр Карьера",
  telegram: "Telegram-каналы"
};

export const CATEGORY = {
  analysis: "анализ",
  analytics: "аналитика",
  architecture: "архитектура",
  bi: "BI",
  cloud: "облака",
  data: "данные",
  database: "базы данных",
  design: "дизайн",
  devops: "DevOps",
  embedded: "встраиваемые системы",
  framework: "фреймворк",
  frontend: "фронтенд",
  gamedev: "геймдев",
  language: "язык программирования",
  language_human: "иностранный язык",
  methodology: "методология",
  ml: "ML",
  mobile: "мобильная разработка",
  network: "сети",
  os: "ОС",
  platform: "платформа",
  security: "безопасность",
  soft: "гибкие навыки",
  testing: "тестирование",
  tools: "инструменты"
};

const integer = new Intl.NumberFormat("ru-RU");
const percent = new Intl.NumberFormat("ru-RU", {style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1});

export const int = (n) => integer.format(n);
export const pct = (share) => percent.format(share);
// A share is never printed without its base: "37 из 174 (21,3 %)".
export const share = (n, of) => `${int(n)} из ${int(of)} (${pct(of ? n / of : 0)})`;

// Dates of the marts are calendar days; parse and print them in UTC so no time zone shifts a day.
export const day = (iso) => new Date(`${iso}T00:00:00Z`);
// "8 октября 2026" without the trailing "г.", which doubles the full stop at the end of a sentence.
export const dateLong = (d) =>
  (typeof d === "string" ? day(d) : d)
    .toLocaleDateString("ru-RU", {day: "numeric", month: "long", year: "numeric", timeZone: "UTC"})
    .replace(/\s*г\.$/, "");
export const dateShort = (d) =>
  (typeof d === "string" ? day(d) : d).toLocaleDateString("ru-RU", {day: "numeric", month: "short", timeZone: "UTC"});
export const monthTick = (d) => d.toLocaleDateString("ru-RU", {month: "short", year: "2-digit", timeZone: "UTC"});

export const GRADE = {
  intern: "стажёр",
  junior: "джуниор",
  middle: "мидл",
  senior: "сеньор",
  lead: "лид",
  unknown: "грейд не указан"
};
export const GRADE_ORDER = ["intern", "junior", "middle", "senior", "lead", "unknown"];

// region_group of mart_salaries: remote wins over the city when the work format is remote.
export const REGION = {
  moscow: "Москва",
  spb: "Санкт-Петербург",
  other: "другие города",
  remote: "удалённо",
  unknown: "место не указано"
};
export const REGION_ORDER = ["moscow", "spb", "other", "remote", "unknown"];

// Role groups of the marts (seed role_groups: salary totals, lifetime) and of the survival model; coarser
// than ROLE_GROUP above, which only colours the weekly chart.
export const MART_GROUP = {
  dev: "Разработка, включая 1С",
  ml_ds_data: "ML, DS, данные",
  analyst: "Бизнес- и системный анализ",
  qa_devops_sys: "QA, DevOps, админы, ИБ",
  other_it: "Дизайн, управление, прочие IT"
};
export const MART_GROUP_ORDER = ["dev", "ml_ds_data", "analyst", "qa_devops_sys", "other_it"];

// Groups and levels of the lifetime model (src/radar/models/survival.py, LABELS_RU there).
export const LIFETIME = {
  overall: "все IT-вакансии",
  role: "Роль",
  experience: "Требуемый опыт",
  salary: "Зарплата в объявлении",
  region: "Регион",
  all: "все",
  dev: "разработка, включая 1С",
  ml_ds_data: "ML, DS, данные",
  analyst: "бизнес- и системный анализ",
  qa_devops_sys: "QA, DevOps, админы, ИБ",
  other_it: "дизайн, управление, прочие IT",
  none: "без опыта",
  "1-2": "1-2 года",
  "3-5": "3-5 лет",
  "6+": "6 лет и больше",
  given: "указана",
  not_given: "не указана",
  moscow: "Москва",
  spb: "Санкт-Петербург",
  other: "другие регионы",
  remote: "удалённо"
};

const money = new Intl.NumberFormat("ru-RU", {maximumFractionDigits: 0});
export const rub = (x) => (x === null || x === undefined ? "" : `${money.format(x)} ₽`);

// "из N каналов" agrees with N: из 51 канала, из 47 каналов.
export const ofChannels = (n) => `${int(n)} ${n % 10 === 1 && n % 100 !== 11 ? "канала" : "каналов"}`;
