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

// The dictionary names skills in English; concepts get a Russian label on the site, technology and
// product names stay as they are. The digest uses the same labels (a test keeps them equal).
export const SKILL = {
  "API integration": "Интеграции по API",
  "A/B tests": "A/B-тесты",
  "Algorithms": "Алгоритмы",
  "Antivirus": "Антивирусная защита",
  "Clean architecture": "Чистая архитектура",
  "Communication": "Коммуникация",
  "Computer vision": "Компьютерное зрение",
  "Cryptography": "Криптография",
  "Deep learning": "Глубокое обучение",
  "Design patterns": "Паттерны проектирования",
  "Embedded": "Встраиваемые системы",
  "English": "Английский язык",
  "Firewalls": "Межсетевые экраны",
  "Gradient boosting": "Градиентный бустинг",
  "Information security standards": "Стандарты ИБ",
  "Machine learning": "Машинное обучение",
  "Manual testing": "Ручное тестирование",
  "Microservices": "Микросервисы",
  "Multithreading": "Многопоточность",
  "Networking": "Компьютерные сети",
  "Pentest": "Пентест",
  "Product analytics": "Продуктовая аналитика",
  "Prompt engineering": "Промпт-инжиниринг",
  "Recommender systems": "Рекомендательные системы",
  "Requirements": "Работа с требованиями",
  "Speech recognition": "Распознавание речи",
  "Statistics": "Статистика",
  "System design": "Проектирование систем",
  "Teamwork": "Командная работа",
  "Test automation": "Автоматизация тестирования",
  "Time series": "Временные ряды",
  "Vector databases": "Векторные БД",
  "Virtualization": "Виртуализация"
};

export const withSkillLabels = (rows) => rows.map((d) => ({...d, skill: SKILL[d.skill] ?? d.skill}));

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
export const thousands = (x) => `${int(Math.round(x / 1000))} тыс.`;

// Russian noun agreement with a number: plural(n, ["день", "дня", "дней"]).
export function plural(n, [one, few, many]) {
  const k = Math.abs(Math.round(n));
  if (k % 10 === 1 && k % 100 !== 11) return one;
  if (k % 10 >= 2 && k % 10 <= 4 && (k % 100 < 12 || k % 100 > 14)) return few;
  return many;
}
// "из N каналов" agrees with N: из 51 канала, из 47 каналов.
export const ofChannels = (n) => `${int(n)} ${n % 10 === 1 && n % 100 !== 11 ? "канала" : "каналов"}`;
export const days = (n) => `${int(n)} ${plural(n, ["день", "дня", "дней"])}`;
// "+2,7 п. п.": change of a share in percentage points.
const oneDecimal = new Intl.NumberFormat("ru-RU", {minimumFractionDigits: 1, maximumFractionDigits: 1});
// The sign follows the rounded number, so a change that rounds to zero reads "0,0", not "-0,0".
const signed = (x) => {
  const text = oneDecimal.format(Math.abs(x));
  return `${text === oneDecimal.format(0) ? "" : x > 0 ? "+" : "-"}${text}`;
};
export const points = (delta) => `${signed(delta * 100)} п. п.`;
export const signedPct = (x) => `${signed(x)} %`;

// One colour per source on every page; everything else is neutral, beginners get a colour no source has.
export const SOURCE_COLOR = {trudvsem: "#4269d0", trudvsem_snapshot: "#4269d0", habr: "#efb118", telegram: "#3ca951"};
export const FAMILY_ORDER = ["trudvsem", "habr", "telegram"];
export const SOURCE_SHORT = {trudvsem: "Работа в России", trudvsem_snapshot: "Работа в России", habr: "Хабр Карьера", telegram: "Telegram"};
export const COLOR = {
  bar: "#4a4a4a",
  all: "#b8b8b8",
  junior: "#8e44ad",
  accent: "#d1495b",
  from: "#9a9a9a",
  to: "#3a3a3a"
};

// Roles of the extraction schema (mart_extracted_salaries: Telegram salaries).
export const XROLE = {
  ml_engineer: "ML-инженер",
  data_scientist: "Data Scientist",
  mlops: "MLOps",
  data_engineer: "Инженер данных",
  data_analyst: "Аналитик данных",
  business_analyst: "Бизнес-аналитик",
  system_analyst: "Системный аналитик",
  backend: "Backend",
  frontend: "Frontend",
  fullstack: "Fullstack",
  mobile: "Мобильная разработка",
  software_engineer: "Программист без уточнения",
  developer_1c: "1С",
  embedded: "Встраиваемые системы",
  devops: "DevOps",
  qa: "Тестирование",
  sysadmin: "Системное администрирование",
  support: "Поддержка",
  security: "Информационная безопасность",
  designer: "Дизайн",
  manager: "Управление",
  other_it: "Прочее IT",
  unknown: "роль не указана"
};

export const WORK_FORMAT = {office: "офис", hybrid: "гибрид", remote: "удалённо", unknown: "не указан"};

// A source picker: radio buttons on a wide screen, a select on a phone, where three long labels wrap.
export function sourcePicker(Inputs, sources, value, label = "Источник") {
  const options = new Map(sources.map((s) => [SOURCE_SHORT[s], s]));
  return window.matchMedia("(max-width: 640px)").matches
    ? Inputs.select(options, {value, label})
    : Inputs.radio(options, {value, label});
}
