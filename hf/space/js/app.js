import {extract, isCached, isLoaded, loadInfo, loadModel} from "./extract.js";

const ROLE = {
  ml_engineer: "ML-инженер", data_scientist: "Data Scientist", mlops: "MLOps", data_engineer: "Инженер данных",
  data_analyst: "Аналитик данных", business_analyst: "Бизнес-аналитик", system_analyst: "Системный аналитик",
  backend: "Backend", frontend: "Frontend", fullstack: "Fullstack", mobile: "Мобильная разработка",
  software_engineer: "Программист без уточнения", developer_1c: "Разработчик 1С", embedded: "Встраиваемые системы",
  devops: "DevOps", qa: "Тестирование", sysadmin: "Системный администратор", support: "Техподдержка",
  security: "Информационная безопасность", designer: "Дизайн", manager: "Управление", other_it: "Прочее IT",
  non_it: "Не IT"
};
const GRADE = {intern: "стажёр", junior: "junior", middle: "middle", senior: "senior", lead: "lead"};
const FORMAT = {office: "офис", hybrid: "гибрид", remote: "удалённо"};
const EMPLOYMENT = {labor_contract: "трудовой договор", gph: "ГПХ", self_employed: "самозанятость", internship: "стажировка"};
const LANG = {ru: "русский", en: "английский", other: "другой"};
const PERIOD = {month: "в месяц", year: "в год", day: "в день", hour: "в час"};
const GROSS = {gross: "до вычета налога", net: "на руки"};

const $ = (id) => document.getElementById(id);
const nf = new Intl.NumberFormat("ru-RU");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const none = '<span class="empty">не указано</span>';
const longDate = (iso) => new Date(iso.length === 10 ? iso + "T00:00:00Z" : iso)
  .toLocaleDateString("ru-RU", {day: "numeric", month: "long", year: "numeric", timeZone: "UTC"}).replace(/\s*г\.$/, "");
const dec = (x, d) => x.toFixed(d).replace(".", ",");
const mib = (bytes) => nf.format(Math.round(bytes / 2 ** 20));

let info = null;
let exampleIndex = 0;
let busy = false;

function setStatus(id, text, error = false) {
  $(id).className = error ? "status error" : "status";
  $(id).textContent = text;
}

function salaryText(s) {
  if (!s) return none;
  const parts = [];
  if (s.from != null && s.to != null) parts.push(`от ${nf.format(s.from)} до ${nf.format(s.to)}`);
  else if (s.from != null) parts.push(`от ${nf.format(s.from)}`);
  else if (s.to != null) parts.push(`до ${nf.format(s.to)}`);
  if (s.currency) parts.push(s.currency);
  if (s.period) parts.push(PERIOD[s.period]);
  if (s.gross_net) parts.push(GROSS[s.gross_net]);
  return esc(parts.join(" ")) || none;
}

function experienceText(e) {
  if (!e) return none;
  if (e.min != null && e.max != null) return `от ${e.min} до ${e.max} лет`;
  if (e.min != null) return `от ${e.min} лет`;
  return `до ${e.max} лет`;
}

function tags(list) {
  return list.length ? `<div class="tags">${list.map((x) => `<span>${esc(x)}</span>`).join("")}</div>` : none;
}

function showFields(r) {
  if (!r.valid) {
    $("fields").innerHTML = '<p class="status error">Ответ модели не прошёл проверку схемы, поля не извлечены.</p>';
    return;
  }
  const f = r.fields;
  const rows = [
    ["Роль", f.role ? esc(ROLE[f.role]) : none],
    ["Грейд", f.grade ? esc(GRADE[f.grade]) : none],
    ["Навыки", tags(f.skills)],
    ["Зарплата в тексте", salaryText(f.salary)],
    ["Формат работы", f.work_format ? esc(FORMAT[f.work_format]) : none],
    ["Города", tags(f.cities)],
    ["Опыт", experienceText(f.experience_years)],
    ["Занятость", f.employment_type ? esc(EMPLOYMENT[f.employment_type]) : none],
    ["Язык текста", esc(LANG[f.lang])]
  ];
  let html = "<table>" + rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("") + "</table>";
  if (r.input_truncated) {
    html += `<p class="note">Текст длиннее, чем модель читает: прочитано начало, ${nf.format(info.prompt.max_input_tokens)} токенов.</p>`;
  }
  $("fields").innerHTML = html;
}

function showSalary(r) {
  if (r.salary_note === "non_it") {
    $("salary").innerHTML = '<p class="empty">Модель зарплаты оценивает только IT-вакансии, а эта размечена как не IT.</p>';
    return;
  }
  if (!r.salary) {
    $("salary").innerHTML = '<p class="empty">Без извлечённых полей оценки нет.</p>';
    return;
  }
  const s = r.salary;
  const level = Math.round(s.level * 100);
  $("salary").innerHTML = `
    <div class="big">${nf.format(s.point)} ₽ в месяц</div>
    <p>Интервал ${level}%: от ${nf.format(s.low)} до ${nf.format(s.high)} ₽.<br>
    Типичная вилка в таких вакансиях: от ${nf.format(s.typical_from)} до ${nf.format(s.typical_to)} ₽.</p>
    <p class="note">Похожих вакансий в обучении (роль, грейд, регион): ${nf.format(s.n_similar)}.
    Оценка по рынку вакансий «Работы в России» с указанной зарплатой: середина опубликованной вилки, суммы как их публикует источник (gross или net он не указывает). Зарплата из текста в оценку не входит.
    На dev-выборке модели интервал ${level}% накрыл ${nf.format(info.salary.dev_coverage.n)} из ${nf.format(info.salary.dev_coverage.denominator)} вакансий.</p>`;
}

export async function runOn(text) {
  const r = await extract(text);
  showFields(r);
  showSalary(r);
  $("json").textContent = JSON.stringify({fields: r.fields, salary: r.salary}, null, 2);
  return r;
}

async function run() {
  const text = $("text").value.trim();
  if (!text) {
    setStatus("status", "Вставьте текст вакансии.");
    return;
  }
  if (busy || !isLoaded()) return;
  busy = true;
  $("run").disabled = true;
  const started = performance.now();
  const tick = () => setStatus("status", `Модель читает текст и пишет ответ, прошло ${Math.round((performance.now() - started) / 1000)} с.`);
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const r = await runOn(text);
    clearInterval(timer);
    setStatus("status", `Ответ за ${dec(r.seconds.total, 1)} с.`);
  } catch (e) {
    clearInterval(timer);
    setStatus("status", "Ошибка: " + e.message, true);
  } finally {
    busy = false;
    $("run").disabled = false;
  }
}

export async function startModel() {
  $("load").disabled = true;
  const started = performance.now();
  const progress = $("progress");
  progress.hidden = false;
  try {
    const cached = await isCached();
    setStatus("load-status", cached ? "Модель читается из хранилища браузера." : "Скачивается модель.");
    const state = await loadModel(({loaded, total}) => {
      if (!total) return;
      progress.max = total;
      progress.value = loaded;
      setStatus("load-status", `Скачано ${mib(loaded)} из ${mib(total)} МиБ.`);
    });
    progress.hidden = true;
    const seconds = (performance.now() - started) / 1000;
    const threads = state.multithread ? `потоков: ${state.threads}` : "один поток";
    setStatus("load-status", `Модель готова за ${dec(seconds, 1)} с, ${threads}.`);
    $("load").hidden = true;
    $("run").disabled = false;
    return {seconds, ...state, cached};
  } catch (e) {
    const quota = e.name === "QuotaExceededError" || /quota/i.test(e.message);
    setStatus(
      "load-status",
      quota
        ? `Браузер не дал места под файл модели. Нужно около ${dec(info.model.gguf_size_bytes / 1e9, 1)} ГБ свободного места на диске; в режиме инкогнито хранилище браузера меньше.`
        : "Модель не загрузилась: " + e.message,
      true
    );
    $("load").disabled = false;
    throw e;
  }
}

function nextExample() {
  const examples = info.examples;
  if (!examples.length) return;
  const ex = examples[exampleIndex++ % examples.length];
  $("text").value = ex.text;
  $("example-source").innerHTML = `Источник: портал «Работа в России», trudvsem.ru, открытые данные Роструда${ex.url ? `, <a href="${esc(ex.url)}" rel="noopener">вакансия</a>` : ""}.`;
}

function describe(cached) {
  const size = nf.format(Math.round(info.model.gguf_size_mib));
  $("load").textContent = cached ? "Запустить модель из хранилища браузера" : `Скачать модель (${size} МиБ) и запустить`;
  let text = `Модель работает в вашем браузере на процессоре, текст никуда не отправляется. При первом запуске браузер скачает файл модели, ${size} МиБ, и сохранит его у себя, следующие запуски берут его оттуда. На телефоне загрузка и ответ могут идти очень долго, а памяти может не хватить.`;
  const s = info.speed;
  if (s) {
    text += ` В проверке на настольном процессоре (${s.cpu}, Chromium, ${s.threads} потоков) половина ответов пришла быстрее ${Math.round(s.extract_p50_s)} с, первый ответ после загрузки дольше, около ${Math.round(s.first_extract_s)} с: модель впервые читает инструкцию. В один поток половина ответов шла дольше ${Math.round(s.extract_single_thread_p50_s)} с. На другом компьютере время будет другим.`;
  }
  $("model-note").textContent = text;
  if (!self.crossOriginIsolated) {
    $("thread-note").hidden = false;
    $("thread-note").innerHTML = `Браузер не разрешил этой странице несколько потоков, модель будет работать в один поток. Так бывает внутри окна huggingface.co; страница напрямую: <a href="${esc(info.links.space_direct)}" rel="noopener">${esc(info.links.space_direct.replace("https://", ""))}</a>.`;
  }
}

function footer() {
  const d = info.dev;
  const links = info.links;
  $("slice").textContent = `Срез рынка проекта: «Работа в России» (trudvsem), Хабр Карьера, SuperJob и публичные Telegram-каналы. hh.ru в выборку не входит. Модель извлечения собрана ${longDate(info.model.built_at)}, модель зарплаты обучена ${longDate(info.salary.trained_at)}.`;
  $("footer").innerHTML = `
    <p>Модель извлечения: ${esc(info.model.base)}, дообученная (LoRA) на разметке двух моделей-учителей, ${esc(info.model.gguf_type)} GGUF в браузере через wllama ${esc(info.libraries["@wllama/wllama"])} (llama.cpp в WebAssembly).
    На dev-выборке эталона (${nf.format(d.n_examples)} вакансий «Работы в России» и Хабр Карьеры, предразметка LLM и ручная проверка) macro F1 ${dec(d.macro_f1, 3)} [${dec(d.ci_low, 3)}; ${dec(d.ci_high, 3)}], валидный JSON ${d.valid_json.n} из ${d.valid_json.denominator}.</p>
    <p>Модель зарплаты: ${esc(info.salary.version)}, градиентный бустинг с конформным интервалом ${Math.round(info.salary.level * 100)}%, обучена на ${nf.format(info.salary.rows_train)} IT-вакансиях с зарплатой в рублях, интервал откалиброван ещё на ${nf.format(info.salary.rows_calib)}.</p>
    <p><a href="${links.site}">Сайт с рынком</a> · <a href="${links.github}">Код и методика</a> · <a href="${links.model}">Модель</a> · <a href="${links.dataset}">Датасет</a> · <a href="NOTICE.md">Лицензии</a></p>`;
}

$("run").addEventListener("click", run);
$("load").addEventListener("click", () => startModel().catch(() => {}));
$("example").addEventListener("click", nextExample);
$("text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
});

info = await loadInfo();
footer();
describe(await isCached().catch(() => false));
$("load").disabled = false;
$("example").disabled = false;
