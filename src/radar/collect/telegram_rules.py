"""Rules that tell a vacancy post from everything else in a job channel, and count vacancies in a post.

A vacancy post names at least one concrete position with an employer or conditions. Ads, resumes,
courses, articles, market news and empty media posts are not vacancies. Rules v2: agreement with
manual labels is reported by scripts/telegram_survey/analyze.py; the rules overrate content channels
without vacancies (see docs/sources.md, Telegram 1.5), so they are used only with the enabled channels.
"""

import re
from dataclasses import dataclass

RULES_VERSION = 3

AD = re.compile(r"\berid\b|#реклама|^\s*реклама\b|партн[её]рский (пост|материал)|на правах рекламы", re.I | re.M)
RESUME = re.compile(
    r"#резюме|#cv\b|#ищу_?работу|#lookingforjob|#resume|#ищу\b|\bищу (работу|позицию|вакансию|проект|заказ)"
    r"|рассмотрю предложения|open to work|в поиске работы|резюме:",
    re.I,
)
# A post that opens with a résumé or a digest of résumés. javascript_jobs_feed publishes "Лучшие резюме и
# вакансии за <день>", "Обзор за <день>" and "Резюме за <день>" since 2025, and RESUME below lets a head
# with "#резюме" through whenever it also carries a marker hashtag such as #junior. The pattern runs on the
# lowercased text; dbt/macros/telegram_rules.sql applies the same one to posts collected under older rules.
RESUME_HEAD = re.compile(r"^[^a-zа-яё0-9]*(резюме|лучшие резюме|топ резюме|обзор за|ищу работу)")
ROLE = (
    r"разработчик|developer|engineer|инженер|аналитик|analyst|тестировщик|\bqa\b|\baqa\b|\bqaa\b|tester|тестиров"
    r"|pentest|пентест|\bsdet\b|devops|программист|дизайнер|designer"
    r"|scientist|researcher|исследовател|архитектор|architect|administrator|администратор|\bdba\b|\bsre\b|\bdata\b"
    r"|\bml\b|\b1[сc]\b|frontend|backend|fullstack|full stack|фронтенд|бэкенд|бекенд|руководител|\bhead\b|\bcto\b"
    r"|\bcpo\b|manager|менеджер|director|директор|officer|specialist|специалист|recruiter|рекрутер|\blead\b|лид\b"
    r"|producer|продюсер|artist|художник|owner|оунер|support|поддержк|оператор|consultant|консультант|маркетолог"
    r"|marketer|copywriter|копирайтер|writer|писател|автор|scrum|coach|mlops|стажер|стажёр|intern\b|верстальщик"
    r"|тимлид"
)
# Seven groups of vacancy signals besides the role; a post with a role and three groups is a vacancy.
SIGNALS = {
    "marker": r"#ваканси|\bваканси[яи]\b|#vacancy|#job\b|#jobs\b|#hiring|\bhiring\b|\bищем\b|в поиске\b|в команду"
    r"|требуется|\bнуж(ен|на|ны)\b|приглашаем|открыт(а|ы)? (позици|набор|ваканси)|стажировк|we are looking"
    r"|looking for|join our team|#стажировк|#internship",
    "salary": r"\d[\d\s.,]*\s?(₽|руб|р\.|k\b|к\b|тыс|000\b)|зарплат|з/п|\bзп\b|вилка|оклад|\bдоход|salary|\$\s?\d"
    r"|\d\s?\$|€|\busd\b|\beur\b|\bgross\b|\bnet\b|по договор[её]нност|по результатам собеседования",
    "requirements": r"требовани|ожидаем|что нужно|мы жд[её]м|нам важно|опыт (работы )?от|опыт коммерческ|requirements"
    r"|must have|you have|навыки|ты (умеешь|знаешь)|будет плюсом|nice to have",
    "duties": r"предстоит|обязанност|задачи|чем (будешь|нужно|придется|придётся) занима|что (нужно )?делать"
    r"|responsibilit|you will",
    "conditions": r"услови[яе]|мы предлагаем|\bпредлагаем|что мы (дадим|предлагаем)|бенефит|\bдмс\b|формат( работы)?"
    r"|удал[её]н|гибрид|\bофис|full[- ]?time|part[- ]?time|полная занятость|частичная занятость|локаци|релокац"
    r"|remote|we offer|график|оформлени|\bтк рф\b|locations?:|languages?:|on-?site|hybrid|relocation|visa\b"
    r"|контракт",
    "apply": r"отклик|резюме (присыла|отправля|направля|жд[её]м|кидай)|пишите|писать (в|сюда|@)|контакт|apply"
    r"|подробнее|hh\.ru/vacancy|career\.habr\.com/vacancies|u\.habr\.com|getmatch\.ru|/vacanc|/jobs?/|@\w{4,}",
    "grade": r"#junior|#middle|#senior|#lead|\bjunior\b|\bmiddle\b|\bsenior\b|\blead\b|\bjun\b|джун|мидл|сеньор"
    r"|синьор|тимлид|team ?lead|стаж[её]р|\bintern\b|грейд",
    "role": ROLE,
}
SIGNALS_RX = {k: re.compile(v, re.I) for k, v in SIGNALS.items()}
ROLE_RX = re.compile(ROLE, re.I)
BOARD_LINK = re.compile(
    r"hh\.ru/vacancy/\d+|career\.habr\.com/vacancies/\d+|u\.habr\.com/\w+|getmatch\.ru/vacancies/[\w-]+"
    r"|geekjob\.ru/vacancy/\w+|superjob\.ru/vakansii/[\w-]+|trudvsem\.ru/vacancy/[\w/-]+",
    re.I,
)
HABR = re.compile(r"career\.habr\.com|u\.habr\.com", re.I)
# Field headers that open one vacancy; the same header repeated means several vacancies in a post.
HEADERS = [
    re.compile(p, re.I | re.M)
    for p in (
        r"^\W*ваканси[яи]\s*:",
        r"^\W*(позиция|должность|position|role)\s*:",
        r"^\W*(компания|company)\s*:",
        r"^\W*(зарплата|з/п|зп|вилка|salary)\s*:",
    )
]
COURSE = re.compile(
    r"\bкурс|поток\b|интенсив|вебинар|буткемп|программ[аеуы] обучения|обучени[ея] с нуля|bootcamp", re.I
)
HIRING = re.compile(r"ваканси|hiring|\bищем\b|looking for|в поиске|приглашаем на (позицию|работу)|требуется", re.I)
COMPANY = re.compile(r"\sв\s+[«\"'A-ZА-ЯЁ]|\b(is|are) (a|an|the)\b|^\W*(компания|company)\s*:|\s@\s", re.M)
ARTICLE = re.compile(r"habr\.com/(ru/)?(articles|companies|news|post)/|vc\.ru/|tproger\.ru/(articles|news)", re.I)
BULLET = re.compile(r"^\s*(🔹|🔸|▪️?|•|–|-|\d+[.)])\s*\S")
# Requirement bullets often mention a role too ("опыт работы DevOps-инженером"); they are not positions.
NOT_POSITION = re.compile(
    r"(опыт|знани|умени|понимани|навык|работа |разработка|настройка|участие|поддержка|ведение|experience|knowledge"
    r"|strong|proficien|hands-on|ability|familiar)",
    re.I,
)
POSITION_AT = re.compile(r"\sв\s+[«\"'A-ZА-ЯЁ]|\s/\s|\s@\s|\sat\s|\s[—–-]\s")
TITLE_AT = re.compile(r"\sв\s+[«\"'A-ZА-ЯЁ]|\s/\s|\s@\s|\sat\s")
HASHTAG_LINE = re.compile(r"^(\s*#\w+)+\s*$")


@dataclass
class Verdict:
    kind: str  # vacancy, no_text, ad, resume, course, article, other
    n_units: int = 0  # vacancies in the post
    habr_link: bool = False

    @property
    def vacancy(self) -> bool:
        return self.kind == "vacancy"

    @property
    def multi(self) -> bool:
        return self.n_units >= 2


def title_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip() and not HASHTAG_LINE.match(line):
            return line.strip()[:120]
    return ""


def signals(blob: str) -> set[str]:
    return {k for k, rx in SIGNALS_RX.items() if rx.search(blob)}


def is_position_bullet(line: str) -> bool:
    if not BULLET.match(line):
        return False
    body = BULLET.sub("", line, count=1).strip()
    if line.lstrip().startswith(("🔹", "🔸")):
        return bool(ROLE_RX.search(body))
    return (
        len(body) < 120
        and not NOT_POSITION.match(body)
        and ROLE_RX.search(body[:40]) is not None
        and POSITION_AT.search(body) is not None
    )


def classify(text: str, links: list[str]) -> Verdict:
    if not text.strip():
        return Verdict("no_text")
    if AD.search(text):
        return Verdict("ad")
    if RESUME_HEAD.match(text.lower()):
        return Verdict("resume")
    blob = text + "\n" + "\n".join(links)
    sig = signals(blob)
    head = text[:300]
    if RESUME.search(head) and "marker" not in signals(head):
        return Verdict("resume")
    title = title_line(text)
    title_role = bool(ROLE_RX.search(title))
    titled_position = title_role and TITLE_AT.search(title) is not None
    if COURSE.search(head) and not HIRING.search(head) and not titled_position:
        return Verdict("course")
    boards = {m.group(0).lower() for m in BOARD_LINK.finditer(blob)}
    if ARTICLE.search(blob) and not boards:
        return Verdict("article")
    titled = title_role and (COMPANY.search(head) is not None or len(sig - {"role", "marker"}) >= 2)
    if not (("role" in sig and len(sig) >= 4) or (len(boards) >= 2 and "role" in sig) or titled):
        return Verdict("other")
    headers = max(len(rx.findall(text)) for rx in HEADERS)
    bullets = sum(1 for line in text.splitlines() if is_position_bullet(line))
    n_units = max(1, len(boards), headers, bullets if bullets >= 2 else 1)
    return Verdict("vacancy", n_units, bool(HABR.search(blob)))
