"""Schema role by rules, for the regex baseline.

The coarse bucket comes from the sampling proxy (`role_proxy.title_bucket`), so the baseline and the
strata read titles the same way; inside a bucket, finer title rules pick the role. Following the
annotation guide, the title decides, and the text is read only when the title is generic
("Программист", "Аналитик") or names a broad family: the stack of a generic developer, SQL or BI
in the requirements of a generic analyst. Habr's own specializations stand in for a title that says
nothing specific.
"""

import re
from collections.abc import Sequence

from radar.extract.role_proxy import (
    ANALYST_OTHER,
    DEV,
    ML_DS_DATA,
    NON_IT_NOISE,
    OTHER_IT,
    QA_DEVOPS_SYS,
    title_bucket,
)


def _any(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


def normalize(text: str | None) -> str:
    return (text or "").lower().replace("ё", "е")


# Titles the proxy leaves in non_it_noise although they name data work.
_DATA_MISSED = _any(r"data[\s-]*инженер", r"data[\s-]*аналитик", r"дата[\s-]*аналитик", r"\bкхд\b", r"хранилищ\w* данн")

_ML_DS_DATA: list[tuple[str, re.Pattern[str]]] = [
    ("mlops", _any(r"mlops", r"\bml[\s-]*ops", r"ml[\s-]*платформ", r"ml[\s-]*platform")),
    (
        "data_scientist",
        _any(
            r"data[\s-]*scien", r"дата[\s-]*са[йе]ен", r"учен\w* по данн", r"исследовател", r"researcher", r"ресе?ре?ч"
        ),
    ),
    (
        "data_engineer",
        _any(
            r"data[\s-]*(?:engineer|инженер)",
            r"дата[\s-]*инженер",
            r"инженер\w* (?:по )?данн",
            r"\bdwh\b",
            r"\betl\b",
            r"\bкхд\b",
            r"хранилищ\w* данн",
            r"data[\s-]*architect",
            r"архитектор\w* данн",
            r"analytics[\s-]+engineer",
        ),
    ),
    ("data_analyst", _any(r"аналит", r"analy", r"\bbi\b", r"power[\s-]*bi")),
    ("data_engineer", _any(r"big[\s-]*data", r"больш\w* данн")),
]

_1C = r"(?<![a-zа-я0-9])1[сc](?![a-zа-я0-9])(?!-битрикс)"
_1C_TEXT = re.compile(
    rf"(?:платформ|язык|программирован|разработк|доработк|конфигураци|сред|программист)\w*\s+(?:(?:на|в|для)\s+)?"
    rf"(?:платформ\w*\s+|сред\w*\s+)?{_1C}|1[сc]\s*:\s*предприяти"
)
_BITRIX = r"битрикс|bitrix"
_GAMEDEV = _any(r"\bunity\b", r"\bunreal\b", r"gamedev", r"геймдев", r"\bигр")
_MOBILE = _any(
    r"\bios\b", r"android", r"андроид", r"flutter", r"mobile", r"мобильн", r"react[\s-]+native", r"\bswift\b"
)
_MOBILE_TEXT = _any(
    r"\bios\b",
    r"android",
    r"андроид",
    r"flutter",
    r"react[\s-]+native",
    r"\bswift(?:ui)?\b",
    r"мобильн\w* (?:приложени|разработк)",
)
_FULLSTACK = _any(r"full[\s-]*stack", r"фулл?[\s-]*стек")
_FRONTEND = _any(r"front", r"фронт", r"верстальщик", r"react", r"\bvue", r"angular", r"javascript", r"typescript")
_EMBEDDED = _any(
    r"embedded", r"встраиваем", r"встроенн\w* (?:по|программ)", r"микроконтроллер", r"\bплис\b", r"\bfpga\b", r"\bплк\b"
)
_EMBEDDED_TEXT = _any(_EMBEDDED.pattern, r"stm32", r"\bavr\b", r"freertos", r"\brtos\b", r"firmware")
_BACKEND_STACK = (
    r"\bpython\b",
    r"\bjava\b(?!script)",
    r"golang",
    r"\bgo\b",
    r"\bphp\b",
    r"c#",
    r"\.net\b",
    r"\bruby\b",
    r"\bscala\b",
    r"node\.?js",
    r"django",
    r"fastapi",
    r"spring",
    r"laravel",
    r"symfony",
    _BITRIX,
)
_BACKEND = _any(r"back", r"бэкенд", r"бекенд", r"серверн", *_BACKEND_STACK)
_BACKEND_TEXT = _any(r"back[\s-]?end", r"бэкенд", r"бекенд", *_BACKEND_STACK)
_SYSADMIN_TEXT = _any(
    r"администрировани\w* (?:сет|сервер|рабоч|пк|локальн|компьютер|пользовател)",
    r"обслуживани\w* (?:компьютер|оргтехник|пк|вычислительн)",
    r"оргтехник",
    r"локальн\w* (?:вычислительн\w* )?сет",
)

_QA_DEVOPS_SYS: list[tuple[str, re.Pattern[str]]] = [
    ("developer_1c", _any(r"администратор\w* 1[сc]\b")),
    (
        "security",
        _any(
            r"безопасн",
            r"кибербез",
            r"\bиб\b",
            r"защит\w* информаци",
            r"пентест",
            r"\bsoc\b",
            r"appsec",
            r"security",
            r"реверс",
        ),
    ),
    ("qa", _any(r"тестиров", r"автотест", r"\ba?qa\b", r"quality[\s-]+assurance", r"\bsdet\b")),
    (
        "devops",
        _any(r"devops", r"девопс", r"\bsre\b", r"site[\s-]+reliability", r"доступност", r"производительност"),
    ),
    (
        "support",
        _any(
            r"поддержк",
            r"helpdesk",
            r"help[\s-]+desk",
            r"service[\s-]*desk",
            r"сервис[\s-]*деск",
            r"support",
            r"(?:первой|второй|третьей) линии",
            r"консультант",
            r"внедрени",
        ),
    ),
]

_ANALYST_SYSTEM = _any(
    r"системн\w* аналитик",
    r"system[\s-]+analy",
    r"аналитик\w* 1[сc]\b",
    r"1[сc][\s-]*аналитик",
    r"аналитик\w* (?:ит|it|информационн\w* систем)\b",
)
_ANALYST_BUSINESS = _any(r"бизнес", r"business")
_DATA_TOOLS = _any(
    r"\bsql\b", r"\bpython\b", r"power[\s-]*bi", r"tableau", r"\bbi\b", r"superset", r"datalens", r"qlik"
)
_ANALYST_NON_IT = _any(
    r"финанс",
    r"кредит",
    r"экономи",
    r"риск",
    r"инвестиц",
    r"маркетинг",
    r"закуп",
    r"тендер",
    r"кадр",
    r"\bhr\b",
    r"юрид",
    r"правов",
    r"рынк",
    r"ценообраз",
    r"продаж",
    r"логист",
    r"склад",
    r"медицин",
    r"казначейств",
)

_DESIGNER = _any(r"дизайн", r"designer", r"\bui\b", r"\bux\b")
_MANAGER = _any(
    r"менеджер\w* (?:ит[\s-]*)?(?:проект|продукт)",
    r"руководител\w* (?:ит[\s-]*)?проект",
    r"project[\s-]+manager",
    r"product[\s-]+(?:manager|owner)",
    r"продакт",
    r"scrum",
    r"agile",
    r"технически\w* директор",
    r"\bcto\b",
    r"delivery[\s-]+manager",
    r"директор\w* по (?:ит|it|информ|цифров)",
)
_AI_ADJACENT_WORK = _any(r"разметчик", r"асессор", r"assessor", r"тренер", r"промпт", r"prompt", r"обучени\w* нейросет")


def _first(rules: Sequence[tuple[str, re.Pattern[str]]], text: str) -> str | None:
    return next((role for role, p in rules if p.search(text)), None)


def _dev_role(title: str, body: str) -> str:
    if re.search(_BITRIX, title):
        return "backend"
    if re.search(_1C, title):
        return "developer_1c"
    if _GAMEDEV.search(title):
        return "software_engineer"
    if _MOBILE.search(title):
        return "mobile"
    if _FULLSTACK.search(title):
        return "fullstack"
    if _EMBEDDED.search(title):
        return "embedded"
    if _FRONTEND.search(title):
        return "frontend"
    if _BACKEND.search(title):
        return "backend"
    # A generic developer title: the stack in the text decides, if it names one.
    if _1C_TEXT.search(body):
        return "developer_1c"
    if _EMBEDDED_TEXT.search(body):
        return "embedded"
    if _MOBILE_TEXT.search(body):
        return "mobile"
    if _BACKEND_TEXT.search(body):
        return "backend"
    if _FRONTEND.search(body):
        return "frontend"
    if _SYSADMIN_TEXT.search(body):
        return "sysadmin"
    return "software_engineer"


_ANALYSTS = ("system_analyst", "business_analyst", "data_analyst")


def _analyst_role(title: str, body: str, specializations: Sequence[str]) -> str:
    if _ANALYST_SYSTEM.search(title):
        return "system_analyst"
    if _ANALYST_BUSINESS.search(title):
        return "business_analyst"
    # A generic "Аналитик": the source's own specialization ("Системный аналитик") is the next title.
    for spec in specializations:
        role = title_role(spec, body)
        if role in _ANALYSTS:
            return role
    if _DATA_TOOLS.search(body):
        return "data_analyst"
    if _ANALYST_NON_IT.search(title):
        return "non_it"
    return "business_analyst"


def _other_it_role(title: str, body: str) -> str:
    if _AI_ADJACENT_WORK.search(title):
        return "other_it"
    if _DESIGNER.search(title):
        return "designer"
    if _MANAGER.search(title):
        return "manager"
    if _SYSADMIN_TEXT.search(body):
        return "sysadmin"
    return "other_it"


def title_role(title: str | None, body: str = "", specializations: Sequence[str] = ()) -> str:
    t, b = normalize(title), normalize(body)
    bucket = title_bucket(title)
    if bucket in (NON_IT_NOISE, DEV, OTHER_IT) and _DATA_MISSED.search(t):
        bucket = ML_DS_DATA
    if bucket == ML_DS_DATA:
        return _first(_ML_DS_DATA, t) or "ml_engineer"
    if bucket == DEV:
        return _dev_role(t, b)
    if bucket == QA_DEVOPS_SYS:
        return _first(_QA_DEVOPS_SYS, t) or "sysadmin"
    if bucket == ANALYST_OTHER:
        return _analyst_role(t, b, specializations)
    if bucket == OTHER_IT:
        return _other_it_role(t, b)
    return "non_it"


_VAGUE = ("other_it", "non_it")


def predict_role(title: str | None, body: str, specializations: Sequence[str] = ()) -> str | None:
    if not (title and title.strip()) and not body.strip():
        return None
    role = title_role(title, body, specializations)
    if role in _VAGUE:
        by_spec = [title_role(s, body) for s in specializations]
        specific = [r for r in by_spec if r not in _VAGUE]
        if specific:
            return specific[0]
    return role
