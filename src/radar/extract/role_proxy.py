"""Coarse role from the title, for stratifying the gold sample only.

The true role comes from labeling (5.3). This proxy decides which stratum a vacancy is drawn from,
so a wrong bucket costs sampling efficiency, not correctness: inclusion probabilities are stored
and the gold labels, not the proxy, are what every metric reads.

Rules run in order and the first match wins. Titles that name AI-adjacent work without building
models («редактор текстов для обучения нейросетей», «специалист по работе с искусственным
интеллектом в управлении персоналом») are caught before the ML rule: they are other_it in the
annotation guide. Habr also states its own specializations; there `ml_ds_data` from either the
title or a specialization wins (oversampling it is the point of the proxy), and a specialization
replaces a title that says nothing specific (other_it or non_it_noise).
"""

import re
from collections.abc import Iterable

ML_DS_DATA = "ml_ds_data"
DEV = "dev"
ANALYST_OTHER = "analyst_other"
QA_DEVOPS_SYS = "qa_devops_sys"
OTHER_IT = "other_it"
NON_IT_NOISE = "non_it_noise"

BUCKETS = (ML_DS_DATA, DEV, ANALYST_OTHER, QA_DEVOPS_SYS, OTHER_IT, NON_IT_NOISE)

# Schema roles each bucket stands for; non_it_noise is non_it.
BUCKET_ROLES = {
    ML_DS_DATA: ("ml_engineer", "data_scientist", "mlops", "data_engineer", "data_analyst"),
    DEV: ("backend", "frontend", "fullstack", "mobile", "software_engineer", "developer_1c", "embedded"),
    ANALYST_OTHER: ("business_analyst", "system_analyst"),
    QA_DEVOPS_SYS: ("devops", "qa", "sysadmin", "support", "security"),
    OTHER_IT: ("designer", "manager", "other_it"),
    NON_IT_NOISE: ("non_it",),
}


def _any(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


# Python's \b is Unicode-aware, so it bounds Cyrillic words as well as Latin ones.
_AI_ADJACENT = _any(
    r"обучени\w* нейросет",
    r"работ\w* с (?:систем\w* )?(?:искусственн|нейросет|нейронн|ии\b|ai\b)",
    r"(?:ai|ии)[\s-]*тренер",
    r"разметчик",
    r"асессор",
    r"assessor",
    r"промпт",
    r"prompt",
    r"ux[\s-]*(?:research|ресерч|ресеч|исследова)",
)

_AI = r"\bai\b|\bии\b|искусственн\w* интеллект|нейросет|нейронн\w* сет"

_ML_DS_DATA = _any(
    r"data[\s-]*scien",
    r"дата[\s-]*са[йе]ен",
    r"machine[\s-]+learning",
    r"машинн\w* обучени",
    r"\bml\b",
    r"mlops",
    r"\bllm\b",
    r"\bnlp\b",
    r"\bcv\b",
    r"computer[\s-]+vision",
    r"компьютерн\w* зрени",
    r"deep[\s-]+learning",
    r"глубок\w* обучени",
    r"\bmle\b",
    # "AI" alone often names the product, not the work ("Product owner (AI)", "DevOps (AI-агенты)"):
    # it counts only next to an engineering noun.
    rf"(?:{_AI})[\s-]*(?:инженер|engineer|разработчик|developer|researcher|исследовател)",
    rf"(?:инженер|engineer|разработчик|developer|исследовател|researcher|специалист)\w*"
    rf"[\s-]+(?:по |в области |моделей )?(?:{_AI})",
    r"data[\s-]*engineer",
    r"дата[\s-]*инженер",
    r"инженер\w* (?:по )?данн",
    r"data[\s-]*architect",
    r"архитектор\w* данн",
    r"\bdwh\b",
    r"\betl\b",
    r"big[\s-]*data",
    r"больш\w* данн",
    r"аналитик\w* (?:по )?данн",
    r"анализ\w* данн",
    r"data[\s-]*analy",
    r"analytics[\s-]+engineer",
    r"\bbi\b",
    r"power[\s-]*bi",
    r"продуктов\w* аналитик",
    r"product[\s-]+analy",
    r"веб[\s-]*аналитик",
    r"web[\s-]*analy",
    r"учен\w* по данн",
    r"исследовател\w* данн",
    r"ресечер",
    r"ресерчер",
    r"researcher",
    r"рекомендательн",
    r"recsys",
)

_QA_DEVOPS_SYS = _any(
    r"тестировщ",
    r"тестировани",
    r"автотест",
    r"\ba?qa\b",
    r"quality[\s-]+assurance",
    r"\bsdet\b",
    r"devops",
    r"девопс",
    r"\bsre\b",
    r"site[\s-]+reliability",
    r"доступност\w* сервис",
    r"производительност\w* (?:систем|прилож|сервис)|инженер по производительности",
    r"системн\w* администратор",
    r"сисадмин",
    r"администратор\w* (?:сет|сервер|баз\w* данн|вычислительн|информационн|систем|linux|windows|ит\b|it\b|1с)",
    r"\bdba\b",
    r"сетев\w* (?:инженер|администратор)",
    r"network[\s-]+engineer",
    r"системн\w* инженер",
    r"system[\s-]+(?:engineer|administrator)",
    r"инженер\w* цод",
    r"поддержк",
    r"helpdesk",
    r"help[\s-]+desk",
    r"service[\s-]*desk",
    r"сервис[\s-]*деск",
    r"support",
    r"(?:первой|второй|третьей) линии",
    r"информационн\w* безопасн",
    r"безопасност\w* (?:информационн|ит\b|it\b)",
    r"кибербезопасн",
    r"\bиб\b",
    r"защит\w* информаци",
    r"пентест",
    r"\bsoc\b",
    r"appsec",
    r"devsecops",
    r"security",
    r"реверс",
    r"консультант\w* 1с",
    r"1с[\s-]*консультант",
    r"внедрени\w* 1с",
)

_ANALYST_OTHER = _any(r"аналитик", r"analyst")

# CNC "программист" and "наладчик" are machine tool operators, not software developers.
_NOT_DEV = _any(r"чпу", r"станк", r"\bcnc\b")

_DEV = _any(
    r"программист",
    r"разработчик",
    r"developer",
    r"software[\s-]+engineer",
    r"programmer",
    r"\bpython\b",
    r"\bjava\b",
    r"javascript",
    r"typescript",
    r"golang",
    r"\bgo\b",
    r"\bphp\b",
    r"c\+\+",
    r"c#",
    r"\.net\b",
    r"kotlin",
    r"\bswift\b",
    r"\bios\b",
    r"android",
    r"flutter",
    r"react",
    r"frontend",
    r"front[\s-]+end",
    r"фронтенд",
    r"backend",
    r"back[\s-]+end",
    r"бэкенд",
    r"бекенд",
    r"full[\s-]*stack",
    r"фулл?стек",
    r"верстальщик",
    r"веб[\s-]*программ",
    r"битрикс",
    r"bitrix",
    r"gamedev",
    r"\bunity\b",
    r"\bunreal\b",
    r"embedded",
    r"встраиваем",
    r"микроконтроллер",
    r"\bплис\b",
    r"\bfpga\b",
)

_SALES = _any(r"продаж", r"\bsales\b", r"по работе с клиентами", r"клиентск\w* менеджер")

_OTHER_IT = _any(
    r"(?:^|[^а-я])ит(?:[^а-я]|$)",
    r"\bit\b",
    r"информационн\w* (?:технолог|систем)",
    r"информатизаци",
    r"цифров\w* (?:трансформац|развити)",
    r"компьютерн",
    r"электронно[\s-]*вычислительн",
    r"\bэвм\b",
    r"дизайнер",
    r"designer",
    r"дизайн\w* .*интерфейс",
    r"\bui\b",
    r"\bux\b",
    r"менеджер\w* (?:ит[\s-]*)?(?:проект|продукт)",
    r"руководител\w* (?:ит[\s-]*)?проект",
    r"project[\s-]+manager",
    r"product[\s-]+(?:manager|owner)",
    r"продакт",
    r"scrum",
    r"agile",
    r"технически\w* писател",
    r"technical[\s-]+writer",
    r"технически\w* директор",
    r"\bcto\b",
    r"архитектор",
    r"architect",
    r"пресейл",
    r"presale",
    r"деврел",
    r"devrel",
)


def _normalize(title: str | None) -> str:
    return (title or "").lower().replace("ё", "е")


def title_bucket(title: str | None) -> str:
    t = _normalize(title)
    if not t.strip():
        return NON_IT_NOISE
    if _AI_ADJACENT.search(t):
        return OTHER_IT
    if _ML_DS_DATA.search(t):
        return ML_DS_DATA
    if _QA_DEVOPS_SYS.search(t):
        return QA_DEVOPS_SYS
    if _ANALYST_OTHER.search(t):
        return ANALYST_OTHER
    if _DEV.search(t) and not _NOT_DEV.search(t):
        return DEV
    if _SALES.search(t):
        return NON_IT_NOISE
    if _OTHER_IT.search(t):
        return OTHER_IT
    return NON_IT_NOISE


_VAGUE = (OTHER_IT, NON_IT_NOISE)


def role_proxy(title: str | None, specializations: Iterable[str] = ()) -> str:
    """Bucket from the title, refined by the source's own specializations where it has them."""
    by_title = title_bucket(title)
    by_spec = [title_bucket(s) for s in specializations]
    if by_title == ML_DS_DATA or ML_DS_DATA in by_spec:
        return ML_DS_DATA
    if by_title in _VAGUE:
        specific = [b for b in by_spec if b not in _VAGUE]
        if specific:
            return specific[0]
        if OTHER_IT in by_spec:
            return OTHER_IT
    return by_title
