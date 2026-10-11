"""What a suspicious vacancy is, and the labeling functions that vote on it.

Unit: a vacancy card, `(family, source_id)`. The trudvsem API and the daily snapshots share ids, so
their rows of one id are one card, with the fields of its latest row (the API first on the same
day). The key does not depend on the dedup, so the frozen sample survives a rerun of it.

A card is suspicious when it is not an honest offer of the job its title names, on the terms it
states. Four kinds, each judged from the card alone:

- course: the card sells training. The candidate pays for a course, an internship, materials or a
  test, or the card advertises an educational programme rather than a job. A paid internship of an
  employer with training inside it is a job.
- bait: the "dream job". Income is promised without a job behind it: high or unlimited pay with no
  experience and no concrete duties, passive income, network marketing, an investment from the
  candidate; or an IT title over duties that are cold calling or sales.
- grey: grey employment. Steady duties without a contract, pay in an envelope or in cash only,
  a request to open a bank card for the employer or to pass on documents, or an offer to invent
  work experience for the CV.
- formal: a placeholder rather than an opening. The text is a job-classifier template with
  nothing about this employer's work and the pay does not fit the duties (a kernel developer at a
  fifth of the market), or the card itself says it is posted for a formality.

A repost is not a kind of suspicious: a portal card expires after about 30 days (the closure spike
of 7.3) and employers post it again. Reposts come from the history (`staging.hist_reposts`: the same
employer, place, title and text under a new id after the old one closed) and are reported per card.
A long repost chain is a labeling function that is measured but left out of the vote: the cards it
fires on are mostly the mass hiring of large employers. So is a salary far above its group: it
fires on senior roles in the regions far more often than on promises.

Labeling functions return SUSPICIOUS, OK or ABSTAIN per card. They see source fields, the text and
the history; no model and no GPU. Their patterns were written by reading matches in the whole card
population before the labeling sample was drawn, never in the sample.
"""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ABSTAIN = -1
OK = 0
SUSPICIOUS = 1

Votes = NDArray[np.int8]

CATEGORIES = ("course", "bait", "grey", "formal")

# Texts shorter than this (letters and digits only) are not grouped: "Обязанности: по собеседованию"
# would join unrelated employers.
MIN_GROUP_TEXT = 200
# Salary z: log monthly salary against the median of its group, scaled by IQR / 1.349. Groups are
# role bucket x region group (Moscow, Petersburg, other, remote), then role bucket, then all; a group
# needs MIN_SALARY_GROUP salaries. The scale has a floor: a group where one employer posts hundreds
# of identical salaries (Sber) would otherwise make every other salary an outlier.
MIN_SALARY_GROUP = 30
MIN_SALARY_SCALE = 0.2
SALARY_OUTLIER_Z = 3.0
# One text from this many employers, this share of its cards in one region: mass posting. Templates
# of the portal's job classifier spread over many regions and stay out.
MASS_EMPLOYERS = 5
MASS_REGION_SHARE = 0.8
# A chain of this many ids of one vacancy within the history window.
REPOST_CHAIN = 4
CONCRETE_SKILLS = 3
# An employer with this many cards in the window (by INN or Habr company id) hires at scale.
LARGE_EMPLOYER_CARDS = 50
# Technical roles of the marts buckets (macros/marts.sql): no support desks, no security sales.
TECH_ROLES = frozenset(
    {
        "ml_ds",
        "data_engineering",
        "data_analytics",
        "software_engineer",
        "backend",
        "frontend",
        "fullstack",
        "mobile",
        "embedded",
        "developer_1c",
        "business_system_analysis",
        "qa",
        "devops",
    }
)
# Words that turn a grey-scheme phrase into its denial: "никаких зарплат в конверте".
NEGATION = re.compile(r"\b(?:никаких|никакой|нет|не|без|исключ\w*)\b[^.;!?]{0,25}$")


def _rx(*parts: str) -> str:
    return "|".join(f"(?:{p})" for p in parts)


# Patterns run on the normalized body (lowercase, ё as е, spaces collapsed).
PATTERNS: dict[str, str] = {
    # The candidate pays; "оплата обучения за счёт компании" and "бесплатное обучение" are benefits.
    "course_pay": _rx(
        r"\bстоимост\w* (?:обучени|курс|стажировк|программ)",
        r"\bплатн\w* (?:обучени|курс|стажировк)",
        r"\b(?:обучени|курс|стажировк)\w* (?:платн|за свой сч)",
        r"\b(?:оплатить|оплачиваете|внести|вносите) (?:\w+ )?(?:обучени|курс|взнос|предоплат|залог|депозит)",
        r"\bпредоплат\w* за (?:обучени|курс|материал)",
        r"\bв рассрочку",
        r"\bвернем (?:\w+ )?(?:деньги|стоимость) за (?:обучени|курс)",
    ),
    "training_then_job": _rx(
        r"\b(?:обучени|курс)\w* с (?:последующим|дальнейшим|гарантированным) трудоустройств",
        r"\bтрудоустройств\w* после (?:прохождения )?(?:обучени|курс)",
        r"\bгарантир\w* трудоустройств",
    ),
    "income_bait": _rx(
        r"\bпассивн\w* доход",
        r"\bсетев\w* маркетинг",
        r"\bmlm\b",
        r"\bмлм\b",
        r"\bарбитраж\w* трафик",
        r"\bфинансов\w* свобод",
        r"\bбез вложений",
        r"\bдоход от \d[\d ]* (?:в день|в неделю)",
    ),
    "unlimited_income": _rx(
        r"\bдоход\w* не ограничен",
        r"\bнеограниченн\w* (?:доход|заработ)",
        r"\b(?:доход|заработ)\w* без (?:потолка|ограничений)",
    ),
    "grey": _rx(
        r"\bбез (?:официального |трудового )?(?:оформлени|трудоустройств|договор)",
        r"\bв конверте",
        r"\bнеофициальн\w* (?:трудоустройств|оформлени|заработн|зарплат|оплат|работ)",
        r"\bсер\w* (?:зарплат|заработн|схем)",
        r"\bоплата (?:только )?наличными",
        r"\bналичными (?:на руки|ежедневно|в конце)",
        r"\b(?:оформить|открыть|передать) (?:на себя |свою |свои )?(?:банковск\w* )?карт",
        r"\b(?:накрут|нарис|легенд)\w* (?:\w+ )?(?:опыт|резюме)",
    ),
    "sales_duties": _rx(
        r"\bхолодн\w* звонк",
        r"\b(?:активн|прям)\w* продаж",
        r"\bплан\w* продаж",
    ),
    "formal_note": _rx(
        r"\b(?:для|под) квот\w* (?:на )?(?:иностран|мигрант|разрешени)",
        r"\bсобеседовани\w* не (?:проводится|требуется)",
        r"\bвакансия (?:размещена )?(?:для|в целях) (?:отчетност|формальн)",
    ),
    "no_experience": _rx(
        r"\bбез опыта",
        r"\bопыт (?:работы )?не (?:нужен|требуется|обязателен|важен)",
    ),
    "remote": _rx(r"\bудаленн", r"\bиз дома\b"),
    # Low pay that the card explains: a trainee or a part-time rate.
    "low_pay_explained": _rx(
        r"\b(?:стажер|стажир|ассистент|помощник|младш|начинающ|junior|intern|trainee)",
        r"\b0[,.]\d+ ставк",
        r"\bнеполн\w* (?:рабоч|занятост|ставк)",
        r"\bподработк",
    ),
    "public_sector": _rx(
        r"\b(?:государственн|муниципальн|федеральн|областн|краев|бюджетн|казенн|автономн)\w* "
        r"(?:\w+ ){0,4}(?:учреждени|предприяти)",
        r"^(?:гбу|гку|гау|мбу|мку|мау|мбоу|мкоу|маоу|мбдоу|мадоу|фгбу|фгуп|гуп|муп|гбуз|гауз|гбоу|гбпоу|"
        r"гапоу|фку|фгбоу|фгаоу|огбу|огку|кгбу|кгку)\b",
    ),
}

_COMPILED = {name: re.compile(p) for name, p in PATTERNS.items()}
# Patterns whose phrases are often denied in the same sentence.
_NEGATABLE = frozenset({"grey"})


def normalize(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def text_group(text: str | None) -> str | None:
    """Hash of the text up to case, punctuation and digits; None for short texts."""
    letters = re.sub(r"[^a-zа-я0]+", " ", re.sub(r"\d", "0", normalize(text))).strip()
    if len(letters) < MIN_GROUP_TEXT:
        return None
    return hashlib.sha1(letters.encode("utf-8")).hexdigest()[:16]


def _found(rx: re.Pattern[str], body: str, negatable: bool) -> bool:
    return any(
        not negatable or not NEGATION.search(body[max(0, m.start() - 40) : m.start()]) for m in rx.finditer(body)
    )


def matches(body: pd.Series, name: str) -> NDArray[np.bool_]:
    rx = _COMPILED[name]
    negatable = name in _NEGATABLE
    return np.fromiter((_found(rx, b, negatable) for b in body), dtype=bool, count=len(body))


def salary_z(log_salary: pd.Series, groups: list[pd.Series]) -> pd.Series:
    """Robust z of each log salary within the finest of `groups` with MIN_SALARY_GROUP values."""
    z = pd.Series(np.nan, index=log_salary.index)
    present = log_salary.notna()
    for keys in [*groups, pd.Series("all", index=log_salary.index)]:
        frame = pd.DataFrame({"v": log_salary[present], "k": keys[present]})
        by = frame.groupby("k")["v"]
        stats = pd.DataFrame({"median": by.median(), "count": by.size(), "iqr": by.quantile(0.75) - by.quantile(0.25)})
        usable = stats[stats["count"] >= MIN_SALARY_GROUP]
        todo = present & z.isna() & keys.isin(usable.index)
        if not todo.any():
            continue
        scale = np.maximum(keys[todo].map(usable["iqr"]) / 1.349, MIN_SALARY_SCALE)
        z[todo] = (log_salary[todo] - keys[todo].map(usable["median"])) / scale
    return z


def repost_chains(cards: pd.DataFrame) -> pd.Series:
    """Ids in the repost chain of each card (1 when it was never reposted)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for new, old in zip(cards["source_id"], cards["repost_of"], strict=True):
        if isinstance(old, str) and old:
            ra, rb = find(new), find(old)
            if ra != rb:
                parent[ra] = rb
    roots = pd.Series([find(s) for s in cards["source_id"]], index=cards.index)
    return roots.map(roots.value_counts()).astype(int)


def add_context(cards: pd.DataFrame) -> pd.DataFrame:
    """Columns the votes need beyond the card itself: body, salary z, text spread, repost chain.

    Expects one row per card with the fields of `radar.models.suspicious.CARD_COLUMNS`.
    """
    out = cards.copy()
    out["body"] = [normalize(f"{t or ''}\n{x or ''}") for t, x in zip(out["title"], out["text"], strict=True)]
    rub = out["salary_currency_code"].fillna("RUB").eq("RUB")
    salary = out["salary_from"].where(out["salary_from"].notna(), out["salary_to"])
    out["salary_month"] = salary.where(rub & (salary > 0))
    out["salary_z"] = salary_z(
        np.log(out["salary_month"]), [out["role_bucket"] + "|" + out["region_group"], out["role_bucket"]]
    )
    out["text_group"] = [text_group(t) for t in out["text"]]
    place = out["region"].fillna(out["region_group"])
    employer = out["employer_key"].where(out["employer_key"].notna() & ~out["is_sole_proprietor"].astype(bool))
    grouped = pd.DataFrame({"g": out["text_group"], "e": employer, "r": place}).dropna(subset=["g"])
    spread = grouped.groupby("g").agg(
        group_employers=("e", "nunique"),
        group_region_share=("r", lambda r: float(r.value_counts(normalize=True).iloc[0]) if r.notna().any() else 0.0),
    )
    out = out.join(spread, on="text_group")
    out["employer_cards"] = employer.map(employer.value_counts()).fillna(0).astype(int)
    out["group_employers"] = out["group_employers"].fillna(0).astype(int)
    out["group_region_share"] = out["group_region_share"].astype(float).fillna(0.0)
    out["repost_chain"] = repost_chains(out)
    return out


@dataclass(frozen=True)
class LabelingFunction:
    name: str
    vote: int
    category: str | None
    rule: str
    apply: Callable[[pd.DataFrame], NDArray[np.bool_]]
    # False: measured and scored on the sample, left out of the weak label.
    combine: bool = True


def _keyword(name: str) -> Callable[[pd.DataFrame], NDArray[np.bool_]]:
    return lambda c: matches(c["body"], name)


def _sole_remote_no_experience(c: pd.DataFrame) -> NDArray[np.bool_]:
    remote = c["work_format"].eq("remote").to_numpy() | matches(c["body"], "remote")
    return c["is_sole_proprietor"].astype(bool).to_numpy() & remote & matches(c["body"], "no_experience")


def _salary_high(c: pd.DataFrame) -> NDArray[np.bool_]:
    return (c["salary_z"] >= SALARY_OUTLIER_Z).to_numpy()


def _salary_low(c: pd.DataFrame) -> NDArray[np.bool_]:
    low = ((c["salary_z"] <= -SALARY_OUTLIER_Z) & c["role_bucket"].isin(TECH_ROLES)).to_numpy()
    return low & ~matches(c["body"], "low_pay_explained")


def _no_experience_high_pay(c: pd.DataFrame) -> NDArray[np.bool_]:
    return matches(c["body"], "no_experience") & (c["salary_z"] >= SALARY_OUTLIER_Z - 1).to_numpy()


def _mass_text(c: pd.DataFrame) -> NDArray[np.bool_]:
    return ((c["group_employers"] >= MASS_EMPLOYERS) & (c["group_region_share"] >= MASS_REGION_SHARE)).to_numpy()


def _repost_chain(c: pd.DataFrame) -> NDArray[np.bool_]:
    return (c["repost_chain"] >= REPOST_CHAIN).to_numpy()


def _title_sales(c: pd.DataFrame) -> NDArray[np.bool_]:
    return c["role_bucket"].isin(TECH_ROLES).to_numpy() & matches(c["body"], "sales_duties")


def _public_sector(c: pd.DataFrame) -> NDArray[np.bool_]:
    return matches(c["employer_name"].map(normalize), "public_sector")


def _large_employer(c: pd.DataFrame) -> NDArray[np.bool_]:
    return (c["employer_cards"] >= LARGE_EMPLOYER_CARDS).to_numpy()


def _concrete(c: pd.DataFrame) -> NDArray[np.bool_]:
    experienced = c["experience_min_years"].fillna(0).ge(1) | (
        c["grade_is_stated"].fillna(0).astype(bool) & c["grade"].isin(["middle", "senior", "lead"])
    )
    return (experienced & c["skills"].map(len).ge(CONCRETE_SKILLS)).to_numpy()


LFS: tuple[LabelingFunction, ...] = (
    LabelingFunction("course_pay", SUSPICIOUS, "course", "the candidate pays for training", _keyword("course_pay")),
    LabelingFunction(
        "training_then_job",
        SUSPICIOUS,
        "course",
        "training with employment promised after it",
        _keyword("training_then_job"),
    ),
    LabelingFunction(
        "income_bait",
        SUSPICIOUS,
        "bait",
        "passive income, MLM, traffic arbitrage, no investment",
        _keyword("income_bait"),
    ),
    LabelingFunction("unlimited_income", SUSPICIOUS, "bait", "income without a ceiling", _keyword("unlimited_income")),
    LabelingFunction(
        "no_experience_high_pay",
        SUSPICIOUS,
        "bait",
        f"the text says no experience is needed, salary z >= {SALARY_OUTLIER_Z - 1}",
        _no_experience_high_pay,
    ),
    LabelingFunction(
        "sole_remote_no_experience",
        SUSPICIOUS,
        "bait",
        "sole proprietor, remote, the text says no experience is needed",
        _sole_remote_no_experience,
    ),
    LabelingFunction("salary_high", SUSPICIOUS, "bait", f"salary z >= {SALARY_OUTLIER_Z}", _salary_high, combine=False),
    LabelingFunction(
        "title_sales", SUSPICIOUS, "bait", "technical IT role by title, cold calls or a sales plan", _title_sales
    ),
    LabelingFunction(
        "grey", SUSPICIOUS, "grey", "no contract, envelope, cash, bank card, invented experience", _keyword("grey")
    ),
    LabelingFunction(
        "salary_low",
        SUSPICIOUS,
        "formal",
        f"technical IT role, salary z <= -{SALARY_OUTLIER_Z}, not a trainee or part-time card",
        _salary_low,
    ),
    LabelingFunction(
        "mass_text",
        SUSPICIOUS,
        "formal",
        f"same text from >= {MASS_EMPLOYERS} employers, >= {MASS_REGION_SHARE} of its cards in one region",
        _mass_text,
    ),
    LabelingFunction(
        "formal_note", SUSPICIOUS, "formal", "the card says it is posted for a formality", _keyword("formal_note")
    ),
    LabelingFunction(
        "repost_chain",
        SUSPICIOUS,
        None,
        f">= {REPOST_CHAIN} ids of one vacancy in the history",
        _repost_chain,
        combine=False,
    ),
    LabelingFunction("public_sector", OK, None, "state or municipal institution", _public_sector),
    LabelingFunction(
        "large_employer", OK, None, f">= {LARGE_EMPLOYER_CARDS} cards of the employer in the window", _large_employer
    ),
    LabelingFunction(
        "concrete_requirements",
        OK,
        None,
        f"a year of experience or more (or a stated middle+ grade) and >= {CONCRETE_SKILLS} dictionary skills",
        _concrete,
    ),
)


def apply_lfs(cards: pd.DataFrame, lfs: tuple[LabelingFunction, ...] = LFS) -> Votes:
    """Vote matrix, cards x labeling functions, of SUSPICIOUS, OK and ABSTAIN. `cards` has context."""
    votes = np.full((len(cards), len(lfs)), ABSTAIN, dtype=np.int8)
    for j, lf in enumerate(lfs):
        votes[lf.apply(cards), j] = lf.vote
    return votes


def combined(lfs: tuple[LabelingFunction, ...] = LFS) -> NDArray[np.bool_]:
    """Mask of the functions that enter the weak label."""
    return np.array([lf.combine for lf in lfs])
