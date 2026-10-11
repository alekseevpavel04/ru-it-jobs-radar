"""Regex baseline for the extraction schema: the floor every teacher and student is
compared with on dev.

Reads the same input as every other system (`radar.extract.inputs.input_for_record`: a header of the
source's structured fields, then the free text) and follows the annotation guide: an explicit
statement or null, never a guess. Field by field:

- role: title rules (`role_rules`), Habr specializations for a vague title;
- grade: grade words in the title, Habr's qualification, trudvsem's "Стажировка";
- skills: the skill dictionary v0 (`skill_dictionary`);
- salary: the header amount (`salary.parse_salary`), gross or net and period from a salary phrase or
  a money line of the text that repeats a header amount; without a header, the first salary phrase
  of the text;
- work_format: phrases of the text, the most remote one offered; the header only when the text says
  nothing;
- cities: settlements of the header address or Habr's location list;
- experience_years: experience phrases of the text, the largest minimum; the header otherwise;
- employment_type: "Стажировка" in the title or header is internship, then a labor contract wherever
  named, then ГПХ or self-employment, whichever comes first;
- lang: share of Latin letters in the text.

    python -m radar.extract.baseline --split dev           # predictions, scored if gold exists
    python -m radar.extract.baseline --train-sample 300    # validity and fill rates on train_pool

The test split is not served here: its predictions are made once, in the final test.
"""

import argparse
import json
import logging
import random
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from radar.config import data_dir
from radar.eval.cli import load_config
from radar.eval.cli import main as evaluate
from radar.extract.role_rules import predict_role
from radar.extract.salary import parse_salary
from radar.extract.schema import (
    GRADES,
    EmploymentType,
    Experience,
    Grade,
    Lang,
    Salary,
    VacancyFields,
    WorkFormat,
    dump,
    parse,
)
from radar.extract.skill_dictionary import find_skills

log = logging.getLogger("radar.extract.baseline")

SYSTEM = "regex"

# Labels the input header uses (radar.extract.inputs).
HEADER_LABELS = frozenset(
    {
        "Должность",
        "Зарплата",
        "Адрес",
        "Требуемый опыт",
        "Тип занятости",
        "График",
        "Специализация",
        "Квалификация",
        "Навыки",
        "Местоположение",
        "Формат работы",
    }
)

_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−"} | {c: " " for c in "   "})


def normalize(text: str) -> str:
    """Lowercase, ё as е, one kind of dash and space; same length as the input, so spans line up."""
    lowered = "".join(c.lower() if len(c.lower()) == 1 else c for c in text)
    return lowered.replace("ё", "е").translate(_DASHES)


@dataclass(frozen=True)
class Input:
    header: dict[str, str]
    body: str

    @property
    def title(self) -> str | None:
        return self.header.get("Должность")


def split_input(text: str) -> Input:
    """Header and free text. A first block that is not all `Label: value` lines is free text."""
    first, _, rest = text.strip().partition("\n\n")
    header: dict[str, str] = {}
    for line in first.split("\n"):
        label, sep, value = line.partition(": ")
        if not sep or label not in HEADER_LABELS:
            return Input({}, text.strip())
        header[label] = value.strip()
    return Input(header, rest.strip())


# --- grade --------------------------------------------------------------------------------------

_GRADE_WORDS: list[tuple[Grade, re.Pattern[str]]] = [
    ("intern", re.compile(r"стаж[её]р|стажировк|\bintern\b|internship|\btrainee\b|для студентов")),
    ("junior", re.compile(r"junior|\bджун|младш|начинающ|без опыта")),
    ("middle", re.compile(r"middle|\bмидл")),
    (
        "senior",
        re.compile(
            r"senior|сеньор|синьор|старш|ведущ"
            r"|главн\w* (?:специалист|инженер|программист|аналитик|эксперт|разработчик|архитектор|администратор)"
        ),
    ),
    (
        "lead",
        re.compile(
            r"\blead\b|team[\s-]*lead|teamlead|тимлид|техлид|тех[\s-]+лид|техническ\w* лидер|head of"
            r"|руководител\w* (?:\w+ )?(?:групп|отдел|направлени|подразделени|департамент|управлени|служб|сектор"
            r"|команд|разработ)"
            r"|начальник\w* (?:\w+ )?(?:отдел|сектор|управлени|служб|групп|лаборатори)"
            r"|заведующ\w* (?:отдел|сектор|лаборатори)"
        ),
    ),
]
# Habr's qualification: "Старший (Senior)", "Ведущий (Lead)".
_QUALIFICATION: dict[str, Grade] = {f"({g})": g for g in ("intern", "junior", "middle", "senior", "lead")}


def predict_grade(inp: Input, title_norm: str) -> Grade | None:
    """Grade words in the title; two grades give the lower one, as the guide asks."""
    found = [g for g, p in _GRADE_WORDS if p.search(title_norm)]
    if found:
        return min(found, key=GRADES.index)
    qualification = normalize(inp.header.get("Квалификация", ""))
    for word, grade in _QUALIFICATION.items():
        if word in qualification:
            return grade
    if normalize(inp.header.get("Тип занятости", "")) == "стажировка":
        return "intern"
    return None


# --- salary -------------------------------------------------------------------------------------

_SALARY_WORD = re.compile(
    r"зарплат|заработн\w* плат|\bз/?п\b|оклад|доход|вознаграждени|оплат\w* труда|\bsalary|compensation"
)
_SALARY_WINDOW = 80
# "руб." ends an amount, not a salary phrase, so only a line break or a semicolon ends one.
_PHRASE_END = re.compile(r"\n|;")
_SENTENCE_END = re.compile(r"\n|;|\.\s")
_MIN_TEXT_AMOUNT = 1000


_CURRENCY_MARK = re.compile(r"₽|\$|€|руб|\bр\.|\bт\.\s?р")
_MAX_LINE = 160


def salary_phrases(body: str) -> list[str]:
    """From each salary word to the end of its line, at most _SALARY_WINDOW characters."""
    out = []
    for m in _SALARY_WORD.finditer(body):
        window = body[m.start() : m.start() + _SALARY_WINDOW]
        end = _PHRASE_END.search(window)
        out.append(window[: end.start()] if end else window)
    return out


def money_lines(body: str) -> list[str]:
    """Short lines with a currency: "120 000 ₽ за месяц на руки" repeats the header amount without
    a salary word."""
    return [line for line in body.split("\n") if len(line) <= _MAX_LINE and _CURRENCY_MARK.search(line)]


def _parsed(phrases: Iterable[str]) -> list[Salary]:
    return [s for s in (parse_salary(p) for p in phrases) if s is not None]


def predict_salary(inp: Input, body_norm: str) -> Salary | None:
    head = parse_salary(inp.header.get("Зарплата"))
    if head is None:
        stated = _parsed(salary_phrases(body_norm))
        return next((s for s in stated if max(s.from_ or 0, s.to or 0) >= _MIN_TEXT_AMOUNT), None)
    # The text adds gross or net and the period only where it repeats an amount of the header; another
    # amount is another payment (a bonus, the income after probation, an ad for the trade).
    bounds = {head.from_, head.to} - {None}
    candidates = _parsed([*salary_phrases(body_norm), *money_lines(body_norm)])
    same = [s for s in candidates if bounds & ({s.from_, s.to} - {None})]
    gross_net = head.gross_net or next((s.gross_net for s in same if s.gross_net), None)
    period = head.period or next((s.period for s in same if s.period), None)
    return head.model_copy(update={"gross_net": gross_net, "period": period})


# --- work format --------------------------------------------------------------------------------

_NO_REMOTE = re.compile(
    r"(?:нет|без) удаленн\w*(?: работы)?"
    r"|удаленн\w* (?:работ\w* )?не (?:предусмотрен|рассматрива|возмож)\w*"
)
_HYBRID = re.compile(
    r"гибрид\w*|\bhybrid\b|частично удаленн\w*"
    r"|(?:удаленн\w*|удаленк\w*)(?: работ\w*| формат\w*(?: работы)?)?(?: возмож\w*| допуска\w*)?"
    r" после (?:окончания |прохождения )?(?:испытательн\w*|ис\b)"
    r"|(?:\d|один|два|три|несколько|пару)\s+дн\w* (?:в неделю )?(?:из дома|удаленно)"
)
_REMOTE = re.compile(
    r"удаленн\w* (?:работ|формат|режим|занятост|основ)|удаленк"
    r"|(?:работ\w*|можно|возможно|полностью|формат\w*(?: работы)?:?|-)\s+удаленно|удаленно\s*/|\bremote\b"
    r"|из любой точки|из любого города|дистанционн\w* (?:работ|формат|режим|занятост)|work from home"
)
_OFFICE = re.compile(
    r"(?:работ\w*|формат\w*(?: работы)?:?) (?:в|из) офис|офисн\w* (?:формат|режим)|full[\s-]*office"
    r"|\bon[\s-]?site\b|\bin[\s-]office\b|\bочн\w* (?:формат|работ)|работать очно"
)


def _mask(pattern: re.Pattern[str], text: str) -> tuple[bool, str]:
    found = bool(pattern.search(text))
    return found, pattern.sub(" ", text)


def predict_work_format(inp: Input, body_norm: str) -> WorkFormat | None:
    """Of the formats the text offers, the most remote; "удаленно после испытательного срока" is hybrid
    and "нет удаленной работы" is office, so both are cut out before looking for remote work."""
    no_remote, text = _mask(_NO_REMOTE, body_norm)
    hybrid, text = _mask(_HYBRID, text)
    if _REMOTE.search(text):
        return "remote"
    if hybrid:
        return "hybrid"
    if no_remote or _OFFICE.search(text):
        return "office"
    head = normalize(inp.header.get("Тип занятости", "") + " " + inp.header.get("Формат работы", ""))
    if "удален" in head:
        return "remote"
    return None


# --- cities -------------------------------------------------------------------------------------

_FEDERAL = {"москва": "Москва", "санкт-петербург": "Санкт-Петербург", "севастополь": "Севастополь"}
_ALIASES = _FEDERAL | {
    "moscow": "Москва",
    "спб": "Санкт-Петербург",
    "питер": "Санкт-Петербург",
    "saint petersburg": "Санкт-Петербург",
    "st. petersburg": "Санкт-Петербург",
}
_NAME = r"(?P<name>[A-Za-zА-Яа-яЁё][^,()]*?)"
_SETTLEMENT_PREFIX = re.compile(
    rf"^(?:(?:г|гор|пгт|пос|п|с|д|дер|ст|х|рп|р\.п)\.\s*|(?:г|город|пгт|поселок|село|деревня|станица|хутор|рп|аул)\s+){_NAME}$",
    re.IGNORECASE,
)
_SETTLEMENT_SUFFIX = re.compile(
    rf"^{_NAME}\s+(?:город|поселок городского типа|рабочий поселок|курортный поселок|дачный поселок|поселок|село"
    rf"|деревня|станица|хутор|аул|слобода|пгт|рп)$",
    re.IGNORECASE,
)


def _city_name(part: str) -> str | None:
    part = re.sub(r"\([^)]*\)", "", part).strip(" .;")
    key = part.lower().replace("ё", "е")
    if key in _ALIASES:
        return _ALIASES[key]
    for pattern in (_SETTLEMENT_PREFIX, _SETTLEMENT_SUFFIX):
        m = pattern.match(part)
        if m:
            name = m.group("name").strip(" .")
            name = name[:1].upper() + name[1:] if name.islower() else name
            return _ALIASES.get(name.lower().replace("ё", "е"), name)
    return None


def address_cities(address: str) -> list[str]:
    """The first settlement of each address; trudvsem joins several addresses with "; ". A region
    without a settlement gives none."""
    out = []
    for one in address.split("; "):
        for part in one.split(","):
            city = _city_name(part)
            if city:
                out.append(city)
                break
    return out


# Habr writes a settlement as "Томилино, пгт": the type is a list item of its own.
_TYPE_ONLY = re.compile(r"^(?:г|город|пгт|п|пос|поселок|рп|с|село|д|деревня|ст|станица|х|хутор|аул)\.?$")


def predict_cities(inp: Input) -> list[str]:
    cities: list[str] = []
    if "Местоположение" in inp.header:
        for part in inp.header["Местоположение"].split(","):
            name = part.strip()
            key = name.lower().replace("ё", "е")
            if not _TYPE_ONLY.match(key):
                cities.append(_ALIASES.get(key, name))
    elif "Адрес" in inp.header:
        cities = address_cities(inp.header["Адрес"])
    unique = list(dict.fromkeys(c for c in cities if 0 < len(c) <= 80))
    return unique[:30]


# --- experience ---------------------------------------------------------------------------------

_WORD_NUMBERS = {
    "одного": 1,
    "один": 1,
    "двух": 2,
    "два": 2,
    "трех": 3,
    "три": 3,
    "четырех": 4,
    "четыре": 4,
    "пяти": 5,
    "пять": 5,
    "шести": 6,
    "шесть": 6,
    "семи": 7,
    "семь": 7,
    "восьми": 8,
    "восемь": 8,
    "десяти": 10,
    "десять": 10,
}
_N = r"(\d+(?:[.,]\d+)?|\b(?:" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")\b)"
_SUFFIX = r"(?:-?(?:х|ти|ми|го|ух|ех|и|ой))?"
_YEARS = r"(?:лет|года?|год|years?)\b"
# "Стаж" is length of service; "стажировка" and "стажер" are not.
_EXP_WORD = re.compile(r"опыт\w*|стаж(?!ир|ер)\w*|experience")
# "от года", "не менее полугода": the unit is the number.
_EXP_UNIT = re.compile(r"(?:от|не менее|более|свыше|больше|минимум)\s+(полугода|года)\b")
_UNIT_YEARS = {"полугода": 0.5, "года": 1.0}
_EXP_ZERO = re.compile(
    r"без опыта|опыт\w*(?: работы)? не (?:требуется|обязател\w*|нужен|важен)|не требуется опыт"
    r"|no (?:prior )?experience|experience (?:is )?not required"
)
_EXP_RANGE = re.compile(rf"(?:от\s+)?{_N}{_SUFFIX}\s*(?:-|до|to)\s*{_N}{_SUFFIX}\s*{_YEARS}")
_EXP_LOWER = re.compile(
    rf"(?:от|не менее|более|свыше|больше|минимум|at least|over|more than)\s+{_N}{_SUFFIX}\s*\+?\s*{_YEARS}"
)
_EXP_PLUS = re.compile(rf"{_N}\s*\+\s*{_YEARS}")
_EXP_BARE = re.compile(rf"(?<![\d.,]){_N}{_SUFFIX}\s*{_YEARS}")
_EXP_EN = re.compile(rf"{_N}\s*\+?\s*(?:-\s*{_N}\s*)?years?(?:\s+of)?(?:\s+\w+){{0,3}}\s+experience")
_EXP_WINDOW = 90


def _number(token: str) -> float:
    return _WORD_NUMBERS[token] if token in _WORD_NUMBERS else float(token.replace(",", "."))


def _experience(lo: float | None, hi: float | None) -> Experience | None:
    if lo is not None and not 0 <= lo <= 50:
        return None
    if hi is not None and not (0 <= hi <= 50 and (lo is None or lo <= hi)):
        hi = None
    if lo is None and hi is None:
        return None
    return Experience(min=lo, max=hi)


def _window_experience(window: str) -> Experience | None:
    m = _EXP_RANGE.search(window)
    if m:
        return _experience(_number(m.group(1)), _number(m.group(2)))
    for pattern in (_EXP_LOWER, _EXP_PLUS, _EXP_BARE):
        m = pattern.search(window)
        if m:
            return _experience(_number(m.group(1)), None)
    m = _EXP_UNIT.search(window)
    return _experience(_UNIT_YEARS[m.group(1)], None) if m else None


def text_experience(body_norm: str) -> list[Experience]:
    found: list[Experience] = []
    for m in _EXP_WORD.finditer(body_norm):
        window = body_norm[m.start() : m.start() + _EXP_WINDOW]
        end = _SENTENCE_END.search(window)
        e = _window_experience(window[: end.start()] if end else window)
        if e is not None:
            found.append(e)
    for m in _EXP_EN.finditer(body_norm):
        e = _experience(_number(m.group(1)), _number(m.group(2)) if m.group(2) else None)
        if e is not None:
            found.append(e)
    if _EXP_ZERO.search(body_norm):
        found.append(Experience(min=0, max=None))
    return found


_HEADER_EXPERIENCE = re.compile(rf"от\s+{_N}")


def predict_experience(inp: Input, body_norm: str) -> Experience | None:
    """Several requirements give the largest minimum; the text is more specific than the header."""
    found = text_experience(body_norm)
    if found:
        return max(found, key=lambda e: (e.min or 0, e.max is not None))
    m = _HEADER_EXPERIENCE.search(normalize(inp.header.get("Требуемый опыт", "")))
    return _experience(_number(m.group(1)), None) if m else None


# --- employment type ----------------------------------------------------------------------------

_INTERNSHIP = re.compile(r"стаж[её]р|стажировк|\bintern\b|internship|\btrainee\b")
_LABOR = re.compile(
    r"\bтк(?: рф)?\b|трудов\w* (?:договор|кодекс)|официальн\w* (?:трудоустройств|оформлени)"
    r"|(?:трудоустройств|оформлени)\w* (?:по|согласно) тк"
)
_OTHER_CONTRACTS: list[tuple[EmploymentType, re.Pattern[str]]] = [
    ("gph", re.compile(r"\bгпх\b|гражданско[\s-]*правов\w* договор|договор\w* подряда")),
    ("self_employed", re.compile(r"самозанят|\bип\b|индивидуальн\w* предпринимател")),
]


def predict_employment(inp: Input, body_norm: str, title_norm: str) -> EmploymentType | None:
    if _INTERNSHIP.search(title_norm) or normalize(inp.header.get("Тип занятости", "")) == "стажировка":
        return "internship"
    if _LABOR.search(body_norm):
        return "labor_contract"
    first: tuple[int, EmploymentType] | None = None
    for kind, pattern in _OTHER_CONTRACTS:
        m = pattern.search(body_norm)
        if m and (first is None or m.start() < first[0]):
            first = (m.start(), kind)
    return first[1] if first else None


# --- language -----------------------------------------------------------------------------------

_CYRILLIC = re.compile(r"[а-яё]")
_LATIN = re.compile(r"[a-z]")
# Letters of Kazakh, Uzbek Cyrillic, Ukrainian and Belarusian that Russian does not have.
_NON_RUSSIAN_CYRILLIC = re.compile(r"[әғқңөұүһіїєўҳҷ]")
_EN_LATIN_SHARE = 0.6
_OTHER_CYRILLIC_SHARE = 0.01
_MIN_LETTERS = 20


def predict_lang(inp: Input) -> Lang:
    text = inp.body.lower()
    if len(_CYRILLIC.findall(text)) + len(_LATIN.findall(text)) < _MIN_LETTERS:
        text = (inp.title or inp.body).lower()
    cyr, lat = len(_CYRILLIC.findall(text)), len(_LATIN.findall(text))
    if cyr + lat == 0:
        return "other"
    if lat / (cyr + lat) > _EN_LATIN_SHARE:
        return "en"
    if cyr and len(_NON_RUSSIAN_CYRILLIC.findall(text)) / cyr > _OTHER_CYRILLIC_SHARE:
        return "other"
    return "ru"


# --- the record ---------------------------------------------------------------------------------


def extract(text: str) -> VacancyFields:
    inp = split_input(text)
    title = inp.title
    if title is None and inp.body:
        title = inp.body.split("\n", 1)[0]
    body_norm = normalize(inp.body)
    title_norm = normalize(title or "")
    lang = predict_lang(inp)

    skill_source = "\n".join(filter(None, (title, inp.header.get("Навыки"), inp.body)))
    specializations = [s.strip() for s in inp.header.get("Специализация", "").split(",") if s.strip()]
    return VacancyFields(
        role=predict_role(title, inp.body, specializations),
        grade=predict_grade(inp, title_norm),
        skills=find_skills(skill_source, normalize(skill_source), english=lang == "en")[:60],
        salary=predict_salary(inp, body_norm),
        work_format=predict_work_format(inp, body_norm),
        cities=predict_cities(inp),
        experience_years=predict_experience(inp, body_norm),
        employment_type=predict_employment(inp, body_norm, title_norm),
        lang=lang,
    )


def output_for(text: str) -> str:
    """The baseline's answer as the raw string every system hands to radar.eval, checked against
    the schema the same way an LLM answer is."""
    out = json.dumps(dump(extract(text)), ensure_ascii=False)
    parse(out)
    return out


# --- runs ---------------------------------------------------------------------------------------


def read_inputs(path: Path) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(str(r["id"]), str(r["input"])) for r in rows]


def write_predictions(rows: Iterable[tuple[str, str]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rid, text in rows:
            f.write(json.dumps({"id": rid, "output": output_for(text)}, ensure_ascii=False) + "\n")
            n += 1
    return n


def train_pool_sample(n: int, seed: int) -> list[tuple[str, str]]:
    corpus = data_dir() / "corpus"
    split = pq.read_table(corpus / "split.parquet", columns=["split", "member_ids"]).to_pylist()
    pool = {rid for row in split if row["split"] == "train_pool" for rid in row["member_ids"]}
    records = pq.read_table(corpus / "corpus.parquet", columns=["id", "input"]).to_pylist()
    rows = sorted((r["id"], r["input"]) for r in records if r["id"] in pool)
    return random.Random(seed).sample(rows, min(n, len(rows)))


def fill_rates(outputs: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Non-null (non-empty for lists) share per field: (count, denominator)."""
    parsed = [parse(o) for o in outputs]
    counts: dict[str, tuple[int, int]] = {}
    for field in VacancyFields.model_fields:
        filled = sum(1 for p in parsed if getattr(p, field) not in (None, []))
        counts[field] = (filled, len(parsed))
    return counts


def cmd_split(split: str) -> int:
    gold_dir = data_dir() / "gold"
    inputs = gold_dir / f"{split}_inputs.jsonl"
    out = data_dir() / "predictions" / split / f"{SYSTEM}.jsonl"
    n = write_predictions(read_inputs(inputs), out)
    log.info("%d predictions -> %s", n, out)
    gold = gold_dir / f"{split}.jsonl"
    if not gold.exists():
        log.info("no %s yet: predictions written, nothing scored and nothing written to results/", gold)
        return 0
    return evaluate(["--system", SYSTEM, "--split", split])


def cmd_train_sample(n: int, seed: int) -> int:
    rows = train_pool_sample(n, seed)
    outputs = []
    invalid = 0
    for _, text in rows:
        try:
            outputs.append(output_for(text))
        except ValueError:
            invalid += 1
    print(f"train_pool sample: {len(rows)} records, seed {seed}; schema-valid {len(outputs)} of {len(rows)}")
    for field, (filled, total) in fill_rates(outputs).items():
        print(f"{field:18} {filled:4} of {total}")
    return 0 if invalid == 0 else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.extract.baseline", description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--split", choices=["dev", "tg_dev"], help="predict data/gold/<split>_inputs.jsonl; scored if gold exists"
    )
    mode.add_argument("--train-sample", type=int, metavar="N", help="validity and fill rates on N train_pool records")
    ap.add_argument("--seed", type=int, help="seed of the train_pool sample; default: the seed of configs/eval.toml")
    args = ap.parse_args(argv)
    if args.split:
        return cmd_split(args.split)
    return cmd_train_sample(args.train_sample, args.seed if args.seed is not None else load_config().seed)


if __name__ == "__main__":
    sys.exit(main())
