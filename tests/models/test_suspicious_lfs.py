import numpy as np
import pandas as pd
import pytest

from radar.models.suspicious_lfs import (
    ABSTAIN,
    LFS,
    MASS_EMPLOYERS,
    OK,
    SUSPICIOUS,
    add_context,
    apply_lfs,
    combined,
    experience_band,
    matches,
    normalize,
    pay_ratio,
    repost_chains,
    salary_z,
    template_groups,
    text_group,
)

TEMPLATE = "Обязанности: " + "проектирование и внедрение средств защиты информации, " * 6
LONG_TEMPLATE = " ".join(f"пункт {w} обязанностей номер {i}" for i, w in enumerate(["сбор", "анализ", "отчет"] * 10))


def card(i: int, **fields: object) -> dict[str, object]:
    base: dict[str, object] = {
        "family": "trudvsem",
        "source_id": f"id{i}",
        "key": f"trudvsem:id{i}",
        "title": "Программист",
        "text": f"Уникальный текст номер {i}",
        "employer_key": f"inn{i}",
        "employer_name": f'ООО "Компания {i}"',
        "is_sole_proprietor": 0,
        "region": "Город Москва",
        "region_group": "moscow",
        "salary_from": 100000.0,
        "salary_to": None,
        "salary_currency_code": "RUB",
        "role_bucket": "software_engineer",
        "grade": "unknown",
        "grade_is_stated": 0,
        "work_format": "office",
        "experience_min_years": 0,
        "no_experience": 0,
        "skills": [],
        "repost_of": None,
    }
    return base | fields


def frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return add_context(pd.DataFrame(rows))


def votes_of(cards: pd.DataFrame, name: str) -> np.ndarray:
    j = [lf.name for lf in LFS].index(name)
    return apply_lfs(cards)[:, j]


def test_normalize_and_text_group_ignore_case_digits_and_punctuation() -> None:
    assert normalize("  Ёлка\n\tИ  ПК ") == "елка и пк"
    a = text_group(TEMPLATE + " Оклад 65 000 руб.")
    b = text_group(TEMPLATE.upper() + " оклад: 70 000 руб")
    assert a is not None and a == b
    assert text_group("Обязанности: по собеседованию") is None


@pytest.mark.parametrize(
    ("name", "text", "hit"),
    [
        ("course_pay", "стоимость обучения 30 000 рублей", True),
        ("course_pay", "бесплатное обучение и курсы за счет компании", False),
        ("course_pay", "платное обучение с выдачей сертификата", True),
        ("grey", "зарплата в конверте два раза в месяц", True),
        ("grey", "официальное трудоустройство, никаких зарплат в конверте", False),
        ("grey", "можем рассмотреть кандидатов и без оформления", True),
        ("income_bait", "пассивный доход без вложений", True),
        ("income_bait", "работа с ip-телефонией и p2p-переводами", False),
        ("no_experience", "опыт не нужен, всему научим", True),
    ],
)
def test_patterns(name: str, text: str, hit: bool) -> None:
    assert matches(pd.Series([normalize(text)]), name)[0] == hit


def test_salary_z_falls_back_to_coarser_group_and_floors_the_scale() -> None:
    fine = pd.Series(["a"] * 40 + ["b"] * 5)
    coarse = pd.Series(["x"] * 45)
    values = pd.Series(np.log([100_000.0] * 40 + [50_000.0, 60_000.0, 70_000.0, 80_000.0, 400_000.0]))
    z = salary_z(values, [fine, coarse])
    # Group "a" has one salary, so IQR is zero and the floor keeps z finite.
    assert np.isfinite(z[:40]).all() and (z[:40] == 0).all()
    # Group "b" is too small: scored against "x", where the median is 100k.
    assert z[44] > 0 > z[40]


def test_repost_chains_count_ids_of_one_vacancy() -> None:
    cards = pd.DataFrame(
        {"source_id": ["a", "b", "c", "d"], "repost_of": [None, "a", "b", None]},
    )
    assert repost_chains(cards).tolist() == [3, 3, 3, 1]


def test_mass_text_needs_many_employers_in_one_region() -> None:
    one_region = [card(i, text=TEMPLATE) for i in range(MASS_EMPLOYERS)]
    spread = [card(100 + i, text=TEMPLATE + " шаблон справочника", region=f"Регион {i}") for i in range(MASS_EMPLOYERS)]
    cards = frame(one_region + spread + [card(200)])
    v = votes_of(cards, "mass_text")
    assert (v[:MASS_EMPLOYERS] == SUSPICIOUS).all()
    assert (v[MASS_EMPLOYERS:] == ABSTAIN).all()


def test_sole_proprietors_do_not_count_as_distinct_employers() -> None:
    rows = [card(i, text=TEMPLATE, employer_key=None, employer_name="ИП", is_sole_proprietor=1) for i in range(8)]
    cards = frame(rows)
    assert (cards["group_employers"] == 0).all()
    assert (votes_of(cards, "mass_text") == ABSTAIN).all()


def test_ok_votes() -> None:
    rows = [
        card(0, employer_name='ГОСУДАРСТВЕННОЕ БЮДЖЕТНОЕ УЧРЕЖДЕНИЕ ЗДРАВООХРАНЕНИЯ "БОЛЬНИЦА"'),
        card(1, employer_name="МБОУ СОШ № 5"),
        card(2, experience_min_years=3, skills=["Python", "SQL", "Docker"]),
        card(3),
    ]
    cards = frame(rows)
    assert votes_of(cards, "public_sector").tolist() == [OK, OK, ABSTAIN, ABSTAIN]
    assert votes_of(cards, "concrete_requirements").tolist() == [ABSTAIN, ABSTAIN, OK, ABSTAIN]


def test_large_employer() -> None:
    rows = [card(i, employer_key="sber") for i in range(50)] + [card(99)]
    v = votes_of(frame(rows), "large_employer")
    assert (v[:50] == OK).all() and v[50] == ABSTAIN


def test_salary_low_skips_trainees() -> None:
    rows = [card(i, salary_from=150_000.0 + 1000 * i) for i in range(40)]
    rows += [card(50, salary_from=20_000.0), card(51, salary_from=20_000.0, title="Стажер-программист")]
    v = votes_of(frame(rows), "salary_low")
    assert v[-2] == SUSPICIOUS and v[-1] == ABSTAIN


def test_repost_chain_and_salary_high_stay_out_of_the_weak_label() -> None:
    names = np.array([lf.name for lf in LFS])
    assert set(names[~combined()]) == {"repost_chain", "salary_high", "pay_far_below", "mass_template"}


def test_experience_band_and_pay_ratio_fall_back_to_coarser_groups() -> None:
    years = pd.Series([0.0, 1.0, 3.0, 6.0, np.nan])
    assert experience_band(years).tolist() == ["0", "1-2", "3-5", "6+", "na"]
    salary = pd.Series([100_000.0] * 40 + [50_000.0])
    fine = pd.Series(["a"] * 40 + ["b"])
    coarse = pd.Series(["x"] * 41)
    ratio = pay_ratio(salary, [fine, coarse])
    assert (ratio[:40] == 1.0).all()
    assert ratio[40] == pytest.approx(0.5)


def test_pay_far_below_and_mass_template_stay_out_of_the_vote() -> None:
    rows = [card(i, salary_from=300_000.0, experience_min_years=3) for i in range(40)]
    rows.append(card(50, salary_from=60_000.0, experience_min_years=3))
    v = votes_of(frame(rows), "pay_far_below")
    assert v[-1] == SUSPICIOUS and (v[:40] == ABSTAIN).all()
    names = [lf.name for lf in LFS]
    assert not combined()[names.index("pay_far_below")]
    assert not combined()[names.index("mass_template")]


def test_template_groups_join_edited_copies() -> None:
    edited = LONG_TEMPLATE + " Оклад обсуждается."
    other = "Совсем другая вакансия про бухгалтерию и отчетность. " * 6
    texts = pd.Series([LONG_TEMPLATE, edited, other])
    groups = pd.Series([text_group(t) for t in texts], dtype=object)
    assert groups[0] != groups[1]
    joined = template_groups(texts, groups)
    assert joined[0] == joined[1] != joined[2]


def test_mass_template_counts_employers_of_near_identical_texts() -> None:
    rows = [card(i, text=LONG_TEMPLATE + f" Вариант {'а' * i}.") for i in range(3)] + [card(9)]
    v = votes_of(frame(rows), "mass_template")
    assert (v[:3] == SUSPICIOUS).all() and v[3] == ABSTAIN
    assert (votes_of(frame(rows), "mass_text") == ABSTAIN).all()
