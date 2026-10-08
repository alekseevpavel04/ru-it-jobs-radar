"""Small hand-made cases; every expected number is counted by hand in the comments."""

from typing import Any

import pytest

from radar.eval.metrics import MACRO_FIELDS, score
from radar.extract.schema import Experience, Salary, VacancyFields

TOL = 0.01


def vf(**kw: Any) -> VacancyFields:
    base: dict[str, Any] = {
        "role": None,
        "grade": None,
        "skills": [],
        "salary": None,
        "work_format": None,
        "cities": [],
        "experience_years": None,
        "employment_type": None,
        "lang": "ru",
    }
    return VacancyFields(**(base | kw))


def sal(lo: int | None, hi: int | None, cur: Any = "RUB", gn: Any = None, per: Any = None) -> Salary:
    return Salary(from_=lo, to=hi, currency=cur, gross_net=gn, period=per)


def value(stats: Any, name: str) -> float:
    return float(stats[name].value())


def test_categorical_with_null() -> None:
    gold = [vf(role="ml_engineer"), vf(role="data_scientist"), vf(role=None), vf(role="ml_engineer")]
    pred = [vf(role="ml_engineer"), vf(role="backend"), vf(role="ml_engineer"), vf(role=None)]
    s = score(gold, pred, TOL)
    # accuracy: only row 1 matches (null == null would count, none here) -> 1/4
    assert value(s, "role_accuracy") == pytest.approx(1 / 4)
    assert s["role_accuracy"].numerator_denominator() == (1, 4)
    # slots: tp 1 (row 1); predicted non-null rows 1, 2, 3; gold non-null rows 1, 2, 4
    assert value(s, "role_precision") == pytest.approx(1 / 3)
    assert value(s, "role_recall") == pytest.approx(1 / 3)
    assert value(s, "role_f1") == pytest.approx(1 / 3)
    # classes: ml_engineer tp 1, pred 2, gold 2 -> F1 0.5; data_scientist 0; backend 0 -> mean 1/6
    assert value(s, "role_macro_f1") == pytest.approx(1 / 6)


def test_null_agreement_counts_for_accuracy_not_for_f1() -> None:
    gold = [vf(grade=None), vf(grade="junior")]
    pred = [vf(grade=None), vf(grade="junior")]
    s = score(gold, pred, TOL)
    assert value(s, "grade_accuracy") == 1.0
    assert value(s, "grade_f1") == 1.0
    # nothing in gold and nothing predicted: F1 is 1 by convention, not 0/0
    s = score([vf()], [vf()], TOL)
    assert value(s, "grade_f1") == 1.0
    assert value(s, "skills_f1") == 1.0


def test_lists_micro_f1_case_insensitive() -> None:
    gold = [vf(skills=["Python", "SQL"]), vf(skills=["Docker"]), vf(skills=[])]
    pred = [vf(skills=["python", "SQL ", "Git"]), vf(skills=[]), vf(skills=[])]
    s = score(gold, pred, TOL)
    # tp 2, predicted 3, gold 3
    assert value(s, "skills_precision") == pytest.approx(2 / 3)
    assert value(s, "skills_recall") == pytest.approx(2 / 3)
    assert value(s, "skills_f1") == pytest.approx(2 / 3)


def test_duplicates_in_a_list_count_once() -> None:
    s = score([vf(cities=["Москва"])], [vf(cities=["Москва", "москва"])], TOL)
    assert value(s, "cities_precision") == 1.0


def test_salary_range_and_parts() -> None:
    gold = [
        vf(salary=sal(100000, 150000, "RUB", "net", "month")),
        vf(salary=sal(None, 200000)),
        vf(salary=None),
        vf(salary=sal(50000, 50000, "USD")),
    ]
    pred = [
        vf(salary=sal(100500, 150000, "RUB", "gross", "month")),  # from within 1%
        vf(salary=sal(None, 250000)),  # wrong upper bound
        vf(salary=sal(1000, None)),  # salary where gold has none
        vf(salary=None),  # missed salary
    ]
    s = score(gold, pred, TOL)
    # range slot: tp row 1; predicted rows 1-3; gold rows 1, 2, 4
    assert value(s, "salary_f1") == pytest.approx(1 / 3)
    assert value(s, "salary_accuracy") == pytest.approx(1 / 4)
    # parts are scored on rows where both have a salary: rows 1 and 2
    assert s["salary_from_accuracy"].numerator_denominator() == (2, 2)  # 100500 ~ 100000, null == null
    assert s["salary_to_accuracy"].numerator_denominator() == (1, 2)
    assert s["salary_currency_accuracy"].numerator_denominator() == (2, 2)
    assert s["salary_gross_net_accuracy"].numerator_denominator() == (1, 2)
    assert s["salary_period_accuracy"].numerator_denominator() == (2, 2)


def test_salary_tolerance_is_relative() -> None:
    gold = [vf(salary=sal(100000, 100000))]
    assert value(score(gold, [vf(salary=sal(100900, 100900))], TOL), "salary_f1") == 1.0
    assert value(score(gold, [vf(salary=sal(102000, 102000))], TOL), "salary_f1") == 0.0
    # an open bound must stay open
    assert value(score(gold, [vf(salary=sal(100000, None))], TOL), "salary_f1") == 0.0


def test_salary_parts_undefined_without_overlap() -> None:
    s = score([vf(salary=sal(1, 2))], [vf()], TOL)
    assert s["salary_currency_accuracy"].numerator_denominator() == (0, 0)
    assert s["salary_currency_accuracy"].value() != s["salary_currency_accuracy"].value()  # nan


def test_experience_is_one_slot() -> None:
    gold = [vf(experience_years=Experience(min=3, max=None)), vf(experience_years=Experience(min=1, max=3))]
    pred = [vf(experience_years=Experience(min=3, max=None)), vf(experience_years=Experience(min=1, max=None))]
    s = score(gold, pred, TOL)
    assert value(s, "experience_years_f1") == pytest.approx(0.5)


def test_invalid_prediction_scores_as_empty() -> None:
    gold = [vf(role="qa", skills=["Selenium"], lang="ru"), vf(role="qa", lang="en")]
    pred: list[VacancyFields | None] = [None, vf(role="qa", lang="en")]
    s = score(gold, pred, TOL)
    assert s["valid_json"].numerator_denominator() == (1, 2)
    # row 1 counts as a miss: recall 1/2 for role, 0/1 for skills, language wrong
    assert value(s, "role_recall") == pytest.approx(1 / 2)
    assert value(s, "role_precision") == 1.0
    assert value(s, "skills_recall") == 0.0
    assert value(s, "lang_accuracy") == pytest.approx(1 / 2)


def test_macro_is_mean_of_field_f1() -> None:
    gold = [vf(role="qa", skills=["Python"], cities=["Казань"]), vf(grade="senior", work_format="remote")]
    pred = [vf(role="qa", skills=["Java"], cities=["Казань"]), vf(grade="middle", work_format="remote")]
    s = score(gold, pred, TOL)
    fields = [value(s, f"{f}_f1") for f in MACRO_FIELDS]
    # role 1, grade 0, skills 0, salary 1 (empty), work_format 1, cities 1, experience 1, employment 1
    assert fields == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert value(s, "macro_f1") == pytest.approx(6 / 8)
    assert value(score(gold, gold, TOL), "macro_f1") == 1.0


def test_lengths_must_match() -> None:
    with pytest.raises(ValueError):
        score([vf()], [], TOL)
