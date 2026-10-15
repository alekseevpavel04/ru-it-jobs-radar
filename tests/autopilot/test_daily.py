"""radar.autopilot.daily: which new posts become weekly units, and what is left for the next run."""

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from radar.autopilot.daily import held_back, process
from radar.autopilot.skillmap import SkillMap, key_hash
from radar.extract import skills
from radar.extract.corpus_run import Answer

BODY = " Требуется опыт коммерческой разработки от трёх лет, участие в проектировании, тестах и код-ревью сервисов."
T0 = datetime(2026, 10, 13, 9, 0, tzinfo=UTC)


def post(source_id: str, text: str, minutes: int = 0, **kw: Any) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "published_at": T0 + timedelta(minutes=minutes),
        "text": text,
        "payload": json.dumps({"channel": source_id.split("/")[0], "job_links": []}),
        "is_vacancy": kw.get("is_vacancy", True),
        "multi_vacancy": kw.get("multi_vacancy", False),
    }


def answer(role: str | None, skill_list: list[str], grade: str | None = "middle") -> Answer:
    fields = {
        "role": role,
        "grade": grade,
        "skills": skill_list,
        "salary": None,
        "work_format": None,
        "cities": [],
        "experience_years": None,
        "employment_type": None,
        "lang": "ru",
    }
    return Answer(json.dumps(fields, ensure_ascii=False), "stop")


@pytest.fixture
def mapper(tmp_path: Path) -> SkillMap:
    seed = tmp_path / "seed.csv"
    rows = ["key_hash,spelling_key,spelling,skill,category,parent,status,origin"]
    for spelling, name in (("python", "Python"), ("postgres", "PostgreSQL"), ("https", "HTTPS")):
        rows.append(f"{key_hash(spelling)},{spelling},{spelling},{name},x,,known,dictionary")
    seed.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return SkillMap(seed, skills.load())


def scripted(*answers: Answer) -> Callable[[str], Answer]:
    it: Iterator[Answer] = iter(answers)
    return lambda text: next(it)


def clock_at(times: list[float]) -> Callable[[], float]:
    it = iter(times)
    last = [0.0]

    def clock() -> float:
        last[0] = next(it, last[0])
        return last[0]

    return clock


def test_units_are_deduplicated_filtered_and_counted(mapper: SkillMap) -> None:
    rows = [
        post("a/10", "Python-разработчик" + BODY + " https://t.me/a", 0),
        # the same vacancy in another channel, signed with its own link: a duplicate
        post("b/7", "python разработчик" + BODY + " https://t.me/b/7 #python", 5),
        post("a/11", "Вакансии недели: backend, frontend, QA" + BODY, 10, multi_vacancy=True),
        post("a/12", "Бухгалтер" + BODY, 20),
        post("b/8", "Data Engineer\nhttps://example.com/job", 30),
        post("b/9", "Не вакансия" + BODY, 40, is_vacancy=False),
        post("a/13", "QA" + BODY, 50),
    ]
    hashes: dict[str, set[str]] = {}
    out = process(
        rows,
        hashes,
        scripted(answer("backend", ["Python", "postgres", "HTTPS"]), answer("non_it", []), Answer("{broken", "stop")),
        mapper,
        {},
        min_content_chars=100,
        deadline=float("inf"),
    )
    assert out.stats["vacancy_posts"] == 6
    assert (out.stats["duplicates"], out.stats["multi_vacancy"], out.stats["not_extracted_link_stub"]) == (1, 1, 1)
    assert (out.stats["extracted"], out.stats["valid"], out.stats["it"]) == (3, 2, 1)
    # HTTPS only from a link: dropped, as int_extracted_fields drops it
    assert out.counts.skills == {
        ("2026-10-12", "a", "backend", "middle", "PostgreSQL"): 1,
        ("2026-10-12", "a", "backend", "middle", "Python"): 1,
    }
    assert out.counts.posts == {("2026-10-12", "a", "backend", "middle"): 1}
    assert len(hashes["2026-10-12"]) == 5
    assert out.left == {}


def test_a_copy_counted_on_an_earlier_day_is_a_duplicate(mapper: SkillMap) -> None:
    rows = [post("b/7", "Python-разработчик" + BODY)]
    first: dict[str, set[str]] = {}
    process(rows, first, scripted(answer("backend", [])), mapper, {}, 100, float("inf"))
    out = process([post("c/1", "PYTHON разработчик" + BODY, 60)], first, scripted(), mapper, {}, 100, float("inf"))
    assert out.stats["duplicates"] == 1 and not out.counts.posts


def test_posts_after_the_budget_stay_above_the_cursor(mapper: SkillMap) -> None:
    rows = [post("a/10", "Backend" + BODY, 0), post("b/5", "Frontend" + BODY, 1), post("a/11", "QA" + BODY, 2)]
    hashes: dict[str, set[str]] = {}
    # the first post starts before the deadline (1.0), the second one after it
    out = process(rows, hashes, scripted(answer("backend", [])), mapper, {}, 100, 1.0, clock_at([0.0, 0.0, 0.5, 2.0]))
    assert out.stats["extracted"] == 1 and out.stats["left_over"] == 2
    assert out.left == {"b": 5, "a": 11}
    assert len(hashes["2026-10-12"]) == 1


def test_a_failed_request_stops_the_run_and_keeps_the_post(mapper: SkillMap) -> None:
    rows = [post("a/10", "Backend" + BODY, 0), post("a/11", "QA" + BODY, 1)]
    out = process(rows, {}, scripted(Answer(None, error="HTTP 500: x")), mapper, {}, 100, float("inf"))
    assert out.stats["extraction_errors"] == 1 and out.left == {"a": 10} and not out.counts.posts


def test_the_channel_language_filters_before_anything_is_counted(mapper: SkillMap) -> None:
    rows = [post("remote/1", "Senior Backend Engineer, fully remote, Go and Kubernetes." * 3)]
    out = process(rows, {}, scripted(), mapper, {"remote": "ru"}, 100, float("inf"))
    assert out.stats["vacancy_posts"] == 0


def test_a_held_back_cursor_rereads_from_the_first_post_left() -> None:
    state = {"last_id": 20, "first_seen_id": 1, "recent": {"18": "x", "19": "y", "20": "z"}, "subscribers": 5}
    assert held_back(state, 19) == {"last_id": 18, "first_seen_id": 1, "recent": {"18": "x"}, "subscribers": 5}


def test_an_unseen_spelling_resolves_as_the_next_seed_would(mapper: SkillMap) -> None:
    assert mapper.names("Python 3.12") == ("Python",)
    assert mapper.unseen == 1
    assert mapper.skills(["https", "python"], "\nпротоколы HTTP/HTTPS") == ["HTTPS", "Python"]
