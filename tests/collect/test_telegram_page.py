from pathlib import Path

import pytest

from radar.collect.telegram_page import Page, job_links, page_url, parse, parse_views, post_url
from radar.collect.telegram_rules import classify

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "telegram"


def _page(name: str) -> Page:
    return parse(200, "2026-10-08T10:00:00+00:00", (FIXTURES / name).read_text(encoding="utf-8"))


def test_recorded_channel_page() -> None:
    page = _page("datasciencejobs_latest.html")
    assert page.exists
    assert page.subscribers == "22.3K"
    assert page.prev_before == 3364
    assert [p.id for p in page.posts] == [3364, 3366, 3370, 3372, 3399]
    first = page.posts[0]
    assert first.date == "2026-09-15T10:02:07+00:00"
    assert first.views == 4910
    assert page.posts[1].edited
    assert page.posts[3].media == ["not_in_preview"]
    assert page.posts[4].media == ["photo"]


def test_structures() -> None:
    page = _page("structures_page.html")
    posts = {p.id: p for p in page.posts}
    assert page.prev_before == 501
    reply = posts[501]
    assert reply.text.startswith("Python-разработчик в ООО «Пример»\n")
    assert "Курс" not in reply.text  # the quote of the replied post is not the post
    assert reply.views == 1200
    assert reply.links == ["https://career.habr.com/vacancies/1000000001?utm_source=tg"]
    channel_forward = posts[502]
    assert (channel_forward.forwarded, channel_forward.forwarded_from) == (True, "other_jobs_channel")
    assert channel_forward.media == ["document"]
    person_forward = posts[503]
    assert (person_forward.forwarded, person_forward.forwarded_from) == (True, None)
    assert person_forward.edited
    assert person_forward.views == 1_200_000
    assert person_forward.media == ["poll"]
    assert posts[504].service
    album = posts[505]
    assert (album.text, album.media) == ("", ["photo", "album"])
    assert not any(p.service for i, p in posts.items() if i != 504)


def test_missing_channel() -> None:
    page = parse(200, "2026-10-08T10:00:00+00:00", "<html><body><div class='tgme_page'></div></body></html>")
    assert not page.exists
    assert not parse(302, "2026-10-08T10:00:00+00:00", "").exists


@pytest.mark.parametrize(("shown", "views"), [("932", 932), ("4.91K", 4910), ("12K", 12000), ("1.2M", 1_200_000)])
def test_parse_views(shown: str, views: int) -> None:
    assert parse_views(shown) == views


def test_job_links_keep_hosts_and_habr_ids_only() -> None:
    links = [
        "https://spb.hh.ru/vacancy/123456",
        "https://hh.ru/resume/abcdef",
        "https://career.habr.com/vacancies/1000000001?utm_source=tg",
        "https://career.habr.com/vacancies/1000000001",
        "https://u.habr.com/AbC12",
        "https://t.me/someone_recruits",
        "mailto:hr@example.com",
        "https://example.com/jobs/1",
    ]
    assert job_links(links) == [
        {"host": "career.habr.com", "habr_vacancy_id": 1000000001},
        {"host": "hh.ru"},
        {"host": "u.habr.com", "habr_short": "AbC12"},
    ]


def test_urls() -> None:
    assert page_url("job_python") == "https://t.me/s/job_python"
    assert page_url("job_python", before=100) == "https://t.me/s/job_python?before=100"
    assert post_url("job_python", 7) == "https://t.me/job_python/7"


def test_rules_on_recorded_posts() -> None:
    ds = {p.id: classify(p.text, p.links) for p in _page("datasciencejobs_latest.html").posts}
    assert ds[3364].vacancy and not ds[3364].multi
    assert ds[3399].kind == "course"
    habr = {p.id: classify(p.text, p.links) for p in _page("habr_career_latest.html").posts}
    assert habr[76083].kind == "no_text"
    assert habr[76084].vacancy and habr[76084].multi and habr[76084].habr_link
    reply = _page("structures_page.html").posts[0]
    assert classify(reply.text, reply.links).vacancy


def test_rules_exclusions() -> None:
    assert classify("Python-разработчик в ООО «Пример»\nРеклама. erid: 2Vtzqx", []).kind == "ad"
    assert classify("#резюме\nИщу работу, Python-разработчик, опыт 3 года", []).kind == "resume"
    assert classify("", []).kind == "no_text"
