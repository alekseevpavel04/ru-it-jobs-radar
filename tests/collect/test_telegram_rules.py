import re

from radar.collect.telegram_rules import RESUME_HEAD, classify
from radar.config import ROOT


def test_resume_heads_and_digests_are_not_vacancies() -> None:
    for text in (
        "🏆 Лучшие резюме и вакансии за 28 мая (3/3)\n\n🔥 Резюме Senior Frontend Engineer, от 300 000₽",
        "📘 Обзор за 16 апреля\n\nВчера на IT-рынке снова был шторм",
        "🌙 Резюме за 11 марта\n\n📃 Резюме Frontend Разработчик (React)",
        "#резюме #junior #backend #csharp\nНачинающий .NET разработчик ищет стажировку",
    ):
        assert classify(text, []).kind == "resume", text
    vacancy = "Senior Python Developer в Яндекс\nЗарплата от 300 000₽, удалённо. Резюме присылайте @hr"
    assert classify(vacancy, []).kind != "resume"


def test_staging_applies_the_same_pattern() -> None:
    macro = (ROOT / "dbt" / "macros" / "telegram_rules.sql").read_text(encoding="utf-8")
    found = re.search(r"match\(lowerUTF8\(\{\{ text \}\}\), '([^']+)'\)", macro)
    assert found is not None and found.group(1) == RESUME_HEAD.pattern
