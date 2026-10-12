from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from radar.extract.telegram_lang import channel_rows, frozen_dropped, latest, measure

RU = "Python-разработчик в финтех, удалённо, от 250 000 рублей #python"
EN = "Senior Python Developer, remote within Europe, B2B contract #python"


def raw(source_id: str, text: str, *, vacancy: bool = True, fetched: int = 10, h: str = "h") -> dict[str, Any]:
    return {
        "source_id": source_id,
        "fetched_at": datetime(2026, 10, 8, fetched, tzinfo=UTC),
        "content_hash": h,
        "is_vacancy": vacancy,
        "text": text,
    }


def test_rows_of_the_ruled_channels_only_and_their_latest_version(tmp_path: Path) -> None:
    for day, rows in {
        "2026-10-08": [raw("remoteit/1", EN, h="old"), raw("jobs/1", EN), raw("remoteitx/1", EN)],
        "2026-10-09": [raw("remoteit/1", RU, fetched=11, h="new")],
    }.items():
        d = tmp_path / "telegram" / f"dt={day}"
        d.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), d / "part-0.parquet")
    rows = list(channel_rows(tmp_path, ["2026-10-08", "2026-10-09"], ["remoteit"]))
    assert sorted(r["source_id"] for r in rows) == ["remoteit/1", "remoteit/1"]
    assert [(p["source_id"], p["content_hash"], p["dt"]) for p in latest(rows)] == [("remoteit/1", "new", "2026-10-09")]


def test_measure_counts_kept_and_dropped_with_denominators() -> None:
    posts = [
        raw("remoteit/1", RU),
        raw("remoteit/2", EN),
        raw("remoteit/3", EN, vacancy=False),
        raw("remoteit/4", "https://telegra.ph/Data-Scientist-06-28", vacancy=False),
        raw("remoteit/5", "Junior Web-разработчик (Fullstack)\nhttps://telegra.ph/Junior-Web-razrabotchik #remote"),
    ]
    m = measure(posts, {"remoteit": "ru"}, 50)["remoteit"]
    assert m["posts_kept"] == {"value": 0.2, "n": 1, "denominator": 5}
    assert m["posts_dropped"]["n"] == 4
    assert [m[f"posts_{x}"]["n"] for x in ("ru", "en", "other")] == [1, 3, 1]
    assert m["vacancy_posts_kept"] == {"value": 0.3333, "n": 1, "denominator": 3}
    assert m["vacancy_posts_with_text_kept"] == {"value": 0.5, "n": 1, "denominator": 2}
    assert m["vacancy_posts_dropped_short_with_cyrillic_word"] == {"value": 0.5, "n": 1, "denominator": 2}


def test_frozen_records_the_rule_would_drop(tmp_path: Path) -> None:
    split = [
        {"group_id": "g1", "split": "test", "gold_id": "telegram:remoteit/1"},
        {"group_id": "g2", "split": "dev", "gold_id": "telegram:remoteit/2"},
        {"group_id": "g3", "split": "train_pool", "gold_id": None},
        {"group_id": "g4", "split": "dev", "gold_id": "telegram:jobs/1"},
    ]
    records = [
        {"id": "telegram:remoteit/1", "channel": "remoteit", "input": RU},
        {"id": "telegram:remoteit/2", "channel": "remoteit", "input": EN},
        {"id": "telegram:remoteit/3", "channel": "remoteit", "input": EN},
        {"id": "telegram:jobs/1", "channel": "jobs", "input": EN},
    ]
    pq.write_table(pa.Table.from_pylist(split), tmp_path / "split.parquet")
    pq.write_table(pa.Table.from_pylist(records), tmp_path / "records.parquet")
    assert frozen_dropped(tmp_path, {"remoteit": "ru"}) == (2, {"test": [], "dev": ["telegram:remoteit/2"]})
