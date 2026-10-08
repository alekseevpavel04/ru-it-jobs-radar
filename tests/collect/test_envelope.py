from datetime import UTC, datetime

import pytest

from radar.collect.envelope import parse_source_time


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-06-05T08:37:49+0300", datetime(2026, 6, 5, 5, 37, 49, tzinfo=UTC)),
        ("2026-06-05T08:37:49+03:00", datetime(2026, 6, 5, 5, 37, 49, tzinfo=UTC)),
        ("2026-06-05T08:37:49.500+0300", datetime(2026, 6, 5, 5, 37, 49, 500000, tzinfo=UTC)),
        ("2026-10-07", datetime(2026, 10, 6, 21, 0, tzinfo=UTC)),
        ("1771493777331", datetime(2026, 2, 19, 9, 36, 17, 331000, tzinfo=UTC)),
        ("", None),
        (None, None),
    ],
)
def test_source_time_formats(value: str | None, expected: datetime | None) -> None:
    parsed = parse_source_time(value)
    assert parsed == expected
    if parsed is not None:
        assert parsed.utcoffset() is not None


@pytest.mark.parametrize("value", ["2026-06-05T08:37:49", "07.10.2026", "не дата", "177149377733"])
def test_source_time_rejects_unknown_shapes(value: str) -> None:
    with pytest.raises(ValueError):
        parse_source_time(value)
