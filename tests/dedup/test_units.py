from pathlib import Path

from radar.dedup.units import Unit, inputs_digest, read_units, representatives, to_units, write_units


def row(source: str, source_id: str, dt: str, **extra: object) -> dict[str, object]:
    return {"source": source, "source_id": source_id, "dt": dt, "title": "Программист", **extra}


def test_api_and_snapshot_with_one_id_form_one_group() -> None:
    units = to_units(
        [
            row("trudvsem_snapshot", "u1", "2026-07-04", text="old"),
            row("trudvsem", "u1", "2026-10-08", text="new"),
            row("trudvsem_snapshot", "u2", "2026-07-01"),
            row("habr", "u1", "2026-10-08", cities=["Москва", "Казань"]),
        ]
    )
    assert [u.group for u in units] == ["trudvsem:u1", "trudvsem:u1", "trudvsem:u2", "habr:u1"]
    reps = representatives(units)
    assert [u.uid for u in reps] == ["habr:u1", "trudvsem:u1", "trudvsem_snapshot:u2"]
    assert reps[0].location == "Москва, Казань"


def test_api_wins_over_snapshot_of_the_same_day() -> None:
    units = to_units([row("trudvsem_snapshot", "u1", "2026-10-08"), row("trudvsem", "u1", "2026-10-08")])
    assert [u.source for u in representatives(units)] == ["trudvsem"]


def test_employer_key_prefers_inn() -> None:
    a, b, c = to_units(
        [
            row("trudvsem", "1", "2026-10-08", employer_name='ООО "Ромашка"', employer_inn="7700000000"),
            row("habr", "2", "2026-10-08", employer_name="Ромашка"),
            row("trudvsem", "3", "2026-10-08", employer_name="ИП"),
        ]
    )
    assert a.employer_key == "inn:7700000000"
    assert a.employer_name_key == b.employer_name_key == "ромашка"
    assert b.employer_key == "name:ромашка"
    assert c.employer_key is None


def test_remote_is_kept_as_stated_or_unknown() -> None:
    units = to_units(
        [
            row("habr", "1", "2026-10-08", remote=True),
            row("trudvsem", "2", "2026-10-08", remote=0),
            row("trudvsem", "3", "2026-10-08", remote=None),
        ]
    )
    assert [u.remote for u in units] == [True, False, None]


def test_document_joins_head_and_text() -> None:
    (u,) = to_units([row("habr", "1", "2026-10-08", employer_name="Ромашка", cities=["Москва"], text="Задачи")])
    assert u.document() == "Программист. Ромашка. Москва.\nЗадачи"


def test_parquet_round_trip(tmp_path: Path) -> None:
    units = to_units([row("habr", "1", "2026-10-08", salary_text="от 100 000")])
    write_units(units, tmp_path / "u.parquet")
    back = read_units(tmp_path / "u.parquet")
    assert back == units
    assert isinstance(back[0], Unit)
    assert inputs_digest(back) == inputs_digest(units)
