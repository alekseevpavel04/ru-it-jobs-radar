import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from radar.config import ROOT
from radar.site import hf_dataset, hf_model, hf_scan, hf_space
from radar.site.hf_common import render_card

FIELDS = {
    "role": "data_scientist",
    "grade": "middle",
    "skills": ["Python", "SQL"],
    "salary_from": 200000.0,
    "salary_to": None,
    "salary_currency": "RUB",
    "salary_gross_net": None,
    "salary_period": "month",
    "work_format": "remote",
    "cities": [],
    "experience_min": 3.0,
    "experience_max": None,
    "employment_type": None,
    "lang": "ru",
    "extraction_valid": True,
}
UUID_A = "2bf568c8-a531-11f1-aa6f-e58ffaa23423"
UUID_B = "61955888-6271-11f1-875d-37533a592c34"


# --- cards ----------------------------------------------------------------------------------------


def results_dir(tmp_path: Path, files: dict[str, Any]) -> Path:
    root = tmp_path / "results"
    for rel, doc in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc), encoding="utf-8")
    return root


def test_card_fills_numbers_and_keeps_pending_until_the_file_exists(tmp_path: Path) -> None:
    template = (
        "F1 <!--n:a.json:metrics.f1.value:.3f|ru-->0<!--/n-->, "
        "test <!--pending:b.json:metrics.t.value:.2f|ru-->ждёт<!--/pending-->"
    )
    card = render_card(template, results_dir(tmp_path, {"a.json": {"metrics": {"f1": {"value": 0.9041}}}}))
    assert "-->0,904<!--/n-->" in card.text
    assert "<!--pending:b.json:metrics.t.value:.2f|ru-->ждёт<!--/pending-->" in card.text
    assert card.numbers == 1
    assert card.pending == ["b.json:metrics.t.value"]

    both = results_dir(tmp_path, {"b.json": {"metrics": {"t": {"value": 0.5}}}})
    card = render_card(template, both)
    assert "<!--n:b.json:metrics.t.value:.2f|ru-->0,50<!--/n-->" in card.text
    assert card.pending == []


def test_card_fails_on_a_key_missing_from_an_existing_file(tmp_path: Path) -> None:
    template = "<!--pending:b.json:metrics.other.value:.2f-->x<!--/pending-->"
    with pytest.raises(KeyError):
        render_card(template, results_dir(tmp_path, {"b.json": {"metrics": {"t": {"value": 0.5}}}}))


@pytest.mark.parametrize("card", ["model", "dataset", "space"])
def test_card_templates_have_no_long_dash(card: str) -> None:
    assert "\u2014" not in (ROOT / "hf" / card / "README.md.in").read_text(encoding="utf-8")


# --- space ----------------------------------------------------------------------------------------


def walk(trees: dict[str, Any], x: list[float]) -> float:
    """SalaryModel.raw of hf/space/js/salary.js, in Python."""
    bounds = {"inf": math.inf, "-inf": -math.inf}
    out = float(trees["baseline"])
    for tree in trees["trees"]:
        node = tree[0]
        while len(node) != 1:
            feature, split, nan_left, left, right = node
            v = x[feature]
            if math.isnan(v):
                go_left = nan_left == 1
            elif isinstance(split, list):
                go_left = v in split
            else:
                go_left = v <= bounds.get(split, split)
            node = tree[left if go_left else right]
        out += node[0]
    return out


@pytest.mark.parametrize("loss", [{"loss": "absolute_error"}, {"loss": "quantile", "quantile": 0.9}])
def test_salary_trees_walk_like_sklearn(loss: dict[str, Any]) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(0)
    n = 600
    # Level 4 of the categorical column exists in the encoder but not in these training rows.
    cat = rng.integers(0, 4, n).astype(float)
    num = rng.normal(size=n)
    num[rng.random(n) < 0.1] = np.nan
    y = cat * 0.5 + np.nan_to_num(num) + rng.normal(scale=0.1, size=n)
    model = HistGradientBoostingRegressor(max_iter=30, categorical_features=[True, False], random_state=0, **loss)
    model.fit(np.column_stack([cat, num]), y)
    trees = json.loads(json.dumps(hf_space.hgb_trees(model, [5, 0]), allow_nan=False))

    rows = [[c, v] for c in (0.0, 1.0, 2.0, 3.0, 4.0, math.nan) for v in (-1.0, 0.0, 0.3, math.nan)]
    want = model.predict(np.array(rows))
    assert [walk(trees, r) for r in rows] == want.tolist()


def test_fields_of_row_rebuilds_the_schema_shape() -> None:
    row = {"source_id": "x", "content_hash": "h", **{k: v for k, v in FIELDS.items() if k != "extraction_valid"}}
    fields = hf_space.fields_of_row(row)
    assert fields["salary"] == {"from": 200000, "to": None, "currency": "RUB", "gross_net": None, "period": "month"}
    assert fields["experience_years"] == {"min": 3.0, "max": None}
    empty = hf_space.fields_of_row({**row, "salary_from": None, "salary_currency": None, "salary_period": None})
    assert empty["salary"] is None


# --- model ----------------------------------------------------------------------------------------


def test_model_plan_never_takes_the_rendered_gold_prompts(tmp_path: Path) -> None:
    models = tmp_path / "models"
    name = "m"
    for f in hf_model.MERGED_FILES:
        (models / "merged" / name).mkdir(parents=True, exist_ok=True)
        (models / "merged" / name / f).write_text("x", encoding="utf-8")
    (models / "merged" / name / "prompts").mkdir()
    (models / "merged" / name / "prompts" / "dev.jsonl").write_text("{}", encoding="utf-8")
    for f in hf_model.ADAPTER_FILES:
        (models / "sft" / name / "adapter").mkdir(parents=True, exist_ok=True)
        (models / "sft" / name / "adapter" / f).write_text("x", encoding="utf-8")
    (models / "gguf" / name).mkdir(parents=True)
    for q in hf_model.GGUF_TYPES:
        (models / "gguf" / name / f"{name}-{q}.gguf").write_text("x", encoding="utf-8")
    files = hf_model.plan(name, models, tmp_path / "out")
    assert not any("prompts" in src.parts for src in files.values())
    assert tmp_path / "out" / "gguf" / "m-Q4_K_M.gguf" in files
    (models / "gguf" / name / f"{name}-Q8_0.gguf").unlink()
    with pytest.raises(SystemExit, match="Q8_0"):
        hf_model.plan(name, models, tmp_path / "out")


# --- dataset --------------------------------------------------------------------------------------


class FakeClickHouse:
    def __init__(self, rows: dict[str, list[dict[str, Any]]]) -> None:
        self.rows = rows

    def query(self, sql: str, **params: str) -> str:
        if "stg_trudvsem_api" in sql:
            key = "trudvsem"
        elif "stg_trudvsem_snapshot" in sql:
            key = "trudvsem_snapshot"
        else:
            key = next(s for s in ("habr", "superjob", "telegram") if f"source = '{s}'" in sql)
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in self.rows.get(key, []))


def trudvsem_row(feed: str, vacancy_id: str, day: str, **over: Any) -> dict[str, Any]:
    return {
        "feed": feed,
        "vacancy_id": vacancy_id,
        "version_day": day,
        "published_day": "2026-09-01",
        "url": f"https://trudvsem.ru/vacancy/card/1027700000000/{vacancy_id}",
        "title": "Аналитик данных",
        "text": "SQL, Python. Звоните +7 912 345-67-89",
        "source_type": "Вакансия работодателя",
        "region": "Москва",
        "employer_name": 'ООО "Ромашка"',
        "employer_inn": "7700000000",
        "stated_salary_from": 100000.0,
        "stated_salary_to": None,
        **FIELDS,
        **over,
    }


def test_trudvsem_rows_follow_the_publication_rules() -> None:
    ch = FakeClickHouse(
        {
            "trudvsem_snapshot": [
                trudvsem_row("trudvsem_snapshot", UUID_A, "2026-10-10"),
                trudvsem_row("trudvsem_snapshot", UUID_B, "2026-10-12", source_type="Вакансия интернет ресурса"),
            ],
            "trudvsem": [
                trudvsem_row("trudvsem", UUID_A, "2026-10-10", employer_name="ИП", employer_inn="770000000000"),
            ],
        }
    )
    found: Counter[str] = Counter()
    rows = {r["vacancy_id"]: r for r in hf_dataset.trudvsem_rows(ch, "m", found)}  # type: ignore[arg-type]
    assert rows[UUID_A]["feed"] == "trudvsem"  # same day in both feeds: the API version
    assert rows[UUID_A]["employer_name"] is None  # a 12-digit INN is a person
    assert rows[UUID_A]["text"] is not None and "[phone]" in rows[UUID_A]["text"]
    assert rows[UUID_B]["text"] is None and rows[UUID_B]["title"] is None
    assert rows[UUID_B]["text_withheld"] == "reposted_from_other_site"
    assert rows[UUID_B]["skills"] == ["Python", "SQL"]  # fields stay
    assert found["phone"] == 1 and found["employer_person"] == 1
    assert rows[UUID_A]["salary_from"] == 200000 and isinstance(rows[UUID_A]["salary_from"], int)


def test_later_version_wins_across_feeds() -> None:
    ch = FakeClickHouse(
        {
            "trudvsem_snapshot": [trudvsem_row("trudvsem_snapshot", UUID_A, "2026-10-12")],
            "trudvsem": [trudvsem_row("trudvsem", UUID_A, "2026-10-01")],
        }
    )
    (row,) = hf_dataset.trudvsem_rows(ch, "m", Counter())  # type: ignore[arg-type]
    assert row["feed"] == "trudvsem_snapshot"


def test_fields_only_rows_carry_no_id_text_or_link() -> None:
    post = {"channel": "datasciencejobs", "version_day": "2026-10-10", "published_day": "2025-03-02", **FIELDS}
    ch = FakeClickHouse({"telegram": [post], "superjob": [{**post, "channel": None}]})
    (tg,) = hf_dataset.fields_only_rows(ch, "m", "telegram")  # type: ignore[arg-type]
    (sj,) = hf_dataset.fields_only_rows(ch, "m", "superjob")  # type: ignore[arg-type]
    assert tg["channel"] == "datasciencejobs" and tg["_month"] == "2025-03"
    assert "channel" not in sj
    assert set(hf_dataset.SCHEMAS) == {"trudvsem", "superjob"}  # Telegram leaves as aggregates only
    for schema in (hf_dataset.TELEGRAM_SCHEMA, hf_dataset.SCHEMAS["superjob"]):
        assert not {"text", "title", "url", "source_id", "vacancy_id", "employer_name"} & set(schema.names)


def test_telegram_aggregates_pool_channels_by_week_and_drop_small_cells() -> None:
    def post(day: str, channel: str = "datasciencejobs", **over: Any) -> dict[str, Any]:
        return {"channel": channel, "version_day": "2026-10-12", "published_day": day, **FIELDS, **over}

    # 2026-10-05 is a Monday, 2026-10-11 the Sunday of the same week; 1970-01-01 means no date.
    posts = [post("2026-10-05") for _ in range(hf_dataset.MIN_CELL - 2)]
    posts += [post("2026-10-11", channel="ml_jobs"), post("2026-10-07", channel="ml_jobs")]
    posts += [post("1970-01-01", role="qa"), post("2026-10-06", extraction_valid=False)]
    rows = hf_dataset.fields_only_rows(FakeClickHouse({"telegram": posts}), "m", "telegram")  # type: ignore[arg-type]
    cells, dropped = hf_dataset.telegram_aggregates(rows)
    week = {"week": date(2026, 10, 5)}
    assert week | {"dimension": "role", "value": "data_scientist", "posts": 5, "posts_in_week": 6} in cells
    assert week | {"dimension": "skill", "value": "SQL", "posts": 5, "posts_in_week": 6} in cells
    assert not any(c["value"] == "qa" for c in cells)  # one post, and in the week of its raw version
    assert "channel" not in hf_dataset.TELEGRAM_SCHEMA.names and all("channel" not in c for c in cells)
    assert dropped >= 1
    assert all(c["posts"] >= hf_dataset.MIN_CELL for c in cells)
    # Pooled, the five data_scientist posts of two channels reach the floor; per channel neither does.
    assert hf_dataset.role_kept(rows, lambda r: (hf_dataset.week_of(r),)) == 5
    assert hf_dataset.role_kept(rows, lambda r: (hf_dataset.week_of(r), r["channel"])) == 0


def test_habr_aggregates_drop_small_cells() -> None:
    rows = [{"version_day": "2026-10-10", "published_day": "2026-10-05", **FIELDS} for _ in range(hf_dataset.MIN_CELL)]
    rows.append({"version_day": "2026-10-10", "published_day": "2026-10-06", **FIELDS, "role": "qa", "skills": []})
    cells, dropped, vacancies = hf_dataset.habr_aggregates(FakeClickHouse({"habr": rows}), "m")  # type: ignore[arg-type]
    assert vacancies == hf_dataset.MIN_CELL + 1
    assert {"month": "2026-10", "dimension": "role", "value": "data_scientist"} | {
        "vacancies": hf_dataset.MIN_CELL,
        "vacancies_in_month": hf_dataset.MIN_CELL + 1,
    } in cells
    assert not any(c["value"] == "qa" for c in cells)
    assert dropped >= 1
    assert all(c["vacancies"] >= hf_dataset.MIN_CELL for c in cells)


# --- scan -----------------------------------------------------------------------------------------


def test_scan_row_checks_identifier_shapes_and_free_text() -> None:
    url = f"https://trudvsem.ru/vacancy/card/1027700000000/{UUID_A}"
    assert hf_scan.scan_row({"vacancy_id": UUID_A, "url": url, "text": "Python"}) == Counter()
    assert hf_scan.scan_row({"vacancy_id": UUID_A, "url": url, "text": "пишите на hr@example.ru"})["email"] == 1
    ogrnip = f"https://trudvsem.ru/vacancy/card/312770000000000/{UUID_A}"
    assert hf_scan.scan_row({"vacancy_id": UUID_A, "url": ogrnip})["url_shape"] == 1
    assert hf_scan.scan_row({"vacancy_id": "79123456789"})["vacancy_id_shape"] == 1


def test_scan_refuses_an_unknown_kind_of_file(tmp_path: Path) -> None:
    (tmp_path / "notes.csv").write_text("a,b", encoding="utf-8")
    with pytest.raises(SystemExit, match="no rule"):
        hf_scan.scan(tmp_path)


def front_matter(card: str) -> dict[str, Any]:
    import yaml

    text = (ROOT / "hf" / card / "README.md.in").read_text(encoding="utf-8")
    head, sep, _ = text.removeprefix("---\n").partition("\n---\n")
    assert text.startswith("---\n") and sep
    meta: dict[str, Any] = yaml.safe_load(head)
    return meta


def test_card_headers_match_the_layout() -> None:
    from radar.site.hf_common import load_config

    repos = load_config()["repos"]
    space = front_matter("space")
    assert space["sdk"] == "static" and space["app_file"] == "index.html"
    # wllama runs several threads only on a cross-origin isolated page
    assert space["custom_headers"]["cross-origin-embedder-policy"] == "require-corp"
    assert space["custom_headers"]["cross-origin-opener-policy"] == "same-origin"
    assert space["models"] == [repos["model"]] and space["datasets"] == [repos["dataset"]]
    assert len(space["short_description"]) <= 60
    model = front_matter("model")
    assert model["base_model"] == load_config()["model"]["base"] and model["datasets"] == [repos["dataset"]]
    dataset = front_matter("dataset")
    configs = {c["config_name"]: c["data_files"] for c in dataset["configs"]}
    assert configs == {
        "trudvsem": "data/trudvsem/*.parquet",
        "superjob": "data/superjob/*.parquet",
        "telegram_aggregates": "aggregates/telegram_weekly.parquet",
        "habr_aggregates": "aggregates/habr_monthly.parquet",
    }
    assert set(hf_dataset.ROW_SOURCES) | {"telegram_aggregates", "habr_aggregates"} == set(configs)


def test_published_card_has_values_only(tmp_path: Path) -> None:
    template = (
        "| a | b |\n|---|---|\n| <!--n:a.json:metrics.f1.value:.3f|ru-->0<!--/n--> | "
        "<!--pending:b.json:metrics.t.value:.2f|ru-->ждёт<!--/pending--> |\n"
    )
    card = render_card(template, results_dir(tmp_path, {"a.json": {"metrics": {"f1": {"value": 0.9041}}}}))
    assert card.published == "| a | b |\n|---|---|\n| 0,904 | ждёт |\n"
