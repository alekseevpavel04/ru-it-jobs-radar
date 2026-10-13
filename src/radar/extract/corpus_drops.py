"""Telegram versions the corpus run never extracts, with the rule that drops them, into ClickHouse
`extract.vacancy_dropped` (infra/clickhouse/003_extract_vacancy_dropped.sql), so the marts can tell a
post the run has not reached yet from a post it never will (dbt mart_extraction_coverage).

The versions are those `corpus_run run` enumerates for Telegram: the latest stored version of every
vacancy post with one vacancy (ClickHouseStore.telegram_posts), checked by the same
radar.extract.telegram_corpus.drop_reason with the same configuration: a digest, a post with almost
no text of its own (link_stub), a post outside its channel's language (channel_lang). The run itself
is not touched; this only repeats its enumeration.

Each check is one insert with its own check_id, after which the rows of earlier checks are deleted;
readers take the rows of the latest check_id, so a dbt build during a check sees the previous one.

    python -m radar.extract.corpus_run dropped
"""

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from radar.extract.telegram_corpus import channel_langs, drop_reason
from radar.extract.telegram_corpus import load_config as telegram_config
from radar.results import write_result
from radar.stream.clickhouse import DDL_DIR, statements

TABLE = "extract.vacancy_dropped"
DDL = DDL_DIR / "003_extract_vacancy_dropped.sql"


class Posts(Protocol):
    def telegram_posts(self, since: Any, part: int, parts: int) -> list[dict[str, Any]]: ...


class Query(Protocol):
    def query(self, sql: str, **params: str) -> str: ...


def rules_sha256(min_content_chars: int, langs: Mapping[str, str]) -> str:
    """What decides a drop besides the code: the threshold and the channel languages."""
    rules = {"min_content_chars": min_content_chars, "channel_langs": dict(sorted(langs.items()))}
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode("utf-8")).hexdigest()


def dropped_rows(
    store: Posts, parts: int, min_content_chars: int, langs: Mapping[str, str], check_id: str, checked_at: str
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Every dropped version with its reason, and the count of posts checked per outcome."""
    rules = rules_sha256(min_content_chars, langs)
    rows: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    for part in range(parts):
        for post in store.telegram_posts(None, part, parts):
            source_id = str(post["source_id"])
            why = drop_reason(post, min_content_chars, langs.get(source_id.rsplit("/", 1)[0]))
            outcomes[why or "kept"] += 1
            if why is None:
                continue
            rows.append(
                {
                    "source": "telegram",
                    "source_id": source_id,
                    "content_hash": str(post["content_hash"]),
                    "dt": str(post["dt"]),
                    "reason": why,
                    "rules_sha256": rules,
                    "check_id": check_id,
                    "checked_at": checked_at,
                }
            )
    return rows, outcomes


def ensure_table(ch: Query) -> None:
    for statement in statements(DDL.read_text(encoding="utf-8")):
        ch.query(statement)


def replace_check(ch: Query, rows: Sequence[Mapping[str, Any]], check_id: str) -> None:
    """One insert for the whole check (atomic below max_insert_block_size), then the older checks go."""
    if rows:
        lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        ch.query(f"INSERT INTO {TABLE} FORMAT JSONEachRow\n{lines}\n")
    ch.query(
        f"ALTER TABLE {TABLE} DELETE WHERE check_id != {{check:String}} SETTINGS mutations_sync = 1", check=check_id
    )


def metrics(outcomes: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    total = sum(outcomes.values())
    out: dict[str, dict[str, Any]] = {"telegram_posts_checked": {"value": total}}
    for reason, n in sorted(outcomes.items()):
        out[f"telegram_{reason}"] = {"value": round(n / total, 4) if total else 0, "n": n, "denominator": total}
    return out


def run(store: Posts, ch: Query, parts: int, now: datetime | None = None) -> dict[str, dict[str, Any]]:
    moment = now or datetime.now(UTC)
    check_id = moment.strftime("%Y%m%dT%H%M%S")
    cfg = telegram_config()
    langs = channel_langs()
    rows, outcomes = dropped_rows(
        store, parts, cfg.min_content_chars, langs, check_id, moment.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    )
    ensure_table(ch)
    replace_check(ch, rows, check_id)
    return metrics(outcomes)


def write_report(result: Mapping[str, dict[str, Any]]) -> Iterable[str]:
    path = write_result(
        "extract/corpus_dropped.json",
        dict(result),
        source_script="src/radar/extract/corpus_drops.py",
        table=TABLE,
        method="latest version of every Telegram vacancy post with one vacancy, as corpus_run run enumerates "
        "them; drop_reason of radar.extract.telegram_corpus with configs/telegram_sample.toml and the "
        "channel languages; kept is what the run extracts",
    )
    yield str(path)
    for key, value in result.items():
        shown = f"{value['n']} of {value['denominator']}" if "denominator" in value else ""
        yield f"{key:40} {value['value']:>10} {shown}"
