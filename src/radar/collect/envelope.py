"""The raw-layer envelope: one row of `data/raw/<source>/dt=YYYY-MM-DD/part-*.parquet`."""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION: Literal[1] = 1

Source = Literal["trudvsem", "trudvsem_snapshot", "habr", "telegram", "superjob"]


class RawRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = SCHEMA_VERSION
    source: Source
    source_id: str = Field(min_length=1)
    fetched_at: AwareDatetime
    published_at: AwareDatetime | None
    modified_at: AwareDatetime | None
    url: str | None
    title: str | None
    text: str | None
    payload: dict[str, Any]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pii_removed: dict[str, int]
    is_vacancy: bool | None = None
    multi_vacancy: bool | None = None

    @field_validator("fetched_at", "published_at", "modified_at")
    @classmethod
    def _to_utc(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(UTC) if value is not None else None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.source, self.source_id, self.content_hash


_SPACE = re.compile(r"\s+")


def _normalize(text: str | None) -> str:
    return _SPACE.sub(" ", text or "").strip().lower()


def content_hash(title: str | None, text: str | None, payload: Mapping[str, Any], ignore: Iterable[str] = ()) -> str:
    """Hash of what the vacancy says: normalized title and text plus payload without bookkeeping fields.

    `ignore` lists payload keys that change without the vacancy changing (modification stamps), so
    an untouched vacancy keeps its hash from one day to the next.
    """
    skip = set(ignore)
    body = {k: v for k, v in payload.items() if k not in skip}
    blob = json.dumps(
        [_normalize(title), _normalize(text), body], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def parse_source_time(value: str | None) -> datetime | None:
    """Timestamps as sources write them: `2026-06-05T08:37:49+0300`, `...+03:00`, a bare date, or
    Unix milliseconds (`1771493777331`, seen in a few trudvsem snapshot records).

    A bare date has no time of day; it is read as midnight Moscow time, the sources' own zone.
    """
    if not value:
        return None
    value = value.strip()
    if re.fullmatch(r"\d{13}", value):
        return datetime.fromtimestamp(int(value) / 1000, tz=UTC)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return datetime.fromisoformat(value + "T00:00:00+03:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp without zone: {value!r}")
    return parsed
