"""Gold and prediction files.

Gold, `data/gold/<split>.jsonl`, one vacancy per line:
    {"id": "<sample id fixed when the samples were drawn>", "source": "trudvsem_snapshot", "fields": {...schema...}}
`source` is optional and only splits the report by source. Extra keys (annotator notes, input text)
are kept and ignored here.

Predictions, `data/predictions/<split>/<system>.jsonl`, one line per gold id:
    {"id": "...", "output": "<the system's raw answer as a string>"}
`output` is the raw text, not parsed JSON, so a broken answer stays visible and counts as invalid;
null means the system produced nothing. Extra keys (latency, model version) are allowed.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from radar.extract.schema import VacancyFields, try_parse


class GoldRecord(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    id: str = Field(min_length=1)
    source: str | None = None
    fields: VacancyFields


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    id: str = Field(min_length=1)
    output: str | None


@dataclass(frozen=True)
class Aligned:
    gold: list[GoldRecord]
    outputs: list[str | None]
    parsed: list[VacancyFields | None]
    missing: int
    extra: int


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lines(path: Path) -> list[tuple[int, dict[str, object]]]:
    out = []
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            out.append((no, json.loads(line)))
    return out


def load_gold(path: Path) -> list[GoldRecord]:
    records: list[GoldRecord] = []
    seen: set[str] = set()
    for no, row in _lines(path):
        try:
            rec = GoldRecord.model_validate(row)
        except ValidationError as e:
            raise ValueError(f"{path}:{no}: gold row does not match the schema\n{e}") from e
        if rec.id in seen:
            raise ValueError(f"{path}:{no}: duplicate gold id {rec.id}")
        seen.add(rec.id)
        records.append(rec)
    if not records:
        raise ValueError(f"{path}: no gold rows")
    return records


def load_predictions(path: Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for no, row in _lines(path):
        rec = PredictionRecord.model_validate(row)
        if rec.id in out:
            raise ValueError(f"{path}:{no}: duplicate prediction id {rec.id}")
        out[rec.id] = rec.output
    return out


def align(gold: list[GoldRecord], predictions: dict[str, str | None]) -> Aligned:
    """Predictions in gold order; a gold id without a prediction scores as an invalid answer."""
    outputs = [predictions.get(g.id) for g in gold]
    gold_ids = {g.id for g in gold}
    return Aligned(
        gold=gold,
        outputs=outputs,
        parsed=[try_parse(o) for o in outputs],
        missing=sum(g.id not in predictions for g in gold),
        extra=sum(pid not in gold_ids for pid in predictions),
    )
