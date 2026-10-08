"""Extraction schema: the contract shared by teachers, the gold set, the regex baseline,
fine-tuned students, the API and the Space.

Rules that the JSON Schema cannot express are enforced here and count towards "valid JSON":
a salary has at least one bound and `from <= to`, the same for experience. Definitions of every field
and of null are in docs/annotation_guide.md.

    python -m radar.extract.schema --json-schema            # print
    python -m radar.extract.schema --json-schema --out configs/extraction_schema.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

SCHEMA_VERSION = 1

Role = Literal[
    "ml_engineer",
    "data_scientist",
    "mlops",
    "data_engineer",
    "data_analyst",
    "business_analyst",
    "system_analyst",
    "backend",
    "frontend",
    "fullstack",
    "mobile",
    "software_engineer",
    "developer_1c",
    "embedded",
    "devops",
    "qa",
    "sysadmin",
    "support",
    "security",
    "designer",
    "manager",
    "other_it",
    "non_it",
]
Grade = Literal["intern", "junior", "middle", "senior", "lead"]
Currency = Literal["RUB", "USD", "EUR", "KZT", "BYN", "UZS", "CNY", "GBP"]
GrossNet = Literal["gross", "net"]
Period = Literal["month", "year", "day", "hour"]
WorkFormat = Literal["office", "hybrid", "remote"]
EmploymentType = Literal["labor_contract", "gph", "self_employed", "internship"]
Lang = Literal["ru", "en", "other"]

ROLES: tuple[str, ...] = get_args(Role)
GRADES: tuple[str, ...] = get_args(Grade)

Item = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


class Salary(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    from_: int | None = Field(alias="from", gt=0)
    to: int | None = Field(gt=0)
    currency: Currency | None
    gross_net: GrossNet | None
    period: Period | None

    @model_validator(mode="after")
    def _bounds(self) -> "Salary":
        if self.from_ is None and self.to is None:
            raise ValueError("salary needs at least one of from, to; no amount means salary is null")
        if self.from_ is not None and self.to is not None and self.from_ > self.to:
            raise ValueError("salary from > to")
        return self


class Experience(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min: float | None = Field(ge=0, le=50)
    max: float | None = Field(ge=0, le=50)

    @model_validator(mode="after")
    def _bounds(self) -> "Experience":
        if self.min is None and self.max is None:
            raise ValueError("experience_years needs at least one of min, max; otherwise it is null")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("experience_years min > max")
        return self


class VacancyFields(BaseModel):
    """Every key is required: null (or an empty list) is an explicit "not stated in the text"."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    role: Role | None
    grade: Grade | None
    skills: list[Item] = Field(max_length=60)
    salary: Salary | None
    work_format: WorkFormat | None
    cities: list[Item] = Field(max_length=30)
    experience_years: Experience | None
    employment_type: EmploymentType | None
    lang: Lang


def parse(text: str) -> VacancyFields:
    """Parse a raw model answer; raises ValidationError on broken JSON or a schema violation."""
    return VacancyFields.model_validate_json(text, by_alias=True)


def try_parse(text: str | None) -> VacancyFields | None:
    if text is None:
        return None
    try:
        return parse(text)
    except ValidationError:
        return None


def dump(fields: VacancyFields) -> dict[str, Any]:
    return fields.model_dump(mode="json", by_alias=True)


def json_schema() -> dict[str, Any]:
    return VacancyFields.model_json_schema(by_alias=True, mode="validation")


def schema_text() -> str:
    return json.dumps(json_schema(), ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m radar.extract.schema", description=__doc__.splitlines()[0])
    ap.add_argument("--json-schema", action="store_true", help="print the JSON Schema for constrained decoding")
    ap.add_argument("--out", type=Path, help="write the schema to this file instead of stdout")
    args = ap.parse_args(argv)
    if not args.json_schema:
        ap.print_help()
        return 2
    text = schema_text()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
