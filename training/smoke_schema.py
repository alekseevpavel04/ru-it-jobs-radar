"""JSON schema of the draft extraction fields, used to test constrained decoding in the smoke run."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Salary(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: int | None = Field(alias="from")
    to: int | None
    currency: Literal["RUB", "USD", "EUR", "KZT", "BYN", "OTHER"] | None
    gross_net: Literal["gross", "net", "unknown"]
    period: Literal["month", "year", "hour"] | None


class Experience(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: int | None
    max: int | None


class VacancyFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["ml_engineer", "data_scientist", "data_analyst", "data_engineer", "mlops", "backend", "other"]
    grade: Literal["intern", "junior", "middle", "senior", "lead"] | None
    skills: list[str]
    salary: Salary | None
    work_format: Literal["office", "hybrid", "remote"] | None
    cities: list[str]
    experience_years: Experience | None
    employment_type: Literal["labor_contract", "gph", "self_employed", "internship"] | None


def json_schema() -> dict[str, Any]:
    return VacancyFields.model_json_schema(by_alias=True)


def is_valid(text: str) -> bool:
    try:
        VacancyFields.model_validate_json(text, by_alias=True)
    except ValueError:
        return False
    return True
