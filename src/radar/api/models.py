"""Request and response bodies of the API, shared with the MCP server. /match is fixed here
ahead of its model."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from radar.extract.schema import VacancyFields

# A bound on request size: the student reads only its first max_input_tokens anyway, while the regex
# and the tokenizer call scan the whole text.
MAX_TEXT_CHARS = 30_000

VacancyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)]


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: VacancyText


class ExtractResponse(BaseModel):
    fields: VacancyFields
    backend: Literal["student", "regex"]
    model: str = Field(description="model id the student server reported, or the regex baseline's version")
    schema_version: int
    prompt_sha256: str | None
    truncated_input: bool = Field(description="the input was cut to the student's max_input_tokens")
    fallback_reason: str | None = Field(description="why the regex answered for a configured student")
    cached: bool


SalarySource = Literal["trudvsem", "habr", "superjob"]


class SalaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: VacancyFields = Field(description="extraction fields, e.g. the fields of POST /extract")
    source: SalarySource = Field(
        default="trudvsem",
        description="whose market the estimate is for; trudvsem holds most of the training vacancies",
    )


class SalaryResponse(BaseModel):
    point: int = Field(description="median estimate, RUB per month, rounded to 1000")
    low: int
    high: int
    level: float = Field(description="nominal coverage of [low, high], split conformal (CQR) on held-out vacancies")
    typical_from: int = Field(description="estimated lower bound of the range a vacancy like this states")
    typical_to: int = Field(description="estimated upper bound of the range a vacancy like this states")
    n_similar: int = Field(
        description="training vacancies with the same role bucket, grade and region group; 0 means the "
        "estimate rests on coarser patterns only"
    )
    source: SalarySource
    currency: Literal["RUB"] = "RUB"
    period: Literal["month"] = "month"
    basis: Literal["as_stated"] = Field(
        default="as_stated", description="amounts as the sources state them: gross or net is not specified"
    )
    model: str = Field(description="model version")


# Values of the marts' dimensions (dbt/models/marts/_marts.yml).
Grade = Literal["intern", "junior", "middle", "senior", "lead", "unknown"]
RegionGroup = Literal["moscow", "spb", "other", "remote", "unknown"]
WorkFormat = Literal["office", "hybrid", "remote", "unknown"]


class VacancyItem(BaseModel):
    """One vacancy as the services may show it: a title and an employer name only where the source's
    terms allow publishing them (trudvsem open data), null otherwise."""

    vacancy_entity_id: str
    source: str
    title: str | None
    employer_name: str | None
    role_bucket: str
    grade: str
    region_group: str
    work_format: str
    salary_from: float | None = Field(description="RUB per month as stated")
    salary_to: float | None = Field(description="RUB per month as stated")
    url: str | None
    first_seen: str
    last_seen: str


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resume_text: VacancyText
    k: int = Field(default=10, ge=1, le=50)


class MatchItem(VacancyItem):
    rank: int = Field(ge=1)
    score: float = Field(description="relevance of the vacancy to the resume, higher is closer")


class MatchResponse(BaseModel):
    items: list[MatchItem]
    model: str


class Problem(BaseModel):
    detail: str
