"""Request and response bodies of the API. /salary and /match are fixed here ahead of their
models (7.4) so the MCP server and the end-to-end test can target them."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from radar.extract.schema import Currency, GrossNet, Period, VacancyFields

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


class SalaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: VacancyFields


class SalaryResponse(BaseModel):
    low: float
    median: float
    high: float
    coverage: float = Field(description="nominal coverage of the [low, high] interval")
    currency: Currency
    period: Period
    gross_net: GrossNet | None
    model: str


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resume_text: VacancyText
    k: int = Field(default=10, ge=1, le=50)


class MatchItem(BaseModel):
    vacancy_entity_id: str
    score: float
    source: str
    url: str | None


class MatchResponse(BaseModel):
    items: list[MatchItem]
    model: str


class Problem(BaseModel):
    detail: str
