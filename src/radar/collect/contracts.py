"""What each source must send for a collector to trust it.

Models check the fields the collectors depend on and let everything else through (`extra="allow"`),
so a new field at the source is not an error, while a missing or retyped required field is: the row
goes to the rejects log instead of the raw layer.
"""

import json
import re
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field, StrictBool, StrictInt

from radar.collect.envelope import parse_source_time

SourceTime = Annotated[AwareDatetime, BeforeValidator(lambda v: parse_source_time(v) if isinstance(v, str) else v)]
DigitString = Annotated[str, Field(pattern=r"^\d+$")]
NumberString = Annotated[str, Field(pattern=r"^\d+(?:\.\d+)?$")]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# trudvsem API: GET /api/v1/vacancies


class TrudvsemRegion(_Open):
    region_code: DigitString
    name: str


class TrudvsemCompany(_Open):
    name: str
    inn: DigitString | None = None
    ogrn: DigitString | None = None
    companycode: str | None = None


class TrudvsemApiVacancy(_Open):
    id: str = Field(min_length=1)
    source: str
    region: TrudvsemRegion
    company: TrudvsemCompany
    creation_date: SourceTime = Field(alias="creation-date")
    date_modify: SourceTime
    job_name: str = Field(alias="job-name", min_length=1)
    vac_url: str | None = None
    salary_min: StrictInt | None = None
    salary_max: StrictInt | None = None


class TrudvsemApiItem(_Open):
    vacancy: dict[str, Any]


class TrudvsemApiResults(_Open):
    vacancies: list[TrudvsemApiItem] = []


class TrudvsemApiMeta(_Open):
    total: StrictInt
    limit: StrictInt


class TrudvsemApiResponse(_Open):
    status: Literal["200"]
    meta: TrudvsemApiMeta
    # An empty page comes without `vacancies`.
    results: TrudvsemApiResults = TrudvsemApiResults()


# trudvsem open-data XML snapshot: one <vacancy> turned into a dict, every leaf a string


class TrudvsemSnapshotCompany(_Open):
    inn: DigitString | None = None
    ogrn: DigitString | None = None
    companyCode: str | None = None


class TrudvsemSnapshotVacancy(_Open):
    """Required is only what identifies and dates a record: a snapshot row rejected here is lost for
    that day, so everything else is optional but type-checked when present."""

    id: str = Field(min_length=1)
    vacancyName: str = Field(min_length=1)
    creationDate: SourceTime
    dateModify: SourceTime
    deleted: Literal["true", "false"]
    datePublished: SourceTime | None = None
    changeTime: SourceTime | None = None
    status: str | None = None
    regionName: str | None = None
    stateRegionCode: DigitString | None = None
    fullCompanyName: str | None = None
    company: TrudvsemSnapshotCompany | None = None
    sourceType: str | None = None
    salaryMin: NumberString | None = None
    salaryMax: NumberString | None = None


# Habr Career: GET /api/frontend/vacancies and the card page /vacancies/<id>


class HabrCompany(_Open):
    id: StrictInt | None = None
    title: str
    alias_name: str | None = None


class HabrPublished(_Open):
    date: SourceTime


class HabrSalary(_Open):
    from_: StrictInt | None = Field(default=None, alias="from")
    to: StrictInt | None = None
    currency: str | None = None


class HabrTitled(_Open):
    title: str
    href: str | None = None


class HabrListItem(_Open):
    id: StrictInt
    href: str
    title: str = Field(min_length=1)
    publishedDate: HabrPublished
    company: HabrCompany
    salary: HabrSalary | None = None
    qualification: str | None = None
    divisions: list[HabrTitled] = []
    skills: list[HabrTitled] = []
    locations: list[HabrTitled] | None = None
    remoteWork: StrictBool | None = None
    archived: StrictBool


class HabrListMeta(_Open):
    currentPage: StrictInt
    perPage: StrictInt
    totalPages: StrictInt
    totalResults: StrictInt


class HabrListResponse(_Open):
    list_: list[dict[str, Any]] = Field(alias="list")
    meta: HabrListMeta


class HabrCardVacancy(HabrListItem):
    description: str


class HabrJobPosting(_Open):
    type_: Literal["JobPosting"] = Field(alias="@type")
    title: str
    description: str
    datePosted: str
    validThrough: str | None = None


class HabrCard(BaseModel):
    vacancy: HabrCardVacancy
    company: dict[str, Any]
    job_posting: HabrJobPosting


_LD_JSON = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.DOTALL)
_SSR_STATE = re.compile(r'<script type="application/json" data-ssr-state="true">(.*?)</script>', re.DOTALL)


def parse_habr_card(html: str) -> HabrCard:
    """Pull the SSR state and the JSON-LD JobPosting out of a card page; ValueError if either is gone."""
    ld = _LD_JSON.search(html)
    ssr = _SSR_STATE.search(html)
    if ld is None or ssr is None:
        raise ValueError("card page without JSON-LD or SSR state")
    posting = json.loads(ld.group(1))
    state = json.loads(ssr.group(1))
    return HabrCard.model_validate(
        {"vacancy": state.get("vacancy"), "company": state.get("company") or {}, "job_posting": posting}
    )
