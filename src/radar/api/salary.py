"""The salary model behind POST /salary and the MCP tool estimate_salary.

The only coupling to the model package is `SalaryEstimator.estimate`, one call to
`radar.models.salary.predict`: when the model moves to the student's fields and gains its own
one-row predict, that call is the line to change. Everything else here is the service's contract.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from radar.extract.schema import VacancyFields, dump
from radar.models import salary as salary_model
from radar.models.salary_features import OutOfScope

__all__ = ["DEFAULT_MODEL_PATH", "Estimator", "OutOfScope", "SalaryEstimator", "UnknownSource"]

DEFAULT_MODEL_PATH = salary_model.MODEL_DIR / salary_model.ARTIFACT


class UnknownSource(ValueError):
    """The model has not seen this source in training, so it has no market to estimate for."""


class Estimator(Protocol):
    @property
    def version(self) -> str: ...

    def estimate(self, fields: VacancyFields, source: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SalaryEstimator:
    model: salary_model.SalaryModel

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "SalaryEstimator":
        return cls(salary_model.load_model(path))

    @property
    def version(self) -> str:
        return self.model.version

    @property
    def sources(self) -> tuple[str, ...]:
        return self.model.encoder.levels["source"]

    def estimate(self, fields: VacancyFields, source: str) -> dict[str, Any]:
        """Monthly RUB estimate as `source`'s market pays; raises OutOfScope for a non-IT role.

        An unseen source would reach the boosting models as a missing category and come back as an
        estimate for no market in particular, so it is refused instead."""
        if source not in self.sources:
            raise UnknownSource(f"the salary model knows the sources {', '.join(self.sources)}, not {source}")
        return salary_model.predict(dump(fields), source, model=self.model)
