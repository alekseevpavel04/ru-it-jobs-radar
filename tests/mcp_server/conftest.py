import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor  # type: ignore[import-untyped]

from radar.api.salary import SalaryEstimator
from radar.models.salary import CellMedian, SalaryModel
from radar.models.salary_features import Encoder, feature_row

# Constant models, so the expected answers follow from these numbers: the point is 150 000, the
# interval 100 000..220 000 widened by the conformal margin, rounded to 1000 by the model package.
POINT, LOWER, UPPER, FROM, TO, MARGIN = 150_000, 100_000, 220_000, 130_000, 170_000, 0.05


def stub_model() -> SalaryModel:
    rows = [
        feature_row({"role": "backend", "grade": "middle", "cities": ["Москва"]}, "trudvsem"),
        feature_row({"role": "qa", "grade": "junior", "cities": []}, "habr"),
    ]
    encoder = Encoder.fit(rows, min_skill_count=1)
    X = encoder.tree_matrix(encoder.encode(rows))

    def constant(value: float) -> DummyRegressor:
        return DummyRegressor(strategy="constant", constant=math.log(value)).fit(X, np.zeros(len(rows)))

    return SalaryModel(
        encoder=encoder,
        point=constant(POINT),
        lower=constant(LOWER),
        upper=constant(UPPER),
        bound_from=constant(FROM),
        bound_to=constant(TO),
        margin=MARGIN,
        level=0.8,
        cells=CellMedian(0.8, 1).fit(pd.DataFrame(rows), np.log([POINT, 90_000])),
        version="salary-stub",
        trained_at="2026-10-13T00:00:00+00:00",
    )


@pytest.fixture(scope="session")
def model_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("salary") / "salary.pkl"
    path.write_bytes(pickle.dumps(stub_model()))
    return path


@pytest.fixture
def estimator() -> SalaryEstimator:
    return SalaryEstimator(stub_model())


@pytest.fixture
def expected() -> dict[str, Any]:
    """What the stub model answers for any IT vacancy, before n_similar and the labels."""
    return {
        "point": POINT,
        "low": round(LOWER * math.exp(-MARGIN), -3),
        "high": round(UPPER * math.exp(MARGIN), -3),
        "typical_from": FROM,
        "typical_to": TO,
        "level": 0.8,
    }
