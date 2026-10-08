import math
import random

import numpy as np
import pytest

from radar.eval.bootstrap import bootstrap, paired_bootstrap, resample_weights
from radar.eval.metrics import score
from radar.extract.schema import VacancyFields

ROLES = ["backend", "frontend", "qa", "devops", "data_analyst"]
N = 300
SEED = 7


def vf(role: str | None) -> VacancyFields:
    return VacancyFields(
        role=role,  # type: ignore[arg-type]
        grade=None,
        skills=[],
        salary=None,
        work_format=None,
        cities=[],
        experience_years=None,
        employment_type=None,
        lang="ru",
    )


def system(gold: list[str], accuracy: float, rng: random.Random) -> list[VacancyFields | None]:
    out: list[VacancyFields | None] = []
    for g in gold:
        right = rng.random() < accuracy
        out.append(vf(g if right else rng.choice([r for r in ROLES if r != g])))
    return out


@pytest.fixture(scope="module")
def gold_roles() -> list[str]:
    rng = random.Random(1)
    return [rng.choice(ROLES) for _ in range(N)]


def test_weights_resample_every_row_count() -> None:
    w = resample_weights(50, 1000, SEED)
    assert w.shape == (1000, 50)
    assert np.all(w.sum(axis=1) == 50)
    assert np.array_equal(w, resample_weights(50, 1000, SEED))
    assert not np.array_equal(w, resample_weights(50, 1000, SEED + 1))


def test_interval_contains_point(gold_roles: list[str]) -> None:
    gold = [vf(r) for r in gold_roles]
    stats = score(gold, system(gold_roles, 0.8, random.Random(2)), 0.01)
    iv = bootstrap(stats, resample_weights(N, 1000, SEED), 0.95)["role_accuracy"]
    assert iv is not None
    assert iv.low < iv.value < iv.high
    assert 0.7 < iv.value < 0.9


def test_same_system_gives_zero_difference(gold_roles: list[str]) -> None:
    gold = [vf(r) for r in gold_roles]
    stats = score(gold, system(gold_roles, 0.7, random.Random(3)), 0.01)
    iv = paired_bootstrap(stats, stats, resample_weights(N, 1000, SEED), 0.95)["role_f1"]
    assert iv is not None
    assert (iv.value, iv.low, iv.high) == (0.0, 0.0, 0.0)


def test_equally_good_systems_interval_covers_zero(gold_roles: list[str]) -> None:
    gold = [vf(r) for r in gold_roles]
    a = score(gold, system(gold_roles, 0.7, random.Random(4)), 0.01)
    b = score(gold, system(gold_roles, 0.7, random.Random(5)), 0.01)
    for name in ("role_accuracy", "macro_f1"):
        iv = paired_bootstrap(a, b, resample_weights(N, 1000, SEED), 0.95)[name]
        assert iv is not None
        assert iv.low < 0 < iv.high, name


def test_clearly_better_system_interval_above_zero(gold_roles: list[str]) -> None:
    gold = [vf(r) for r in gold_roles]
    good = score(gold, system(gold_roles, 0.9, random.Random(6)), 0.01)
    weak = score(gold, system(gold_roles, 0.6, random.Random(7)), 0.01)
    iv = paired_bootstrap(good, weak, resample_weights(N, 1000, SEED), 0.95)["role_accuracy"]
    assert iv is not None
    assert 0 < iv.low < iv.value < iv.high
    reverse = paired_bootstrap(weak, good, resample_weights(N, 1000, SEED), 0.95)["role_accuracy"]
    assert reverse is not None and reverse.high < 0


def test_paired_is_narrower_than_independent(gold_roles: list[str]) -> None:
    """Both systems fail on the same hard rows: pairing removes that shared noise."""
    rng = random.Random(8)
    hard = {i for i in range(N) if rng.random() < 0.3}
    gold = [vf(r) for r in gold_roles]

    def answers(extra_errors: float, seed: int) -> list[VacancyFields | None]:
        r = random.Random(seed)
        out: list[VacancyFields | None] = []
        for i, g in enumerate(gold_roles):
            wrong = i in hard or r.random() < extra_errors
            out.append(vf("qa" if wrong and g != "qa" else "backend" if wrong else g))
        return out

    a = score(gold, answers(0.05, 9), 0.01)
    b = score(gold, answers(0.10, 10), 0.01)
    w = resample_weights(N, 2000, SEED)
    paired = paired_bootstrap(a, b, w, 0.95)["role_accuracy"]
    ia = bootstrap(a, w, 0.95)["role_accuracy"]
    ib = bootstrap(b, resample_weights(N, 2000, SEED + 1), 0.95)["role_accuracy"]
    assert paired is not None and ia is not None and ib is not None
    # independent intervals would combine roughly as the root of summed squares
    assert paired.high - paired.low < 0.7 * math.hypot(ia.high - ia.low, ib.high - ib.low)


def test_undefined_metric_has_no_interval() -> None:
    stats = score([vf("qa")], [vf("qa")], 0.01)
    assert bootstrap(stats, resample_weights(1, 1000, SEED), 0.95)["salary_currency_accuracy"] is None
