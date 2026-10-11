from radar.train.guard import StepGuard, vram_limit_mib


def guard(limit: float | None = 6000) -> StepGuard:
    return StepGuard(limit_mib=limit, factor=3.0, reference_steps=3, patience=2)


def warm(g: StepGuard, seconds: float = 1.0, tokens: int = 1000) -> None:
    g.check(20.0, tokens, 3000)  # first step: compilation, ignored
    for _ in range(3):
        assert g.check(seconds, tokens, 3000) is None


def test_reserved_memory_above_the_limit_stops_at_once() -> None:
    g = guard()
    assert g.check(1.0, 1000, 5000) is None
    reason = g.check(1.0, 1000, 6100)
    assert reason is not None and "6100" in reason
    assert g.peak_reserved_mib == 6100


def test_no_limit_on_the_cpu() -> None:
    assert guard(None).check(1.0, 1000, 10**6) is None


def test_reference_skips_the_first_step() -> None:
    g = guard()
    warm(g)
    assert g.reference == 1.0


def test_a_lasting_slowdown_stops_but_one_slow_step_does_not() -> None:
    g = guard()
    warm(g)
    assert g.check(4.0, 1000, 3000) is None
    assert g.check(1.0, 1000, 3000) is None
    assert g.check(4.0, 1000, 3000) is None
    reason = g.check(4.0, 1000, 3000)
    assert reason is not None and "system RAM" in reason


def test_time_is_taken_per_token() -> None:
    g = guard()
    warm(g)
    for _ in range(5):
        assert g.check(4.0, 4000, 3000) is None  # four times the tokens, four times the time


def test_without_token_counts_the_step_time_is_used() -> None:
    g = guard()
    warm(g, seconds=2.0, tokens=0)
    assert g.reference == 2.0
    g.check(7.0, 0, 3000)
    assert g.check(7.0, 0, 3000) is not None


def test_limit_leaves_out_what_was_in_use_and_the_margin() -> None:
    assert vram_limit_mib(8192, 900, 512) == 6780
