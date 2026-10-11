from radar.agent.compare import results_match, value_match


def test_order_of_rows_and_columns_is_free_extra_columns_allowed() -> None:
    ref = [["habr", 25], ["trudvsem", 131]]
    pred = [[131, "trudvsem", 0.5], [25, "habr", 0.1]]
    assert results_match(ref, pred)


def test_row_count_must_match() -> None:
    assert not results_match([["habr", 25]], [["habr", 25], ["trudvsem", 131]])
    assert not results_match([[1]], [])


def test_integers_are_exact() -> None:
    assert results_match([[52]], [[52]])
    assert results_match([[52]], [[52.0]])
    assert not results_match([[52]], [[53]])


def test_share_tolerates_rounding_and_percent() -> None:
    ref = [[0.4423076923076923]]
    assert results_match(ref, [[0.44]])
    assert results_match(ref, [[44.23]])
    assert results_match(ref, [[44.2]])
    assert not results_match(ref, [[0.5]])
    assert not results_match(ref, [[23]])


def test_percent_scale_only_for_shares() -> None:
    assert not results_match([[232500.0]], [[23250000.0]])


def test_strings_compare_case_insensitively() -> None:
    assert results_match([["Python"], ["SQL"]], [["sql"], ["python "]])
    assert not results_match([["Python"]], [["Java"]])


def test_null_matches_only_null() -> None:
    assert value_match(None, None)
    assert not value_match(None, 0)
    assert not value_match(5.0, None)


def test_number_does_not_match_string() -> None:
    assert not value_match(52, "52")


def test_columns_are_matched_jointly_not_one_by_one() -> None:
    ref = [["a", 1], ["b", 2]]
    assert not results_match(ref, [["a", 2], ["b", 1]])
    assert results_match(ref, [["b", 2], ["a", 1]])


def test_empty_reference_matches_empty_result() -> None:
    assert results_match([], [])
