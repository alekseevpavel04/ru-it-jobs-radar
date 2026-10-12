from radar.dedup.entities import Edge, Traits, components, mapping, split_large, stable_id, traits_of
from radar.dedup.units import to_units

NONE = Traits(None, None, None, None)


def test_components_follow_links_transitively() -> None:
    groups, refused = components(["a", "b", "c", "d"], [Edge("a", "b", 0.9, 0.0), Edge("b", "c", 0.8, 0.0)])
    assert groups == [["a", "b", "c"], ["d"]]
    assert refused == 0


def test_a_card_without_settlement_does_not_join_two_towns() -> None:
    # b states only the region; a and c are in different towns. The stronger link a-b wins.
    traits = {
        "a": Traits("1", None, "чита", None),
        "b": Traits("1", None, None, None),
        "c": Traits("1", None, "нерчинск", None),
    }
    edges = [Edge("a", "b", 0.95, 0.0), Edge("b", "c", 0.9, 0.0)]
    groups, refused = components(["a", "b", "c"], edges, traits=traits)
    assert groups == [["a", "b"], ["c"]]
    assert refused == 1


def test_two_inns_are_never_one_entity() -> None:
    traits = {"a": Traits("1", None, None, None), "b": Traits("2", None, None, None)}
    groups, _ = components(["a", "b"], [Edge("a", "b", 1.0, 1.0)], traits=traits)
    assert groups == [["a"], ["b"]]


def test_oversized_component_is_recut_by_a_stricter_jaccard() -> None:
    nodes = [str(i) for i in range(6)]
    edges = [Edge("0", "1", 0.95, 0), Edge("1", "2", 0.95, 0), Edge("2", "3", 0.72, 0), Edge("3", "4", 0.95, 0)]
    groups, refused, recut = split_large(nodes, edges, start=0.7, max_size=3, traits={n: NONE for n in nodes})
    assert groups == [["0", "1", "2"], ["3", "4"], ["5"]]
    assert (refused, recut) == (0, 1)


def test_remote_hides_place_except_for_a_sole_proprietor() -> None:
    company, proprietor = to_units(
        [
            {"source": "trudvsem", "source_id": "1", "dt": "2026-10-08", "region": "Город Москва",
             "address": "г Москва, Тверская улица, 1", "remote": True, "employer_name": "Ромашка",
             "employer_inn": "77"},
            {"source": "trudvsem", "source_id": "2", "dt": "2026-10-08", "region": "Город Москва",
             "address": "г Москва", "remote": True, "employer_name": "ИП"},
        ]
    )  # fmt: skip
    assert traits_of(company) == Traits("77", None, None, None)
    assert traits_of(proprietor) == Traits(None, None, "москва", None)


def test_superjob_employer_is_a_trait_of_its_own() -> None:
    one, other, habr = to_units(
        [
            {"source": "superjob", "source_id": "1", "dt": "2026-10-12", "cities": ["Подольск"],
             "address": "Московская обл., г. Подольск", "employer_name": 'ООО "Ромашка"'},
            {"source": "superjob", "source_id": "2", "dt": "2026-10-12", "employer_name": "Лютик"},
            {"source": "habr", "source_id": "3", "dt": "2026-10-12", "employer_name": "Ромашка Групп"},
        ]
    )  # fmt: skip
    assert traits_of(one) == Traits(None, None, "подольск", None, "ромашка")
    traits = {u.uid: traits_of(u) for u in (one, other, habr)}
    edges = [Edge(one.uid, other.uid, 0.9, 0.9), Edge(one.uid, habr.uid, 0.8, 0.9)]
    groups, refused = components([u.uid for u in (one, other, habr)], edges, traits=traits)
    # two SuperJob employers never merge; another site's spelling of the name does not block a link
    assert groups == [["habr:3", "superjob:1"], ["superjob:2"]]
    assert refused == 1


def test_mapping_spreads_to_id_linked_records_and_ids_are_stable() -> None:
    units = to_units(
        [
            {"source": "trudvsem", "source_id": "u1", "dt": "2026-10-08"},
            {"source": "trudvsem_snapshot", "source_id": "u1", "dt": "2026-07-01"},
            {"source": "habr", "source_id": "9", "dt": "2026-10-08"},
        ]
    )
    rows = mapping(units, [["habr:9", "trudvsem:u1"]], [["habr:9"], ["trudvsem:u1"]])
    entity = {(r["source"], r["source_id"]): r["vacancy_entity_id"] for r in rows}
    text = {(r["source"], r["source_id"]): r["text_cluster_id"] for r in rows}
    assert len(set(entity.values())) == 1
    assert entity[("habr", "9")] == stable_id("ve_", ["habr:9", "trudvsem:u1", "trudvsem_snapshot:u1"])
    assert text[("trudvsem", "u1")] == text[("trudvsem_snapshot", "u1")] != text[("habr", "9")]
