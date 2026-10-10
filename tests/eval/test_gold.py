from collections import Counter

from radar.eval.gold import normalize_skills


def test_skill_rules_drop_classes_prefixes_and_add_platform() -> None:
    changes: Counter[str] = Counter()
    skills = ["Apache Kafka", "VPN", "LLM", "Microsoft Excel", "1С:ERP", "ГОСТ 34", "Python", "Kafka"]
    assert normalize_skills(skills, changes) == ["1С", "Kafka", "Excel", "1С:ERP", "Python"]
    assert changes == Counter({"removed": 3, "prefix_dropped": 2, "1c_added": 1})


def test_skills_without_rules_stay_as_labeled() -> None:
    changes: Counter[str] = Counter()
    assert normalize_skills(["SQL", "Power BI"], changes) == ["SQL", "Power BI"]
    assert not changes
