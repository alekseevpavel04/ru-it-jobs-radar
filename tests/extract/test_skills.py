from pathlib import Path

import pytest

from radar.extract import skills as sk
from radar.extract.skill_dictionary import v0_rows

CATEGORIES = {
    "language", "framework", "database", "data", "ml", "bi", "cloud", "devops", "os", "network", "security",
    "testing", "tools", "office", "1c", "platform", "methodology", "hardware", "design", "other",
}  # fmt: skip


def toy(tmp_path: Path, body: str) -> sk.Dictionary:
    path = tmp_path / "skills.toml"
    path.write_text(body, encoding="utf-8")
    return sk.load(path)


def test_dictionary_loads_with_known_categories() -> None:
    d = sk.load()
    assert len(d.skills) > 900
    assert {s.category for s in d.skills.values()} <= CATEGORIES


def test_every_name_and_alias_resolves_to_its_own_skill() -> None:
    d = sk.load()
    for skill in d.skills.values():
        for spelling in (skill.name, *skill.aliases):
            assert sk.resolve(spelling, d).canonical == skill.name, spelling


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        # guide, skills: the canonical spellings
        ("Postgres", "PostgreSQL"),
        ("k8s", "Kubernetes"),
        ("sklearn", "scikit-learn"),
        ("MS SQL", "MS SQL Server"),
        ("1C", "1С"),
        ("Битрикс", "1С-Битрикс"),  # case 25
        ("1С:Управление холдингом", "1С:ERP УХ"),
        ("A/B testing", "A/B-тесты"),
        # case and separators need no alias
        ("Node JS", "Node.js"),
        ("NODEJS", "Node.js"),
        ("Power-BI", "Power BI"),
        # inflection regexes
        ("питона", "Python"),
        ("постгресом", "PostgreSQL"),
        ("1С:Бухгалтерия предприятия 3.0", "1С:Бухгалтерия"),
    ],
)
def test_known_spellings(raw: str, canonical: str) -> None:
    r = sk.resolve(raw)
    assert (r.canonical, r.status) == (canonical, "known")


@pytest.mark.parametrize(
    ("raw", "canonical", "rule"),
    [
        ("Python 3.10", "Python", "version"),
        ("C++11/14", "C++", "version"),
        ("Java 8+", "Java", "version"),
        ("PHPStan Level 9", "PHPStan", "version"),
        ("Apache Kafka", "Kafka", None),
        ("Apache Flume 1.11", "Flume", None),
        ("Apache Druid", None, None),
        ("Microsoft PowerPoint", "PowerPoint", None),
        ("Microsoft SQL Server 2019", "MS SQL Server", "version"),
        ("Yandex DataLens", "DataLens", None),
        ("Adobe InDesign", "InDesign", None),
        ("Hashicorp Consul", "Consul", "vendor_prefix"),
        ("СУБД ClickHouse", "ClickHouse", "generic_word"),
        ("фреймворк Django", "Django", "generic_word"),
        ("Laravel framework", "Laravel", "generic_word"),
        ("Jupyter Notebooks", "Jupyter", "plural"),
        ("PHP v8", "PHP", "version"),
        ("Рython", "Python", "lookalike"),  # Cyrillic Р
    ],
)
def test_rules(raw: str, canonical: str | None, rule: str | None) -> None:
    r = sk.resolve(raw)
    assert r.canonical == canonical
    if rule is not None:
        assert (r.status, r.rule) == ("rule", rule)


@pytest.mark.parametrize(
    "raw",
    [
        "Machine Learning",  # case 74
        "Data Science",
        "LLM",  # case 48
        "NLP",
        "ГОСТ Р 57580",
        "ISO 27001",
        "152-ФЗ",
        "CVSS",
        "CCNA",  # case 58
        "AWS Certified Solutions Architect",
        "VPN",  # case 55
        "Wi-Fi",
        "zero-shot",  # case 57
        "ERP",  # case 35
        "NoSQL",
        "коммуникабельность",
        "Английский язык B2",
        "Upper-Intermediate",
    ],
)
def test_non_skills(raw: str) -> None:
    assert sk.resolve(raw).status == "non_skill"
    assert sk.normalize(raw) is None


def test_prefix_needs_its_space() -> None:
    assert sk.resolve("isotope").status == "unknown"


def test_unknown_spelling_is_kept_as_written() -> None:
    r = sk.resolve("  Zorblax   DB ")
    assert (r.status, r.canonical) == ("unknown", None)
    assert sk.normalize("  Zorblax   DB ") == "Zorblax DB"


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        # docs/annotation_guide.md, cases 17, 23, 35, 48, 53-57, 61, 62
        ("numpy", "NumPy"),
        ("pandas", "pandas"),
        ("catboost", "CatBoost"),
        ("matplotlib", "matplotlib"),
        ("C++", "C++"),
        ("STL", "STL"),
        ("OPC UA", "OPC UA"),
        ("CANopen", "CANopen"),
        ("U-Boot", "U-Boot"),
        ("CI/CD", "CI/CD"),
        ("MLOps", "MLOps"),
        ("RAG", "RAG"),
        ("LoRA", "LoRA"),
        ("User story", "User story"),
        ("ETL", "ETL"),
        ("IaC", "IaC"),
        ("Data Quality", "Data Quality"),
        ("prompt engineering", "Prompt engineering"),
        ("пентест", "Pentest"),
        ("threat hunting", "Threat hunting"),
        ("DFIR", "DFIR"),
        ("SEO", "SEO"),
        ("Agile", "Agile"),
        ("Scrum", "Scrum"),
        ("ITIL", "ITIL"),
        ("PMBOK", "PMBOK"),
        ("OWASP", "OWASP"),
        ("ООП", "ООП"),
        ("паттерны проектирования", "Design patterns"),
        ("микросервисы", "Microservices"),
        ("TCP/IP", "TCP/IP"),
        ("BGP", "BGP"),
        ("Modbus", "Modbus"),
        ("Mikrotik", "MikroTik"),
        ("Zynq", "Zynq"),
        ("BERT", "BERT"),
        ("GPT-4", "GPT"),
        ("YandexGPT", "YandexGPT"),
        ("fine-tuning", "Fine-tuning"),
        ("Apache Spark", "Spark"),
        ("Microsoft Excel", "Excel"),
        ("1С:Розница", "1С:Розница"),
        ("МСФО", "МСФО"),
        ("СБиС", "СБИС"),
        ("Честный знак", "Честный знак"),
        ("PHPUnit", "PHPUnit"),
        ("memcached", "Memcached"),
    ],
)
def test_guide_examples(raw: str, canonical: str) -> None:
    assert sk.resolve(raw).canonical == canonical


def test_normalize_list_splits_glued_spellings_and_drops_non_skills() -> None:
    raws = ["C#/.NET", "sklearn/catboost", "SQL (PostgreSQL, MySQL)", "коммуникабельность", "Python", "python3"]
    assert sk.normalize_list(raws) == ["C#", ".NET", "scikit-learn", "CatBoost", "SQL", "PostgreSQL", "MySQL", "Python"]


def test_normalize_list_keeps_cpp_and_known_slashes() -> None:
    assert sk.normalize_list(["C++/C#", "TCP/IP", "PHP + Java"]) == ["C++", "C#", "TCP/IP", "PHP", "Java"]


def test_normalize_list_leaves_an_unknown_glue_whole() -> None:
    assert sk.normalize_list(["Python/Zorblax"]) == ["Python/Zorblax"]


def test_configuration_implies_1c() -> None:
    assert sk.normalize_list(["1С:ЗУП"]) == ["1С", "1С:ЗУП"]
    assert sk.complete(["1С-Битрикс"]) == ["1С-Битрикс"]


def test_eval_key() -> None:
    assert sk.eval_key("Postgres") == sk.eval_key("PostgreSQL") == "postgresql"
    assert sk.eval_key("Machine Learning") == "machine learning"
    assert sk.eval_key("Zorblax  DB") == "zorblax db"


def test_loader_rejects_a_spelling_of_two_skills(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="already a spelling"):
        toy(tmp_path, '[skills.database]\n"PostgreSQL" = ["pg"]\n"Greenplum" = ["PG"]\n')


def test_loader_rejects_a_non_skill_that_is_a_skill(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-skills"):
        toy(tmp_path, '[skills.database]\n"Redis" = []\n[non_skills]\nnames = ["redis"]\n')


def test_loader_rejects_unknown_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parents"):
        toy(tmp_path, '[skills.os]\n"Ubuntu" = []\n[parents]\n"Ubuntu" = "Linux"\n')
    with pytest.raises(ValueError, match="search"):
        toy(tmp_path, '[skills.os]\n"Ubuntu" = []\n[search]\nnames_not_searched = ["Linux"]\n')


def test_loader_rejects_an_unknown_text_check_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="text_check"):
        toy(tmp_path, '[skills.network]\n"HTTP" = []\n[text_check]\noutside_links = ["HTTPS"]\n')


def test_http_and_https_stay_skills_checked_against_links() -> None:
    d = sk.load()
    assert d.outside_links == {"HTTP", "HTTPS"}
    assert sk.normalize_list(["HTTPS", "HTTP"], d) == ["HTTPS", "HTTP"]
    # scoring keeps comparing the extractor's own answer
    assert sk.eval_key("HTTPS") == "https"


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("Откликнуться: https://example.com/jobs", set()),
        ("Отклик: HTTPS://EXAMPLE.COM/A и www.example.com/https", set()),
        ("Понимание протоколов HTTP/HTTPS, TCP", {"HTTP", "HTTPS"}),
        ("HTTP-запросы; анкета https://example.com/form?x=http", {"HTTP"}),
        ("aiohttp, httpx, OkHttp", set()),
        ("Ссылка http://example.com\nзнание HTTPS", {"HTTPS"}),
    ],
)
def test_named_outside_links(text: str, named: set[str]) -> None:
    assert {n for n in ("HTTP", "HTTPS") if sk.named_outside_links(n, text)} == named


def test_named_outside_links_reads_aliases(tmp_path: Path) -> None:
    d = toy(tmp_path, '[skills.network]\n"HTTP" = ["хттп"]\n[text_check]\noutside_links = ["HTTP"]\n')
    assert sk.named_outside_links("HTTP", "протокол ХТТП", d)
    assert not sk.named_outside_links("HTTP", "https://example.com/хттп", d)


def test_without_drops_entries(tmp_path: Path) -> None:
    d = toy(tmp_path, '[skills.os]\n"Ubuntu" = ["убунту"]\n"Linux" = []\n[patterns]\n"Ubuntu" = [\'убунт[а-я]*\']\n')
    smaller = d.without(["Ubuntu"])
    assert sk.resolve("убунту", smaller).status == "unknown"
    assert sk.resolve("убунтой", smaller).status == "unknown"
    assert sk.resolve("Linux", smaller).canonical == "Linux"


def test_v0_spellings_still_resolve() -> None:
    """Every fixed v0 synonym is a v1 skill or a non-skill, except spellings dropped as too ambiguous."""
    unresolved = [
        synonym
        for row in v0_rows()
        for synonym in row["synonyms"].split("|")
        if not any(ch in synonym for ch in "[]()*?|") and sk.resolve(synonym).status == "unknown"
    ]
    # "argo" is also Argo Workflows and an English word; "ad ds" finds "adds" once separators are optional.
    assert unresolved == ["argo", "ad ds"]


def test_eval_option_selects_the_skill_comparison(tmp_path: Path) -> None:
    from radar.eval import cli
    from radar.eval.metrics import normalize_item

    configured = cli.load_config().skill_normalization
    base = cli.CONFIG.read_text(encoding="utf-8").replace(f'normalization = "{configured}"', "normalization = {}")
    assert cli.SKILL_NORMALIZERS["casefold"] is normalize_item
    path = tmp_path / "eval.toml"
    path.write_text(base.replace("normalization = {}", 'normalization = "casefold"'), encoding="utf-8")
    assert cli.SKILL_NORMALIZERS[cli.load_config(path).skill_normalization] is normalize_item
    path.write_text(base.replace("normalization = {}", 'normalization = "dictionary"'), encoding="utf-8")
    assert cli.SKILL_NORMALIZERS[cli.load_config(path).skill_normalization] is sk.eval_key
    path.write_text(base.replace("normalization = {}", 'normalization = "fuzzy"'), encoding="utf-8")
    with pytest.raises(ValueError, match="normalization"):
        cli.load_config(path)
