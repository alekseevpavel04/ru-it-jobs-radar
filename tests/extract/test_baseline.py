import json
from pathlib import Path

import pytest

from radar import config, results
from radar.extract import baseline
from radar.extract.baseline import extract, output_for, split_input
from radar.extract.inputs import build_input
from radar.extract.role_rules import predict_role
from radar.extract.schema import parse
from radar.extract.skill_dictionary import find_skills


def page(title: str | None = "Программист", text: str = "", **facts: str) -> str:
    labels = {
        "salary": "Зарплата",
        "address": "Адрес",
        "experience": "Требуемый опыт",
        "busy": "Тип занятости",
        "spec": "Специализация",
        "qualification": "Квалификация",
        "skills": "Навыки",
        "location": "Местоположение",
        "remote": "Формат работы",
    }
    return build_input(title, text, [(labels[k], v) for k, v in facts.items()])


def skills(text: str) -> list[str]:
    return find_skills(text, baseline.normalize(text))


def test_split_input_header_and_text() -> None:
    inp = split_input(page("Тестировщик", "Обязанности: тесты", salary="50 000 руб."))
    assert inp.title == "Тестировщик"
    assert inp.header["Зарплата"] == "50 000 руб."
    assert inp.body == "Обязанности: тесты"


def test_text_without_header_is_all_body() -> None:
    inp = split_input("Ищем Python-разработчика\n\nЗарплата: 200к")
    assert inp.header == {}
    assert inp.body.startswith("Ищем Python")


@pytest.mark.parametrize(
    ("title", "text", "specs", "role"),
    [
        ("Data Scientist", "", (), "data_scientist"),
        ("ML-инженер (NLP)", "", (), "ml_engineer"),
        ("MLOps engineer", "", (), "mlops"),
        ("Middle data-инженер", "", (), "data_engineer"),
        ("Разработчик КХД", "", (), "data_engineer"),
        ("Аналитик данных", "", (), "data_analyst"),
        ("Аналитик (казначейство)", "SQL и Python для анализа данных", (), "data_analyst"),
        ("Кредитный аналитик", "финансовая отчетность заемщиков", (), "non_it"),
        ("Аналитик 1С", "", (), "system_analyst"),
        ("Аналитик", "", ("Системный аналитик",), "system_analyst"),
        ("Бизнес-аналитик", "SQL", (), "business_analyst"),
        ("Frontend-разработчик", "", (), "frontend"),
        ("Android-разработчик", "", (), "mobile"),
        ("Bitrix-разработчик", "", (), "backend"),
        ("Программист 1С", "", (), "developer_1c"),
        ("Fullstack-разработчик", "", (), "fullstack"),
        ("Инженер-программист", "Java, Spring, Hibernate, PostgreSQL", (), "backend"),
        ("Инженер-программист", "разработка ПО для микроконтроллеров (STM32), C#", (), "embedded"),
        ("Инженер-программист", "обслуживание компьютерной техники и оргтехники", (), "sysadmin"),
        ("Программист", "разработка программного кода", (), "software_engineer"),
        ("Тестировщик", "", (), "qa"),
        ("DevOps-инженер", "", (), "devops"),
        ("Инженер по защите информации", "", (), "security"),
        ("Специалист технической поддержки", "", (), "support"),
        ("Системный администратор", "", (), "sysadmin"),
        ("UX/UI дизайнер", "", (), "designer"),
        ("Менеджер проектов", "", (), "manager"),
        ("Редактор текстов для обучения нейросетей (AI-тренер)", "", (), "other_it"),
        ("Оператор станка с ЧПУ", "", (), "non_it"),
        ("Водитель", "", (), "non_it"),
        ("Руководитель группы разработки", "", ("Бэкенд разработчик",), "backend"),
    ],
)
def test_role(title: str, text: str, specs: tuple[str, ...], role: str) -> None:
    assert predict_role(title, text, specs) == role


def test_role_null_only_without_title_and_text() -> None:
    assert predict_role(None, "") is None
    assert extract("").role is None


@pytest.mark.parametrize(
    ("title", "facts", "grade"),
    [
        ("Senior Python Developer", {}, "senior"),
        ("Старший системный аналитик", {}, "senior"),
        ("Ведущий программист 1С", {}, "senior"),
        ("Middle+/Senior DevOps", {}, "middle"),
        ("Junior+/Middle аналитик", {}, "junior"),
        ("Стажер Data Science", {}, "intern"),
        ("Руководитель группы разработки", {}, "lead"),
        ("Team Lead backend", {}, "lead"),
        ("Начальник отдела информатизации", {}, "lead"),
        ("Инженер-программист 1 категории", {}, None),
        ("Менеджер продукта", {"qualification": "Старший (Senior)"}, "senior"),
        ("Специалист", {"busy": "Стажировка"}, "intern"),
        ("Программист", {}, None),
    ],
)
def test_grade(title: str, facts: dict[str, str], grade: str | None) -> None:
    assert extract(page(title, "текст вакансии", **facts)).grade == grade


def test_skills_dictionary_order_and_canonical_names() -> None:
    assert skills("Опыт: Python 3.10, Postgres, k8s, sklearn") == ["Python", "PostgreSQL", "Kubernetes", "scikit-learn"]


def test_skills_split_glued_and_keep_cpp_apart() -> None:
    assert skills("C#/.NET, C++11/14 и C") == ["C#", ".NET", "C++"]


def test_skills_longer_match_wins() -> None:
    assert skills("MS SQL и 1С-Битрикс") == ["MS SQL Server", "1С-Битрикс"]


def test_skills_leave_out_what_the_guide_does_not_count() -> None:
    assert skills("Английский B2, коммуникабельность, алгоритмы, сбор требований, нейросети") == []


def test_skills_english_words_in_lowercase_english_text() -> None:
    text = "We go further than the rest. Go, REST"
    assert find_skills(text, baseline.normalize(text), english=True) == ["Go", "REST"]


@pytest.mark.parametrize(
    ("facts", "text", "expected"),
    [
        ({"salary": "от 81 000 до 100 000 руб."}, "", (81000, 100000, "RUB", None, None)),
        (
            {"salary": "от 200 000 до 250 000 руб."},
            "Зарплата от 200 000 руб. после вычета налогов",
            (200000, 250000, "RUB", "net", None),
        ),
        ({"salary": "120 000 ₽"}, "120 000 ₽ за месяц на руки", (120000, 120000, "RUB", "net", "month")),
        # Another amount in the text is another payment: it adds nothing to the header.
        (
            {"salary": "от 80 000 до 200 000 руб."},
            "средний доход агентов от 720 000 ₽ в год",
            (80000, 200000, "RUB", None, None),
        ),
        ({}, "Зарплата: 300-400к net", (300000, 400000, None, "net", None)),
        ({}, "Заработная плата обсуждается по итогам собеседования", None),
        ({}, "Выплата зарплаты 2 раза в месяц", None),
        (
            {"salary": "от 25 000 до 30 000 руб."},
            "заработная плата обсуждается по результатам собеседования",
            (25000, 30000, "RUB", None, None),
        ),
    ],
)
def test_salary(facts: dict[str, str], text: str, expected: tuple[object, ...] | None) -> None:
    s = extract(page("Программист", text, **facts)).salary
    assert (None if s is None else (s.from_, s.to, s.currency, s.gross_net, s.period)) == expected


@pytest.mark.parametrize(
    ("facts", "text", "expected"),
    [
        ({}, "Возможность удаленной работы на территории РФ", "remote"),
        ({}, "Гибридный/удаленный формат работы: 5/2", "remote"),
        ({}, "Удаленный формат работы возможен после испытательного срока", "hybrid"),
        ({}, "Гибридный график работы", "hybrid"),
        ({}, "Готовность работать очно: у нас нет удаленной работы", "office"),
        ({}, "Работа в офисе на Таганке", "office"),
        ({}, "Удаленная поддержка пользователей, настройка удаленного доступа", None),
        ({"busy": "Дистанционная (удаленная) работа"}, "", "remote"),
        ({"remote": "Можно удаленно"}, "Офис в центре города", "remote"),
        ({"address": "Город Москва, Тверская улица, 1"}, "", None),
    ],
)
def test_work_format(facts: dict[str, str], text: str, expected: str | None) -> None:
    assert extract(page("Программист", text, **facts)).work_format == expected


@pytest.mark.parametrize(
    ("facts", "cities"),
    [
        ({"address": "Нижегородская область, Город Дзержинск, Зеленая улица, 10"}, ["Дзержинск"]),
        ({"address": "Тюменская область, г Тюмень, Одесская улица, дом Д. 14"}, ["Тюмень"]),
        ({"address": "Приморский край, г. Уссурийск, Калинина улица, 52"}, ["Уссурийск"]),
        ({"address": "Республика Бурятия, Муйский район, Таксимо поселок городского типа, 1"}, ["Таксимо"]),
        (
            {"address": "Липецкая область, Добровский район, Большой Хомутец село, Зарникова улица, 14"},
            ["Большой Хомутец"],
        ),
        ({"address": "Город Москва"}, ["Москва"]),
        ({"address": "Москва"}, ["Москва"]),
        ({"address": "Херсонская область"}, []),
        (
            {"address": "Город Москва, Тверская улица, 1; Свердловская область, г Екатеринбург, Ленина, 5"},
            ["Москва", "Екатеринбург"],
        ),
        ({"location": "Москва, Самара"}, ["Москва", "Самара"]),
        ({"location": "Томилино, пгт"}, ["Томилино"]),
        ({}, []),
    ],
)
def test_cities(facts: dict[str, str], cities: list[str]) -> None:
    assert extract(page("Программист", "текст", **facts)).cities == cities


@pytest.mark.parametrize(
    ("facts", "text", "expected"),
    [
        ({"experience": "от 1 года"}, "", (1, None)),
        ({"experience": "от 1 года"}, "Опыт работы более двух лет", (2, None)),
        ({}, "Опыт работы от 3 лет", (3, None)),
        ({}, "Опыт коммерческой разработки 1-3 года", (1, 3)),
        ({}, "опыт от 5+ лет (Middle+/Senior)", (5, None)),
        ({}, "Опыт администрирования от 1,5 лет", (1.5, None)),
        ({}, "Опыт работы от полугода", (0.5, None)),
        ({}, "Опыт работы приветствуется, но возможен прием без опыта с обучением", (0, None)),
        ({}, "Опыт работы не менее 1 года; опыт с Kubernetes от 3 лет", (3, None)),
        ({}, "3+ years of commercial experience with Go", (3, None)),
        ({}, "Опыт работы приветствуется", None),
        ({}, "Возможна стажировка 2 года", None),
        ({}, "Требования не предъявляются", None),
    ],
)
def test_experience(facts: dict[str, str], text: str, expected: tuple[float, float | None] | None) -> None:
    e = extract(page("Программист", text, **facts)).experience_years
    assert (None if e is None else (e.min, e.max)) == expected


@pytest.mark.parametrize(
    ("title", "facts", "text", "expected"),
    [
        ("Программист", {}, "Оформление по ТК РФ или ИП/самозанятость на выбор", "labor_contract"),
        ("Программист", {}, "Срочный трудовой договор на период декретного отпуска", "labor_contract"),
        ("Программист", {}, "Работа по договору ГПХ или как самозанятый", "gph"),
        ("Программист", {}, "Сотрудничество с самозанятыми и ИП", "self_employed"),
        ("Стажер-разработчик", {}, "Официальное трудоустройство", "internship"),
        ("Программист", {"busy": "Стажировка"}, "", "internship"),
        ("Программист", {"busy": "Полная занятость"}, "", None),
    ],
)
def test_employment_type(title: str, facts: dict[str, str], text: str, expected: str | None) -> None:
    assert extract(page(title, text, **facts)).employment_type == expected


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Разработка сервисов на Python, Docker и Kubernetes, работа с PostgreSQL и Kafka", "ru"),
        ("We are looking for a backend engineer to build services in Go and run them on Kubernetes", "en"),
        ("Біз Python әзірлеушісін іздейміз, жұмыс Алматы қаласында, тәжірибе қажет", "other"),
    ],
)
def test_lang(text: str, lang: str) -> None:
    assert extract(page("Программист", text)).lang == lang


def test_output_is_schema_valid_json() -> None:
    out = output_for(page("Senior Python Developer", "Опыт от 3 лет. Удаленно/гибрид", salary="от 200 000 ₽"))
    fields = parse(out)
    assert fields.grade == "senior"
    assert json.loads(out)["salary"]["from"] == 200000


def test_split_command_writes_predictions_and_skips_scoring_without_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RADAR_DATA_DIR", str(tmp_path))
    gold = tmp_path / "gold"
    gold.mkdir()
    rows = [{"id": "a:1", "input": page("Тестировщик", "Опыт от 1 года")}, {"id": "a:2", "input": ""}]
    (gold / "dev_inputs.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), "utf-8")

    assert baseline.main(["--split", "dev"]) == 0

    lines = (tmp_path / "predictions" / "dev" / "regex.jsonl").read_text("utf-8").splitlines()
    preds = [json.loads(line) for line in lines]
    assert [p["id"] for p in preds] == ["a:1", "a:2"]
    assert parse(preds[0]["output"]).role == "qa"


def test_split_command_scores_when_gold_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RADAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    gold = tmp_path / "data" / "gold"
    gold.mkdir(parents=True)
    text = page("Тестировщик", "Опыт от 1 года", salary="50 000 руб.")
    (gold / "dev_inputs.jsonl").write_text(json.dumps({"id": "a:1", "input": text}, ensure_ascii=False), "utf-8")
    fields = json.loads(output_for(text)) | {"grade": "junior"}
    (gold / "dev.jsonl").write_text(json.dumps({"id": "a:1", "fields": fields}, ensure_ascii=False), "utf-8")

    assert baseline.main(["--split", "dev"]) == 0

    doc = json.loads((tmp_path / "results" / "extract" / "dev_regex.json").read_text("utf-8"))
    assert doc["system"] == "regex"
    assert Path(doc["predictions_file"]).parts[-3:] == ("predictions", "dev", "regex.jsonl")
    assert doc["metrics"]["valid_json"]["n"] == 1
    assert doc["metrics"]["role_f1"]["value"] == 1.0
    assert doc["metrics"]["grade_recall"]["value"] == 0.0


def test_test_split_is_not_served() -> None:
    with pytest.raises(SystemExit):
        baseline.main(["--split", "test"])
