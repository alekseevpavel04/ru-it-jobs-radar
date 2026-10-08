import pytest

from radar.extract.role_proxy import (
    ANALYST_OTHER,
    BUCKET_ROLES,
    BUCKETS,
    DEV,
    ML_DS_DATA,
    NON_IT_NOISE,
    OTHER_IT,
    QA_DEVOPS_SYS,
    role_proxy,
    title_bucket,
)
from radar.extract.schema import ROLES


@pytest.mark.parametrize(
    ("title", "bucket"),
    [
        ("Data Scientist", ML_DS_DATA),
        ("Старший дата-сайентист", ML_DS_DATA),
        ("Аналитик данных", ML_DS_DATA),
        ("Data Analyst (SQL, Python)", ML_DS_DATA),
        ("Продуктовый аналитик", ML_DS_DATA),
        ("BI-аналитик (Power BI)", ML_DS_DATA),
        ("Инженер данных", ML_DS_DATA),
        ("Senior Data Engineer", ML_DS_DATA),
        ("Разработчик DWH", ML_DS_DATA),
        ("MLOps-инженер", ML_DS_DATA),
        ("ML-инженер (Python, PyTorch)", ML_DS_DATA),
        ("NLP-разработчик", ML_DS_DATA),
        ("Инженер по компьютерному зрению", ML_DS_DATA),
        ("Инженер машинного обучения", ML_DS_DATA),
        ("AI-инженер", ML_DS_DATA),
        ("Python developer / AI engineer", ML_DS_DATA),
        ("Специалист по искусственному интеллекту", ML_DS_DATA),
        ("Специалист по нейросетям", ML_DS_DATA),
        # AI names the product, not the work
        ("Product owner (AI)", OTHER_IT),
        ("DevOps инженер (AI-агенты)", QA_DEVOPS_SYS),
        ("Full-stack разработчик - MOST AI", DEV),
        # AI-adjacent work without models is other_it in the annotation guide
        ("Редактор текстов для обучения нейросетей (GigaChat)", OTHER_IT),
        ("Специалист по работе с искусственным интеллектом в управлении персоналом", OTHER_IT),
        ("Специалист по работе с нейросетями", OTHER_IT),
        ("Промпт-инженер", OTHER_IT),
        ("Программист 1С", DEV),
        ("Инженер-программист", DEV),
        ("Техник-программист", DEV),
        ("Java-разработчик", DEV),
        ("HTML-верстальщик", DEV),
        ("Инженер-программист станков с ЧПУ", NON_IT_NOISE),
        ("Системный аналитик", ANALYST_OTHER),
        ("Бизнес-аналитик", ANALYST_OTHER),
        ("Аналитик 1С", ANALYST_OTHER),
        ("Тестировщик (верификатор) / QA engineer", QA_DEVOPS_SYS),
        ("Разработчик автотестов", QA_DEVOPS_SYS),
        ("Системный администратор", QA_DEVOPS_SYS),
        ("Специалист технической поддержки", QA_DEVOPS_SYS),
        ("Специалист по защите информации", QA_DEVOPS_SYS),
        ("Специалист по безопасности информационных технологий", QA_DEVOPS_SYS),
        ("SRE", QA_DEVOPS_SYS),
        ("IT-специалист", OTHER_IT),
        ("Инженер ИТ", OTHER_IT),
        ("Технический писатель", OTHER_IT),
        ("Менеджер проектов", OTHER_IT),
        ("Специалист по дизайну графических и пользовательских интерфейсов", OTHER_IT),
        ("Менеджер по продажам IT-решений", NON_IT_NOISE),
        ("Старший менеджер по работе с клиентами", NON_IT_NOISE),
        ("Инженер по обслуживанию слаботочных систем безопасности", NON_IT_NOISE),
        ("Почтальон", NON_IT_NOISE),
        ("Инженер", NON_IT_NOISE),
        ("", NON_IT_NOISE),
        (None, NON_IT_NOISE),
    ],
)
def test_title_bucket(title: str | None, bucket: str) -> None:
    assert title_bucket(title) == bucket


def test_word_bounds_keep_latin_and_cyrillic_apart() -> None:
    assert title_bucket("HTML-верстальщик") != ML_DS_DATA
    assert title_bucket("Специалист по ИИ") == ML_DS_DATA
    assert title_bucket("Специалист бИИ") == NON_IT_NOISE


@pytest.mark.parametrize(
    ("title", "specs", "bucket"),
    [
        ("Специалист ИТ", ["Системный администратор"], QA_DEVOPS_SYS),
        ("Python developer", ["ML разработчик"], ML_DS_DATA),
        ("Data Scientist", ["Бэкенд разработчик"], ML_DS_DATA),
        ("Бэкенд-разработчик", ["Менеджер проекта"], DEV),
        ("Специалист", ["Менеджер проекта"], OTHER_IT),
        ("Специалист", ["Копирайтер"], NON_IT_NOISE),
        ("Специалист", [], NON_IT_NOISE),
    ],
)
def test_specializations_refine_vague_titles(title: str, specs: list[str], bucket: str) -> None:
    assert role_proxy(title, specs) == bucket


def test_buckets_cover_every_schema_role_once() -> None:
    covered = [role for bucket in BUCKETS for role in BUCKET_ROLES[bucket]]
    assert sorted(covered) == sorted(ROLES)
