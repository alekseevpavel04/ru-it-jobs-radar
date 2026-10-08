"""Synthetic vacancy texts with their target JSON for the GPU smoke test.

Companies and texts are invented, nothing comes from real postings. Fields follow the draft schema in
the final schema is fixed in radar.extract.schema, so this module is only for measuring memory and speed.
Every sixth text is padded with company boilerplate so that some sequences reach the 2048-token limit.
"""

import json
import random
from typing import Any

SYSTEM_PROMPT = (
    "Ты извлекаешь структуру из текста вакансии. Верни только JSON по схеме: role, grade, skills, salary "
    "(from, to, currency, gross_net, period), work_format, cities, experience_years (min, max), employment_type. "
    "Если поля нет в тексте, ставь null. Не угадывай."
)

COMPANIES = [
    "Кедровая Аналитика",
    "Лунный Кот Софт",
    "Пиксельная Мельница",
    "Тихая Гавань Системс",
    "Облачный Сад",
    "Синяя Цапля Данные",
    "Медный Ключ",
    "Северный Ветер Технологии",
]

ROLES: dict[str, tuple[str, list[str]]] = {
    "ml_engineer": ("ML-инженер", ["Python", "PyTorch", "Docker", "MLflow", "Kubernetes", "Airflow", "SQL", "Git"]),
    "data_scientist": ("Data Scientist", ["Python", "pandas", "scikit-learn", "CatBoost", "SQL", "A/B-тесты"]),
    "data_analyst": ("Аналитик данных", ["SQL", "Python", "Excel", "Superset", "ClickHouse", "статистика"]),
    "data_engineer": ("Data Engineer", ["Python", "Airflow", "dbt", "ClickHouse", "Spark", "Kafka", "SQL"]),
    "mlops": ("MLOps-инженер", ["Kubernetes", "Docker", "Terraform", "Prometheus", "Grafana", "MLflow", "Helm"]),
    "backend": ("Backend-разработчик", ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Git"]),
}

GRADES = {
    "intern": "Стажёр",
    "junior": "Junior",
    "middle": "Middle",
    "senior": "Senior",
    "lead": "Lead",
}

CITIES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск", "Екатеринбург", "Нижний Новгород", "Томск"]

WORK_FORMATS = {
    "office": "Работа в офисе пять дней в неделю.",
    "hybrid": "Гибрид: два дня в офисе, остальное из дома.",
    "remote": "Полностью удалённая работа.",
}

EMPLOYMENT = {
    "labor_contract": "Оформление по трудовому договору.",
    "gph": "Оформление по договору ГПХ.",
    "self_employed": "Работаем с ИП или самозанятыми.",
    "internship": "Оплачиваемая стажировка на три месяца.",
}

BOILERPLATE = [
    "Компания разрабатывает внутренние сервисы для логистики и складского учёта.",
    "Команда состоит из двенадцати инженеров, трёх аналитиков и одного продакт-менеджера.",
    "Мы выпускаем релиз каждые две недели и проводим ретроспективы после каждого спринта.",
    "У нас принято писать тесты и проводить код-ревью для каждого изменения.",
    "Офис находится рядом с метро, есть кухня, переговорные и парковка для велосипедов.",
    "Продукт используют несколько сотен компаний малого и среднего бизнеса.",
    "Мы ценим прозрачность, поэтому все решения по архитектуре записываем в общий документ.",
    "За последний год команда выросла вдвое и запустила два новых направления.",
    "Наставник помогает новому сотруднику пройти адаптацию в первые месяцы.",
    "Раз в квартал проходит внутренний хакатон, лучшие идеи попадают в план разработки.",
    "Данные хранятся в собственном кластере, доступ к ним выдаётся по ролям.",
    "Мы поддерживаем участие в конференциях и оплачиваем профильные курсы.",
    "Рабочий день начинается в удобное время, ядро встреч с одиннадцати до шестнадцати.",
    "Инфраструктура описана как код, изменения проходят через пулл-реквесты.",
    "Поддержка пользователей работает круглосуточно, инженеры дежурят по графику.",
]


def _salary(rng: random.Random) -> tuple[dict[str, Any] | None, str]:
    kind = rng.choice(["rub_range", "rub_from", "usd_to", "negotiable", "absent"])
    if kind == "rub_range":
        lo = rng.randrange(80, 300, 10) * 1000
        hi = lo + rng.randrange(30, 150, 10) * 1000
        net = rng.random() < 0.5
        text = f"Зарплата от {lo:,} до {hi:,} руб. {'на руки' if net else 'до вычета НДФЛ'}.".replace(",", " ")
        return {
            "from": lo,
            "to": hi,
            "currency": "RUB",
            "gross_net": "net" if net else "gross",
            "period": "month",
        }, text
    if kind == "rub_from":
        lo = rng.randrange(100, 350, 10) * 1000
        return {"from": lo, "to": None, "currency": "RUB", "gross_net": "unknown", "period": "month"}, (
            f"Доход от {lo // 1000}к в месяц."
        )
    if kind == "usd_to":
        hi = rng.randrange(2000, 7000, 500)
        return {"from": None, "to": hi, "currency": "USD", "gross_net": "gross", "period": "month"}, (
            f"Вилка до ${hi} gross в месяц."
        )
    if kind == "negotiable":
        return None, "Зарплата по итогам собеседования."
    return None, ""


def _experience(rng: random.Random, grade: str) -> tuple[dict[str, int | None] | None, str]:
    if grade == "intern" or rng.random() < 0.25:
        return None, ""
    lo = {"junior": 1, "middle": 2, "senior": 4, "lead": 5}[grade]
    if rng.random() < 0.5:
        return {"min": lo, "max": None}, f"Опыт работы от {lo} лет."
    return {"min": lo, "max": lo + 2}, f"Опыт работы от {lo} до {lo + 2} лет."


def make_example(rng: random.Random, long: bool) -> dict[str, Any]:
    company = rng.choice(COMPANIES)
    role = rng.choice(list(ROLES))
    title, pool = ROLES[role]
    grade = rng.choice(list(GRADES))
    skills = rng.sample(pool, k=rng.randint(3, min(6, len(pool))))
    salary, salary_text = _salary(rng)
    work_format = rng.choice([*WORK_FORMATS, None])
    cities = rng.sample(CITIES, k=rng.choice([0, 1, 1, 2]))
    experience, experience_text = _experience(rng, grade)
    employment = "internship" if grade == "intern" else rng.choice(["labor_contract", "gph", "self_employed", None])

    n_boiler = rng.randint(55, 70) if long else rng.randint(2, 12)
    about = " ".join(rng.choice(BOILERPLATE) for _ in range(n_boiler))
    lines = [
        f"{GRADES[grade]} {title} в компанию «{company}».",
        about,
        "Что нужно делать: развивать сервисы команды, участвовать в планировании и ревью.",
        "Что ждём: " + ", ".join(skills) + ".",
        experience_text,
        salary_text,
        WORK_FORMATS[work_format] if work_format else "",
        ("Город: " + ", ".join(cities) + ".") if cities else "",
        EMPLOYMENT[employment] if employment else "",
    ]
    fields = {
        "role": role,
        "grade": grade,
        "skills": skills,
        "salary": salary,
        "work_format": work_format,
        "cities": cities,
        "experience_years": experience,
        "employment_type": employment,
    }
    return {"text": "\n".join(line for line in lines if line), "fields": fields}


def build_examples(n: int = 48, seed: int = 13) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [make_example(rng, long=(i % 6 == 5)) for i in range(n)]


def messages(text: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]


def target(fields: dict[str, Any]) -> str:
    return json.dumps(fields, ensure_ascii=False)
