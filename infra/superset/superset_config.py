"""Superset settings of the local stack (compose profile bi), read through SUPERSET_CONFIG_PATH."""

import os

SECRET_KEY = os.environ.get("SUPERSET_SECRET_KEY", "")
if not SECRET_KEY:
    raise RuntimeError("SUPERSET_SECRET_KEY is empty: set it in .env (see .env.example)")

# One local user and every chart, dataset and dashboard re-created from infra/superset/assets, so the
# metadata database is SQLite in the superset-home volume rather than one more Postgres.
SQLALCHEMY_DATABASE_URI = "sqlite:////app/superset_home/superset.db?check_same_thread=false"

APP_NAME = "Радар IT-вакансий"
ROW_LIMIT = 10000

# Russian number and date formats in charts: 12 345, 12,5 %, «окт» on time axes.
D3_FORMAT = {"decimal": ",", "thousands": " ", "grouping": [3], "currency": ["", " ₽"]}
D3_TIME_FORMAT = {
    "dateTime": "%A, %e %B %Y г. %X",
    "date": "%d.%m.%Y",
    "time": "%H:%M:%S",
    "periods": ["", ""],
    "days": ["воскресенье", "понедельник", "вторник", "среда", "четверг", "пятница", "суббота"],
    "shortDays": ["вс", "пн", "вт", "ср", "чт", "пт", "сб"],
    "months": [
        "январь",
        "февраль",
        "март",
        "апрель",
        "май",
        "июнь",
        "июль",
        "август",
        "сентябрь",
        "октябрь",
        "ноябрь",
        "декабрь",
    ],
    "shortMonths": ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"],
}
