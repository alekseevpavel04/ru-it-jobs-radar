# API и MCP-сервер

Два процесса из одного образа (`infra/compose/api/Dockerfile`, профиль compose `api`): HTTP API на FastAPI и MCP-сервер. Оба только читают: витрины через read-only пользователя ClickHouse, модель зарплаты из файла. Запуск вместе с ядром стека и проверка главного пути одной командой:

```
bash scripts/e2e.sh            # scripts/up.sh core api, затем tests/api/test_e2e.py
bash scripts/e2e.sh --no-up    # только проверка против того, что уже запущено
```

Локально без Docker: `python -m radar.api` (порт 8000) и `python -m radar.mcp` (stdio) или `python -m radar.mcp --http` (порт 8001), настройки из `.env`.

## HTTP API

| Метод | Что отдаёт |
|---|---|
| `GET /health` | доступность ClickHouse, Redis, ученика (если задан) и версию загруженной модели зарплаты; `degraded`, если чего-то нет |
| `POST /extract {text}` | поля схемы 5.2, бэкенд (`student` или `regex`), версию модели, причину отката на регулярки |
| `GET /stats/weekly`, `/stats/salaries`, `/stats/skills` | агрегаты витрин, у каждой доли знаменатель, кэш в Redis |
| `POST /salary {fields, source}` | оценку в рублях в месяц: `point`, интервал `low..high` на номинальном уровне `level`, типичную вилку, `n_similar`, версию модели |
| `POST /match {resume_text, k}` | 501 до модели подбора; формат запроса и ответа уже зафиксирован |

`/salary` принимает поля целиком, как их возвращает `/extract`; указанная в полях зарплата не читается. `source` по умолчанию `trudvsem`: на нём почти все обучающие вакансии. Источник, которого модель не видела в обучении, и роль `non_it` получают 422, а не оценку. `n_similar` это число обучающих вакансий с той же ролью, грейдом и регионом; 0 показывается как есть и значит, что оценка держится только на более грубых закономерностях. Суммы как их публикуют источники, gross или net они не указывают.

## MCP

| Инструмент | Что делает |
|---|---|
| `ask_market(question)` | вопрос агенту-аналитику (LangGraph): один SELECT по `marts.*` от read-only пользователя, ответ с цифрами, SQL и строками результата |
| `estimate_salary(role, grade, cities, skills, work_format, experience_min_years, employment_type, source)` | та же функция, что за `POST /salary` |
| `find_vacancies(source, role_bucket, grade, region_group, work_format, skill, min_salary, limit)` | открытые вакансии по фильтрам, `limit` до 50, рядом число всех подходящих |

Все три помечены как read-only. Фильтры доходят до ClickHouse параметрами запроса, текст аргумента в SQL не попадает. `ask_market` нужен локальный LLM по `LLM_BASE_URL`; из контейнера сервер на хосте Windows недоступен (WSL в режиме NAT), поэтому в compose этот инструмент отвечает ошибкой, пока LLM не слушает адрес, видимый из контейнеров. `find_vacancies` читает витрину `marts.mart_open_vacancies`; пока её нет, инструмент так и отвечает.

Подключение настольного клиента по stdio:

```json
{
  "mcpServers": {
    "radar": {
      "command": "<репозиторий>/.venv/Scripts/python.exe",
      "args": ["-m", "radar.mcp"]
    }
  }
}
```

Вызов одного инструмента из консоли: `python -m radar.mcp call estimate_salary '{"role": "ml_engineer", "grade": "junior"}'` (по stdio) или с `--url http://127.0.0.1:8001/mcp` (по HTTP к сервису compose).

## Модель зарплаты

Файл `models/salary/salary.pkl` (вне git) сохраняет `python -m radar.models.salary`; в compose он монтируется только для чтения. API и MCP загружают модель при старте, поэтому после переобучения оба сервиса перезапускаются. Нет файла: `/salary` отвечает 503, остальное API работает, `/health` показывает `salary_model: missing`.
