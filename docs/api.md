# API и MCP-сервер

Два процесса из одного образа (`infra/compose/api/Dockerfile`, профиль compose `api`): HTTP API на FastAPI и MCP-сервер. Оба только читают: витрины через read-only пользователя ClickHouse, модель зарплаты из файла. Запуск вместе с ядром стека и проверка главного пути одной командой:

```
bash scripts/e2e.sh            # scripts/up.sh core api, затем tests/api/test_e2e.py
bash scripts/e2e.sh --no-up    # только проверка против того, что уже запущено
```

В профиле `api` три сервиса: `student` (llama.cpp на CPU с учеником Q4_K_M, наружу не публикуется), `api` и `mcp`. `/extract` в compose отвечает учеником, регулярки отвечают только пока `student` недоступен; `/health` показывает это в `extract_backend` и `checks.student`, а ответ `/extract` в `backend` и `fallback_reason`. Задержка `/extract` на CPU в `results/api/extract_latency.json` (`python -m radar.api.latency`).

Локально без Docker: `python -m radar.api` (порт 8000) и `python -m radar.mcp` (stdio) или `python -m radar.mcp --http` (порт 8001), настройки из `.env`.

## HTTP API

| Метод | Что отдаёт |
|---|---|
| `GET /health` | доступность ClickHouse, Redis, ученика (если задан) и версии загруженных модели зарплаты и индекса подбора; `degraded`, если чего-то нет |
| `POST /extract {text}` | поля схемы 5.2, бэкенд (`student` или `regex`), версию модели, причину отката на регулярки |
| `GET /stats/weekly`, `/stats/salaries`, `/stats/skills` | агрегаты витрин, у каждой доли знаменатель, кэш в Redis |
| `POST /salary {fields, source}` | оценку в рублях в месяц: `point`, интервал `low..high` на номинальном уровне `level`, типичную вилку, `n_similar`, версию модели |
| `POST /match {resume_text, k}` | до `k` (не больше 50) открытых вакансий, ближайших к тексту резюме: ранг, оценку и поля витрины `marts.mart_open_vacancies` по правилам публикации источника; версию индекса |

`/salary` принимает поля целиком, как их возвращает `/extract`; указанная в полях зарплата не читается. `source` по умолчанию `trudvsem`: на нём почти все обучающие вакансии. Источник, которого модель не видела в обучении, и роль `non_it` получают 422, а не оценку. `n_similar` это число обучающих вакансий с той же ролью, грейдом и регионом; 0 показывается как есть и значит, что оценка держится только на более грубых закономерностях. Суммы как их публикуют источники, gross или net они не указывают.

## MCP

| Инструмент | Что делает |
|---|---|
| `ask_market(question)` | вопрос агенту-аналитику (LangGraph): один SELECT по `marts.*` от read-only пользователя, ответ с цифрами, SQL и строками результата |
| `estimate_salary(role, grade, cities, skills, work_format, experience_min_years, employment_type, source)` | та же функция, что за `POST /salary` |
| `find_vacancies(source, role_bucket, grade, region_group, work_format, skill, min_salary, limit)` | открытые вакансии по фильтрам, `limit` до 50, рядом число всех подходящих |

Все три помечены как read-only. Фильтры доходят до ClickHouse параметрами запроса, текст аргумента в SQL не попадает. `ask_market` нужен локальный LLM по `LLM_BASE_URL` (как его подключить в compose, ниже). `find_vacancies` читает витрину `marts.mart_open_vacancies`; пока её нет, инструмент так и отвечает.

### LLM для ask_market в compose

Агент ходит в `LLM_BASE_URL` (в compose `http://host.docker.internal:8080/v1`, блок `x-api-env`) и до первого вопроса проверяет, что сервер отдаёт модель `Qwen3-8B-Q4_K_M`: на ней посчитан контрольный набор агента. Имя `host.docker.internal` в контейнерах `api` и `mcp` ведёт на шлюз Docker внутри ВМ WSL, а не в Windows. Поэтому llama-server, запущенный на Windows (`scripts/llm_server.sh qwen`), из контейнера не виден: WSL работает в режиме NAT, и связь идёт только из Windows в ВМ через проброс localhost.

Рабочий путь это сервер внутри дистрибутива `radar`, который слушает все адреса ВМ:

```
# в WSL: llama-server Linux-сборки с CUDA (или образ ghcr.io/ggml-org/llama.cpp:server-cuda-b<N>)
llama-server -m /mnt/d/VScode_projects/assets/llm/qwen3-8b/Qwen3-8B-Q4_K_M.gguf \
    --alias Qwen3-8B-Q4_K_M --host 0.0.0.0 --port 8080 -ngl 99 -fa on --jinja -c 8192
```

Флаги те же, что в `scripts/llm_server.sh qwen`: сервер с другой моделью или другим шаблоном чата даёт другие ответы, чем на контрольном наборе. vLLM в WSL подходит так же (`--host 0.0.0.0`, `--served-model-name Qwen3-8B-Q4_K_M`), но только с теми же весами в GGUF. Чекпойнт AWQ под тем же именем был бы другой моделью под чужой этикеткой. Если порт или адрес другие, поменяйте `LLM_BASE_URL` в `x-api-env` и пересоздайте `mcp` (`scripts/up.sh api`). Проверка `python -m radar.mcp call ask_market '{"question": "Сколько вакансий ML на Хабр Карьере?"}' --url http://127.0.0.1:8001/mcp`.

Порт ВМ на 0.0.0.0 виден контейнерам и через проброс localhost Windows, в локальную сеть он не выходит. Режим `networkingMode=mirrored` в `.wslconfig` открыл бы контейнерам сервер на Windows, но меняет сеть WSL для всей машины, поэтому здесь не используется. GPU одна: сервер на 8B занимает около 6 ГБ и не уживается с прогоном vLLM корпуса.

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

## Индекс подбора

`/match` отвечает индексом `models/matching/index.pkl` (`python -m radar.models.matching evaluate`, вне git) и энкодером запросов multilingual-e5-small в ONNX рядом с ним, `models/matching/onnx/` (`python -m radar.models.matching_onnx export` в WSL, где стоит torch). В образе API нет torch: запрос кодирует onnxruntime, токенизирует tokenizers, а префикс `query: `, среднее по токенам и L2-нормировка те же, что при построении индекса. Совпадение с векторами torch и nDCG@10 системы на dev с каждым из энкодеров в `results/models/matching_onnx.json` (`python -m radar.models.matching_onnx check`), задержка `/match` и память контейнера в `results/api/match_latency.json` (`python -m radar.api.match_latency`). В compose каталог `models/matching` монтируется только для чтения. Веса лежат отдельным файлом `model.onnx.data`: onnxruntime отображает его в память, а не копирует граф в кучу, и таблица эмбеддингов читается только по встреченным токенам. В образе выставлен `ORT_DISABLE_TELEMETRY=1`: Linux-сборка onnxruntime без него отправляет события использования в Microsoft. Нет индекса или экспорта: `/match` отвечает 503, `/health` показывает `match_index: missing`.
