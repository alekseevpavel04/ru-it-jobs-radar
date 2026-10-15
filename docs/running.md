# Как запустить

Проверка кода и чисел без моих данных, свой сбор и стек. Коротко в [README](../README.md).

Без моих данных можно проверить код и все опубликованные числа. Нужны Python 3.11 и uv:

```bash
uv sync --frozen
bash scripts/check.sh
```

`check.sh` запускает ruff, mypy, pytest и `verify_numbers`. Тесты в сеть не ходят: коллекторы проверяются на записанных ответах источников, где тексты, компании и контакты заменены выдуманными ([`tests/fixtures/README.md`](../tests/fixtures/README.md)). `verify_numbers` сверяет каждое число в README, `docs/` и на сайте с `results/` и проверяет, что у каждого файла результатов есть команда, коммит и знаменатели.

Данные (сырой слой, эталон, silver, предсказания) версионируются DVC, но хранилище DVC локальное и не опубликовано. Поэтому переобучение и пересчёт метрик требуют своего сбора. В каждом файле `results/` записаны команда, коммит и sha256 входов, по ним видно, чем число получено.

Свой сбор и стек:

```bash
cp .env.example .env                 # User-Agent с контактом, пароли стека
bash scripts/daily_collect.sh        # все источники по очереди, Parquet в data/raw
bash scripts/up.sh core              # ClickHouse, Redpanda, Redis; профили ml, orchestration, mlops, bi
uv run python -m radar.stream.replay --source habr --all   # и так для каждого источника
bash scripts/dbt.sh build
bash scripts/e2e.sh                  # API и MCP, затем сквозной тест
bash scripts/k8s_smoke.sh            # Helm-чарт в одноразовом кластере kind
```

Стеку нужен Docker Engine с Compose. Стек поднимался только на Windows: `up.sh` из Git Bash перезапускает себя внутри WSL-дистрибутива `radar` с Docker Engine, а Airflow монтирует `~/.ssh` с алиасом сервера сбора. На Linux `up.sh` работает напрямую, но на чистой машине этот путь не проверялся. Сколько времени и памяти нужно стеку, не измерено. Снимки портала тяжёлые: каждый скачивается целиком и фильтруется потоково. SuperJob без ключа API пропускается.
