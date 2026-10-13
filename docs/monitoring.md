# Мониторинг, реестр моделей и Kubernetes

Три части MLOps-слоя: реестр моделей в MLflow, метрики и дашборды Prometheus и Grafana (9.3), Helm-чарт сервисов с проверкой в локальном kind (11.1). Всё описано файлами в репозитории и поднимается командами ниже.

## Реестр моделей MLflow

Что лежит в реестре, записано в `infra/compose/mlflow/registry.toml`. Команда

```bash
python -m radar.monitor.registry          # регистрирует новое и выставляет алиасы
python -m radar.monitor.registry --list   # модели, версии, алиасы и главные метрики
```

делает из каждой записи прогон эксперимента `radar-registry` и версию зарегистрированной модели. В прогоне лежат файлы модели (`model/`), файлы `results/`, из которых взяты метрики (`results/`), и файлы происхождения (`provenance/`); метрики прогона это числа из тех же файлов `results/`, с границами бутстрап-интервалов. Источник версии это `model/` прогона, поэтому `models:/radar-salary@champion` указывает на конкретные байты в хранилище артефактов MLflow, а не на путь на диске.

| Модель | Что в версии | Откуда метрики |
|---|---|---|
| `radar-extractor` | адаптер LoRA, GGUF Q4_K_M для llama-server и Space, AWQ W4A16 для vLLM | `results/extract/dev_sft_1p7b_r16_clean.json`, `tg_dev_sft_1p7b_r16_clean.json`, `quant_summary.json` |
| `radar-salary` | `salary.pkl` (бустинг с конформными квантилями) | `results/models/salary.json`, только метрики dev |

Вместо стадий (Staging, Production) алиасы: MLflow с версии 2.9 считает стадии устаревшими. `champion` это версия в работе, `challenger` кандидат, которого сравнивают с ней. Алиасы выставляются ровно так, как записано в файле: алиас, которого в файле нет, снимается с модели.

Повторный запуск безопасен. Уже зарегистрированная запись находится по ключу и не перечитывается: в реестре своя копия файлов, так что обучение может перезаписать `models/salary/` следующей версией (13.10 новое обучение перезаписало `salary.pkl` через 13 минут после регистрации v0, копия v0 в реестре цела). Новая запись получает отпечаток (sha256 по путям и содержимому файлов модели), и те же файлы под вторым ключом скрипт не примет. Новая версия модели это новая запись `[[version]]` с новым ключом; алиас переносится правкой файла и повторным запуском.

Отпечаток версии проверяется и при скачивании: `python -m radar.monitor.fetch_model --model radar-salary --alias champion --out DIR` берёт версию по алиасу через REST API MLflow и кладёт файлы в `DIR`, только если отпечаток совпал. Так модель зарплаты получает под Kubernetes init-контейнер чарта.

Скриншоты: `node infra/compose/mlflow/screenshot.mjs` пишет `docs/img/mlflow_registry.png`, `mlflow_extractor.png`, `mlflow_salary.png`.

## Prometheus и Grafana

Сервисы профиля `mlops`: `prometheus` (127.0.0.1:9090), `grafana` (127.0.0.1:3030, только просмотр без входа), `radar-exporter` (127.0.0.1:9188). Весь профиль поднимает `scripts/up.sh mlops`, вместе с n8n; без n8n

```bash
docker compose -f infra/compose/docker-compose.yml --env-file .env --profile mlops up -d radar-exporter prometheus grafana
```

(внутри WSL, как остальные команды compose). У сервисов нет зависимостей от профилей `core`, `api` и `orchestration`: недоступная цель видна как `down`, а запуск мониторинга не пересоздаёт ClickHouse.

Что собирается (`infra/prometheus/prometheus.yml`):

| Цель | Откуда | Что даёт |
|---|---|---|
| `api:8000/metrics` | `radar.monitor.api_metrics.instrument(app)` | запросы по шаблону маршрута и коду ответа, гистограмма задержки, запросы в работе, память и CPU процесса |
| `statsd-exporter:9102` | Airflow по StatsD | DAG-раны и их длительность, успехи и отказы задач, отставание сырых данных из DAG `radar_sla` |
| `radar-exporter:9188` | `radar.monitor.exporter`, агрегатные запросы к ClickHouse раз в минуту | свежесть каждого источника (`max(dt)`), строки слоёв и время последней сборки dbt, прогресс извлечения студентом, Kafka-движок ClickHouse, `system.metrics` и `system.events` |
| `redpanda:9644/public_metrics` | Redpanda | offset топиков и группы `clickhouse_raw`, из них отставание консьюмера |

Правила в `infra/prometheus/rules.yml`: три записывающих (возраст партиции в днях, скорость извлечения в строках в минуту, отставание консьюмера) и четыре алерта (источник без новой партиции больше 3 дней, ClickHouse не отвечает экспортёру, консьюмер стоит при непрочитанных сообщениях, больше 5% ответов API с ошибкой 5xx). Alertmanager нет: алерты видны на странице Alerts в Prometheus.

Дашборды лежат JSON-файлами в `infra/grafana/dashboards`, источник данных в `infra/grafana/provisioning`; Grafana перечитывает их каждые 30 секунд, правки через интерфейс не сохраняются.

| Дашборд | Что на нём | Скриншот |
|---|---|---|
| Радар: данные и свежесть | возраст последней партиции по источникам, извлечение корпуса и его скорость, отставание консьюмера, строки `raw.vacancies`, часы с последней сборки таблиц, память и запросы ClickHouse | [grafana_data.png](img/grafana_data.png) |
| Радар: пайплайны Airflow | успешные и упавшие DAG-раны, отставание по SLA, срабатывания SLA, длительность ранов, задачи, разбор файлов DAG, цикл планировщика | [grafana_airflow.png](img/grafana_airflow.png) |
| Радар: API | доступность, запросы в секунду, доля 5xx, p50 и p95 по маршрутам, память и CPU | [grafana_api.png](img/grafana_api.png) |

Скриншоты: `node infra/grafana/screenshot.mjs [--from now-24h]`; панели без данных скрипт перечисляет и завершается с кодом 1.

Собственный эндпоинт Prometheus у ClickHouse выключен: он требует файла в `config.d` и пересоздания сервера. Пока его нет, те же `system.metrics` и `system.events` отдаёт `radar-exporter` с префиксами эндпоинта ClickHouse.

## Kubernetes: Helm-чарт и проверка в kind

Чарт `deploy/helm/radar` ставит три Deployment с Service из одного образа API (`radar/api`): API, MCP-сервер и экспортёр данных. ClickHouse, Redis и MLflow остаются снаружи, их адреса в `values.yaml`. Пароли ClickHouse приходят из Secret `radar-clickhouse`, который чарт не создаёт. У API и MCP init-контейнер скачивает из реестра модель зарплаты по алиасу `champion` с проверкой отпечатка, так что смена модели в работе это перенос алиаса и перезапуск Deployment. Контейнеры работают от `nobody` с корневой файловой системой только для чтения.

`scripts/k8s_smoke.sh` работает только с кластером kind `radar` (контекст `kind-radar` зашит в скрипт): создаёт его, если нет, подключает узел к сети compose (поды видят `clickhouse`, `redis`, `mlflow` по именам), загружает образ, создаёт Secret из `.env`, ставит чарт с ожиданием готовности и проверяет через port-forward, что `/health` видит модель из реестра и ClickHouse, `/extract` отвечает, `/metrics` API и экспортёра отдают метрики. Флаг `--delete` удаляет кластер в конце. Нужны профили `core` и `ml`, заполненный реестр и собранный образ API. Каждый прогон, удачный или нет, пишет `results/k8s/smoke.json`: версии kind, kubectl, helm и Kubernetes, образ, время каждого шага (создание кластера, загрузка образа, установка чарта с ожиданием готовности, удаление), исход и время каждой проверки, состояние подов и память узла kind.

`helm lint` и `helm template` проверяет `tests/deploy/test_chart.py`, когда `helm` есть в PATH (в CI); без него эти тесты пропускаются, остальные проверки чарта идут всегда.

Установка инструментов в WSL (версии закреплены, контрольные суммы сверяются; в `~/.local/bin`, его добавляет в PATH сам `scripts/k8s_smoke.sh`):

```bash
mkdir -p ~/.local/bin && cd "$(mktemp -d)"
curl -fsSLO https://kind.sigs.k8s.io/dl/v0.33.0/kind-linux-amd64
curl -fsSLO https://kind.sigs.k8s.io/dl/v0.33.0/kind-linux-amd64.sha256sum
sha256sum -c kind-linux-amd64.sha256sum && install -m 0755 kind-linux-amd64 ~/.local/bin/kind
curl -fsSLO https://dl.k8s.io/release/v1.37.1/bin/linux/amd64/kubectl
curl -fsSLO https://dl.k8s.io/release/v1.37.1/bin/linux/amd64/kubectl.sha256
echo "$(cat kubectl.sha256)  kubectl" | sha256sum -c && install -m 0755 kubectl ~/.local/bin/kubectl
curl -fsSLO https://get.helm.sh/helm-v4.3.0-linux-amd64.tar.gz
curl -fsSLO https://get.helm.sh/helm-v4.3.0-linux-amd64.tar.gz.sha256sum
sha256sum -c helm-v4.3.0-linux-amd64.tar.gz.sha256sum && tar -xzf helm-v4.3.0-linux-amd64.tar.gz linux-amd64/helm
install -m 0755 linux-amd64/helm ~/.local/bin/helm
```
