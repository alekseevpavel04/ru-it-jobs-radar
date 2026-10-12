# Ежедневный сбор на сервере

Коллекторы trudvsem (API и ежедневные снимки), Хабр Карьеры, SuperJob и Telegram работают на отдельном домашнем сервере (Ubuntu 24.04, контейнер Incus, 2 ядра, 3 ГБ памяти) от непривилегированного пользователя. Стека там нет: только Python и Parquet. Локальная машина забирает готовые партиции и уже у себя кладёт их в Redpanda и ClickHouse.

## Что где лежит

| | сервер | локально |
|---|---|---|
| код | `~/ru-it-jobs-radar` (копия `src/`, `scripts/daily_collect.sh`, `pyproject.toml`, `uv.lock`) | репозиторий |
| окружение | `~/.local/bin/uv`, Python 3.11 от uv, `.venv` проекта | `.venv` |
| настройки | `.env` с `RADAR_USER_AGENT`, `RADAR_DATA_DIR` и `SUPERJOB_APP_KEY` (с 2026-10-12), права 600 | `.env` |
| данные | `data/raw/{trudvsem,habr,trudvsem_snapshot,superjob,telegram}` | `data/raw/...` |
| логи | `logs/daily/<дата>.log`, `logs/daily/cron-<месяц>.log` | `logs/daily/` |

`REDPANDA_BROKERS` на сервере не задан: каждый коллектор пишет одно предупреждение и завершается с кодом 0. Снимки trudvsem сервер берёт начиная с версии 2026-10-09 (`RADAR_SNAPSHOT_SINCE` в строке cron), более старые догружает локальный бэкфилл.

## Доступ

Скрипты обращаются к серверу по ssh-алиасу `radar-vps` (или по `RADAR_VPS`). Блок для `~/.ssh/config`, адрес, порт и пользователь из своих записей о сервере:

```
Host radar-vps
    HostName <адрес>
    Port <порт бокса>
    User <пользователь бокса>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

## Расписание

cron пользователя, раз в сутки в 21:30 по Москве (на сервере UTC, поэтому `30 18 * * *`). Запуск занят своим замком `logs/daily/.lock`, второй параллельно не стартует. Логи старше 60 дней скрипт удаляет сам (`RADAR_LOG_DAYS`). Данные на сервере не удаляются.

```
ssh radar-vps crontab -l                       # посмотреть
ssh radar-vps 'crontab -l | grep -v ru-it-jobs-radar | crontab -'   # остановить
ssh radar-vps 'tail -n 5 ru-it-jobs-radar/logs/daily/$(date +%F).log'
```

## Обновить код

Из Git Bash в корне репозитория:

```
ssh radar-vps 'rm -rf ru-it-jobs-radar/src'
tar cf - --exclude=__pycache__ src configs scripts/daily_collect.sh pyproject.toml uv.lock README.md | ssh radar-vps 'tar xf - -C ru-it-jobs-radar'
ssh radar-vps 'cd ru-it-jobs-radar && ~/.local/bin/uv sync --frozen --no-dev'
```

Не во время запуска: в 21:30 и примерно час после.

## Забрать данные

`bash scripts/sync_from_vps.sh` из Git Bash. Скрипт только добавляет: забирает партиции с `_SUCCESS`, которых нет локально или которые на сервере выросли из локальной копии, новые строки `_meta/*.jsonl` и `_rejects/*.jsonl`. Локальные part-файлы не перезаписываются, локально ничего не удаляется. День, который собрали обе машины независимо, пропускается с сообщением `skip`. После синхронизации числа в `results/collect/` пересчитывает `python -m radar.collect.<источник> --summarize`.
