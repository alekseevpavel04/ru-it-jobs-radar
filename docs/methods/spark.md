# Spark и ClickHouse на истории вакансий

Вопрос: окупается ли в этом проекте Spark, если пересчитать им ту же историю вакансий, которую dbt строит в ClickHouse. Ответ: на объёме проекта нет. PySpark в локальном режиме прочитал все <!--n:spark/bench.json:setup.snapshot_days:d-->105<!--/n--> ежедневных снимков trudvsem (<!--n:spark/bench.json:metrics.raw_rows.value:,d|ru-->54 969 344<!--/n--> строк) и получил тот же результат, что ClickHouse, строка в строку. На тяжёлом шаге, который читает весь сырой слой, движки идут наравне, на остальной истории Spark медленнее в разы, а памяти ему нужно заметно больше. Данные помещаются на одну машину, и тяжёлая часть упирается в процессор, так что выигрывать Spark здесь нечем.

Код `jobs/spark/history.py` (вычисление), `jobs/spark/rules.py` (правила, прочитанные из dbt), `jobs/spark/bench.py` (замер и сверка), результат `results/spark/bench.json`, пробы на трёх днях `results/spark/probe_mnt_d.json` и `results/spark/probe_ext4.json`.

## Что считается

История вакансий снимков trudvsem, те же три шага, что в `dbt/models/history`, для источника `trudvsem_snapshot`:

1. Наблюдения (`stg_trudvsem_snapshot` и `hist_snapshot_days`). Из каждого дневного снимка всех профессий берётся последняя за день строка вакансии, разбирается JSON `payload`, применяется IT-фильтр (сфера, типовая должность, регулярное выражение по названию) и считаются поля истории. Это единственный шаг, который читает весь сырой объём.
2. Версии SCD2 (`hist_calendar` и `hist_vacancy_scd2`): полные дни снимков, 7 полных дней отсутствия до закрытия, переоткрытия, список изменённых полей.
3. Время жизни (`hist_vacancy_lifetime`) с цензурированием, без колонок перепубликаций (`hist_reposts` не переносился).

Правила, которые в dbt заданы данными (доля полного дня, дни ожидания закрытия, шаг между днями, регулярное выражение IT-фильтра, список типовых должностей), Spark читает из макросов и seed dbt, поэтому разойтись с ними молча не может; `tests/spark/test_rules.py` сверяет регулярное выражение с тем, что dbt компилирует в SQL. Сам SQL переведён в DataFrame API вручную, поэтому каждая строка результата сверяется с ClickHouse.

ClickHouse исполняет SQL, который скомпилировал dbt (`dbt/target/compiled`), так же, как его запускает dbt: наблюдения по одному снимку за запрос (<!--n:spark/bench.json:clickhouse.observations_queries.queries:d-->105<!--/n--> запросов), потом SCD2 и время жизни. Настройки как в конфигурации моделей (`max_block_size` 8192), потоков столько же, сколько ядер у Spark (<!--n:spark/bench.json:setup.cores:d-->4<!--/n-->). Spark читает сырой слой из Parquet (`data/raw/trudvsem_snapshot`, <!--n:spark/bench.json:setup.raw_parquet_bytes:,d|ru-->38 968 450 815<!--/n--> байт), ClickHouse из своей таблицы `raw.vacancies`. Куча драйвера Spark 2 ГБ: больше на виртуальной машине WSL с 12 ГБ памяти не выделить, пока на ней идут задачи на GPU.

## Результат

| Шаг | Spark | ClickHouse |
|-----|-------|------------|
| Наблюдения, с | <!--n:spark/bench.json:metrics.spark_observations_s.value:,.0f\|ru-->1 424<!--/n--> | <!--n:spark/bench.json:metrics.clickhouse_observations_s.value:,.0f\|ru-->1 495<!--/n--> |
| SCD2 и время жизни, с | <!--n:spark/bench.json:metrics.spark_history_s.value:.1f\|ru-->8,8<!--/n--> | <!--n:spark/bench.json:metrics.clickhouse_history_s.value:.1f\|ru-->1,6<!--/n--> |
| Пик памяти, МиБ | <!--n:spark/bench.json:metrics.spark_peak_rss_mib.value:,d\|ru-->2 848<!--/n--> | <!--n:spark/bench.json:metrics.clickhouse_peak_query_memory_mib.value:,d\|ru-->995<!--/n--> |

Память меряется по-разному, и сравнивать её нужно с этой оговоркой. У Spark это пик резидентной памяти процесса драйвера вместе с JVM (выборка раз в 0,2 с), у ClickHouse наибольшая `memory_usage` одного запроса из `system.query_log`, без памяти самого сервера.

Совпадение результатов:

| Что сравнивается | Строк слева | Строк справа | Только слева | Только справа | Из них вакансий нет в сыром слое |
|------------------|-------------|--------------|--------------|---------------|----------------------------------|
| Наблюдения, Spark и ClickHouse | <!--n:spark/bench.json:metrics.observations_spark_vs_clickhouse.rows_left.value:,d\|ru-->995 762<!--/n--> | <!--n:spark/bench.json:metrics.observations_spark_vs_clickhouse.rows_right.value:,d\|ru-->995 762<!--/n--> | <!--n:spark/bench.json:metrics.observations_spark_vs_clickhouse.only_left.value:d-->0<!--/n--> | <!--n:spark/bench.json:metrics.observations_spark_vs_clickhouse.only_right.value:d-->0<!--/n--> | |
| Наблюдения, Spark и таблица dbt | <!--n:spark/bench.json:metrics.observations_spark_vs_dbt.rows_left.value:,d\|ru-->995 762<!--/n--> | <!--n:spark/bench.json:metrics.observations_spark_vs_dbt.rows_right.value:,d\|ru-->995 764<!--/n--> | <!--n:spark/bench.json:metrics.observations_spark_vs_dbt.only_left.value:d-->0<!--/n--> | <!--n:spark/bench.json:metrics.observations_spark_vs_dbt.only_right.value:d-->2<!--/n--> | <!--n:spark/bench.json:metrics.observations_spark_vs_dbt.only_right_not_in_raw.value:d-->2<!--/n--> |
| SCD2, Spark и ClickHouse | <!--n:spark/bench.json:metrics.scd2_spark_vs_clickhouse.rows_left.value:,d\|ru-->38 730<!--/n--> | <!--n:spark/bench.json:metrics.scd2_spark_vs_clickhouse.rows_right.value:,d\|ru-->38 731<!--/n--> | <!--n:spark/bench.json:metrics.scd2_spark_vs_clickhouse.only_left.value:d-->0<!--/n--> | <!--n:spark/bench.json:metrics.scd2_spark_vs_clickhouse.only_right.value:d-->1<!--/n--> | <!--n:spark/bench.json:metrics.scd2_spark_vs_clickhouse.only_right_not_in_raw.value:d-->1<!--/n--> |
| SCD2, ClickHouse и таблица dbt | <!--n:spark/bench.json:metrics.scd2_clickhouse_vs_dbt.rows_left.value:,d\|ru-->38 731<!--/n--> | <!--n:spark/bench.json:metrics.scd2_clickhouse_vs_dbt.rows_right.value:,d\|ru-->38 731<!--/n--> | <!--n:spark/bench.json:metrics.scd2_clickhouse_vs_dbt.only_left.value:d-->0<!--/n--> | <!--n:spark/bench.json:metrics.scd2_clickhouse_vs_dbt.only_right.value:d-->0<!--/n--> | |
| Время жизни, Spark и ClickHouse | <!--n:spark/bench.json:metrics.lifetime_spark_vs_clickhouse.rows_left.value:,d\|ru-->25 796<!--/n--> | <!--n:spark/bench.json:metrics.lifetime_spark_vs_clickhouse.rows_right.value:,d\|ru-->25 797<!--/n--> | <!--n:spark/bench.json:metrics.lifetime_spark_vs_clickhouse.only_left.value:d-->0<!--/n--> | <!--n:spark/bench.json:metrics.lifetime_spark_vs_clickhouse.only_right.value:d-->1<!--/n--> | <!--n:spark/bench.json:metrics.lifetime_spark_vs_clickhouse.only_right_not_in_raw.value:d-->1<!--/n--> |

Строка совпадает, только если совпали все сравниваемые колонки: даты версий, причины, закрытия, список изменённых полей, зарплаты, навыки. Наблюдения Spark и свежий пересчёт ClickHouse из того же сырого слоя совпали полностью. Все остальные расхождения это строки, которых нет у Spark, и их вакансий нет в сыром слое вовсе (последний столбец). Это одна вакансия с синтетическим идентификатором `00000000-0000-4000-8000-0000000000a1` (список в `diffs` результата): она лежит в таблице `staging.hist_snapshot_days`, построенной dbt, и оттуда попадает в SCD2 и время жизни ClickHouse, которые читают таблицы dbt. Это остаток тестовых данных в рабочей таблице, а не разница движков. Дубликатов вакансии внутри одного снимка в сыром слое нет (<!--n:spark/bench.json:metrics.duplicate_vacancy_days.value:d-->0<!--/n-->), так что правило «последняя строка за день» на этих данных ничего не отбрасывало.

Хэш текста считается разными функциями (в Spark нет `cityHash64`, взят `xxhash64`), поэтому `text_hash` сравнивается только как наличие текста. Сам хэш нужен истории только для сравнения версий между собой, и список изменённых полей, где он участвует, совпал.

## Почему Spark здесь не выигрывает

Объём помещается на одну машину. После первого шага от сырого слоя остаётся <!--n:spark/bench.json:metrics.observations_spark_vs_clickhouse.rows_left.value:,d|ru-->995 762<!--/n--> строк IT-наблюдений, и SCD2 со временем жизни на них занимают секунды в обоих движках. На таком объёме у Spark доминируют собственные расходы (планирование задач, перемешивание, запись промежуточного Parquet): <!--n:spark/bench.json:metrics.spark_history_s.value:.1f|ru-->8,8<!--/n--> с против <!--n:spark/bench.json:metrics.clickhouse_history_s.value:.1f|ru-->1,6<!--/n--> с.

Тяжёлый шаг упирается в процессор: разбор JSON и регулярное выражение по названию для каждой строки сырого слоя. Здесь Spark и ClickHouse на одинаковом числе потоков почти равны, но ClickHouse обходится меньшей памятью. Parquet лежит на диске Windows и читается из WSL через `/mnt/d`; копия трёх снимков на диске WSL дала Spark <!--n:spark/probe_ext4.json:metrics.spark_observations_s.value:.0f-->44<!--/n--> с против <!--n:spark/probe_mnt_d.json:metrics.spark_observations_s.value:.0f-->53<!--/n--> с, но ClickHouse, которому место Parquet безразлично, в тех же двух прогонах тоже ускорился, с <!--n:spark/probe_mnt_d.json:metrics.clickhouse_observations_s.value:.0f-->38<!--/n--> до <!--n:spark/probe_ext4.json:metrics.clickhouse_observations_s.value:.0f-->30<!--/n--> с. Разница того же порядка, что колебания нагрузки машины, и диск узким местом не выглядит.

Наконец, Spark требует JVM и отдельной кучи в памяти, которую в проекте делят ClickHouse, Airflow и модели на GPU.

## Когда Spark окупится

- Сырой слой перестаёт помещаться на один сервер ClickHouse по диску или по памяти (dbt уже строит наблюдения по одному дню, потому что запрос по всем дням не помещается в бюджет памяти сервера, см. `dbt/macros/incremental_by_day.sql`). Spark масштабируется добавлением узлов; локальный режим здесь меряет только накладные расходы движка, а не масштабирование.
- Данные лежат в объектном хранилище и в базу не загружаются: Spark читает Parquet там, где он лежит. В проекте ClickHouse и так держит сырой слой, и этот довод не работает.
- Обработка на строку, которую не выразить SQL: токенизация, признаки для моделей, применение модели к корпусу через pandas UDF. Здесь такая обработка идёт отдельными заданиями на Python и GPU, и узкое место в них модель, а не движок данных.

Пока ни одно из условий не выполнено, ClickHouse хватает: один движок, один язык (SQL в dbt), данные уже в нём. Spark-версия остаётся в репозитории как проверенный второй путь для истории: она даёт тот же результат, и при росте объёма с неё можно начать.

## Ограничения

- Один полный прогон на нагруженной машине: на той же виртуальной машине WSL работали задачи на GPU с процессорной частью. Разброс между прогонами повторениями не оценивался, а пробы выше показывают, что он сравним с разницей движков на тяжёлом шаге. Поэтому равенство на этом шаге надёжнее, чем знак разницы.
- Spark в локальном режиме на одной машине, 4 ядра и 2 ГБ кучи; кластер не проверялся.
- ClickHouse считает наблюдения по дню, как в dbt; один запрос на все дни не проверялся.
- Время жизни сравнивается без колонок перепубликаций.

## Как запустить

Нужны Java 21 и отдельное окружение со Spark в дистрибутиве WSL (версии и пути в `D:/VScode_projects/assets/INSTALLED.md`), поднятый ClickHouse с построенными моделями истории и свежий `dbt compile`: скрипт отказывается работать со скомпилированным SQL старше моделей или их макросов.

```bash
export JAVA_HOME=$HOME/.local/opt/jdk-21.0.12.1+1-jre PYTHONPATH=src
~/venvs/spark/bin/python jobs/spark/bench.py --days 3 --result probe_mnt_d.json   # проба на трёх днях, несколько минут
~/venvs/spark/bin/python jobs/spark/bench.py                                      # все снимки, около часа
~/venvs/spark/bin/python -m pytest --noconftest tests/spark                       # тесты на маленьких примерах
```
