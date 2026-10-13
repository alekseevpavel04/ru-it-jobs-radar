# Лицензия

У весов этой модели два источника условий.

## Qwen3-1.7B, Apache 2.0

Модель дообучена из `Qwen/Qwen3-1.7B`. Её веса распространяются по лицензии Apache License 2.0, полный текст в [licenses/Qwen3-1.7B-LICENSE.txt](licenses/Qwen3-1.7B-LICENSE.txt).

> Qwen3-1.7B, Copyright 2024 Alibaba Cloud. Licensed under the Apache License, Version 2.0.

Изменения относительно базы: LoRA-дообучение на извлечение полей из вакансий, слияние адаптера с весами, квантизация в GGUF.

## YandexGPT-5-Lite-8B, лицензионное соглашение Яндекса

Часть обучающей разметки (посты Telegram) построена на ответах модели YandexGPT-5-Lite-8B-instruct: в разметку шли значения, в которых согласились Qwen3-8B и YandexGPT-5-Lite-8B. Лицензионное соглашение YandexGPT-5-Lite-8B относит модели, созданные на основе выходных данных его модели, к производным произведениям. Поэтому эта модель, адаптер и GGUF распространяются и на условиях этого соглашения, а пользоваться ими можно только с соблюдением его ограничений (раздел II) и условий коммерческого использования (раздел I). Копия соглашения в [licenses/YandexGPT-5-Lite-8B-LICENSE.txt](licenses/YandexGPT-5-Lite-8B-LICENSE.txt), оригинал на странице [yandex/YandexGPT-5-Lite-8B-instruct](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct/blob/main/LICENSE).

> YandexGPT-5-Lite-8B распространяется на условиях Лицензионного соглашения YandexGPT-5-Lite-8B. Copyright (c) 2025, ООО «ЯНДЕКС». Все права защищены.

## Код

Код обучения и оценки лежит в [репозитории проекта](https://github.com/alekseevpavel04/ru-it-jobs-radar) под лицензией MIT.
