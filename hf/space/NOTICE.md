# Лицензии

Код этого Space распространяется под лицензией MIT.

Страница скачивает в браузер GGUF модели [ru-it-vacancy-extractor-qwen3-1.7b](https://huggingface.co/alekseevpavel04/ru-it-vacancy-extractor-qwen3-1.7b) и показывает её выход. У модели смешанная лицензия.

- База `Qwen/Qwen3-1.7B` распространяется по лицензии Apache License 2.0, текст в [licenses/Qwen3-1.7B-LICENSE.txt](licenses/Qwen3-1.7B-LICENSE.txt). Токенизатор в `assets/tokenizer/` взят из той же базы.

  > Qwen3-1.7B, Copyright 2024 Alibaba Cloud. Licensed under the Apache License, Version 2.0.

- Часть обучающей разметки модели (посты Telegram) построена на ответах YandexGPT-5-Lite-8B-instruct. Лицензионное соглашение YandexGPT-5-Lite-8B относит модели, созданные на основе выходных данных его модели, к производным произведениям, поэтому модель распространяется и на его условиях. Копия соглашения в [licenses/YandexGPT-5-Lite-8B-LICENSE.txt](licenses/YandexGPT-5-Lite-8B-LICENSE.txt).

  > YandexGPT-5-Lite-8B распространяется на условиях Лицензионного соглашения YandexGPT-5-Lite-8B. Copyright (c) 2025, ООО «ЯНДЕКС». Все права защищены.

Библиотеки, которые лежат в `vendor/` без изменений, со своими лицензиями рядом:

- [wllama](https://github.com/ngxson/wllama) (llama.cpp в WebAssembly), MIT, [vendor/wllama/LICENCE](vendor/wllama/LICENCE).
- [tokenizers.js](https://github.com/huggingface/tokenizers.js), Apache License 2.0, [vendor/tokenizers/LICENSE](vendor/tokenizers/LICENSE).
