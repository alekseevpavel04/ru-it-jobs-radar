# radar.monitor.exporter: the project's src/ is mounted read-only, the image only carries the two
# libraries the module imports besides the standard library (versions as in uv.lock).
FROM python:3.11.16-alpine3.24

RUN pip install --no-cache-dir httpx==0.28.1 prometheus-client==0.26.0
