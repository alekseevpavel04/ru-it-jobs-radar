# n8n with a bare CPython for the Execute Command steps of the weekly digest: radar.site.digest needs
# the standard library only. The n8n image is a hardened Alpine without a package manager, so the
# interpreter comes from the official Python image of the same Alpine release; it links against musl
# and nothing else (ldd of python3.11 and libpython3.11.so).
FROM python:3.11.16-alpine3.24 AS python

FROM n8nio/n8n:2.41.3
COPY --from=python /usr/local/lib/libpython3.11.so.1.0 /usr/local/lib/
COPY --from=python /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=python /usr/local/bin/python3.11 /usr/local/bin/python3
