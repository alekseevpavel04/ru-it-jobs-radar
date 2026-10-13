# llama-server from the llama.cpp release the quantized student was scored with, CPU only. Every
# CPU variant is built and the server loads the best one the host supports, so the image runs on
# whatever processor the Space lands on.
FROM debian:bookworm-slim AS llama
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG LLAMA_CPP_TAG=b11052
RUN git clone --depth 1 --branch "${LLAMA_CPP_TAG}" https://github.com/ggml-org/llama.cpp /src
WORKDIR /src
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build build -j"$(nproc)" \
    && mkdir -p /out \
    && cp build/bin/llama-server /out/ \
    && find build -name "*.so*" -exec cp -P {} /out/ \;

FROM python:3.11.16-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=llama /out/ /opt/llama/
ENV PATH=/opt/llama:$PATH \
    LD_LIBRARY_PATH=/opt/llama

RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONPATH=/home/user/app:/home/user/app/vendor \
    PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["bash", "start.sh"]
