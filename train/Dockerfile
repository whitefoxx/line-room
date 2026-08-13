# jaxlib publishes no macOS x86_64 wheels — only manylinux, Apple Silicon and
# Windows — so fine-tuning cannot run natively on an Intel Mac. The host is
# x86_64, so a linux/amd64 container runs at native speed with no emulation.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir cactus-needle

ENV HF_HOME=/cache/huggingface
WORKDIR /work
