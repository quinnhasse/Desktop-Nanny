# Multi-stage build — select VARIANT=cpu (default) or VARIANT=cuda at build time.
#
#   docker build --build-arg VARIANT=cpu -t desktop-nanny:cpu .
#   docker build --build-arg VARIANT=cuda -t desktop-nanny:cuda .

ARG VARIANT=cpu

# ── CPU base ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base-cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── CUDA base ─────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base-cuda

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-distutils \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency builder (shared logic) ────────────────────────────────────────
FROM base-${VARIANT} AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Runtime image ─────────────────────────────────────────────────────────────
FROM base-${VARIANT} AS runtime

# Copy installed packages from builder
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY nanny.py detector.py actions.py config.yaml ./

# Prometheus metrics endpoint
EXPOSE 8000

ENTRYPOINT ["python", "nanny.py"]
CMD ["--source", "screen", "--config", "config.yaml", "--metrics-port", "8000"]
