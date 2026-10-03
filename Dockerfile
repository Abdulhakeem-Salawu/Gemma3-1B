# Prebuilt BASE image: Python + build tools + a compiled llama-cpp-python.
# (Copy of Dockerfile.base placed at the repo root of this throwaway branch so
# `gcloud builds submit --tag` can build it; see main's Dockerfile.base.)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl \
    && rm -rf /var/lib/apt/lists/*

# Don't bake CPU-specific instructions into the build - the Cloud Build
# machine's CPU may differ from the Cloud Run runtime's. Small portability
# cost, avoids "illegal instruction" crashes at runtime.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF"

ARG LLAMA_CPP_PYTHON_VERSION=0.3.36
RUN pip install --no-cache-dir llama-cpp-python==${LLAMA_CPP_PYTHON_VERSION}
