FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl \
    && rm -rf /var/lib/apt/lists/*

# Don't bake CPU-specific instructions into the build — the Cloud Build
# machine's CPU may differ from the Cloud Run runtime's. Small portability
# cost, avoids "illegal instruction" crashes at runtime.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY static ./static

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
