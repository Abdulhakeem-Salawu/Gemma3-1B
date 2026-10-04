# Declared before the first FROM so it's in global scope: an ARG used in a
# later `FROM ${BASE_IMAGE}` must be, or Docker sees it as blank.
ARG BASE_IMAGE=us-central1-docker.pkg.dev/pioneering-axe-233302/cloud-run-source-deploy/gemma-agent-base:current
ARG MODEL_BUCKET=pioneering-axe-233302-gemma-models
ARG MODEL_FILE=Qwen3-4B-Q4_K_M.gguf

# --- Model download stage -------------------------------------------------
# Pulls the GGUF from GCS at build time so it ships inside the image instead
# of being read through a gcsfuse mount on every cold start. Auth comes from
# the Cloud Build service account via the metadata server, which is only
# reachable when the build runs with `docker build --network cloudbuild`
# (the same flag the existing build command already uses).
# The mtime is pinned so the resulting layer has the same digest on every
# build of the same model file: Artifact Registry then stores and pushes the
# ~2.5 GB layer once instead of once per build.
FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:slim AS model
ARG MODEL_BUCKET
ARG MODEL_FILE
RUN mkdir -p /models \
    && gcloud storage cp "gs://${MODEL_BUCKET}/${MODEL_FILE}" "/models/${MODEL_FILE}" \
    && chmod 0644 "/models/${MODEL_FILE}" \
    && touch -d @0 "/models/${MODEL_FILE}" /models

# --- Frontend build stage -------------------------------------------------
# React + assistant-ui chat UI, built to static assets and baked into the
# final image below. Kept as its own stage so a Python-only dependency
# change doesn't force a full npm reinstall, and vice versa.
FROM node:22-slim AS web-build
WORKDIR /web
COPY web/package.json ./
RUN npm install
COPY web/ ./
RUN npm run build

# --- Application image -----------------------------------------------------
# The slow llama-cpp-python compile lives in a prebuilt base image
# (Dockerfile.base, built once with cloudbuild.base.yaml), so this build only
# installs the light dependencies and copies the code (~1-3 min) every time.
FROM ${BASE_IMAGE}

WORKDIR /app

# Model first: it's the biggest layer and changes least, so it sits below
# everything that changes on every commit.
ARG MODEL_FILE
COPY --from=model /models /models
ENV MODEL_PATH=/models/${MODEL_FILE}

# llama-cpp-python is already installed in the base image, so pip treats it
# as satisfied and does not recompile it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py chatlog.py ./
COPY --from=web-build /web/dist ./static

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
