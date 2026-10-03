# Declared before the first FROM so it's in global scope: an ARG used in a
# later `FROM ${BASE_IMAGE}` must be, or Docker sees it as blank.
ARG BASE_IMAGE=us-central1-docker.pkg.dev/pioneering-axe-233302/cloud-run-source-deploy/gemma-agent-base:current

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

# llama-cpp-python is already installed in the base image, so pip treats it
# as satisfied and does not recompile it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py chatlog.py ./
COPY --from=web-build /web/dist ./static

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
