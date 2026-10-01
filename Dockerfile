# The slow llama-cpp-python compile lives in a prebuilt base image
# (Dockerfile.base, built once with cloudbuild.base.yaml), so this build only
# installs the light dependencies and copies the code (~1-2 min) every time.
ARG BASE_IMAGE=us-central1-docker.pkg.dev/pioneering-axe-233302/cloud-run-source-deploy/gemma-agent-base:current
FROM ${BASE_IMAGE}

WORKDIR /app

# llama-cpp-python is already installed in the base image, so pip treats it as
# satisfied and does not recompile it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py chatlog.py ./
COPY static ./static

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
