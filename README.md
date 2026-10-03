# Gemma3-1B agent

A small FastAPI service that runs a Qwen/Gemma GGUF model (via llama-cpp-python)
behind a React chat UI (assistant-ui), with tool calling, document RAG, and MCP
server access, deployable to Cloud Run.

## Layout

- `app.py` — FastAPI backend: loads the model, runs the tool-call loop, serves
  the chat UI.
- `chatlog.py` — best-effort logging of every chat turn to Firestore (for eval),
  written in a background task after the reply has finished streaming.
- `web/` — the React + assistant-ui chat frontend (built by the Dockerfile's
  Node stage into `static/` — nothing under `web/dist` is committed).
- `eval/` — local-only tooling that judges logged chats with a local model and
  records scores in MLflow. See `eval/README.md` (it also covers the one-time
  Firestore setup this logging needs).
- `Dockerfile` — app image: builds the frontend, then builds `FROM` the
  prebuilt base image below. The model weights are **not** baked into the
  image; they're mounted at runtime from a Cloud Storage bucket.
- `Dockerfile.base` + `cloudbuild.base.yaml` — the llama-cpp-python base image
  (the slow ~12 min compile), built once and reused by every app build.
- `requirements.txt` — Python dependencies.

## Deploy

1. Accept the license and download the GGUF model from Hugging Face.
2. Upload it to your bucket: `gcloud storage cp model.gguf gs://YOUR_BUCKET/`
3. **One time only:** build the llama-cpp-python base image (~12 min). Skip
   this if `gemma-agent-base:current` already exists in your Artifact Registry.
   From a checkout of this repo:
   ```
   gcloud builds submit . --region=us-central1 --config=cloudbuild.base.yaml
   ```
   (`--config` reads the file from your machine, so it won't work when pointed
   at a git URL.) Every build after that skips the compile. To upgrade
   llama-cpp-python, rerun with `--substitutions=_LLAMA_VERSION=<new version>`
   and bump the pin in `requirements.txt`.
4. Build the app (a few minutes, even with `--tag`, which disables layer
   caching — the base image is unaffected either way):
   ```
   gcloud builds submit https://github.com/Abdulhakeem-Salawu/Gemma3-1B \
     --git-source-revision=main --region=us-central1 \
     --tag=us-central1-docker.pkg.dev/PROJECT/cloud-run-source-deploy/gemma-agent
   ```
5. Deploy. Two flags matter for latency and correctness:
   - `--concurrency 1`: one llama.cpp instance isn't safe to share between
     simultaneous requests — without it, a second request is accepted by the
     container and silently queues behind a Python lock for minutes, with no
     Cloud Run-level timeout handling.
   - `--no-cpu-throttling`: chat logging runs in a background task *after* the
     reply finishes; that needs CPU to stay allocated once the response ends.
   ```
   gcloud run deploy gemma-agent \
     --image us-central1-docker.pkg.dev/PROJECT/cloud-run-source-deploy/gemma-agent \
     --region us-central1 --execution-environment gen2 \
     --cpu 4 --memory 6Gi --cpu-boost --no-cpu-throttling \
     --concurrency 1 --max-instances 1 --timeout 600 \
     --add-volume name=models,type=cloud-storage,bucket=YOUR_BUCKET,readonly=true \
     --add-volume-mount volume=models,mount-path=/mnt/models \
     --set-env-vars MODEL_PATH=/mnt/models/your-model.gguf,APP_API_KEY=pick-a-secret \
     --allow-unauthenticated
   ```

Optional env var: `MCP_SERVER_URLS` (comma-separated SSE URLs). Be careful
which servers you attach — the model reads untrusted web text, so don't give
it a tool that can change infrastructure.

Add business-intelligence tools by adding a function + JSON-schema entry to
`BUILTIN_TOOLS` in `app.py` — nothing else needs to change.

## Frontend development

```
cd web
npm install
npm run dev      # local dev server — point it at a deployed backend URL or
                 # run app.py locally too
npm run build    # production build — this is what the Dockerfile runs
```
