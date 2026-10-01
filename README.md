# Gemma3-1B agent

A small FastAPI service that runs Gemma 3 1B (via llama-cpp-python) behind a
chat UI, with tool calling and MCP server access, deployable to Cloud Run.

## Layout

- `app.py` — FastAPI backend: loads the model, runs the tool-call loop,
  serves the chat UI.
- `chatlog.py` — best-effort logging of every chat turn to Firestore (for eval).
- `static/index.html` — the chat UI.
- `eval/` — local-only tooling that judges logged chats with a local model and
  records scores in MLflow. See `eval/README.md` (it also covers the one-time
  Firestore setup this logging needs).
- `Dockerfile` — container build. The model weights are **not** baked into
  the image; they're mounted at runtime from a Cloud Storage bucket.
- `requirements.txt` — Python dependencies.

## Deploy

1. Accept the license and download `gemma-3-1b-it-q4_0.gguf` from
   `google/gemma-3-1b-it-qat-q4_0-gguf` on Hugging Face.
2. Upload it to your bucket: `gcloud storage cp gemma-3-1b-it-q4_0.gguf gs://YOUR_BUCKET/`
3. Build straight from this repo (the llama.cpp compile takes ~15 min):
   ```
   gcloud builds submit https://github.com/Abdulhakeem-Salawu/Gemma3-1B \
     --git-source-revision=main --region=us-central1 \
     --tag=us-central1-docker.pkg.dev/PROJECT/cloud-run-source-deploy/gemma-agent
   ```
4. Deploy (`--concurrency=1` matters: one llama.cpp instance isn't safe to
   share between simultaneous requests):
   ```
   gcloud run deploy gemma-agent \
     --image us-central1-docker.pkg.dev/PROJECT/cloud-run-source-deploy/gemma-agent \
     --region us-central1 --execution-environment gen2 \
     --cpu 4 --memory 4Gi --cpu-boost --concurrency 1 --timeout 600 \
     --add-volume name=models,type=cloud-storage,bucket=YOUR_BUCKET,readonly=true \
     --add-volume-mount volume=models,mount-path=/mnt/models \
     --set-env-vars MODEL_PATH=/mnt/models/gemma-3-1b-it-q4_0.gguf,APP_API_KEY=pick-a-secret \
     --allow-unauthenticated
   ```

Optional env var: `MCP_SERVER_URLS` (comma-separated SSE URLs). Be careful
which servers you attach — the model reads untrusted web text, so don't give
it a tool that can change infrastructure.

Add business-intelligence tools by adding a function + JSON-schema entry to
`BUILTIN_TOOLS` in `app.py` — nothing else needs to change.
