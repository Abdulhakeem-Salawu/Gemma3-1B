# Gemma3-1B agent

A small FastAPI service that runs Gemma 3 1B (via llama-cpp-python) behind a
chat UI, with tool calling and MCP server access, deployable to Cloud Run.

## Layout

- `app.py` — FastAPI backend: loads the model, runs the tool-call loop,
  serves the chat UI.
- `static/index.html` — the chat UI.
- `Dockerfile` — container build. The model weights are **not** baked into
  the image; they're mounted at runtime from a Cloud Storage bucket.
- `requirements.txt` — Python dependencies.

## Deploy

1. Accept the license and download `gemma-3-1b-it-q4_0.gguf` from
   `google/gemma-3-1b-it-qat-q4_0-gguf` on Hugging Face.
2. Upload it to a bucket: `gsutil cp gemma-3-1b-it-q4_0.gguf gs://YOUR_BUCKET/`
3. Build: `gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/REPO/gemma-agent`
4. Deploy:
   ```
   gcloud run deploy gemma-agent \
     --image REGION-docker.pkg.dev/PROJECT/REPO/gemma-agent \
     --region us-central1 --cpu 4 --memory 4Gi \
     --add-volume mount-path=/mnt/models,type=cloud-storage,bucket=YOUR_BUCKET,readonly=true \
     --set-env-vars MODEL_PATH=/mnt/models/gemma-3-1b-it-q4_0.gguf,APP_API_KEY=pick-a-secret,MCP_SERVER_URLS=https://your-mcp-server/sse \
     --allow-unauthenticated
   ```

Add business-intelligence tools later by adding a function + JSON-schema
entry to `BUILTIN_TOOLS` in `app.py` — nothing else needs to change.
