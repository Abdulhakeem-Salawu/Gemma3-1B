# Gemma3-1B agent

A small FastAPI service that runs a Qwen/Gemma GGUF model (via llama-cpp-python)
behind a React chat UI (assistant-ui), with tool calling, long-conversation
compaction, document RAG, and MCP server access, deployable to Cloud Run.

## Layout

- `app.py` — FastAPI backend: loads the model, runs the tool-call loop, serves
  the chat UI.
- `compaction.py` — pure logic for summarization-based conversation compaction
  (when to compact, what to fold, the summary prompt, chunking a large fold,
  reading the model's output). No heavy imports, so `tests/` exercise it
  without a model.
- `documents.py` — pure logic for attached documents (text cleanup, chunking,
  outline, BM25 retrieval, per-question selection within a token budget,
  validation). Same no-heavy-imports approach as `compaction.py`.
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
- `tests/` — `pytest` tests for `compaction.py` and `documents.py`, plus
  app-level tests (`llama_cpp` stubbed, no model needed). Run from the repo
  root: `pip install pytest httpx` (the rest of `requirements.txt` minus
  `llama-cpp-python`, which the stub replaces), then `python -m pytest tests`.
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

### Conversation compaction

Long chats keep going without the model forgetting early turns. When the
history reaches about 75% of its token budget, the browser calls
`POST /chat/compact` to condense the older messages into a short summary (the
composer locks meanwhile, with a progress banner). Later requests send the
summary plus the recent messages. The browser owns the summary — it sends it
as `summary` / `summary_covers` with every chat request — so the server stays
stateless; the older, char-based history shortening (`fit_history` in
`app.py`) is still the fallback when no summary is sent or compaction fails.

| Env var | Default | Meaning |
|---|---|---|
| `COMPACT_AT_FRACTION` | `0.75` | share of the history budget that triggers compaction |
| `KEEP_RECENT_MESSAGES` | `4` (`2` when `N_CTX` < 6144) | newest messages kept verbatim |
| `SUMMARY_MAX_TOKENS` | `320` | generation cap for one summary |
| `SUMMARY_MAX_CHARS` | `2500` | longest summary accepted or produced |
| `COMPACT_INPUT_MAX_TOKENS` | `3000` | per-call input cap; larger folds are chunked |

### Documents (attachments and long pastes)

Uploading a PDF/DOCX/TXT/MD file, or pasting more than ~3000 characters into
the composer, sends it to `POST /kb/extract`, which only extracts, cleans
(de-ligatures, rejoins PDF-wrapped lines, strips page numbers) and chunks the
text — nothing is stored server-side. The browser keeps the chunks in
IndexedDB and sends the chat's documents with every `/chat` request (an empty
`documents: []` means nothing is attached; omitting the field is the old,
server-side-index behaviour, still supported for `/kb/upload` clients). This
is what makes documents survive a restart or a second Cloud Run instance:
nothing depends on which process saw the upload.

Per question, `documents.select_context` in `documents.py` picks what goes in
the prompt, within a token budget computed from what's left of `N_CTX` after
instructions, any summary and the reply: an outline plus excerpts sampled
across the document for an overview question ("summarize this"), the matching
section for "section 12", otherwise BM25 over the chunks (with the previous
question folded in for a short follow-up like "tell me more"). A greeting
adds no document text at all.

| Env var | Default | Meaning |
|---|---|---|
| `DOC_CONTEXT_TOKENS` | `1100` (`1800` when `N_CTX` >= 6144) | token budget for document text in one prompt |
| `MAX_DOC_BYTES` | 15 MB | per-file upload limit |
| `MAX_EXTRACT_CHARS` | 2,000,000 | pasted-text limit before any processing |

`documents.py`'s own limits: `MAX_DOCS` (20 per chat), `MAX_DOC_CHARS` (about
80 pages per document), `MAX_TOTAL_CHARS` (about 80 pages per chat, across all
attached documents).

## Frontend development

```
cd web
npm install
npm run dev      # local dev server — point it at a deployed backend URL or
                 # run app.py locally too
npm run build    # production build — this is what the Dockerfile runs
```
