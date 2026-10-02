# Eval: judge real chats with a local model

```
Cloud Run app --(writes each turn)--> Firestore chat_logs
                                          |
        your PC:  judge_logs.py  <--------+   (reads a sample)
                      |  asks
                      v
              llama-server (local judge model)
                      |
                      v
              local MLflow (eval/mlflow.db)  -> `mlflow ui`
```

Nothing here needs your PC on while the app runs. The app logs on its own;
you run the judge when you feel like it.

## One-time cloud setup

```bash
PROJECT=your-project-id
gcloud services enable firestore.googleapis.com --project $PROJECT

# Only one database per project gets the free quota: use the "(default)" one.
gcloud firestore databases list --project $PROJECT
gcloud firestore databases create --location=us-central1 --project $PROJECT   # skip if one exists

# Let the Cloud Run service write to it (check which service account the service uses).
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:RUNTIME_SERVICE_ACCOUNT" --role=roles/datastore.user

# Auto-delete logs after LOG_TTL_DAYS (default 60) so you stay under 1 GiB.
gcloud firestore fields ttls update expire_at --collection-group=chat_logs --enable-ttl --project $PROJECT
```

Then redeploy the app (rebuild is ~1 min; the llama.cpp layer is cached) using
the commands in the main README. Send a chat, then check the `chat_logs`
collection in the Firestore console.

App env vars (all optional): `LOG_ENABLED=0` turns logging off,
`LOG_COLLECTION` (default `chat_logs`), `LOG_TTL_DAYS` (default `60`).

**Privacy:** logs contain user questions, answers, tool results and short
excerpts of uploaded documents. They stay in your Firestore and are judged by a
model on your own PC.

## One-time PC setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r eval/requirements.txt
gcloud auth application-default login                  # lets the script read Firestore
```

## Run an eval

1. Start the judge model (about 14 GB RAM; fine on a 24 GB PC, CPU-only is OK):
   ```bash
   llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 8192 --port 8080
   ```
   Any GGUF works: use `-m path/to/model.gguf` instead of `-hf`. A stronger option
   that is tighter on 24 GB is a Gemma 4 26B-A4B quant (16-18 GB). If answers come
   back malformed or very slow, restart with thinking turned off (recent
   llama.cpp builds have `--reasoning-budget 0`).
2. Judge a sample of recent chats:
   ```bash
   python eval/judge_logs.py --limit 30 --since-days 7 --project your-project-id
   ```
3. Look at the results (run from the repo root):
   ```bash
   mlflow ui --backend-store-uri sqlite:///eval/mlflow.db
   ```
   Open http://localhost:5000, experiment `gemma-agent-prod-judging`.

Already-judged logs are remembered in `eval/judged_ids.txt` and skipped next
time (`--rejudge` to override). Logs where the judge returned unusable output are
retried on the next run. No Firestore yet? `--from-jsonl file.jsonl` judges an
exported file instead.

## Reading the scores

| Metric | Meaning |
|---|---|
| `pass_rate` | share of answers with faithfulness >= 4 and relevance >= 3 |
| `hallucination_rate` | share of answers where the judge listed at least one unsupported claim |
| `mean_faithfulness` / `mean_relevance` | 1-5 averages |
| `pass_rate_tool_answers` / `pass_rate_direct_answers` | split by whether the agent used a tool |
| `parse_failure_rate` | judge calls that returned unusable output (should be near 0) |

A local judge is still a model. Open `results_table.json` on the run's Artifacts
tab and read a few FAILs (and a few PASSes) yourself before trusting a trend.
Always compare runs that share the same `judge_model` and `judge_prompt_version`
params; bump `JUDGE_PROMPT_VERSION` in `judge_logs.py` whenever you edit the rubric.

## Free-tier budget

Firestore free quota: 1 GiB stored, 20k writes/day, 50k reads/day. One chat = one
write; one judge run reads up to `--limit x 4` documents.
