"""Judge logged production chats with a local model and record scores in MLflow.

Runs on YOUR PC, not in the cloud:
  Firestore (chat_logs)  ->  local judge model (llama-server)  ->  local MLflow

The judge checks each logged answer against the evidence the agent actually had
(tool results + retrieved document chunks) and scores:
  - faithfulness 1-5: is every factual claim supported by that evidence
                      (or uncontroversial common knowledge)?
  - relevance    1-5: does it answer what was asked?
  - unsupported_claims: the specific claims it could not support (hallucinations)

See eval/README.md for setup. Quick start:
  llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 8192 --port 8080
  python eval/judge_logs.py --limit 30
  mlflow ui --backend-store-uri sqlite:///eval/mlflow.db
"""

import argparse
import json
import re
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DB_URI = f"sqlite:///{(HERE / 'mlflow.db').as_posix()}"
ARTIFACT_DIR = HERE / "mlartifacts"
JUDGED_IDS_FILE = HERE / "judged_ids.txt"
EXPERIMENT = "gemma-agent-prod-judging"

# Bump this whenever you edit the rubric/prompt below, so scores from
# different rubrics are never compared by accident.
JUDGE_PROMPT_VERSION = "v1"

MAX_EVIDENCE_CHARS = 9000  # keeps the judge prompt inside an 8k-token context

# Key order matters: llama.cpp emits keys in this order, so the judge lists the
# unsupported claims and its rationale BEFORE committing to scores.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
        "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
        "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["unsupported_claims", "rationale", "faithfulness", "relevance"],
}

JUDGE_SYSTEM = """You are a strict, fair evaluator of an AI assistant's answers.
You are shown the user's question, the EVIDENCE the assistant had (document context and tool results), and the assistant's ANSWER.

The assistant's own rules: never invent prices, dates or news; anything of that kind must come from a tool result or document context. If the documents do not cover the question, the assistant must say so rather than guess.

Judge only the ANSWER against the EVIDENCE:
- unsupported_claims: list each specific factual claim in the ANSWER that is neither supported by the EVIDENCE nor uncontroversial common knowledge. Quote it briefly. Use an empty list if there are none. If there is no EVIDENCE at all, any specific price, date, statistic or news claim is unsupported.
- rationale: one or two sentences.
- faithfulness: 5 = every claim supported or common knowledge; 4 = one trivial unsupported detail; 3 = some unsupported detail; 2 = a material unsupported claim; 1 = invents facts or contradicts the EVIDENCE.
- relevance: 5 = directly answers the question; 3 = partly; 1 = off-topic, or refuses/deflects when the EVIDENCE could have answered.
Do not penalize brevity. Do not reward length. Respond with JSON only."""


def clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + " ...[clipped]"


def build_judge_messages(log: dict) -> list[dict]:
    tool_blocks = []
    for t in log.get("tool_calls") or []:
        tool_blocks.append(f"[{t.get('name')} {t.get('args', '')}]\n{clip(t.get('result', ''), 2500)}")
    evidence = (
        "DOCUMENT CONTEXT:\n" + (clip(log.get("context", ""), 4000) or "(none)")
        + "\n\nTOOL RESULTS:\n" + ("\n\n".join(tool_blocks) or "(none)")
    )
    evidence = clip(evidence, MAX_EVIDENCE_CHARS)
    user = (
        f"USER QUESTION:\n{clip(log.get('question', ''), 2000)}\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f"ANSWER:\n{clip(log.get('final_answer', ''), 3000)}"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)  # last resort: first {...} block
        if not m:
            raise
        return json.loads(m.group(0))


def judge_one(base_url: str, model: str, log: dict, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": build_judge_messages(log),
        "temperature": 0,
        "max_tokens": 1500,
        "response_format": {"type": "json_object", "schema": VERDICT_SCHEMA},
    }
    resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"].get("content") or ""
    v = parse_json(content)
    return {
        "unsupported_claims": [str(c) for c in v.get("unsupported_claims", [])],
        "rationale": str(v.get("rationale", "")),
        "faithfulness": int(v["faithfulness"]),
        "relevance": int(v["relevance"]),
    }


def passed(v: dict) -> bool:
    return v["faithfulness"] >= 4 and v["relevance"] >= 3


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def fetch_from_firestore(args) -> list[dict]:
    from google.cloud import firestore
    from google.cloud.firestore_v1.base_query import FieldFilter

    db = firestore.Client(project=args.project) if args.project else firestore.Client()
    since = datetime.now(timezone.utc) - timedelta(days=args.since_days)
    # Single-field range + order on `ts` needs no composite index. Status is
    # filtered in Python. Over-fetch a bit so skipped/already-judged logs
    # don't leave us short (still only a few hundred reads, vs 50k/day free).
    query = (
        db.collection(args.collection)
        .where(filter=FieldFilter("ts", ">=", since))
        .order_by("ts", direction=firestore.Query.DESCENDING)
        .limit(args.limit * 4)
    )
    out = []
    for snap in query.stream():
        d = snap.to_dict()
        d["id"] = snap.id
        out.append(d)
    return out


def fetch_from_jsonl(path: str) -> list[dict]:
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            d = json.loads(line)
            d.setdefault("id", f"jsonl-{i}")
            out.append(d)
    return out


def load_judged_ids() -> set[str]:
    if not JUDGED_IDS_FILE.exists():
        return set()
    return set(JUDGED_IDS_FILE.read_text().split())


# ---------------------------------------------------------------------------


def detect_judge_name(base_url: str, fallback: str) -> str:
    try:
        r = requests.get(f"{base_url}/models", timeout=5)
        data = r.json().get("data") or r.json().get("models") or []
        if data:
            first = data[0]
            return str(first.get("id") or first.get("model") or first.get("name") or fallback)
    except Exception:
        pass
    return fallback


def require_judge(base_url: str) -> None:
    try:
        requests.get(f"{base_url}/models", timeout=5).raise_for_status()
    except Exception as exc:
        raise SystemExit(
            f"Judge server not reachable at {base_url} ({exc.__class__.__name__}).\n"
            "Start it first, e.g.:  llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 8192 --port 8080"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=30, help="max logs to judge this run")
    p.add_argument("--since-days", type=int, default=7, help="only logs newer than this")
    p.add_argument("--judge-url", default="http://127.0.0.1:8080/v1", help="OpenAI-compatible base URL of llama-server")
    p.add_argument("--judge-name", default=None, help="label for MLflow (default: asked from the server)")
    p.add_argument("--timeout", type=int, default=900, help="seconds per judge call (CPU can be slow)")
    p.add_argument("--collection", default="chat_logs")
    p.add_argument("--project", default=None, help="GCP project id (default: from your gcloud/ADC config)")
    p.add_argument("--from-jsonl", default=None, help="judge logs from a JSONL file instead of Firestore")
    p.add_argument("--rejudge", action="store_true", help="also re-judge logs that were already judged")
    args = p.parse_args()

    import mlflow

    logs = fetch_from_jsonl(args.from_jsonl) if args.from_jsonl else fetch_from_firestore(args)
    already = set() if args.rejudge else load_judged_ids()
    todo = [d for d in logs if d.get("status") == "ok" and d.get("final_answer") and d["id"] not in already]
    todo = todo[: args.limit]
    print(f"Fetched {len(logs)} logs; judging {len(todo)} new ok-status logs.")
    if not todo:
        return

    require_judge(args.judge_url)
    judge_name = args.judge_name or detect_judge_name(args.judge_url, "local-judge")
    results = []
    for n, log in enumerate(todo, 1):
        t0 = time.monotonic()
        try:
            v = judge_one(args.judge_url, judge_name, log, args.timeout)
            v["parse_ok"] = True
        except Exception as exc:  # network error, malformed JSON, ... -> count it, keep going
            print(f"  ! judge failed on {log['id']}: {exc!r}")
            v = {"unsupported_claims": [], "rationale": f"judge failed: {exc!r}", "faithfulness": 0,
                 "relevance": 0, "parse_ok": False}
        v["seconds"] = round(time.monotonic() - t0, 1)
        v["pass"] = v["parse_ok"] and passed(v)
        v["log_id"] = log["id"]
        v["question"] = log.get("question", "")
        v["answer"] = log.get("final_answer", "")
        v["app_model"] = log.get("model", "")
        v["used_tools"] = bool(log.get("tool_calls"))
        results.append(v)
        flag = "PASS" if v["pass"] else ("ERR " if not v["parse_ok"] else "FAIL")
        print(f"[{n}/{len(todo)}] {flag} faith={v['faithfulness']} rel={v['relevance']} "
              f"unsupported={len(v['unsupported_claims'])} ({v['seconds']}s)")

    ok = [r for r in results if r["parse_ok"]]
    if not ok:
        raise SystemExit("Every judge call failed, so nothing was recorded. Check the llama-server output.")
    metrics = {
        "n_logs": len(results),
        "n_judged_ok": len(ok),
        "parse_failure_rate": 1 - len(ok) / len(results),
        "mean_judge_seconds": statistics.mean(r["seconds"] for r in results),
    }
    if ok:
        metrics.update(
            mean_faithfulness=statistics.mean(r["faithfulness"] for r in ok),
            mean_relevance=statistics.mean(r["relevance"] for r in ok),
            pass_rate=sum(r["pass"] for r in ok) / len(ok),
            hallucination_rate=sum(bool(r["unsupported_claims"]) for r in ok) / len(ok),
        )
        for label, flag in (("tool_answers", True), ("direct_answers", False)):
            group = [r for r in ok if r["used_tools"] == flag]
            if group:
                metrics[f"pass_rate_{label}"] = sum(r["pass"] for r in group) / len(group)

    mlflow.set_tracking_uri(DB_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"judge-{datetime.now():%Y%m%d-%H%M}"):
        mlflow.log_params({
            "judge_model": judge_name,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_temperature": 0,
            "since_days": args.since_days,
            "source": "jsonl" if args.from_jsonl else "firestore",
        })
        mlflow.set_tag("app_models", ",".join(sorted({r["app_model"] for r in results if r["app_model"]})))
        mlflow.log_metrics(metrics)
        mlflow.log_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), "judged.jsonl")
        mlflow.log_table(
            data={k: [r[k] for r in results] for k in
                  ("log_id", "question", "answer", "faithfulness", "relevance", "pass", "rationale")},
            artifact_file="results_table.json",
        )

    with open(JUDGED_IDS_FILE, "a") as f:
        f.writelines(r["log_id"] + "\n" for r in results if r["parse_ok"])  # failed ones get retried next run

    print("\nSummary:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}, indent=2))
    print("View results:  mlflow ui --backend-store-uri", DB_URI)


if __name__ == "__main__":
    main()
