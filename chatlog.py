"""Best-effort logging of every chat turn to Firestore, for offline eval.

Design rules:
- Logging must NEVER break or slow a reply. Every failure is caught and printed.
- Docs stay small (long fields are truncated) so they fit the free tier:
  1 GiB stored, 20k writes/day.
- Each doc carries `expire_at`; enable a Firestore TTL policy on that field
  and old logs delete themselves (see eval/README.md).

Env vars:
  LOG_ENABLED     "0"/"false" turns logging off (default: on)
  LOG_COLLECTION  Firestore collection name (default: chat_logs)
  LOG_TTL_DAYS    days until a log expires (default: 60)
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

LOG_ENABLED = os.getenv("LOG_ENABLED", "1").lower() not in ("0", "false", "no", "off")
LOG_COLLECTION = os.getenv("LOG_COLLECTION", "chat_logs")
LOG_TTL_DAYS = int(os.getenv("LOG_TTL_DAYS", "60"))

MAX_QUESTION_CHARS = 4000
MAX_CONTEXT_CHARS = 6000
MAX_TOOL_RESULT_CHARS = 2500
MAX_ANSWER_CHARS = 8000

_client: Any = None
_client_failed = False


def _truncate(text: Any, limit: int) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _get_client():
    """Lazy init, so a missing library/credential only disables logging
    instead of crashing app startup."""
    global _client, _client_failed
    if _client is not None or _client_failed:
        return _client
    try:
        from google.cloud import firestore

        _client = firestore.Client()  # uses the Cloud Run service account
    except Exception as exc:
        _client_failed = True
        print(f"[chatlog] Firestore unavailable, logging disabled: {exc}")
    return _client


def new_log(model_path: str, instance_id: str, session_id: str | None) -> dict:
    """Skeleton filled in as the request progresses; written once at the end."""
    return {
        "model": os.path.basename(model_path),
        "instance_id": instance_id,
        "session_id": session_id,
        "question": "",
        "context": "",
        "tool_calls": [],
        "thinking": "",
        "final_answer": "",
        "hops": 0,
        "status": "incomplete",  # becomes "ok" or "error"; stays "incomplete" if the client left early
        "error": "",
        "latency_s": None,
    }


def _build_doc(log: dict) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "ts": now,
        "expire_at": now + timedelta(days=LOG_TTL_DAYS),
        "model": log["model"],
        "instance_id": log["instance_id"],
        "session_id": log["session_id"],
        "question": _truncate(log["question"], MAX_QUESTION_CHARS),
        "context": _truncate(log["context"], MAX_CONTEXT_CHARS),
        # tool args stored as a JSON string: Firestore map keys are picky and
        # a small model can emit odd argument shapes.
        "tool_calls": [
            {
                "name": t["name"],
                "args": _truncate(json.dumps(t["args"], default=str), 1000),
                "result": _truncate(t["result"], MAX_TOOL_RESULT_CHARS),
            }
            for t in log["tool_calls"]
        ],
        "thinking": _truncate(log["thinking"], 1000),
        "final_answer": _truncate(log["final_answer"], MAX_ANSWER_CHARS),
        "hops": log["hops"],
        "status": log["status"],
        "error": _truncate(log["error"], 500),
        "latency_s": log["latency_s"],
    }


def _write(doc: dict) -> None:
    client = _get_client()
    if client is None:
        return
    client.collection(LOG_COLLECTION).add(doc)


async def persist(log: dict) -> None:
    """Awaited from the request's own `finally`, so the write happens while
    Cloud Run still counts the request as active (a fire-and-forget write
    after the response can be CPU-throttled). Never raises."""
    if not LOG_ENABLED or not log.get("question"):
        return
    try:
        doc = _build_doc(log)
        loop = asyncio.get_running_loop()
        # Submitted to the thread pool immediately, so the write still runs
        # even if this await is cancelled by a client disconnect.
        await asyncio.wait_for(loop.run_in_executor(None, _write, doc), timeout=3)
    except Exception as exc:
        print(f"[chatlog] failed to write log: {exc!r}")
