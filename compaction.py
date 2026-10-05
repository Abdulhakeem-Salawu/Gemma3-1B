"""Summarization-based conversation compaction: the pure logic.

Deliberately free of heavy imports (no llama_cpp, no FastAPI) so it can be unit
tested without the model. Everything that needs a tokenizer takes a
`count_tokens` callable; app.py wires in the real one.

How it fits together (see the plan in the repo history / PR description):

- The CLIENT owns the summary. It sends `summary` and `summary_covers` (how
  many leading messages of its raw message list the summary already covers)
  with every chat request. The server stays stateless.
- After each answer the server reports context usage in the `done` event. When
  it is nearly full the client calls POST /chat/compact, which folds
  `previous summary + older messages` into one new summary.
- Chat requests then send the summary plus only the not-yet-summarized
  messages. `fit_history` in app.py stays on as the safety net.

All indices below refer to the RAW client message list (before
normalize_history), so they stay stable however normalize_history merges or
drops messages later.
"""

from __future__ import annotations

import json
import re
from typing import Callable

CountTokens = Callable[[str], int]

# Schema for the compaction call. Same grammar-constrained JSON technique as
# chat: it forces a JSON object, so Qwen3 cannot spend minutes (at ~3 tokens/s)
# on a free-text <think> block first.
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

SUMMARY_HEADER = (
    "Summary of the earlier conversation (older messages were condensed; treat it as accurate background):"
)
SUMMARY_RULE = (
    "- Use the summary of the earlier conversation and the recent messages to stay consistent; "
    "if asked about something that is in neither, say you don't have it instead of guessing.\n"
)

_PROMPT_HEAD = (
    "You are condensing a conversation so it can continue with limited memory.\n"
    "Write an updated summary that keeps everything needed to continue naturally.\n"
)
_PROMPT_RULES = """Rules:
- Keep every fact, name, number, date, constraint, preference, decision and goal the user stated, exactly as stated.
- Keep the assistant's main points, conclusions, recommendations and the key items of any list it gave, briefly.
- Keep unanswered questions and unfinished tasks.
- Keep what the PREVIOUS SUMMARY says unless the new messages change it.
- Add nothing that is not in the conversation. Do not address the user.
- Plain sentences or short bullets, at most about 200 words.
Respond with ONLY a JSON object: {"summary": "<the updated summary>"}"""

_LABELS = {"user": "User", "assistant": "Assistant"}
_TRUNCATION_MARKER = " […] "
_LINE_OVERHEAD_TOKENS = 4  # "User: " / "Assistant: " plus the newline


# ---------------------------------------------------------------------------
# The summary block used in chat
# ---------------------------------------------------------------------------


def format_summary_block(summary: str) -> str:
    """The block inserted into the final user turn just before `User: ...`."""
    return f"{SUMMARY_HEADER}\n{summary}\n\n"


# ---------------------------------------------------------------------------
# Validation / request resolution
# ---------------------------------------------------------------------------


def validate_summary(summary: object, max_chars: int) -> str:
    """Return the trimmed summary, or raise ValueError (the caller turns it
    into HTTP 400, or into a compaction error for model output)."""
    if not isinstance(summary, str):
        raise ValueError("The summary must be a string.")
    cleaned = summary.strip()
    if not cleaned:
        raise ValueError("The summary is empty.")
    if len(cleaned) > max_chars:
        raise ValueError(f"The summary is too long ({len(cleaned)} characters; the limit is {max_chars}).")
    return cleaned


def clamp_covers(covers: object, n_messages: int, keep_last: int = 1) -> int:
    """A usable `summary_covers` for a raw list of `n_messages`. At least
    `keep_last` messages always stay uncovered (for chat that is the newest
    user question, which must never be sliced away)."""
    try:
        value = int(covers or 0)
    except (TypeError, ValueError):
        value = 0
    return max(0, min(value, max(n_messages - keep_last, 0)))


def resolve_summary(
    summary: str | None, covers: object, n_messages: int, max_chars: int
) -> tuple[str | None, int]:
    """Validate the (summary, covers) pair sent by a client.

    - No summary, or a blank one: (None, 0). A blank summary cannot carry
      anything, so its coverage is ignored rather than slicing messages away.
    - Otherwise the summary is validated (ValueError if oversized/non-string)
      and `covers` is clamped to the list. A summary the server was sent is
      never silently dropped.
    """
    if summary is None or (isinstance(summary, str) and not summary.strip()):
        return None, 0
    return validate_summary(summary, max_chars), clamp_covers(covers, n_messages)


def slice_after_summary(messages: list[dict], covers: int) -> list[dict]:
    """The not-yet-summarized part of the RAW client list. Done before
    normalize_history so indices stay stable. Never removes the last message."""
    return messages[clamp_covers(covers, len(messages)) :]


# ---------------------------------------------------------------------------
# When to compact, and what to fold
# ---------------------------------------------------------------------------


def context_usage(
    *,
    summary_tokens: int,
    history_tokens: int,
    answer_tokens: int,
    instruction_tokens: int,
    n_ctx: int,
    max_generation_tokens: int,
    max_history_tokens: int,
    margin_tokens: int,
) -> tuple[int, int]:
    """(used, budget). `used` is the summary plus every not-yet-summarized
    message at full length, including the answer just produced. `budget` is
    what the history may take: min(MAX_HISTORY_TOKENS, what n_ctx leaves after
    the reply cap, the instruction turn and a safety margin)."""
    used = max(0, summary_tokens) + max(0, history_tokens) + max(0, answer_tokens)
    budget = min(max_history_tokens, n_ctx - max_generation_tokens - instruction_tokens - margin_tokens)
    return used, max(0, budget)


def needs_compaction(used: int, budget: int, fraction: float, can_fold: bool) -> bool:
    """True when usage reached `fraction` of the budget AND there is something
    to fold (otherwise compacting could never help and would just cost the
    user a wait)."""
    return bool(can_fold and budget > 0 and used >= fraction * budget)


def select_fold_range(messages: list[dict], covers: int, keep_recent: int) -> tuple[int, int]:
    """(start, end) of the raw messages to fold; `end == start` means nothing.

    Everything from `covers` up to but excluding the most recent `keep_recent`
    messages is folded. `end` is moved back so the kept tail starts on a user
    message: an exchange is never split and the verbatim window stays
    user/assistant aligned. Folding less than the maximum is always fine.
    """
    n = len(messages)
    start = max(0, min(int(covers or 0), n))
    end = n - max(2, int(keep_recent))
    if end <= start:
        return start, start
    while end > start and messages[end].get("role") != "user":
        end -= 1
    return (start, end) if end > start else (start, start)


# ---------------------------------------------------------------------------
# The compaction prompt
# ---------------------------------------------------------------------------


def build_summary_prompt(previous_summary: str | None, messages: list[dict]) -> str:
    """The single user turn sent for one compaction call. Built by
    concatenation (not str.format) so braces in the conversation are safe."""
    lines = []
    for m in messages:
        label = _LABELS.get(m.get("role"), "User")
        lines.append(f"{label}: {m.get('content', '')}")
    return (
        _PROMPT_HEAD
        + "\nPREVIOUS SUMMARY:\n"
        + ((previous_summary or "").strip() or "(none)")
        + "\n\nNEW MESSAGES TO FOLD IN:\n"
        + "\n".join(lines)
        + "\n\n"
        + _PROMPT_RULES
    )


def truncate_to_tokens(text: str, max_tokens: int, count_tokens: CountTokens) -> str:
    """Cut `text` to roughly `max_tokens`, keeping its start and its end (the
    middle of a very long message is what we give up). Binary search on the
    kept character count, using the real tokenizer."""
    if count_tokens(text) <= max_tokens:
        return text

    def candidate(keep: int) -> str:
        head = int(keep * 0.6)
        tail = keep - head
        return text[:head].rstrip() + _TRUNCATION_MARKER + (text[-tail:].lstrip() if tail else "")

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(candidate(mid)) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return candidate(lo)


def plan_chunks(
    messages: list[dict],
    previous_summary: str | None,
    count_tokens: CountTokens,
    *,
    input_max_tokens: int,
    summary_max_tokens: int,
) -> list[list[dict]]:
    """Split `messages` into sequential chunks so each compaction call stays
    under `input_max_tokens` (prompt scaffold + previous summary + chunk).

    Each call's output becomes the next call's PREVIOUS SUMMARY, so the room
    reserved for the previous summary is the larger of the real one and the
    generation cap. A single message that cannot fit in a chunk on its own is
    truncated (start and end kept) rather than dropped.
    """
    if not messages:
        return []
    scaffold = count_tokens(build_summary_prompt(None, []))
    reserve = max(count_tokens(previous_summary or ""), summary_max_tokens)
    room = max(200, input_max_tokens - scaffold - reserve)

    chunks: list[list[dict]] = []
    current: list[dict] = []
    spent = 0
    for m in messages:
        content = m.get("content", "")
        label = _LABELS.get(m.get("role"), "User")
        cost = count_tokens(f"{label}: {content}") + _LINE_OVERHEAD_TOKENS
        if cost > room:
            content = truncate_to_tokens(content, room - _LINE_OVERHEAD_TOKENS - count_tokens(f"{label}: "), count_tokens)
            m = {"role": m.get("role"), "content": content}
            cost = count_tokens(f"{label}: {content}") + _LINE_OVERHEAD_TOKENS
        if current and spent + cost > room:
            chunks.append(current)
            current, spent = [], 0
        current.append(m)
        spent += cost
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Reading the model's output
# ---------------------------------------------------------------------------

_SUMMARY_KEY = re.compile(r'"summary"\s*:\s*"')


def extract_summary(raw: str) -> str:
    """The text of the "summary" field from the model's raw JSON output.

    Complete JSON is parsed normally. If the output was cut off at the token
    cap (so it is not valid JSON), whatever text of the field was written is
    salvaged instead of throwing the whole compaction away. Returns "" when
    nothing usable is there.
    """
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("summary"), str):
            return obj["summary"].strip()
    except (json.JSONDecodeError, TypeError):
        pass
    m = _SUMMARY_KEY.search(raw or "")
    if not m:
        return ""
    body, i = raw[m.end() :], 0
    while i < len(body):
        if body[i] == "\\":
            i += 2
            continue
        if body[i] == '"':
            break
        i += 1
    segment = body[:i]
    if i >= len(body):  # no closing quote: the output was truncated mid-string
        segment = re.sub(r"\\u[0-9a-fA-F]{0,3}$", "", segment)  # half-written \uXXXX
        trailing = len(segment) - len(segment.rstrip("\\"))
        if trailing % 2:  # a lone backslash at the very end starts an escape that never arrived
            segment = segment[:-1]
    try:
        return json.loads(f'"{segment}"').strip()
    except json.JSONDecodeError:
        return segment.replace("\\n", "\n").replace('\\"', '"').strip()


_SENTENCE_BOUNDARY = re.compile(r"[.!?](?=\s|$)|\n")


def trim_to_sentence(text: str, min_keep: float = 0.5) -> str:
    """For a summary cut off at the token cap: drop the dangling fragment after
    the last sentence end or line break, if that keeps at least `min_keep` of
    the text; otherwise keep it all."""
    last = None
    for m in _SENTENCE_BOUNDARY.finditer(text):
        last = m
    if last is None:
        return text
    cut = text[: last.end()].rstrip()
    return cut if len(cut) >= min_keep * len(text) else text
