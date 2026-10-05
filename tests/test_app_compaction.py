"""App-level tests for compaction: build_turns, the `done.context` event and
POST /chat/compact. llama_cpp is replaced by a stub, so no model is needed.
Run from the repo root (app.py mounts ./static)."""

import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeLlama:
    """Tokenizer: one token per whitespace-separated word. Completions are
    scripted: chat calls answer `answer_text`; compaction calls (recognized by
    the summary schema) pop from `summaries`."""

    def __init__(self, *args, **kwargs):  # accepts Llama(...)'s real constructor arguments
        self.calls: list[dict] = []
        self.answer_text = "Sure, here is the answer."
        self.summaries: list = []  # str, or (raw_json_text, finish_reason)
        self.chat_raw: str | None = None  # override the raw chat output (e.g. malformed)
        self.fail_with: Exception | None = None

    def tokenize(self, data: bytes, add_bos: bool = False):
        return data.decode("utf-8").split()

    def create_chat_completion(self, messages, response_format, temperature, max_tokens, stream):
        schema = response_format["schema"]
        is_compaction = "summary" in schema["properties"]
        self.calls.append(
            {"messages": messages, "compaction": is_compaction, "temperature": temperature, "max_tokens": max_tokens}
        )
        if self.fail_with:
            raise self.fail_with
        finish = "stop"
        if is_compaction:
            item = self.summaries.pop(0) if self.summaries else "A fake summary."
            raw, finish = item if isinstance(item, tuple) else (json.dumps({"summary": item}), "stop")
        elif self.chat_raw is not None:
            raw = self.chat_raw
        else:
            raw = json.dumps(
                {"thinking": "ok", "action": "final_answer", "tool_name": "", "tool_arguments": {},
                 "final_answer": self.answer_text}
            )
        pieces = [raw[i : i + 6] for i in range(0, len(raw), 6)]

        def gen():
            for p in pieces:
                yield {"choices": [{"delta": {"content": p}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": finish}]}

        return gen()


stub = types.ModuleType("llama_cpp")
stub.Llama = FakeLlama
sys.modules.setdefault("llama_cpp", stub)

from fastapi.testclient import TestClient  # noqa: E402

import app as appmod  # noqa: E402
import chatlog  # noqa: E402


@pytest.fixture()
def llm(monkeypatch):
    fake = FakeLlama()
    monkeypatch.setitem(appmod.STATE, "llm", fake)
    monkeypatch.setitem(appmod.STATE, "mcp_tools", {})
    monkeypatch.setattr(appmod, "APP_API_KEY", None)
    return fake


@pytest.fixture()
def client(llm):
    return TestClient(appmod.app)


@pytest.fixture()
def logs(monkeypatch):
    captured: list[dict] = []

    async def fake_persist(log):
        captured.append(log)

    monkeypatch.setattr(chatlog, "persist", fake_persist)
    return captured


def sse_events(resp) -> list[dict]:
    out = []
    for frame in resp.text.split("\n\n"):
        if frame.startswith("data: "):
            out.append(json.loads(frame[6:]))
    return out


def conv(n: int, words: int = 8, start: int = 0) -> list[dict]:
    return [
        {"role": "user" if (i + start) % 2 == 0 else "assistant", "content": f"m{i + start} " + "w " * words}
        for i in range(n)
    ]


# --- backward compatibility ------------------------------------------------------


def test_no_summary_renders_exactly_the_old_template():
    t = appmod.INSTRUCTIONS_TEMPLATE
    kwargs = dict(tools="T", context_block="", question="Q")
    with_empty = t.format(summary_rule="", summary_block="", **kwargs)
    removed = t.replace("{summary_rule}", "").replace("{summary_block}", "").format(**kwargs)
    assert with_empty == removed
    assert "\n\n\n" not in with_empty
    assert "Summary of the earlier" not in with_empty


def test_build_turns_without_summary_is_unchanged(llm):
    messages = conv(5)  # ends on a user message
    turns = appmod.build_turns(messages)
    assert turns[:-1] == messages[:-1]
    assert turns[-1]["content"].endswith("User: " + messages[-1]["content"])
    assert "Summary of the earlier" not in turns[-1]["content"]


def test_old_client_chat_still_works(client, llm):
    r = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "hello there"}]})
    events = sse_events(r)
    assert [e["type"] for e in events if e["type"] in ("token", "done")][-1] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "token") == llm.answer_text


# --- build_turns with a summary ---------------------------------------------------


def test_summary_block_sits_just_before_the_question_with_a_rule(llm):
    messages = conv(7)
    turns = appmod.build_turns(messages, summary="User runs a bakery. Budget 5k.", summary_covers=4)
    content = turns[-1]["content"]
    block = "Summary of the earlier conversation"
    assert content.index(block) < content.index("User: " + messages[-1]["content"])
    assert "User runs a bakery. Budget 5k.\n\nUser: m6" in content
    assert "if asked about something that is in neither, say you don't have it" in content
    # only the uncovered part of the RAW list is replayed
    assert [m["content"] for m in turns[:-1]] == [m["content"] for m in messages[4:-1]]


def test_slicing_uses_raw_indices_even_when_normalize_merges_messages(llm):
    # raw[2] and raw[3] are two assistant messages that normalize_history will merge,
    # and raw[0..1] precede the coverage point. Indices must still refer to the raw list.
    raw = [
        {"role": "user", "content": "q0 first"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "q1 second"},
        {"role": "assistant", "content": "a1 part one"},
        {"role": "assistant", "content": "a1 part two"},
        {"role": "user", "content": "q2 third"},
    ]
    turns = appmod.build_turns(raw, summary="S", summary_covers=2)
    assert [m["content"] for m in turns[:-1]] == ["q1 second", "a1 part one\n\na1 part two"]


def test_blank_or_missing_summary_ignores_coverage(llm):
    messages = conv(5)
    assert appmod.build_turns(messages, summary=None, summary_covers=4)[:-1] == messages[:-1]


def test_summary_tokens_count_toward_the_prompt_budget(llm, monkeypatch):
    monkeypatch.setattr(appmod, "N_CTX", 4096)
    big = "word " * 400
    messages = conv(1) + [{"role": "assistant", "content": "a"}] + [{"role": "user", "content": "q"}]
    usage_plain, usage_sum = {}, {}
    appmod.build_turns(messages, usage=usage_plain)
    appmod.build_turns(messages, summary=big, summary_covers=0, usage=usage_sum)
    assert usage_plain["summary_tokens"] == 0
    assert usage_sum["summary_tokens"] >= 400
    # the instruction turn is measured WITHOUT the summary
    assert usage_sum["instruction_tokens"] == usage_plain["instruction_tokens"]


def test_prompt_too_long_still_raised_with_a_summary(llm):
    huge = [{"role": "user", "content": "word " * 5000}]
    with pytest.raises(appmod.PromptTooLong):
        appmod.build_turns(huge, summary="S", summary_covers=0)


def test_fit_history_still_applies_as_the_safety_net(llm, monkeypatch):
    monkeypatch.setattr(appmod, "MAX_HISTORY_TOKENS", 300)
    messages = conv(9, words=200)  # far over budget
    turns = appmod.build_turns(messages, summary="S", summary_covers=0)
    replayed = sum(len(m["content"].split()) for m in turns[:-1])
    assert replayed < sum(len(m["content"].split()) for m in messages[:-1])
    assert turns[0]["role"] == "user"


# --- request validation ------------------------------------------------------------


def test_oversized_summary_is_rejected_with_400(client):
    r = client.post(
        "/chat/stream",
        json={"messages": conv(3), "summary": "x" * (appmod.SUMMARY_MAX_CHARS + 1), "summary_covers": 2},
    )
    assert r.status_code == 400
    assert "too long" in r.json()["detail"]


def test_non_string_summary_is_422(client):
    r = client.post("/chat/stream", json={"messages": conv(3), "summary": 7, "summary_covers": 2})
    assert r.status_code == 422


def test_compact_requires_the_api_key(client, monkeypatch):
    monkeypatch.setattr(appmod, "APP_API_KEY", "secret")
    r = client.post("/chat/compact", json={"messages": conv(8)})
    assert r.status_code == 401
    ok = client.post("/chat/compact", json={"messages": conv(8)}, headers={"x-api-key": "secret"})
    assert ok.status_code == 200


# --- the `done` event reports context usage ----------------------------------------


def test_done_reports_context_and_flags_a_full_context(client, llm, monkeypatch):
    monkeypatch.setattr(appmod, "KEEP_RECENT_MESSAGES", 2)
    short = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    done = sse_events(short)[-1]
    assert done["type"] == "done"
    assert set(done["context"]) == {"used", "budget", "needs_compaction"}
    assert done["context"]["needs_compaction"] is False  # nothing foldable, nowhere near full

    long_chat = conv(11, words=250)  # ends on user; long enough to cross 75% of the budget
    full = sse_events(client.post("/chat/stream", json={"messages": long_chat}))[-1]
    assert full["context"]["used"] >= 0.75 * full["context"]["budget"]
    assert full["context"]["needs_compaction"] is True


def test_context_not_flagged_when_there_is_nothing_to_fold(client, llm, monkeypatch):
    monkeypatch.setattr(appmod, "KEEP_RECENT_MESSAGES", 4)
    llm.answer_text = "w " * 3000  # one enormous answer, but only 2 messages in total
    done = sse_events(client.post("/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}))[-1]
    assert done["context"]["used"] > done["context"]["budget"]
    assert done["context"]["needs_compaction"] is False


def test_no_context_on_error_paths(client, llm):
    llm.chat_raw = "this is not json"
    events = sse_events(client.post("/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}))
    assert events[-1] == {"type": "done"}

    llm.chat_raw = None
    llm.fail_with = RuntimeError("boom")
    events = sse_events(client.post("/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}))
    assert events[-1]["type"] == "error"
    assert not any("context" in e for e in events)


def test_context_counts_the_summary(client, llm, monkeypatch):
    monkeypatch.setattr(appmod, "KEEP_RECENT_MESSAGES", 2)
    base = {"messages": conv(5)}
    a = sse_events(client.post("/chat/stream", json=base))[-1]["context"]["used"]
    b = sse_events(client.post("/chat/stream", json={**base, "summary": "word " * 300, "summary_covers": 0}))[-1][
        "context"
    ]["used"]
    assert b - a >= 300


# --- POST /chat/compact --------------------------------------------------------------


def compact(client, **body):
    return sse_events(client.post("/chat/compact", json=body))


def test_compact_happy_path(client, llm, logs, monkeypatch):
    monkeypatch.setattr(appmod, "KEEP_RECENT_MESSAGES", 4)
    messages = conv(10)  # u a u a u a u a u a
    llm.summaries = ["The user asked three things; budget 5k."]
    events = compact(client, messages=messages, session_id="s1")
    types_ = [e["type"] for e in events]
    assert types_[0] == "started" and types_[-2:] == ["summary", "done"]
    started, summary = events[0], events[-2]
    assert started["fold_messages"] == 6 and started["chunks"] == 1 and started["input_tokens"] > 0
    assert summary["summary"] == "The user asked three things; budget 5k."
    assert summary["summary_covers"] == 6
    assert summary["summary_tokens"] > 0

    call = llm.calls[0]
    assert call["compaction"] and call["temperature"] == 0.2 and call["max_tokens"] == appmod.SUMMARY_MAX_TOKENS
    prompt = call["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:\n(none)" in prompt
    for m in messages[:6]:
        assert m["content"] in prompt
    assert messages[6]["content"] not in prompt  # the recent window stays out

    assert len(logs) == 1
    doc = logs[0]
    assert doc["status"] == "compaction_ok" and doc["final_answer"] == summary["summary"]
    assert doc["extra"]["fold_messages"] == 6 and doc["extra"]["output_tokens"] > 0
    assert "first_token_s" in doc["extra"]


def test_compact_is_a_rewrite_that_includes_the_previous_summary(client, llm, monkeypatch):
    monkeypatch.setattr(appmod, "KEEP_RECENT_MESSAGES", 4)
    messages = conv(14)
    llm.summaries = ["Newer summary."]
    events = compact(client, messages=messages, summary="Older summary: budget 5k.", summary_covers=6)
    assert events[0]["fold_messages"] == 4  # 14 - 4 recent - 6 already covered
    assert events[-2]["summary_covers"] == 10
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:\nOlder summary: budget 5k." in prompt
    assert messages[5]["content"] not in prompt and messages[6]["content"] in prompt


def test_compact_chunks_chain_summaries(client, llm, monkeypatch):
    monkeypatch.setattr(appmod, "KEEP_RECENT_MESSAGES", 2)
    monkeypatch.setattr(appmod, "COMPACT_INPUT_MAX_TOKENS", 900)
    monkeypatch.setattr(appmod, "SUMMARY_MAX_TOKENS", 100)
    monkeypatch.setattr(appmod, "COMPACT_PROGRESS_TOKENS", 1)
    messages = conv(12, words=120)
    llm.summaries = ["after chunk one", "after chunk two", "after chunk three", "after chunk four", "x5", "x6"]
    events = compact(client, messages=messages)
    chunks = events[0]["chunks"]
    assert chunks >= 2
    prompts = [c["messages"][0]["content"] for c in llm.calls]
    assert len(prompts) == chunks
    assert "PREVIOUS SUMMARY:\n(none)" in prompts[0]
    assert "PREVIOUS SUMMARY:\nafter chunk one" in prompts[1]
    assert events[-2]["type"] == "summary"
    assert [e["chunk"] for e in events if e["type"] == "progress"] != []


def test_compact_nothing_to_fold_is_a_graceful_noop(client, llm, logs, monkeypatch):
    monkeypatch.setattr(appmod, "KEEP_RECENT_MESSAGES", 4)
    events = compact(client, messages=conv(4))
    assert events == [{"type": "started", "fold_messages": 0, "input_tokens": 0, "chunks": 0}, {"type": "done"}]
    assert llm.calls == [] and logs == []
    assert compact(client, messages=[])[-1] == {"type": "done"}


def test_compact_rejects_an_oversized_previous_summary(client):
    r = client.post(
        "/chat/compact",
        json={"messages": conv(10), "summary": "x" * (appmod.SUMMARY_MAX_CHARS + 1), "summary_covers": 2},
    )
    assert r.status_code == 400


@pytest.mark.parametrize("bad", ["", "   ", "x" * 5000])
def test_compact_bad_model_output_is_an_error_not_a_summary(client, llm, logs, monkeypatch, bad):
    monkeypatch.setattr(appmod, "KEEP_RECENT_MESSAGES", 2)
    llm.summaries = [bad]
    events = compact(client, messages=conv(8))
    assert events[-1]["type"] == "error" and "Could not condense" in events[-1]["text"]
    assert not any(e["type"] == "summary" for e in events)
    assert logs[0]["status"] == "compaction_error"


def test_compact_model_exception_is_an_error_event(client, llm, logs, monkeypatch):
    monkeypatch.setattr(appmod, "KEEP_RECENT_MESSAGES", 2)
    llm.fail_with = RuntimeError("llama blew up")
    events = compact(client, messages=conv(8))
    assert events[-1]["type"] == "error" and "llama blew up" in events[-1]["text"]
    assert logs[0]["status"] == "compaction_error"


def test_compact_salvages_a_summary_cut_off_at_the_token_cap(client, llm, monkeypatch):
    monkeypatch.setattr(appmod, "KEEP_RECENT_MESSAGES", 2)
    llm.summaries = [('{"summary": "User runs a bakery. Budget is 5k. Asked abo', "length")]
    events = compact(client, messages=conv(8))
    assert events[-2]["summary"] == "User runs a bakery. Budget is 5k."


def test_compact_queues_behind_a_running_answer_and_sends_heartbeats(client, llm, monkeypatch):
    monkeypatch.setattr(appmod, "KEEP_RECENT_MESSAGES", 2)
    monkeypatch.setattr(appmod, "COMPACT_PROGRESS_SECONDS", 0.05)
    result: dict = {}

    def run():
        result["events"] = compact(client, messages=conv(8))

    appmod.INFERENCE_LOCK.acquire()  # a chat answer is "running"
    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.4)
    assert t.is_alive() and llm.calls == []  # still waiting for the lock
    appmod.INFERENCE_LOCK.release()
    t.join(5)
    events = result["events"]
    assert events[-2]["type"] == "summary"
    assert sum(1 for e in events if e["type"] == "progress" and e["tokens"] == 0) >= 2  # heartbeats while queued


def test_cancelled_compaction_releases_the_lock_and_never_generates():
    cancel = threading.Event()
    cancel.set()
    fake = FakeLlama()
    raw, finish = appmod.compact_generate_stream(fake, "prompt", lambda p: None, cancel)
    assert (raw, finish) == ("", None) and fake.calls == []
    assert appmod.INFERENCE_LOCK.acquire(blocking=False)  # released
    appmod.INFERENCE_LOCK.release()


def test_cancel_mid_generation_stops_early_and_releases_the_lock():
    cancel = threading.Event()
    fake = FakeLlama()
    fake.summaries = ["word " * 200]
    seen: list = []

    def emit(piece):
        if piece is not None:
            seen.append(piece)
            if len(seen) == 3:
                cancel.set()  # the client disconnects

    raw, _ = appmod.compact_generate_stream(fake, "prompt", emit, cancel)
    assert len(seen) == 3
    assert appmod.INFERENCE_LOCK.acquire(blocking=False)
    appmod.INFERENCE_LOCK.release()
