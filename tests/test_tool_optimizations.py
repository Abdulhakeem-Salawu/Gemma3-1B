"""Tests for tool-loop optimizations: fuzzy repeat detection, force_finalize
(forced synthesis instead of a canned apology when tools are exhausted), and
the in-loop token-budget guard. llama_cpp is replaced by a stub, so no model
is needed. Run from the repo root (app.py mounts ./static)."""

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class SequencedFakeLlama:
    """Tokenizer: one token per whitespace-separated word. Each chat call pops
    the next scripted decision dict from `decisions` (by call index; the last
    one repeats if exhausted). A compaction call (recognized by the summary
    schema) is not exercised here and just gets a trivial summary."""

    def __init__(self, *args, **kwargs):
        self.calls: list[dict] = []
        self.decisions: list[dict] = []

    def tokenize(self, data: bytes, add_bos: bool = False):
        return data.decode("utf-8").split()

    def create_chat_completion(self, messages, response_format, temperature, max_tokens, stream):
        schema = response_format["schema"]
        is_compaction = "summary" in schema["properties"]
        self.calls.append({"messages": messages, "schema": schema, "compaction": is_compaction})
        if is_compaction:
            raw = json.dumps({"summary": "S"})
        elif self.decisions:
            idx = min(len(self.calls) - 1, len(self.decisions) - 1)
            raw = json.dumps(self.decisions[idx])
        else:
            raw = json.dumps(
                {
                    "thinking": "ok",
                    "action": "final_answer",
                    "tool_name": "",
                    "tool_arguments": {},
                    "final_answer": "default answer",
                }
            )
        pieces = [raw[i : i + 6] for i in range(0, len(raw), 6)]

        def gen():
            for p in pieces:
                yield {"choices": [{"delta": {"content": p}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        return gen()


stub = types.ModuleType("llama_cpp")
stub.Llama = SequencedFakeLlama
sys.modules.setdefault("llama_cpp", stub)

from fastapi.testclient import TestClient  # noqa: E402

import app as appmod  # noqa: E402
import chatlog  # noqa: E402


@pytest.fixture()
def llm(monkeypatch):
    fake = SequencedFakeLlama()
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


@pytest.fixture(autouse=True)
def fake_tool_handlers(monkeypatch):
    """Tool handlers never hit the real network in these tests."""
    monkeypatch.setitem(
        appmod.BUILTIN_TOOLS,
        "web_search",
        {**appmod.BUILTIN_TOOLS["web_search"], "handler": lambda query: f"result for {query}"},
    )
    monkeypatch.setitem(
        appmod.BUILTIN_TOOLS,
        "get_market_snapshot",
        {**appmod.BUILTIN_TOOLS["get_market_snapshot"], "handler": lambda ticker: f"price for {ticker}"},
    )


def sse_events(resp) -> list[dict]:
    out = []
    for frame in resp.text.split("\n\n"):
        if frame.startswith("data: "):
            out.append(json.loads(frame[6:]))
    return out


def tool_call(name="web_search", **args):
    return {"thinking": "t", "action": "tool_call", "tool_name": name, "tool_arguments": args, "final_answer": ""}


def final(text):
    return {"thinking": "t", "action": "final_answer", "tool_name": "", "tool_arguments": {}, "final_answer": text}


# --- is_repeat_tool_call (pure function) --------------------------------------------


def test_identical_args_is_a_repeat():
    args = json.dumps({"query": "X"}, sort_keys=True)
    assert appmod.is_repeat_tool_call("web_search", args, [("web_search", args)])


def test_near_duplicate_args_is_a_repeat():
    # mirrors a real production chat_logs entry: successive rewordings of one search
    q1 = json.dumps(
        {"query": "AI tools for import-export that integrate with stock market analysis"}, sort_keys=True
    )
    q2 = json.dumps(
        {"query": "AI tools for import-export that integrate with stock market analysis or financial tools"},
        sort_keys=True,
    )
    assert appmod.is_repeat_tool_call("web_search", q2, [("web_search", q1)])


def test_different_tool_name_is_never_a_repeat():
    args = json.dumps({"query": "X"}, sort_keys=True)
    assert not appmod.is_repeat_tool_call("read_webpage", args, [("web_search", args)])


def test_clearly_different_query_is_not_a_repeat():
    q1 = json.dumps({"query": "weather in Lagos"}, sort_keys=True)
    q2 = json.dumps({"query": "stock price of AAPL"}, sort_keys=True)
    assert not appmod.is_repeat_tool_call("web_search", q2, [("web_search", q1)])


# --- force_finalize on a stuck or exhausted tool loop ---------------------------------


def test_near_duplicate_loop_is_finalized_not_apologized(client, llm, logs):
    llm.decisions = [
        tool_call(query="AI tools for import-export that integrate with stock market analysis"),
        tool_call(query="AI tools for import-export that integrate with stock market analysis or financial tools"),
        final("Here is what I found: three platforms do this."),
    ]
    r = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "find competitors"}]})
    events = sse_events(r)
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer == "Here is what I found: three platforms do this."
    assert events[-1]["type"] == "done"
    assert logs[0]["status"] == "tool_loop_finalized"
    assert logs[0]["final_answer"] == answer
    assert len(llm.calls) == 3  # 2 tool-call hops (2nd rejected as a repeat) + 1 forced finalize


def test_forced_finalize_cannot_itself_request_a_tool(client, llm, logs):
    llm.decisions = [tool_call(query="a"), tool_call(query="a"), final("done")]
    client.post("/chat/stream", json={"messages": [{"role": "user", "content": "x"}]})
    assert llm.calls[-1]["schema"]["properties"]["action"]["enum"] == ["final_answer"]


def test_hop_limit_is_finalized_not_apologized(client, llm, logs, monkeypatch):
    monkeypatch.setattr(appmod, "MAX_TOOL_HOPS", 2)
    llm.decisions = [
        tool_call(query="alpha"),
        tool_call(query="beta"),  # distinct args — never trips the loop guard
        final("Best answer with what I have."),
    ]
    r = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "x"}]})
    events = sse_events(r)
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer == "Best answer with what I have."
    assert logs[0]["status"] == "tool_limit_finalized"
    assert len(llm.calls) == 3  # 2 hops (the cap) + 1 forced finalize


def test_finalize_failure_falls_back_to_a_short_apology(client, llm, logs):
    llm.decisions = [tool_call(query="a"), tool_call(query="a"), final("")]  # forced call returns empty
    r = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "x"}]})
    events = sse_events(r)
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert "try rephrasing" in answer
    assert logs[0]["status"] == "tool_loop"  # not "_finalized": the forced attempt itself failed


# --- in-loop token-budget guard -------------------------------------------------------


def test_tool_loop_finalizes_before_overflowing_the_context_window(client, llm, logs, monkeypatch):
    llm.decisions = [tool_call(query="alpha"), final("Finalized before overflow.")]

    def big_result(query):
        return "word " * 3000  # far bigger than any hop budget has room for

    monkeypatch.setitem(
        appmod.BUILTIN_TOOLS,
        "web_search",
        {**appmod.BUILTIN_TOOLS["web_search"], "handler": big_result},
    )
    r = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "x"}]})
    events = sse_events(r)
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer == "Finalized before overflow."
    assert logs[0]["status"] == "tool_budget_finalized"
    assert len(llm.calls) == 2  # 1 tool-call hop + 1 forced finalize (no 2nd real hop)
