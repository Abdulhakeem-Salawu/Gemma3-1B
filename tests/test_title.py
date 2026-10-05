"""Tests for POST /chat/title (names a saved chat from its first question).
llama_cpp is replaced by a stub, so no model is needed.
Run from the repo root (app.py mounts ./static)."""

import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _StubLlama:  # only so `from llama_cpp import Llama` works if this file runs first
    def __init__(self, *args, **kwargs):
        pass


stub = types.ModuleType("llama_cpp")
stub.Llama = _StubLlama
sys.modules.setdefault("llama_cpp", stub)

from fastapi.testclient import TestClient  # noqa: E402

import app as appmod  # noqa: E402


class TitleLlm:
    """Answers non-streaming completions with `content`, or raises `exc`."""

    def __init__(self, content='{"title": "BTC price check"}', exc: Exception | None = None):
        self.content = content
        self.exc = exc
        self.calls: list[dict] = []

    def create_chat_completion(self, messages, response_format, temperature, max_tokens, stream=False):
        self.calls.append(
            {
                "messages": messages,
                "response_format": response_format,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
        )
        if self.exc:
            raise self.exc
        return {"choices": [{"message": {"content": self.content}}]}


@pytest.fixture()
def llm(monkeypatch):
    fake = TitleLlm()
    monkeypatch.setitem(appmod.STATE, "llm", fake)
    monkeypatch.setattr(appmod, "APP_API_KEY", None)
    return fake


@pytest.fixture()
def client(llm):
    return TestClient(appmod.app)


def post(client, question, **kwargs):
    return client.post("/chat/title", json={"question": question}, **kwargs)


def test_returns_the_model_title_and_asks_for_json_only(client, llm):
    r = post(client, "What is the price of BTC-USD right now?")
    assert r.status_code == 200
    assert r.json() == {"title": "BTC price check"}

    (call,) = llm.calls
    assert len(call["messages"]) == 1 and call["messages"][0]["role"] == "user"
    assert "What is the price of BTC-USD right now?" in call["messages"][0]["content"]
    assert call["response_format"]["type"] == "json_object"
    assert "title" in call["response_format"]["schema"]["properties"]
    assert call["max_tokens"] == appmod.TITLE_MAX_TOKENS
    assert not call["stream"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"title": "\\"Bitcoin price today.\\""}', "Bitcoin price today"),
        ('{"title": "  Spaced   out\\ntitle  "}', "Spaced out title"),
        ('{"title": "How to bake sourdough bre', "How to bake sourdough bre"),  # cut off by max_tokens
        ('{"title": ""}', None),
        ("not json at all", None),
        ('{"title": "' + "x" * 200 + '"}', "x" * appmod.TITLE_MAX_CHARS),
    ],
)
def test_cleans_up_the_output(client, llm, raw, expected):
    llm.content = raw
    assert post(client, "q").json() == {"title": expected}


def test_braces_in_the_question_do_not_break_the_prompt(client, llm):
    r = post(client, "what does {x} mean in {{y}} and {0}?")
    assert r.json() == {"title": "BTC price check"}
    assert "{x} mean in {{y}} and {0}" in llm.calls[0]["messages"][0]["content"]


def test_only_the_opening_of_a_long_question_is_sent(client, llm):
    post(client, "word " * 1000)
    sent = llm.calls[0]["messages"][0]["content"].split("Message: ", 1)[1]
    assert len(sent) <= appmod.TITLE_QUESTION_CHARS


def test_rejects_an_empty_question(client):
    assert post(client, "   ").status_code == 400
    assert client.post("/chat/title", json={}).status_code == 422


def test_503_while_the_model_is_loading(client, monkeypatch):
    monkeypatch.delitem(appmod.STATE, "llm")
    assert post(client, "hi").status_code == 503


def test_busy_model_gives_up_quickly_instead_of_queueing(client, llm, monkeypatch):
    monkeypatch.setattr(appmod, "TITLE_LOCK_WAIT_S", 0.2)
    held, release = threading.Event(), threading.Event()

    def hold_the_model():
        with appmod.INFERENCE_LOCK:
            held.set()
            release.wait(5)

    t = threading.Thread(target=hold_the_model)
    t.start()
    assert held.wait(2)
    try:
        started = time.monotonic()
        r = post(client, "hello")
        assert r.json() == {"title": None}
        assert time.monotonic() - started < 2
        assert llm.calls == []  # never touched the model
    finally:
        release.set()
        t.join()
    assert post(client, "hello").json() == {"title": "BTC price check"}  # usable again once free


def test_model_failure_is_a_null_title_not_a_500_and_releases_the_lock(client, llm):
    llm.exc = RuntimeError("boom")
    assert post(client, "hello").json() == {"title": None}
    assert appmod.INFERENCE_LOCK.acquire(timeout=0.5)
    appmod.INFERENCE_LOCK.release()


def test_requires_the_api_key_when_one_is_configured(client, monkeypatch):
    monkeypatch.setattr(appmod, "APP_API_KEY", "secret")
    assert post(client, "hi").status_code == 401
    assert post(client, "hi", headers={"x-api-key": "wrong"}).status_code == 401
    assert post(client, "hi", headers={"x-api-key": "secret"}).json() == {"title": "BTC price check"}
