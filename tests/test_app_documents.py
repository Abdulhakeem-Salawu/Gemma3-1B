"""App-level tests for client-owned documents: /kb/extract, the `documents` field of a chat
request, the stateless two-instance behaviour, the old-client fallback, window safety and logging.
llama_cpp is replaced by a stub, so no model is needed. Run from the repo root."""

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


class FakeLlama:
    """One token per whitespace-separated word. Every chat completion answers 'Answer.'"""

    def __init__(self, *args, **kwargs):
        self.calls: list[dict] = []

    def tokenize(self, data: bytes, add_bos: bool = False):
        return data.decode("utf-8").split()

    def create_chat_completion(self, messages, response_format, temperature, max_tokens, stream=False):
        self.calls.append({"messages": messages})
        raw = json.dumps(
            {"thinking": "ok", "action": "final_answer", "tool_name": "", "tool_arguments": {}, "final_answer": "Answer."}
        )
        pieces = [raw[i : i + 8] for i in range(0, len(raw), 8)]

        def gen():
            for p in pieces:
                yield {"choices": [{"delta": {"content": p}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        return gen()


stub = types.ModuleType("llama_cpp")
stub.Llama = FakeLlama
sys.modules.setdefault("llama_cpp", stub)

from fastapi.testclient import TestClient  # noqa: E402

import app as appmod  # noqa: E402
import chatlog  # noqa: E402
import documents as D  # noqa: E402
from test_documents import SECTIONS, sample_raw, tiny_pdf  # noqa: E402


@pytest.fixture()
def llm(monkeypatch):
    fake = FakeLlama()
    monkeypatch.setitem(appmod.STATE, "llm", fake)
    monkeypatch.setitem(appmod.STATE, "mcp_tools", {})
    monkeypatch.setattr(appmod, "APP_API_KEY", None)
    appmod.SESSIONS.clear()
    appmod.SESSION_DOC_NAMES.clear()
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
    return [json.loads(f[6:]) for f in resp.text.split("\n\n") if f.startswith("data: ")]


def payload(built: dict) -> dict:
    """A document as the browser sends it with a chat request."""
    return {k: built[k2] for k, k2 in (("id", "doc_id"), ("name", "name"), ("chunks", "chunks"),
                                       ("overlaps", "overlaps"), ("outline", "outline"))}


def blue_doc() -> dict:
    return payload(D.build_document("Blue.pdf", sample_raw(), pages=4))


def ask(client, question: str, documents=..., history=(), **extra):
    messages = list(history) + [{"role": "user", "content": question}]
    body = {"messages": messages, "session_id": "s1", **extra}
    if documents is not ...:
        body["documents"] = documents
    return client.post("/chat", json=body)


def prompt(llm) -> str:
    return llm.calls[-1]["messages"][-1]["content"]


# --- /kb/extract ---------------------------------------------------------------------------


def test_extract_pasted_text_returns_a_document_and_stores_nothing(client):
    r = client.post("/kb/extract", json={"text": sample_raw(), "name": "Pasted text"})
    assert r.status_code == 200
    doc = r.json()
    assert doc["name"] == "Pasted text" and doc["chars"] > 1000 and len(doc["chunks"]) >= 3
    assert doc["doc_id"] and len(doc["overlaps"]) == len(doc["chunks"]) and doc["outline"]
    assert not appmod.SESSIONS and not appmod.SESSION_DOC_NAMES  # stateless


def test_extract_the_same_text_twice_gives_the_same_id(client):
    a = client.post("/kb/extract", json={"text": sample_raw(), "name": "a"}).json()
    b = client.post("/kb/extract", json={"text": sample_raw(), "name": "b"}).json()
    assert a["doc_id"] == b["doc_id"]


def test_extract_a_real_pdf_upload(client):
    r = client.post("/kb/extract", files={"file": ("Blue economy .pdf", tiny_pdf("Verification is mandatory"), "application/pdf")})
    assert r.status_code == 200
    doc = r.json()
    assert doc["name"] == "Blue economy .pdf" and "Verification is mandatory" in " ".join(doc["chunks"])
    assert doc["pages"] == 1  # reported for PDFs, so the UI can say "4 pages"


def test_extract_text_file_upload(client):
    r = client.post("/kb/extract", files={"file": ("notes.txt", "Bees make honey.".encode(), "text/plain")})
    assert r.status_code == 200 and r.json()["chunks"] == ["Bees make honey."]


@pytest.mark.parametrize(
    "request_kwargs, status, fragment",
    [
        ({"json": {"name": "x"}}, 400, "text must be a string"),
        ({"json": {"text": 5}}, 400, "text must be a string"),
        ({"content": b"not json", "headers": {"content-type": "application/json"}}, 400, ""),
        ({"json": {"text": "   \n  "}}, 400, "No text found"),
        ({"files": {"file": ("x.exe", b"MZ", "application/octet-stream")}}, 400, "Unsupported file type"),
        ({"files": {"file": ("bad.pdf", b"this is not a pdf", "application/pdf")}}, 400, "Could not read that file"),
        ({"data": {"nope": "1"}}, 400, "No file received"),
    ],
)
def test_extract_errors_are_clear_400s(client, request_kwargs, status, fragment):
    r = client.post("/kb/extract", **request_kwargs)
    assert r.status_code == status and fragment in r.json()["detail"]


def test_extract_scanned_pdf_gets_the_no_text_message(client):
    r = client.post("/kb/extract", files={"file": ("scan.pdf", tiny_pdf(""), "application/pdf")})
    assert r.status_code == 400 and "no selectable text" in r.json()["detail"]


def test_extract_size_limits_give_413(client, monkeypatch):
    monkeypatch.setattr(appmod, "MAX_EXTRACT_CHARS", 1000)
    assert client.post("/kb/extract", json={"text": "w " * 600}).status_code == 413
    monkeypatch.setattr(appmod, "MAX_DOC_BYTES", 10)
    assert client.post("/kb/extract", files={"file": ("a.txt", b"x" * 50, "text/plain")}).status_code == 413
    monkeypatch.setattr(appmod, "MAX_DOC_BYTES", 15 * 1024 * 1024)
    r = client.post("/kb/extract", json={"text": "word " * 90_000})
    assert r.status_code == 413 and "too large" in r.json()["detail"]


def test_extract_requires_the_api_key(client, monkeypatch):
    monkeypatch.setattr(appmod, "APP_API_KEY", "secret")
    assert client.post("/kb/extract", json={"text": "hello world"}).status_code == 401
    ok = client.post("/kb/extract", json={"text": "hello world"}, headers={"x-api-key": "secret"})
    assert ok.status_code == 200


def test_upload_route_keeps_working_and_now_also_returns_the_document(client):
    r = client.post("/kb/upload?session_id=s9", files={"file": ("a.txt", sample_raw().encode(), "text/plain")})
    body = r.json()
    assert r.status_code == 200 and body["filename"] == "a.txt" and body["chunks_added"] >= 1  # old fields
    assert body["doc_id"] and body["chunks"] and body["outline"]  # new fields
    assert "s9" in appmod.SESSIONS  # and the old in-memory index is still filled


# --- chat with documents ---------------------------------------------------------------------


def test_specific_question_puts_the_matching_passage_in_the_prompt(client, llm):
    ask(client, "How does verification work?", [blue_doc()])
    p = prompt(llm)
    assert "Context from documents uploaded this session:" in p and "Verification" in p and "12. VERIFICATION" in p


def test_overview_question_samples_the_whole_document(client, llm):
    ask(client, "Summarize this for me", [blue_doc()])
    p = prompt(llm)
    assert "excerpts" in p
    for heading, _ in SECTIONS:
        assert heading in p  # the outline is there


def test_greeting_adds_no_document_text(client, llm):
    ask(client, "Hi", [blue_doc()])
    assert "Context from documents" not in prompt(llm)


def test_followup_uses_the_previous_question(client, llm):
    history = [
        {"role": "user", "content": "what are the weather risks?"},
        {"role": "assistant", "content": "Regulatory change and weather."},
    ]
    ask(client, "tell me more about that", [blue_doc()], history=history)
    assert "weather" in prompt(llm)


def test_nothing_matching_tells_the_model_so(client, llm):
    ask(client, "what is the capital of Mongolia", [blue_doc()])
    assert "no passage matched" in prompt(llm)


def test_empty_documents_list_means_no_document_context_even_if_the_server_has_a_legacy_index(client, llm):
    client.post("/kb/upload?session_id=s1", files={"file": ("old.txt", b"The legacy zebra document.", "text/plain")})
    ask(client, "tell me about the zebra")  # old client: field absent
    assert "legacy zebra" in prompt(llm)
    ask(client, "tell me about the zebra", documents=[])  # new client with nothing attached
    assert "legacy zebra" not in prompt(llm) and "Context from documents" not in prompt(llm)


def test_old_client_without_the_field_still_gets_the_in_memory_index(client, llm):
    client.post("/kb/upload?session_id=s1", files={"file": ("old.txt", b"The legacy zebra document.", "text/plain")})
    ask(client, "what about the zebra")
    assert "legacy zebra" in prompt(llm)


def test_documents_work_on_a_different_instance_than_the_upload(client, llm, monkeypatch):
    """The reported bug: upload landed on instance A, the question on instance B (nothing in memory).
    Documents ride along with every request, so no instance needs to have seen the upload."""
    extracted = client.post("/kb/extract", json={"text": sample_raw(), "name": "Blue.pdf"}).json()
    doc = payload(extracted)
    monkeypatch.setattr(appmod, "INSTANCE_ID", "a-different-process")
    appmod.SESSIONS.clear()
    appmod.SESSION_DOC_NAMES.clear()
    questions = [
        ("How does verification work?", "VERIFICATION"),
        ("what are the weather risks?", "RISKS"),
        ("tell me about section 2", "MARKET OVERVIEW"),
        ("what is the notification process", "Notifications"),
        ("Summarize this", "INTRODUCTION"),
    ]
    for q, expect in questions:
        r = ask(client, q, [doc])
        assert r.status_code == 200
        assert expect in prompt(llm), q
        assert "Context from documents uploaded this session:" in prompt(llm), q


def test_two_documents_in_one_chat_are_both_available(client, llm):
    a = payload(D.build_document("A.txt", "Alpha document about volcanoes and lava flows."))
    b = payload(D.build_document("B.txt", "Beta document about glaciers and ice sheets."))
    ask(client, "tell me about glaciers", [a, b])
    assert "glaciers" in prompt(llm) and "[B.txt]" in prompt(llm)
    ask(client, "summarize both", [a, b])
    assert "volcanoes" in prompt(llm) and "glaciers" in prompt(llm)


# --- validation -------------------------------------------------------------------------------


def test_bad_documents_get_400_and_oversized_ones_413(client):
    assert ask(client, "hi", [{"name": "no id", "chunks": ["a"]}]).status_code == 400
    assert ask(client, "hi", [{"id": "a", "chunks": []}]).status_code == 400
    assert ask(client, "hi", "nope").status_code == 422
    too_many = [{"id": f"d{i}", "name": "n", "chunks": ["x"]} for i in range(D.MAX_DOCS + 1)]
    r = ask(client, "hi", too_many)
    assert r.status_code == 413 and "Too many documents" in r.json()["detail"]
    huge = [{"id": "big", "name": "b", "chunks": ["y" * 2900] * 150}]
    r = ask(client, "hi", huge)
    assert r.status_code == 413 and "too large" in r.json()["detail"]


# --- window safety ------------------------------------------------------------------------------


def test_document_text_can_never_push_the_prompt_past_the_window(llm, monkeypatch):
    big = payload(D.build_document("Big.txt", "\n\n".join(f"{n}. TOPIC {n}\n" + ("lorem ipsum dolor sit amet " * 60) for n in range(1, 30))))
    for n_ctx in (2500, 3000, 4096, 6144):
        monkeypatch.setattr(appmod, "N_CTX", n_ctx)
        monkeypatch.setattr(appmod, "DOC_CONTEXT_TOKENS", 1100 if n_ctx < 6144 else 1800)
        for q in ("Summarize this", "tell me about lorem ipsum", "section 7"):
            turns = appmod.build_turns([{"role": "user", "content": q}], "s", None, documents=D.validate_documents([big]))
            used = appmod.count_tokens(llm, turns[-1]["content"]) + appmod.PROMPT_MARGIN_TOKENS
            assert used <= n_ctx - appmod.MAX_GENERATION_TOKENS, (n_ctx, q, used)


def test_when_nothing_is_left_for_documents_only_a_note_is_sent(llm, monkeypatch):
    # The instructions fit (no "message too long"), but after the reply, the margin and the room history
    # always keeps there are fewer than MIN_BUDGET_TOKENS left for document text.
    monkeypatch.setattr(appmod, "N_CTX", 2100)
    doc = D.validate_documents([blue_doc()])
    turns = appmod.build_turns([{"role": "user", "content": "how does verification work?"}], "s", None, documents=doc)
    assert "no room" in turns[-1]["content"] and "12. VERIFICATION" not in turns[-1]["content"]


def test_a_summary_and_documents_both_fit(client, llm):
    r = ask(client, "how does verification work?", [blue_doc()], summary="User asked about ports.", summary_covers=0)
    assert r.status_code == 200
    p = prompt(llm)
    assert "User asked about ports." in p and "Context from documents" in p


# --- logging ---------------------------------------------------------------------------------------


def test_the_log_records_what_the_model_was_shown(client, llm, logs):
    doc = blue_doc()
    ask(client, "how does verification work?", [doc], summary="S", summary_covers=0)
    log = logs[-1]
    extra = log["extra"]
    assert extra["doc_mode"] == "search" and extra["doc_ids"] == doc["id"]
    assert extra["doc_chunks_total"] == len(doc["chunks"]) and extra["doc_chunks_selected"] >= 1
    assert extra["doc_context_tokens"] > 0
    assert extra["summary_covers"] == 0  # the summary's numbers survive next to the document's
    assert "12. VERIFICATION" in log["context"]


def test_a_greeting_is_logged_as_no_document_use(client, llm, logs):
    ask(client, "Hi", [blue_doc()])
    assert logs[-1]["extra"]["doc_mode"] == "none" and logs[-1]["extra"]["doc_context_tokens"] == 0
