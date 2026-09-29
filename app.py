import asyncio
import functools
import io
import ipaddress
import json
import math
import os
import re
import socket
import threading
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from ddgs import DDGS
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from llama_cpp import Llama
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/models/gemma-3-1b-it-q4_0.gguf")
APP_API_KEY = os.getenv("APP_API_KEY")  # set in production; unset = no auth (local dev only)
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
MCP_SERVER_URLS = [u.strip() for u in os.getenv("MCP_SERVER_URLS", "").split(",") if u.strip()]
MAX_TOOL_HOPS = 4
MAX_HISTORY_MESSAGES = 24  # crude guard against overflowing n_ctx
N_CTX = int(os.getenv("N_CTX", "4096"))
LLAMA_THREADS = int(os.getenv("LLAMA_THREADS", "4"))  # match the Cloud Run --cpu value
INFERENCE_LOCK = threading.Lock()  # one llama.cpp instance is not thread-safe

STATE: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Tools. To add a business-intelligence function later, write a plain python
# function plus a JSON-schema description here — nothing else in this file
# needs to change.
# ---------------------------------------------------------------------------


def get_market_snapshot(ticker: str) -> str:
    """Latest price + recent headlines for a ticker (stocks, ETFs, FX, or
    crypto pairs like BTC-USD)."""
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(f"{ticker} financial news market", max_results=3))
        news = "\n".join(f"- {h['title']}: {h['body'][:200]}" for h in hits) or "No recent headlines."
    except Exception as exc:  # a flaky search shouldn't crash the tool call
        news = f"News lookup failed: {exc}"

    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if hist.empty:
            price = "No pricing data available."
        else:
            last = hist.iloc[-1]
            price = f"Close {last['Close']:.2f} | High {last['High']:.2f} | Low {last['Low']:.2f}"
    except Exception as exc:
        price = f"Price lookup failed: {exc}"

    return f"PRICE: {price}\nNEWS:\n{news}"


def web_search(query: str) -> str:
    """General web search for current events, news and facts."""
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=4))
    except Exception as exc:
        return f"Web search failed: {exc}"
    if not hits:
        return "No results found."
    return "\n".join(f"- {h['title']}: {h['body'][:250]}" for h in hits)


def _is_public_url(url: str) -> bool:
    """Only public http(s) hosts — never loopback, private, link-local or the
    cloud metadata server (this service is reachable from the internet)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def read_webpage(url: str) -> str:
    """Extract the readable text of a public web page."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        for _ in range(4):  # follow a few redirects manually, re-checking every hop
            if not _is_public_url(url):
                return "Refused: only public http(s) URLs can be read."
            resp = requests.get(url, headers=headers, timeout=10, allow_redirects=False, stream=True)
            if resp.is_redirect:
                url = urljoin(url, resp.headers.get("Location", ""))
                resp.close()
                continue
            break
        else:
            return "Too many redirects."
        body = b""
        for chunk in resp.iter_content(65536):
            body += chunk
            if len(body) > 1_000_000:  # never pull more than ~1 MB
                break
        resp.close()
        soup = BeautifulSoup(body, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        return text[:3000] or "The page had no readable text."
    except Exception as exc:
        return f"Failed to read webpage: {exc}"


BUILTIN_TOOLS: dict[str, dict[str, Any]] = {
    "get_market_snapshot": {
        "handler": get_market_snapshot,
        "description": "Get the latest price and recent headlines for a ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "e.g. BTC-USD, AAPL, XAUUSD=X"}},
            "required": ["ticker"],
        },
    },
    "web_search": {
        "handler": web_search,
        "description": "Search the web for current events, news or facts you cannot know offline.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "search query"}},
            "required": ["query"],
        },
    },
    "read_webpage": {
        "handler": read_webpage,
        "description": "Read the text of one public web page. Use after web_search to open a result.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "full http(s) URL"}},
            "required": ["url"],
        },
    },
}

# ---------------------------------------------------------------------------
# Session-scoped document knowledge base. A client-generated session_id keys
# an in-memory BM25 index of chunks from whatever PDFs/DOCX/TXT/MD the user
# uploaded this session. Deliberately no embedding model here: this box is
# already CPU-bound and memory-tight running one 4B GGUF (production replies
# take 25-150s per the Cloud Run request logs), and a second model just for
# embeddings would compete with it for the same 4 vCPU / 6 GiB. Plain BM25
# costs microseconds per query against a few hundred chunks, so it doesn't
# add meaningfully to that budget. Trade-off: lexical match only, no semantic
# match — a question that shares no words with the relevant passage may miss.
#
# Nothing here is persisted: it lives in this process's memory only, keyed by
# session_id, and is lost on restart (scale-to-zero after idle, a crash, or a
# new deploy). INSTANCE_ID changes every time this process starts, so a
# client can tell when its uploaded documents didn't survive a restart.
# ---------------------------------------------------------------------------

INSTANCE_ID = uuid.uuid4().hex
MAX_DOC_BYTES = 15 * 1024 * 1024  # 15 MB per uploaded file
MAX_CHUNKS_PER_SESSION = 400  # crude memory guard, independent of any one file's size
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
TOP_K_CHUNKS = 4

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Greedy character-based chunking that prefers to break on a paragraph or
    sentence boundary near the target size. Character-based (not token-based)
    because tokenizing would mean loading a tokenizer just for this."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind("\n\n", start, end)
            if cut == -1 or cut <= start + size // 2:
                cut = text.rfind(". ", start, end)
            if cut != -1 and cut > start + size // 2:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_text(filename: str, data: bytes) -> str:
    """Raises ValueError for an unsupported type, and lets parser errors
    propagate as-is — the caller turns both into an HTTP 400."""
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        from docx import Document

        doc = Document(io.BytesIO(data))
        return "\n\n".join(p.text for p in doc.paragraphs)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    raise ValueError("Unsupported file type. Use PDF, DOCX, TXT or MD.")


class BM25Index:
    """Textbook Okapi BM25 over an in-memory list of chunks. No numpy, no
    external index — a few hundred chunks scored in pure Python is
    microseconds, irrelevant next to a 25-150s model reply."""

    K1, B = 1.5, 0.75

    def __init__(self):
        self.chunks: list[dict] = []  # {"doc_name", "text", "tf": Counter, "length": int}
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def _recompute_df(self) -> None:
        self.df = Counter()
        for c in self.chunks:
            self.df.update(c["tf"].keys())
        self.avgdl = sum(c["length"] for c in self.chunks) / len(self.chunks) if self.chunks else 0.0

    def add(self, doc_name: str, text: str) -> int:
        added = 0
        for piece in _chunk_text(text):
            tokens = _tokenize(piece)
            if not tokens:
                continue
            self.chunks.append({"doc_name": doc_name, "text": piece, "tf": Counter(tokens), "length": len(tokens)})
            added += 1
        self._recompute_df()
        return added

    def remove_doc(self, doc_name: str) -> None:
        self.chunks = [c for c in self.chunks if c["doc_name"] != doc_name]
        self._recompute_df()

    def search(self, query: str, top_k: int = TOP_K_CHUNKS) -> list[dict]:
        if not self.chunks:
            return []
        q_terms = set(_tokenize(query))
        if not q_terms:
            return []
        n = len(self.chunks)
        scores = [0.0] * n
        for term in q_terms:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            for i, c in enumerate(self.chunks):
                tf = c["tf"].get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.K1 * (1 - self.B + self.B * c["length"] / (self.avgdl or 1))
                scores[i] += idf * (tf * (self.K1 + 1)) / (denom or 1)
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:top_k] if scores[i] > 0]


SESSIONS: dict[str, BM25Index] = {}
SESSION_DOC_NAMES: dict[str, list[str]] = {}
SESSIONS_LOCK = threading.Lock()


def retrieve_context(session_id: str | None, question: str) -> str:
    if not session_id:
        return ""
    index = SESSIONS.get(session_id)
    if not index or not index.chunks:
        return ""
    hits = index.search(question)
    if not hits:
        return ""
    return "\n\n---\n\n".join(f"[{h['doc_name']}]\n{h['text']}" for h in hits)


# ---------------------------------------------------------------------------
# MCP: every server is treated as a separate HTTP/SSE service (the same
# pattern as your gcloud-mcp-server on Cloud Run) — never spawned as a local
# subprocess inside this container. A fresh connection is opened per call
# rather than held open, which is more robust across Cloud Run's
# scale-to-zero / multi-instance lifecycle than a long-lived session.
# ---------------------------------------------------------------------------


async def fetch_mcp_tools(url: str) -> dict[str, dict]:
    tools: dict[str, dict] = {}
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            for t in listed.tools:
                tools[t.name] = {
                    "server_url": url,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                }
    return tools


async def call_mcp_tool(url: str, name: str, arguments: dict) -> str:
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=arguments)
            parts = [c.text for c in result.content if getattr(c, "text", None)]
            return "\n".join(parts) if parts else "(tool returned no text content)"


async def dispatch_tool(name: str, arguments: dict) -> str:
    try:
        if name in BUILTIN_TOOLS:
            loop = asyncio.get_event_loop()
            fn = functools.partial(BUILTIN_TOOLS[name]["handler"], **arguments)
            return await loop.run_in_executor(None, fn)
        if name in STATE.get("mcp_tools", {}):
            server_url = STATE["mcp_tools"][name]["server_url"]
            return await call_mcp_tool(server_url, name, arguments)
    except Exception as exc:  # bad arguments from a small model, network errors, ...
        return f"Tool {name} failed: {exc}"
    return f"Unknown tool: {name}"


def tool_directory_text() -> str:
    lines = []
    for name, spec in BUILTIN_TOOLS.items():
        lines.append(f"- {name}({json.dumps(spec['parameters'])}): {spec['description']}")
    for name, spec in STATE.get("mcp_tools", {}).items():
        lines.append(f"- {name}({json.dumps(spec['parameters'])}): {spec['description']}")
    return "\n".join(lines) or "(no tools available)"


# ---------------------------------------------------------------------------
# Model. The chat template is auto-detected from the GGUF's own metadata
# (we don't pass chat_format), and instructions ride on the first user turn
# rather than a "system" message, since that was required for the model this
# was originally built against and was never revisited after switching to
# Qwen3 in production (MODEL_PATH below) — untested whether Qwen3's own
# template would accept a system role instead.
# response_format below grammar-constrains every model turn to valid JSON
# regardless of chat template — that's what makes tool-routing reliable here,
# not a hand-parsed string like `.replace("```json", "")`.
# ---------------------------------------------------------------------------

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        # Listed first: llama.cpp's grammar sampler emits object keys in this
        # declaration order, so "thinking" is generated (and can be streamed
        # to the client) before the model commits to action/final_answer.
        "thinking": {"type": "string"},
        "action": {"type": "string", "enum": ["tool_call", "final_answer"]},
        "tool_name": {"type": "string"},
        "tool_arguments": {"type": "object"},
        "final_answer": {"type": "string"},
    },
    "required": ["thinking", "action", "tool_name", "tool_arguments", "final_answer"],
}

INSTRUCTIONS_TEMPLATE = """You are a concise, friendly assistant. Answer normally from your own knowledge.

You also have these tools:
{tools}
{context_block}
Rules:
- First, fill "thinking" with ONE short sentence (max ~15 words) of your own reasoning about how to answer. The user can see this, so keep it brief.
- Greetings, general questions: answer directly with action "final_answer". Do not use a tool.
- Use "tool_call" ONLY when the user asks for current prices, market data or recent news/facts that a tool can fetch.
- If context from uploaded documents is given above and it answers the question, use it and say so. If it's there but doesn't cover the question, say it doesn't rather than guessing.
- Never invent prices, dates or news. If it did not come from a tool result or the document context above, do not state it.
- Keep answers short.

Respond with ONLY a JSON object of this exact shape:
{{"thinking": "<your brief reasoning>", "action": "tool_call" or "final_answer", "tool_name": "<name or empty string>", "tool_arguments": {{...or empty object}}, "final_answer": "<your reply to the user, or empty string if calling a tool>"}}

Leave the fields you don't need empty rather than omitting them.

User: {question}"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Check the volume mount / MODEL_PATH.")

    STATE["llm"] = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=LLAMA_THREADS,
        use_mmap=False,  # one plain read into RAM at startup, instead of
        # mmap-driven page faults over a gcsfuse-mounted volume
        verbose=False,
    )

    mcp_tools: dict[str, dict] = {}
    for url in MCP_SERVER_URLS:
        try:
            mcp_tools.update(await fetch_mcp_tools(url))
        except Exception as exc:
            print(f"[startup] could not load tools from {url}: {exc}")
    STATE["mcp_tools"] = mcp_tools

    yield
    STATE.clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=False,  # "*" + credentials is rejected by browsers anyway, and unneeded here
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)):
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class ChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant", "content": "..."}] — client keeps history
    session_id: str | None = None  # ties this chat to its uploaded documents, if any


def normalize_history(messages: list[dict]) -> list[dict]:
    """Gemma's chat template needs strict user/assistant alternation that
    starts with 'user'. Merge repeated roles and drop anything else, so a
    malformed client history can never turn into a 500."""
    clean: list[dict] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        if clean and clean[-1]["role"] == role:
            clean[-1]["content"] += "\n\n" + content
        elif clean or role == "user":
            clean.append({"role": role, "content": content})
    return clean


def build_turns(messages: list[dict], session_id: str | None = None) -> list[dict]:
    """Client history -> the turns sent to the model (instructions ride on the
    newest user message). Used by /chat and /chat/stream (now the same
    handler — see chat_stream below)."""
    history = normalize_history(messages[-MAX_HISTORY_MESSAGES:])
    if not history or history[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must have role 'user'.")

    question = history[-1]["content"]
    context = retrieve_context(session_id, question)
    context_block = f"\nContext from documents uploaded this session:\n{context}\n" if context else ""
    turns = history[:-1]  # everything before the newest question, replayed as-is
    turns.append(
        {
            "role": "user",
            "content": INSTRUCTIONS_TEMPLATE.format(
                tools=tool_directory_text(), context_block=context_block, question=question
            ),
        }
    )
    return turns


# ---------------------------------------------------------------------------
# Streaming. Every model turn is a grammar-constrained JSON object, so the
# tokens that arrive are pieces of JSON, not prose. StringFieldStreamer pulls
# one named string field's text out of that JSON as it arrives; /chat/stream
# runs one instance for "thinking" (unconditional — it's always the first
# field written) and one for "final_answer" (gated on action=="final_answer",
# so a tool-routing turn never leaks its empty final_answer text, and a key
# named "final_answer" nested inside tool_arguments can't be mistaken for the
# real one). Both are forwarded to the browser as server-sent events.
# ---------------------------------------------------------------------------


class StringFieldStreamer:
    """Decodes one named JSON string field's value out of an object arriving
    piece by piece. If gate_key/gate_value are given, extraction only starts
    once that other field's value has been seen and matches — otherwise it
    starts as soon as the target key itself is seen."""

    _SIMPLE = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}

    def __init__(self, key: str, gate_key: str | None = None, gate_value: str | None = None):
        self._key_re = re.compile(rf'"{re.escape(key)}"\s*:\s*"')
        self._gate_re = re.compile(rf'"{re.escape(gate_key)}"\s*:\s*"(\w+)"') if gate_key else None
        self.gate_value = gate_value
        self.gate: bool | None = True if gate_key is None else None  # None = unresolved
        self.buf = ""
        self.pos: int | None = None  # index just after the opening quote of the value
        self.done = False
        self.emitted = False

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        if self.done or self.gate is False:
            return ""
        if self.gate is None:
            m = self._gate_re.search(self.buf)
            if not m:
                return ""
            self.gate = m.group(1) == self.gate_value
            if not self.gate:
                return ""
        if self.pos is None:
            m = self._key_re.search(self.buf)
            if not m:
                return ""
            self.pos = m.end()

        buf, i, out = self.buf, self.pos, []
        while i < len(buf):
            c = buf[i]
            if c == '"':
                self.done = True
                i += 1
                break
            if c != "\\":
                out.append(c)
                i += 1
                continue
            if i + 1 >= len(buf):
                break  # escape sequence split across chunks: wait for the rest
            n = buf[i + 1]
            if n in self._SIMPLE:
                out.append(self._SIMPLE[n])
                i += 2
            elif n == "u":
                if i + 6 > len(buf):
                    break
                try:
                    code = int(buf[i + 2 : i + 6], 16)
                    if 0xD800 <= code < 0xDC00:  # high surrogate: needs its low half too
                        if i + 12 > len(buf):
                            break
                        low = int(buf[i + 8 : i + 12], 16)
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        i += 12
                    else:
                        i += 6
                    out.append(chr(code))
                except ValueError:
                    out.append("\ufffd")
                    i += 6
            else:
                out.append(n)
                i += 2
        self.pos = i
        text = "".join(out)
        if text:
            self.emitted = True
        return text


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def generate_stream(llm: Llama, turns: list[dict], emit, cancel: threading.Event) -> str:
    """Runs in a worker thread. Calls emit(piece) for every raw token piece and
    emit(None) once when the turn is over. Returns the full raw text."""
    raw: list[str] = []
    try:
        with INFERENCE_LOCK:
            stream = llm.create_chat_completion(
                messages=turns,
                response_format={"type": "json_object", "schema": DECISION_SCHEMA},
                temperature=0.2,
                max_tokens=5000,
                stream=True,
            )
            try:
                for chunk in stream:
                    if cancel.is_set():  # client went away: stop burning CPU
                        break
                    piece = chunk["choices"][0]["delta"].get("content")
                    if piece:
                        raw.append(piece)
                        emit(piece)
            finally:
                stream.close()
    finally:
        emit(None)
    return "".join(raw)


async def stream_reply(llm: Llama, turns: list[dict]):
    loop = asyncio.get_running_loop()
    cancel = threading.Event()
    try:
        for _ in range(MAX_TOOL_HOPS):
            queue: asyncio.Queue = asyncio.Queue()
            thinking_streamer = StringFieldStreamer("thinking")
            answer_streamer = StringFieldStreamer("final_answer", gate_key="action", gate_value="final_answer")
            thinking_done_sent = False

            def emit(piece):  # called from the worker thread
                loop.call_soon_threadsafe(queue.put_nowait, piece)

            task = loop.run_in_executor(None, generate_stream, llm, turns, emit, cancel)
            while (piece := await queue.get()) is not None:
                think_text = thinking_streamer.feed(piece)
                if think_text:
                    yield sse({"type": "thinking", "text": think_text})
                if thinking_streamer.done and not thinking_done_sent:
                    thinking_done_sent = True
                    yield sse({"type": "thinking_done"})
                answer_text = answer_streamer.feed(piece)
                if answer_text:
                    yield sse({"type": "token", "text": answer_text})
            raw = await task  # re-raises anything the worker hit

            try:
                decision = json.loads(raw)
            except json.JSONDecodeError:
                if not answer_streamer.emitted:
                    yield sse({"type": "token", "text": raw})
                yield sse({"type": "done"})
                return

            if decision.get("action") == "tool_call" and decision.get("tool_name"):
                name = decision["tool_name"]
                args = decision.get("tool_arguments") or {}
                yield sse({"type": "tool", "text": f"{name}({args})"})
                result = await dispatch_tool(name, args)
                turns.append({"role": "assistant", "content": raw})
                turns.append(
                    {
                        "role": "user",
                        "content": f"[Result of {name}]\n{result}\n\nContinue: answer the original question, "
                        f"or call another tool if you still need to, in the same JSON shape.",
                    }
                )
                continue

            if not answer_streamer.emitted:  # fallback if the answer never streamed
                yield sse({"type": "token", "text": decision.get("final_answer", "")})
            yield sse({"type": "done"})
            return

        yield sse({"type": "token", "text": "Reached the tool-call limit without a final answer — try rephrasing."})
        yield sse({"type": "done"})
    except Exception as exc:
        yield sse({"type": "error", "text": str(exc)})
    finally:
        cancel.set()


# /chat now points at the same streaming handler as /chat/stream — there is
# no separate blocking implementation any more. Anything that used to POST
# to /chat gets a Server-Sent Events response instead.
@app.post("/chat", dependencies=[Depends(require_api_key)])
@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
async def chat_stream(req: ChatRequest):
    llm = STATE.get("llm")
    if llm is None:
        raise HTTPException(status_code=503, detail="Model still loading.")

    turns = build_turns(req.messages, req.session_id)
    return StreamingResponse(
        stream_reply(llm, turns),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/session/init")
async def session_init():
    """The client compares this to what it last saw and stored. A mismatch
    means the process restarted since then, so any documents it had uploaded
    are gone (nothing here survives a restart) — the client can tell the
    user their documents need re-uploading, without touching chat history."""
    return {"instance_id": INSTANCE_ID}


class ClearRequest(BaseModel):
    session_id: str


@app.post("/kb/upload", dependencies=[Depends(require_api_key)])
async def kb_upload(session_id: str, file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_DOC_BYTES:
        raise HTTPException(status_code=413, detail="File too large (15 MB max).")

    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(None, extract_text, file.filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # a malformed PDF/DOCX shouldn't 500 the request
        raise HTTPException(status_code=400, detail=f"Could not read that file: {exc}")

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in that file (is it a scanned/image PDF?).",
        )

    with SESSIONS_LOCK:
        index = SESSIONS.setdefault(session_id, BM25Index())
        if len(index.chunks) >= MAX_CHUNKS_PER_SESSION:
            raise HTTPException(
                status_code=413,
                detail="This session's document limit is full. Clear documents to add more.",
            )
        added = await loop.run_in_executor(None, index.add, file.filename, text)
        SESSION_DOC_NAMES.setdefault(session_id, []).append(file.filename)
        total = len(index.chunks)

    return {"filename": file.filename, "chunks_added": added, "total_chunks": total}


@app.post("/kb/clear", dependencies=[Depends(require_api_key)])
async def kb_clear(req: ClearRequest):
    with SESSIONS_LOCK:
        SESSIONS.pop(req.session_id, None)
        SESSION_DOC_NAMES.pop(req.session_id, None)
    return {"cleared": True}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
