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
import time
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
from starlette.background import BackgroundTask
from llama_cpp import Llama
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel

import chatlog
import compaction

MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/models/gemma-3-1b-it-q4_0.gguf")
APP_API_KEY = os.getenv("APP_API_KEY")  # set in production; unset = no auth (local dev only)
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
MCP_SERVER_URLS = [u.strip() for u in os.getenv("MCP_SERVER_URLS", "").split(",") if u.strip()]
MAX_TOOL_HOPS = 15  # raised from 4 — a higher ceiling means a worse-case tool-looping request takes proportionally longer; watch it against timeoutSeconds (600s) and per-hop latency
MAX_HISTORY_MESSAGES = 40  # upper bound on messages considered; the token budget in build_turns is what actually protects n_ctx
N_CTX = int(os.getenv("N_CTX", "4096"))
LLAMA_THREADS = int(os.getenv("LLAMA_THREADS", "4"))  # match the Cloud Run --cpu value
MAX_GENERATION_TOKENS = 1400  # was 5000, then 1024, then 2048. 1024 cut long answers off mid-JSON. Qwen3-4B on 4 vCPU generates only ~3 tokens/s in production (roughly 67s for ~200 tokens in chat_logs), so 2048 could not finish inside the 600s request timeout; 1400 (~470s of generation plus prompt processing) still leaves room for a 500+ word answer. Watch it against timeoutSeconds and N_CTX, which also has to hold the prompt and history
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "2500"))  # cap on replayed chat history; there is no KV-cache reuse across turns, so every history token is re-processed on every turn (slow on CPU)
PROMPT_MARGIN_TOKENS = 64  # slack for the instructions turn and the start of the reply
MESSAGE_OVERHEAD_TOKENS = 12  # chat-template tokens wrapped around each replayed message
KEEP_USER_CHARS = 1200  # older user messages are replayed verbatim up to this length
KEEP_ANSWER_CHARS = 300  # older assistant answers are reduced to their opening sentences, up to this length
KEEP_ANSWER_SENTENCES = 2
INFERENCE_LOCK = threading.Lock()  # one llama.cpp instance is not thread-safe

# --- Summarization-based compaction (see compaction.py) --------------------
# The client owns the summary and sends it with every chat request; the server
# stays stateless. fit_history above remains the safety net: if no summary is
# sent or compaction fails, chats behave exactly as they did before.
COMPACT_AT_FRACTION = float(os.getenv("COMPACT_AT_FRACTION", "0.75"))  # compact when history reaches this share of its budget
# Two exchanges stay verbatim. With n_ctx 4096 the budget is too tight for four
# messages next to a summary, so the default drops to two there.
KEEP_RECENT_MESSAGES = int(os.getenv("KEEP_RECENT_MESSAGES", "4" if N_CTX >= 6144 else "2"))
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "320"))  # generation cap for one summary
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "2500"))  # validation cap on a summary (sent or generated)
# Per-call input cap for a compaction; it also has to leave room in n_ctx for the summary being written.
COMPACT_INPUT_MAX_TOKENS = min(
    int(os.getenv("COMPACT_INPUT_MAX_TOKENS", "3000")), N_CTX - SUMMARY_MAX_TOKENS - 128
)
COMPACT_PROGRESS_SECONDS = 10  # max silence on the /chat/compact stream (also keeps idle-connection timeouts away)
COMPACT_PROGRESS_TOKENS = 8  # a progress event every this many generated tokens

STATE: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Tools. To add a business-intelligence function later, write a plain python
# function plus a JSON-schema entry here — nothing else in this file
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

INSTRUCTIONS_TEMPLATE = """You are a friendly, knowledgeable assistant who explains things clearly and in depth, like a good teacher. Answer normally from your own knowledge.

You also have these tools:
{tools}
{context_block}
Rules:
- First, fill "thinking" with ONE short sentence (max 15 words) of your own reasoning about how to answer. The user can see this, so never write more than that.
- Greetings, general questions: answer directly with action "final_answer". Do not use a tool.
- Use "tool_call" ONLY when the user asks for current prices, market data or recent news/facts that a tool can fetch.
- If context from uploaded documents is given above and it answers the question, use it and say so. If it's there but doesn't cover the question, say it doesn't rather than guessing.
- Never invent prices, dates or news. If it did not come from a tool result or the document context above, do not state it.
- Match the depth of your answer to the question. For greetings and simple facts, one or two sentences. For anything that asks for an explanation, details, examples, a comparison, a how-to, advice, analysis or writing (this includes follow-ups like "tell me more", "explain in detail" or "give examples"), write a full, rich answer of at least 250 words (usually 300-500): open with a direct answer, then develop it by explaining the why, giving 2-4 concrete examples, and ending with a practical tip or takeaway. Never answer a request for detail in just a few sentences.
- For follow-up questions, build on the conversation so far and add NEW detail and examples. Do not repeat your previous answer.
- Inside "final_answer", put each list item on its own line and separate paragraphs with a blank line. Short **bold** headings are fine. This keeps long answers readable.
{summary_rule}
Respond with ONLY a JSON object of this exact shape:
{{"thinking": "<your brief reasoning>", "action": "tool_call" or "final_answer", "tool_name": "<name or empty string>", "tool_arguments": {{...or empty object}}, "final_answer": "<your full reply to the user: thorough, specific and well organized, with examples, for explanations, how-tos, comparisons, advice and writing; just a sentence or two for greetings and simple facts; empty string if calling a tool>"}}

Leave the fields you don't need empty rather than omitting them.

{summary_block}User: {question}"""


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

    try:  # numbers needed to tune compaction; never allowed to block startup
        instruction_tokens = count_tokens(
            STATE["llm"],
            INSTRUCTIONS_TEMPLATE.format(
                tools=tool_directory_text(), context_block="", summary_rule="", summary_block="", question=""
            ),
        )
        history_budget = min(MAX_HISTORY_TOKENS, N_CTX - MAX_GENERATION_TOKENS - instruction_tokens - PROMPT_MARGIN_TOKENS)
        print(
            f"[startup] compaction: instruction_tokens={instruction_tokens} n_ctx={N_CTX} "
            f"history_budget={history_budget} compact_at={COMPACT_AT_FRACTION} keep_recent={KEEP_RECENT_MESSAGES} "
            f"summary_max_tokens={SUMMARY_MAX_TOKENS} input_cap={COMPACT_INPUT_MAX_TOKENS}",
            flush=True,
        )
    except Exception as exc:
        print(f"[startup] could not measure instruction tokens: {exc!r}", flush=True)

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
    # Compaction state, owned by the client (see compaction.py). A request with
    # neither field behaves exactly as before.
    summary: str | None = None  # model-written summary of the first `summary_covers` messages
    summary_covers: int = 0  # how many leading messages of `messages` the summary already covers


def normalize_history(messages: list[dict]) -> list[dict]:
    """Gemma's chat template needs strict user/assistant alternation that
    starts with 'user'. Merge repeated assistant messages, keep the newest of
    repeated user messages, and drop anything else, so a malformed client
    history can never turn into a 500."""
    clean: list[dict] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        if clean and clean[-1]["role"] == role:
            if role == "user":
                # Two user messages in a row means the first never got a reply
                # (its request failed). Keep only the newest: merging them made
                # every retry re-send the failed text, which grew the prompt
                # until every later turn in that chat overflowed n_ctx too.
                clean[-1]["content"] = content
            else:
                clean[-1]["content"] += "\n\n" + content
        elif clean or role == "user":
            clean.append({"role": role, "content": content})
    return clean


class PromptTooLong(Exception):
    """The newest message alone (plus instructions) cannot fit in the context window."""


def count_tokens(llm: Llama, text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))


# A sentence ends at . ! or ? after a letter or closing quote/bracket, so list
# numbers ("1.") and years do not count as sentence ends.
_SENTENCE_END = re.compile(r"(?<=[A-Za-z)\"'\u2019\u201d][.!?])\s+")


def shorten_message(m: dict) -> dict:
    """Compact an older chat message. User messages are the questions, which
    are short and carry the thread, so they are kept verbatim (only a very long
    paste is cut). Assistant answers are long, so only their opening
    sentences are kept, marked with an ellipsis so the model knows they were
    abbreviated."""
    text = m["content"]
    if m["role"] == "user":
        if len(text) <= KEEP_USER_CHARS:
            return m
        cut = text[:KEEP_USER_CHARS].rsplit(" ", 1)[0]
        return {"role": "user", "content": cut + " [\u2026]"}
    flat = " ".join(text.split())
    short = " ".join(_SENTENCE_END.split(flat)[:KEEP_ANSWER_SENTENCES])
    if len(short) > KEEP_ANSWER_CHARS:
        short = short[:KEEP_ANSWER_CHARS].rsplit(" ", 1)[0]
    if short == flat:
        return {"role": "assistant", "content": flat}
    return {"role": "assistant", "content": short + " [\u2026]"}


def fit_history(llm: Llama, older: list[dict], budget: int) -> list[dict]:
    """Make the replayed history fit `budget` tokens. If everything fits it is
    sent untouched. Otherwise it is compacted: the most recent exchange stays
    in full (follow-ups like "continue" or "tell me more" depend on it), every
    older message goes through shorten_message, and if that is still too big
    the oldest messages are dropped."""

    def cost(m: dict) -> int:
        return count_tokens(llm, m["content"]) + MESSAGE_OVERHEAD_TOKENS

    if sum(cost(m) for m in older) <= budget:
        return older
    kept: list[dict] = []
    spent = 0
    for i in range(len(older) - 1, -1, -1):  # newest first, so recent context survives
        m = older[i]
        recent = i >= len(older) - 2
        for option in ([m, shorten_message(m)] if recent else [shorten_message(m)]):
            c = cost(option)
            if spent + c <= budget:
                kept.append(option)
                spent += c
                break
        else:
            break  # nothing older fits either
    kept.reverse()
    while kept and kept[0]["role"] != "user":  # a window must still start on a user turn
        kept.pop(0)
    return kept


def build_turns(
    messages: list[dict],
    session_id: str | None = None,
    log: dict | None = None,
    summary: str | None = None,
    summary_covers: int = 0,
    usage: dict | None = None,
) -> list[dict]:
    """Client history -> the turns sent to the model (instructions ride on the
    newest user message). Used by /chat and /chat/stream (now the same
    handler — see chat_stream below). If `log` is given, the question and the
    retrieved document context are recorded into it for eval.

    `summary`/`summary_covers` are the client's compaction state, already
    validated by the caller: the first `summary_covers` RAW messages are
    replaced by the summary. If `usage` is given it is filled with what
    chat_stream needs to report context usage in the `done` event."""
    if summary:
        # Slice the RAW list first (before normalize_history merges or drops
        # anything), so the client's indices stay valid.
        messages = compaction.slice_after_summary(messages, summary_covers)
    history = normalize_history(messages[-MAX_HISTORY_MESSAGES:])
    if not history or history[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must have role 'user'.")

    question = history[-1]["content"]
    context = retrieve_context(session_id, question)
    if log is not None:
        log["question"] = question
        log["context"] = context
    context_block = f"\nContext from documents uploaded this session:\n{context}\n" if context else ""
    final_turn = {
        "role": "user",
        "content": INSTRUCTIONS_TEMPLATE.format(
            tools=tool_directory_text(),
            context_block=context_block,
            summary_rule=compaction.SUMMARY_RULE if summary else "",
            summary_block=compaction.format_summary_block(summary) if summary else "",
            question=question,
        ),
    }
    older = history[:-1]  # everything before the newest question, replayed as-is

    llm = STATE.get("llm")
    if llm is not None:
        # n_ctx has to hold the prompt AND the reply, so the prompt may only
        # use what MAX_GENERATION_TOKENS leaves free. Compact, then drop, the
        # oldest turns until the history fits (it used to be capped by message
        # count, which let a few long answers overflow the window: "Requested
        # tokens (4288) exceed context window of 4096"). A summary, when there
        # is one, sits inside final_turn, so it is counted here too.
        budget = N_CTX - MAX_GENERATION_TOKENS
        used = count_tokens(llm, final_turn["content"]) + PROMPT_MARGIN_TOKENS
        if used > budget:
            raise PromptTooLong(
                f"That message is too long for me to read in full (about {used} tokens; the limit per "
                f"message is about {budget - 800}). Please shorten it or split it into parts."
            )
        if usage is not None:
            _fill_usage(llm, usage, final_turn["content"], older, question, context_block, summary)
        older = fit_history(llm, older, min(MAX_HISTORY_TOKENS, budget - used))

    return older + [final_turn]


def _fill_usage(
    llm: Llama,
    usage: dict,
    final_content: str,
    older: list[dict],
    question: str,
    context_block: str,
    summary: str | None,
) -> None:
    """What the `done` event needs: the instruction turn's size without the
    summary, the summary's own size, and the not-yet-summarized history at FULL
    length (before fit_history shortens anything), including the question."""
    if summary:
        plain = INSTRUCTIONS_TEMPLATE.format(
            tools=tool_directory_text(), context_block=context_block, summary_rule="", summary_block="", question=question
        )
        instruction_tokens = count_tokens(llm, plain)
        summary_tokens = max(0, count_tokens(llm, final_content) - instruction_tokens)
    else:
        instruction_tokens = count_tokens(llm, final_content)
        summary_tokens = 0
    history_tokens = sum(count_tokens(llm, m["content"]) + MESSAGE_OVERHEAD_TOKENS for m in older)
    history_tokens += count_tokens(llm, question) + MESSAGE_OVERHEAD_TOKENS  # the question is history next turn
    usage.update(
        instruction_tokens=instruction_tokens, summary_tokens=summary_tokens, history_tokens=history_tokens
    )


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
                temperature=0.6,
                max_tokens=MAX_GENERATION_TOKENS,
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


async def stream_reply(llm: Llama, turns: list[dict], log: dict, context_after=None):
    loop = asyncio.get_running_loop()
    cancel = threading.Event()
    started = time.monotonic()
    last_call_signature: tuple[str, str] | None = None  # (name, sorted-args-json) of the previous hop's tool call
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
            log["hops"] += 1

            try:
                decision = json.loads(raw)
            except json.JSONDecodeError:
                log["status"], log["final_answer"] = "malformed", raw
                if not answer_streamer.emitted:
                    yield sse({"type": "token", "text": raw})
                yield sse({"type": "done"})
                return
            log["thinking"] = decision.get("thinking") or ""

            if decision.get("action") == "tool_call" and decision.get("tool_name"):
                name = decision["tool_name"]
                args = decision.get("tool_arguments") or {}
                # A small model can get stuck re-issuing the identical call
                # instead of using its result — with MAX_TOOL_HOPS raised to
                # 15, that would otherwise burn every remaining hop (and its
                # full generation latency) before the fallback below kicks
                # in. Catch it after one repeat rather than fifteen.
                signature = (name, json.dumps(args, sort_keys=True, default=str))
                if signature == last_call_signature:
                    log["status"] = "tool_loop"
                    yield sse(
                        {
                            "type": "token",
                            "text": f"That lookup ({name}) repeated without new information, so I stopped early "
                            "rather than keep retrying — try rephrasing the question.",
                        }
                    )
                    yield sse({"type": "done"})
                    return
                last_call_signature = signature

                tool_call_id = uuid.uuid4().hex
                yield sse({"type": "tool", "tool_call_id": tool_call_id, "name": name, "args": args})
                result = await dispatch_tool(name, args)
                yield sse({"type": "tool_result", "tool_call_id": tool_call_id, "result": result})
                log["tool_calls"].append({"name": name, "args": args, "result": result})
                turns.append({"role": "assistant", "content": raw})
                turns.append(
                    {
                        "role": "user",
                        "content": f"[Result of {name}]\n{result}\n\nContinue: answer the original question, "
                        f"or call another tool if you still need to, in the same JSON shape.",
                    }
                )
                continue

            log["status"], log["final_answer"] = "ok", decision.get("final_answer") or ""
            if not answer_streamer.emitted:  # fallback if the answer never streamed
                yield sse({"type": "token", "text": decision.get("final_answer", "")})
            done_event: dict = {"type": "done"}
            if context_after is not None:  # normal completion only; never allowed to break a reply
                try:
                    done_event["context"] = context_after(log["final_answer"])
                except Exception as exc:
                    print(f"[compaction] could not compute context usage: {exc!r}")
            yield sse(done_event)
            return

        log["status"] = "tool_limit"
        yield sse({"type": "token", "text": "Reached the tool-call limit without a final answer — try rephrasing."})
        yield sse({"type": "done"})
    except Exception as exc:
        log["status"], log["error"] = "error", str(exc)
        yield sse({"type": "error", "text": str(exc)})
    finally:
        cancel.set()
        # Model-side latency only. The Firestore write is NOT awaited here: it
        # runs as a background task after the stream has closed (see
        # chat_stream), so it neither holds the reply open nor delays the next
        # request under concurrency=1.
        log["latency_s"] = round(time.monotonic() - started, 1)


# /chat now points at the same streaming handler as /chat/stream — there is
# no separate blocking implementation any more. Anything that used to POST
# to /chat gets a Server-Sent Events response instead.
@app.post("/chat", dependencies=[Depends(require_api_key)])
@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
async def chat_stream(req: ChatRequest):
    llm = STATE.get("llm")
    if llm is None:
        raise HTTPException(status_code=503, detail="Model still loading.")

    try:
        summary, covers = compaction.resolve_summary(
            req.summary, req.summary_covers, len(req.messages), SUMMARY_MAX_CHARS
        )
    except ValueError as exc:  # oversized or non-string summary: tell the client, never drop it silently
        raise HTTPException(status_code=400, detail=str(exc))

    log = chatlog.new_log(MODEL_PATH, INSTANCE_ID, req.session_id)
    usage: dict = {}
    try:
        turns = build_turns(req.messages, req.session_id, log, summary, covers, usage)
    except PromptTooLong as exc:
        log["status"], log["error"] = "too_long", str(exc)

        async def too_long():
            yield sse({"type": "token", "text": str(exc)})
            yield sse({"type": "done"})

        return StreamingResponse(
            too_long(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            background=BackgroundTask(chatlog.persist, log),
        )
    if summary:
        log["extra"] = {"summary_covers": covers, "summary_tokens": usage.get("summary_tokens")}
    return StreamingResponse(
        stream_reply(llm, turns, log, make_context_after(llm, req.messages, covers, usage, log)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        # Runs after the last byte is sent (also after a client disconnect), so
        # logging adds no reply latency. Relies on the service running with CPU
        # always allocated (--no-cpu-throttling), otherwise Cloud Run may
        # throttle the instance before the write completes.
        background=BackgroundTask(chatlog.persist, log),
    )


def make_context_after(llm: Llama, messages: list[dict], covers: int, usage: dict, log: dict):
    """Returns the callback stream_reply uses to build the `context` object of
    the `done` event once the answer text is known. `needs_compaction` is
    decided with select_fold_range on the same list the client will send to
    /chat/compact (its messages plus this answer), so the two always agree."""

    def context_after(answer: str) -> dict | None:
        if "history_tokens" not in usage:  # no model-side counts (llm missing): nothing to report
            return None
        answer_tokens = count_tokens(llm, answer) + MESSAGE_OVERHEAD_TOKENS
        used, budget = compaction.context_usage(
            summary_tokens=usage["summary_tokens"],
            history_tokens=usage["history_tokens"],
            answer_tokens=answer_tokens,
            instruction_tokens=usage["instruction_tokens"],
            n_ctx=N_CTX,
            max_generation_tokens=MAX_GENERATION_TOKENS,
            max_history_tokens=MAX_HISTORY_TOKENS,
            margin_tokens=PROMPT_MARGIN_TOKENS,
        )
        start, end = compaction.select_fold_range(
            messages + [{"role": "assistant", "content": answer}], covers, KEEP_RECENT_MESSAGES
        )
        flag = compaction.needs_compaction(used, budget, COMPACT_AT_FRACTION, end > start)
        log.setdefault("extra", {}).update(context_used=used, context_budget=budget, needs_compaction=flag)
        return {"used": used, "budget": budget, "needs_compaction": flag}

    return context_after


class CompactRequest(BaseModel):
    messages: list[dict]  # the client's full raw list, ending with the answer just produced
    summary: str | None = None  # previous summary, if any
    summary_covers: int = 0
    session_id: str | None = None


def compact_generate_stream(llm: Llama, prompt: str, emit, cancel: threading.Event) -> tuple[str, str | None]:
    """Worker thread: one grammar-constrained compaction call. Same pattern as
    generate_stream; returns (raw text, finish_reason)."""
    raw: list[str] = []
    finish: str | None = None
    try:
        with INFERENCE_LOCK:
            if cancel.is_set():  # client left while this was queued behind a running answer
                return "", None
            stream = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object", "schema": compaction.SUMMARY_SCHEMA},
                temperature=0.2,  # fidelity over variety
                max_tokens=SUMMARY_MAX_TOKENS,
                stream=True,
            )
            try:
                for chunk in stream:
                    if cancel.is_set():  # client went away: stop burning CPU, release the lock
                        break
                    choice = chunk["choices"][0]
                    piece = choice["delta"].get("content")
                    if piece:
                        raw.append(piece)
                        emit(piece)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
            finally:
                stream.close()
    finally:
        emit(None)
    return "".join(raw), finish


async def stream_compact(
    llm: Llama, previous_summary: str | None, chunks: list[list[dict]], fold_count: int,
    new_covers: int, input_tokens: int, log: dict,
):
    loop = asyncio.get_running_loop()
    cancel = threading.Event()
    started = time.monotonic()
    generated = 0
    summary = previous_summary
    try:
        yield sse({"type": "started", "fold_messages": fold_count, "input_tokens": input_tokens, "chunks": len(chunks)})
        for index, chunk in enumerate(chunks, start=1):
            prompt = compaction.build_summary_prompt(summary, chunk)
            queue: asyncio.Queue = asyncio.Queue()

            def emit(piece):  # called from the worker thread
                loop.call_soon_threadsafe(queue.put_nowait, piece)

            task = loop.run_in_executor(None, compact_generate_stream, llm, prompt, emit, cancel)
            chunk_started = time.monotonic()
            since_event = 0
            while True:
                try:
                    piece = await asyncio.wait_for(queue.get(), timeout=COMPACT_PROGRESS_SECONDS)
                except asyncio.TimeoutError:  # heartbeat: queued behind an answer, or still reading the prompt
                    yield sse({"type": "progress", "tokens": generated, "chunk": index, "chunks": len(chunks)})
                    continue
                if piece is None:
                    break
                if index == 1 and generated == 0:  # prompt-processing time (plus any wait for the lock), for tuning
                    log["extra"]["first_token_s"] = round(time.monotonic() - chunk_started, 1)
                generated += 1
                since_event += 1
                if since_event >= COMPACT_PROGRESS_TOKENS:
                    since_event = 0
                    yield sse({"type": "progress", "tokens": generated, "chunk": index, "chunks": len(chunks)})
            raw, finish = await task  # re-raises anything the worker hit
            text = compaction.extract_summary(raw)
            if finish == "length":  # hit the token cap mid-summary: drop the dangling fragment
                text = compaction.trim_to_sentence(text)
            summary = compaction.validate_summary(text, SUMMARY_MAX_CHARS)  # ValueError -> error event below

        summary_tokens = count_tokens(llm, summary)
        log["status"], log["final_answer"] = "compaction_ok", summary
        log["extra"].update(output_tokens=generated, summary_tokens=summary_tokens, summary_covers=new_covers)
        yield sse({"type": "summary", "summary": summary, "summary_covers": new_covers, "summary_tokens": summary_tokens})
        yield sse({"type": "done"})
    except Exception as exc:
        log["status"], log["error"] = "compaction_error", str(exc)
        yield sse({"type": "error", "text": f"Could not condense the conversation: {exc}"})
    finally:
        cancel.set()
        log["latency_s"] = round(time.monotonic() - started, 1)


@app.post("/chat/compact", dependencies=[Depends(require_api_key)])
async def chat_compact(req: CompactRequest):
    """Fold `previous summary + the older messages` into one new summary and
    stream progress. A separate request from the answer stream on purpose: an
    answer can take ~470s of generation plus prefill, and Cloud Run's request
    limit is 600s, so doing both in one request could time out."""
    llm = STATE.get("llm")
    if llm is None:
        raise HTTPException(status_code=503, detail="Model still loading.")
    try:
        summary, covers = compaction.resolve_summary(
            req.summary, req.summary_covers, len(req.messages) + 1, SUMMARY_MAX_CHARS
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    start, end = compaction.select_fold_range(req.messages, covers, KEEP_RECENT_MESSAGES)
    to_fold = normalize_history(req.messages[start:end])
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    if end <= start or not to_fold:  # nothing to fold: a graceful no-op, not an error

        async def nothing():
            yield sse({"type": "started", "fold_messages": 0, "input_tokens": 0, "chunks": 0})
            yield sse({"type": "done"})

        return StreamingResponse(nothing(), media_type="text/event-stream", headers=headers)

    chunks = compaction.plan_chunks(
        to_fold, summary, lambda t: count_tokens(llm, t),
        input_max_tokens=COMPACT_INPUT_MAX_TOKENS, summary_max_tokens=SUMMARY_MAX_TOKENS,
    )
    input_tokens = (
        count_tokens(llm, compaction.build_summary_prompt(summary, [])) + sum(
            count_tokens(llm, m["content"]) + MESSAGE_OVERHEAD_TOKENS for ch in chunks for m in ch
        )
    )
    log = chatlog.new_log(MODEL_PATH, INSTANCE_ID, req.session_id)
    log["question"] = f"[compaction] folding {end - start} messages in {len(chunks)} chunk(s)"
    log["extra"] = {
        "fold_messages": end - start, "chunks": len(chunks), "input_tokens": input_tokens,
        "previous_summary_tokens": count_tokens(llm, summary) if summary else 0,
    }
    return StreamingResponse(
        stream_compact(llm, summary, chunks, end - start, end, input_tokens, log),
        media_type="text/event-stream",
        headers=headers,
        background=BackgroundTask(chatlog.persist, log),
    )


# ---------------------------------------------------------------------------
# Chat titles. The UI names each saved chat from its first question. It's a
# tiny grammar-constrained generation on the same model, so it has to share
# INFERENCE_LOCK with replies: it waits only briefly, and if a reply is in
# flight it gives up (title: null) and the UI keeps its fallback title rather
# than making anyone queue behind it.
# ---------------------------------------------------------------------------

TITLE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
}
TITLE_PROMPT = """Write a short title (3 to 6 words) for a chat that starts with the message below. Use the same language as the message. No quotes and no trailing punctuation.

Respond with ONLY a JSON object of this exact shape:
{{"title": "<the title>"}}

Message: {question}"""
TITLE_QUESTION_CHARS = 300  # only the opening of a long paste is needed to name a chat
TITLE_MAX_CHARS = 60
TITLE_MAX_TOKENS = 40
TITLE_LOCK_WAIT_S = 3.0

_TITLE_FIELD = re.compile(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)')


def clean_title(raw: str) -> str | None:
    """Pull the title out of the model's JSON. If generation hit the token
    cap mid-string the JSON is cut off, so fall back to a regex over what
    was produced before giving up."""
    try:
        title = json.loads(raw).get("title", "")
    except (json.JSONDecodeError, AttributeError):
        m = _TITLE_FIELD.search(raw)
        title = m.group(1) if m else ""
    title = " ".join(str(title).split()).strip(" \"'\u201c\u201d\u2018\u2019`.")
    return title[:TITLE_MAX_CHARS].strip() or None


def generate_title(llm: Llama, question: str) -> str | None:
    """Runs in a worker thread. None = model busy or no usable output."""
    if not INFERENCE_LOCK.acquire(timeout=TITLE_LOCK_WAIT_S):
        return None
    try:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": TITLE_PROMPT.format(question=question)}],
            response_format={"type": "json_object", "schema": TITLE_SCHEMA},
            temperature=0.3,
            max_tokens=TITLE_MAX_TOKENS,
        )
    finally:
        INFERENCE_LOCK.release()
    return clean_title(out["choices"][0]["message"].get("content") or "")


class TitleRequest(BaseModel):
    question: str


@app.post("/chat/title", dependencies=[Depends(require_api_key)])
async def chat_title(req: TitleRequest):
    llm = STATE.get("llm")
    if llm is None:
        raise HTTPException(status_code=503, detail="Model still loading.")
    question = " ".join(req.question.split())[:TITLE_QUESTION_CHARS]
    if not question:
        raise HTTPException(status_code=400, detail="question is empty.")
    loop = asyncio.get_running_loop()
    try:
        title = await loop.run_in_executor(None, generate_title, llm, question)
    except Exception as exc:  # a title is cosmetic: never surface a 500 for it
        print(f"[title] generation failed: {exc!r}")
        title = None
    return {"title": title}


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
