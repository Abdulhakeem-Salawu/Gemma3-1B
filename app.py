import asyncio
import functools
import ipaddress
import json
import os
import socket
import threading
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from ddgs import DDGS
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
# Model. Gemma's own chat template (auto-detected from the GGUF's metadata,
# since we don't pass chat_format) has no "system" role and raises an error
# if it sees one, so instructions ride on the first user turn instead.
# response_format below grammar-constrains every model turn to valid JSON
# regardless of chat template — that's what makes tool-routing reliable here,
# not a hand-parsed string like `.replace("```json", "")`.
# ---------------------------------------------------------------------------

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["tool_call", "final_answer"]},
        "tool_name": {"type": "string"},
        "tool_arguments": {"type": "object"},
        "final_answer": {"type": "string"},
    },
    "required": ["action", "tool_name", "tool_arguments", "final_answer"],
}

INSTRUCTIONS_TEMPLATE = """You are a concise, friendly assistant. Answer normally from your own knowledge.

You also have these tools:
{tools}

Rules:
- Greetings, general questions: answer directly with action "final_answer". Do not use a tool.
- Use "tool_call" ONLY when the user asks for current prices, market data or recent news/facts that a tool can fetch.
- Never invent prices, dates or news. If it did not come from a tool result, do not state it.
- Keep answers short.

Respond with ONLY a JSON object of this exact shape:
{{"action": "tool_call" or "final_answer", "tool_name": "<name or empty string>", "tool_arguments": {{...or empty object}}, "final_answer": "<your reply to the user, or empty string if calling a tool>"}}

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
    allow_methods=["POST"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)):
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class ChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant", "content": "..."}] — client keeps history


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


def run_completion(llm: Llama, turns: list[dict]) -> dict:
    with INFERENCE_LOCK:
        return llm.create_chat_completion(
            messages=turns,
            response_format={"type": "json_object", "schema": DECISION_SCHEMA},
            temperature=0.2,
            max_tokens=5000,
        )


@app.post("/chat", dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest):
    llm = STATE.get("llm")
    if llm is None:
        raise HTTPException(status_code=503, detail="Model still loading.")

    history = normalize_history(req.messages[-MAX_HISTORY_MESSAGES:])
    if not history or history[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must have role 'user'.")

    question = history[-1]["content"]
    turns = history[:-1]  # everything before the newest question, replayed as-is
    turns.append(
        {
            "role": "user",
            "content": INSTRUCTIONS_TEMPLATE.format(tools=tool_directory_text(), question=question),
        }
    )

    loop = asyncio.get_event_loop()
    tool_trace: list[str] = []

    for _ in range(MAX_TOOL_HOPS):
        completion = await loop.run_in_executor(None, run_completion, llm, turns)
        raw = completion["choices"][0]["message"]["content"]
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            return {"reply": raw, "tool_trace": tool_trace}

        if decision.get("action") == "tool_call" and decision.get("tool_name"):
            name = decision["tool_name"]
            args = decision.get("tool_arguments") or {}
            tool_trace.append(f"{name}({args})")
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

        return {"reply": decision.get("final_answer", ""), "tool_trace": tool_trace}

    return {"reply": "Reached the tool-call limit without a final answer — try rephrasing.", "tool_trace": tool_trace}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
