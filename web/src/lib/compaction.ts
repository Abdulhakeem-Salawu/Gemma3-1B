import { useSyncExternalStore } from "react";

/**
 * Client side of summarization-based compaction (server: compaction.py and
 * /chat/compact in app.py).
 *
 * The CLIENT owns the summary, so the stateless server survives restarts and
 * a second Cloud Run instance. Per chat we keep:
 *   summary  - model-written text covering the first `covers` messages
 *   covers   - how many leading messages of the raw message list (user and
 *              assistant messages with text, see toHistory) it replaces
 *   anchor   - a fingerprint of the last covered message, so an edited,
 *              deleted or truncated history invalidates the summary instead of
 *              silently pairing it with the wrong messages
 * The full transcript stays visible in the UI; only what is sent to the model
 * changes. If anything goes wrong we fall back to plain chats (the server's
 * older history shortening still applies).
 */

const SUMMARY_STORAGE = "agent_summary_v1";

export type HistoryMessage = { role: "user" | "assistant"; content: string };

export type ContextInfo = { used: number; budget: number; needs_compaction: boolean };

type StoredSummary = {
  summary: string;
  covers: number;
  anchor: string;
  /** thread index of the first message the summary does NOT cover (for the divider) */
  dividerIndex: number | null;
};

export type CompactionPhase = "idle" | "running" | "error";

export type CompactionView = {
  phase: CompactionPhase;
  startedAt: number;
  foldMessages: number;
  chunks: number;
  chunk: number;
  tokens: number;
  error: string;
  dividerIndex: number | null;
};

// ---------------------------------------------------------------------------
// History helpers
// ---------------------------------------------------------------------------

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((p): p is { type: "text"; text: string } => p?.type === "text" && typeof p.text === "string")
    .map((p) => p.text)
    .join("\n\n");
}

/** The raw message list sent to the server: user/assistant messages that have
 * text. The summary's `covers` counts into THIS list. */
export function toHistory(messages: readonly { role: string; content: unknown }[]): HistoryMessage[] {
  return messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({ role: m.role as "user" | "assistant", content: textOf(m.content) }))
    .filter((m) => m.content.trim().length > 0);
}

/** Thread index of the `historyIndex`-th message that toHistory keeps. */
function threadIndexOf(messages: readonly { role: string; content: unknown }[], historyIndex: number): number | null {
  let kept = 0;
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if ((m.role === "user" || m.role === "assistant") && textOf(m.content).trim().length > 0) {
      if (kept === historyIndex) return i;
      kept++;
    }
  }
  return null;
}

function hash(text: string): string {
  let h = 5381;
  for (let i = 0; i < text.length; i++) h = ((h << 5) + h + text.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}

function anchorOf(history: readonly HistoryMessage[], covers: number): string {
  const m = history[covers - 1];
  // Whitespace-normalized, so a harmless trim of the stored text between the
  // streamed answer and the saved message can never invalidate the summary.
  return m ? `${covers}:${m.role}:${hash(m.content.replace(/\s+/g, " ").trim())}` : "";
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

function loadStored(): StoredSummary | null {
  try {
    const raw = JSON.parse(localStorage.getItem(SUMMARY_STORAGE) ?? "null");
    if (
      raw &&
      typeof raw.summary === "string" &&
      raw.summary.trim() &&
      Number.isInteger(raw.covers) &&
      raw.covers > 0 &&
      typeof raw.anchor === "string"
    ) {
      return {
        summary: raw.summary,
        covers: raw.covers,
        anchor: raw.anchor,
        dividerIndex: Number.isInteger(raw.dividerIndex) ? raw.dividerIndex : null,
      };
    }
  } catch {
    // fall through: unreadable state is the same as none
  }
  return null;
}

function saveStored(value: StoredSummary) {
  try {
    localStorage.setItem(SUMMARY_STORAGE, JSON.stringify(value));
  } catch {
    // storage full or blocked: the summary still works for this page load
  }
}

// ---------------------------------------------------------------------------
// Observable state (banner, composer, transcript divider)
// ---------------------------------------------------------------------------

function idleView(): CompactionView {
  return {
    phase: "idle",
    startedAt: 0,
    foldMessages: 0,
    chunks: 0,
    chunk: 0,
    tokens: 0,
    error: "",
    dividerIndex: loadStored()?.dividerIndex ?? null,
  };
}

let view: CompactionView = idleView();
const listeners = new Set<() => void>();

function setView(patch: Partial<CompactionView>) {
  view = { ...view, ...patch };
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useCompactionView(): CompactionView {
  return useSyncExternalStore(subscribe, () => view);
}

/** Current state outside React (tests and scripts). */
export function getCompactionView(): CompactionView {
  return view;
}

// ---------------------------------------------------------------------------
// Using and invalidating the stored summary
// ---------------------------------------------------------------------------

/** Forget the summary and stop any compaction in flight. Called on "New chat";
 * the epoch makes a compaction that is still running discard its result. */
let epoch = 0;
let activeAbort: AbortController | null = null;
let pendingChoice: ((choice: "retry" | "continue") => void) | null = null;

export function resetCompaction() {
  epoch++;
  activeAbort?.abort();
  activeAbort = null;
  pendingChoice?.("continue");
  pendingChoice = null;
  try {
    localStorage.removeItem(SUMMARY_STORAGE);
  } catch {
    // ignore
  }
  setView({ ...idleView(), dividerIndex: null });
}

/** Drops a stored summary that no longer matches the messages (history was
 * truncated, edited or cleared). `maxCovers` is how many leading messages a
 * valid summary may cover. */
function validStored(history: readonly HistoryMessage[], maxCovers: number): StoredSummary | null {
  const stored = loadStored();
  if (!stored) return null;
  if (stored.covers > maxCovers || stored.anchor !== anchorOf(history, stored.covers)) {
    resetCompaction();
    return null;
  }
  return stored;
}

/** Called once when persisted chat history is loaded. */
export function reconcileSummary(history: readonly HistoryMessage[]) {
  validStored(history, history.length);
}

/** The fields to add to a chat request. `history` ends with the new question,
 * which must stay uncovered. */
export function summaryPayload(history: readonly HistoryMessage[]): { summary?: string; summary_covers?: number } {
  const stored = validStored(history, history.length - 1);
  return stored ? { summary: stored.summary, summary_covers: stored.covers } : {};
}

// ---------------------------------------------------------------------------
// Running a compaction
// ---------------------------------------------------------------------------

type CompactEvent =
  | { type: "started"; fold_messages: number; input_tokens: number; chunks: number }
  | { type: "progress"; tokens: number; chunk?: number; chunks?: number }
  | { type: "summary"; summary: string; summary_covers: number; summary_tokens: number }
  | { type: "done" }
  | { type: "error"; text: string };

async function* sseEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<CompactEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        if (frame.startsWith("data: ")) yield JSON.parse(frame.slice(6)) as CompactEvent;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

type CompactResult = { summary: string; covers: number } | null; // null: nothing to fold

async function requestCompaction(
  history: readonly HistoryMessage[],
  previous: StoredSummary | null,
  ctx: { apiKey: string; sessionId: string; signal: AbortSignal },
): Promise<CompactResult> {
  const res = await fetch("/chat/compact", {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-api-key": ctx.apiKey },
    body: JSON.stringify({
      messages: history,
      session_id: ctx.sessionId,
      ...(previous ? { summary: previous.summary, summary_covers: previous.covers } : {}),
    }),
    signal: ctx.signal,
  });
  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    let message = detail;
    try {
      message = JSON.parse(detail).detail ?? detail;
    } catch {
      // not JSON: use the raw text
    }
    throw new Error(
      res.status === 401 ? "Wrong API key." : message || `Request failed (${res.status})`,
    );
  }

  let result: CompactResult = null;
  let finished = false;
  for await (const evt of sseEvents(res.body)) {
    switch (evt.type) {
      case "started":
        setView({ foldMessages: evt.fold_messages, chunks: evt.chunks, chunk: evt.chunks ? 1 : 0, tokens: 0 });
        break;
      case "progress":
        setView({ tokens: evt.tokens, ...(evt.chunk ? { chunk: evt.chunk } : {}) });
        break;
      case "summary":
        result = { summary: evt.summary, covers: evt.summary_covers };
        break;
      case "error":
        throw new Error(evt.text);
      case "done":
        finished = true;
        break;
    }
  }
  if (!finished) throw new Error("The connection dropped before the summary arrived.");
  return result;
}

function waitForChoice(signal: AbortSignal): Promise<"retry" | "continue"> {
  return new Promise((resolve) => {
    const finish = (choice: "retry" | "continue") => {
      pendingChoice = null;
      signal.removeEventListener("abort", onAbort);
      resolve(choice);
    };
    const onAbort = () => finish("continue");
    pendingChoice = finish;
    signal.addEventListener("abort", onAbort);
  });
}

export function retryCompaction() {
  pendingChoice?.("retry");
}

/** "Continue anyway": keep chatting without a (new) summary. */
export function skipCompaction() {
  pendingChoice?.("continue");
}

/**
 * Condenses the older messages. `history` is the full raw list INCLUDING the
 * answer just produced; `threadMessages` is the thread as it was when the run
 * started (used to place the divider). Resolves when the user may type again.
 * Never throws: on failure the user chooses Retry or Continue anyway, and
 * "continue" means plain chats from here on.
 */
export async function compactConversation(args: {
  history: readonly HistoryMessage[];
  threadMessages: readonly { role: string; content: unknown }[];
  apiKey: string;
  sessionId: string;
  signal: AbortSignal;
}): Promise<void> {
  const myEpoch = epoch;
  const own = new AbortController();
  const forward = () => own.abort();
  args.signal.addEventListener("abort", forward);
  activeAbort = own;
  const live = () => myEpoch === epoch && !own.signal.aborted;

  setView({ phase: "running", startedAt: Date.now(), foldMessages: 0, chunks: 0, chunk: 0, tokens: 0, error: "" });
  try {
    while (live()) {
      try {
        const previous = loadStored();
        const result = await requestCompaction(args.history, previous, {
          apiKey: args.apiKey,
          sessionId: args.sessionId,
          signal: own.signal,
        });
        if (!live()) return; // New chat or Stop pressed meanwhile: drop the result
        if (result && result.covers > 0 && result.covers < args.history.length) {
          saveStored({
            summary: result.summary,
            covers: result.covers,
            anchor: anchorOf(args.history, result.covers),
            dividerIndex: threadIndexOf(args.threadMessages, result.covers),
          });
          setView({ dividerIndex: threadIndexOf(args.threadMessages, result.covers) });
        }
        return;
      } catch (err) {
        if (!live()) return;
        setView({ phase: "error", error: err instanceof Error ? err.message : String(err) });
        const choice = await waitForChoice(own.signal);
        if (choice === "continue" || !live()) return;
        setView({ phase: "running", startedAt: Date.now(), foldMessages: 0, chunks: 0, chunk: 0, tokens: 0, error: "" });
      }
    }
  } finally {
    args.signal.removeEventListener("abort", forward);
    if (activeAbort === own) activeAbort = null;
    if (myEpoch === epoch) setView({ phase: "idle", error: "" });
  }
}
