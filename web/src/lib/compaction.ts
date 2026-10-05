import { useSyncExternalStore } from "react";
import { useSessionId } from "./session-context";

/**
 * Client side of summarization-based compaction (server: compaction.py and
 * /chat/compact in app.py).
 *
 * The CLIENT owns the summary, so the stateless server survives restarts and
 * a second Cloud Run instance. Everything here is per chat (keyed by session
 * id), because several chats can be open at once: each has its own stored
 * summary, banner state and in-flight compaction. For each chat we keep:
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

/** Where a chat's summary is stored. Chats created before multi-session
 * support kept one global summary under LEGACY_SUMMARY_STORAGE; the session
 * store moves it into the first chat's key on upgrade. */
export const summaryStorageKey = (sessionId: string) => `agent_session_${sessionId}_summary`;
export const LEGACY_SUMMARY_STORAGE = "agent_summary_v1";

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

function loadStored(sessionId: string): StoredSummary | null {
  try {
    const raw = JSON.parse(localStorage.getItem(summaryStorageKey(sessionId)) ?? "null");
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

function saveStored(sessionId: string, value: StoredSummary) {
  try {
    localStorage.setItem(summaryStorageKey(sessionId), JSON.stringify(value));
  } catch {
    // storage full or blocked: the summary still works for this page load
  }
}

// ---------------------------------------------------------------------------
// Observable state (banner, composer, transcript divider)
// ---------------------------------------------------------------------------

function idleView(sessionId: string): CompactionView {
  return {
    phase: "idle",
    startedAt: 0,
    foldMessages: 0,
    chunks: 0,
    chunk: 0,
    tokens: 0,
    error: "",
    dividerIndex: loadStored(sessionId)?.dividerIndex ?? null,
  };
}

/** Per-chat runtime state. `epoch` makes a compaction that is still running
 * discard its result after a reset; `activeAbort` / `pendingChoice` are the
 * handles reset needs to stop it. */
type Controller = {
  view: CompactionView;
  epoch: number;
  activeAbort: AbortController | null;
  pendingChoice: ((choice: "retry" | "continue") => void) | null;
};

let epochCounter = 0; // global, so a deleted-and-recreated controller never reuses an epoch
const controllers = new Map<string, Controller>();
const listeners = new Set<() => void>();

function controller(sessionId: string): Controller {
  let c = controllers.get(sessionId);
  if (!c) {
    c = { view: idleView(sessionId), epoch: ++epochCounter, activeAbort: null, pendingChoice: null };
    controllers.set(sessionId, c);
  }
  return c;
}

function setView(sessionId: string, patch: Partial<CompactionView>) {
  const c = controller(sessionId);
  c.view = { ...c.view, ...patch };
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** A chat's compaction state. Without an argument it is the chat the calling
 * component is rendered in (see SessionIdContext). */
export function useCompactionView(sessionId?: string): CompactionView {
  const fromContext = useSessionId();
  const id = sessionId ?? fromContext;
  return useSyncExternalStore(subscribe, () => controller(id).view);
}

/** Current state outside React (tests and scripts). */
export function getCompactionView(sessionId: string): CompactionView {
  return controller(sessionId).view;
}

// ---------------------------------------------------------------------------
// Using and invalidating the stored summary
// ---------------------------------------------------------------------------

/** Forget a chat's summary and stop any compaction in flight for it. The
 * epoch makes a compaction that is still running discard its result. */
export function resetCompaction(sessionId: string) {
  const c = controller(sessionId);
  c.epoch = ++epochCounter;
  c.activeAbort?.abort();
  c.activeAbort = null;
  c.pendingChoice?.("continue");
  c.pendingChoice = null;
  try {
    localStorage.removeItem(summaryStorageKey(sessionId));
  } catch {
    // ignore
  }
  setView(sessionId, { ...idleView(sessionId), dividerIndex: null });
}

/** A chat was deleted: reset it and forget its controller. */
export function dropCompaction(sessionId: string) {
  resetCompaction(sessionId);
  controllers.delete(sessionId);
}

/** Drops a stored summary that no longer matches the messages (history was
 * truncated, edited or cleared). `maxCovers` is how many leading messages a
 * valid summary may cover. */
function validStored(sessionId: string, history: readonly HistoryMessage[], maxCovers: number): StoredSummary | null {
  const stored = loadStored(sessionId);
  if (!stored) return null;
  if (stored.covers > maxCovers || stored.anchor !== anchorOf(history, stored.covers)) {
    resetCompaction(sessionId);
    return null;
  }
  return stored;
}

/** Called once when a chat's persisted history is loaded. */
export function reconcileSummary(sessionId: string, history: readonly HistoryMessage[]) {
  validStored(sessionId, history, history.length);
}

/** The fields to add to a chat request. `history` ends with the new question,
 * which must stay uncovered. */
export function summaryPayload(
  sessionId: string,
  history: readonly HistoryMessage[],
): { summary?: string; summary_covers?: number } {
  const stored = validStored(sessionId, history, history.length - 1);
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
        setView(ctx.sessionId, {
          foldMessages: evt.fold_messages,
          chunks: evt.chunks,
          chunk: evt.chunks ? 1 : 0,
          tokens: 0,
        });
        break;
      case "progress":
        setView(ctx.sessionId, { tokens: evt.tokens, ...(evt.chunk ? { chunk: evt.chunk } : {}) });
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

function waitForChoice(sessionId: string, signal: AbortSignal): Promise<"retry" | "continue"> {
  const c = controller(sessionId);
  return new Promise((resolve) => {
    const finish = (choice: "retry" | "continue") => {
      c.pendingChoice = null;
      signal.removeEventListener("abort", onAbort);
      resolve(choice);
    };
    const onAbort = () => finish("continue");
    c.pendingChoice = finish;
    signal.addEventListener("abort", onAbort);
  });
}

export function retryCompaction(sessionId: string) {
  controller(sessionId).pendingChoice?.("retry");
}

/** "Continue anyway": keep chatting without a (new) summary. */
export function skipCompaction(sessionId: string) {
  controller(sessionId).pendingChoice?.("continue");
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
  const id = args.sessionId;
  const c = controller(id);
  const myEpoch = c.epoch;
  const own = new AbortController();
  const forward = () => own.abort();
  args.signal.addEventListener("abort", forward);
  c.activeAbort = own;
  // `controllers.get(id) === c` also ends the run if the chat was deleted meanwhile.
  const live = () => controllers.get(id) === c && myEpoch === c.epoch && !own.signal.aborted;

  setView(id, { phase: "running", startedAt: Date.now(), foldMessages: 0, chunks: 0, chunk: 0, tokens: 0, error: "" });
  try {
    while (live()) {
      try {
        const previous = loadStored(id);
        const result = await requestCompaction(args.history, previous, {
          apiKey: args.apiKey,
          sessionId: args.sessionId,
          signal: own.signal,
        });
        if (!live()) return; // New chat or Stop pressed meanwhile: drop the result
        if (result && result.covers > 0 && result.covers < args.history.length) {
          saveStored(id, {
            summary: result.summary,
            covers: result.covers,
            anchor: anchorOf(args.history, result.covers),
            dividerIndex: threadIndexOf(args.threadMessages, result.covers),
          });
          setView(id, { dividerIndex: threadIndexOf(args.threadMessages, result.covers) });
        }
        return;
      } catch (err) {
        if (!live()) return;
        setView(id, { phase: "error", error: err instanceof Error ? err.message : String(err) });
        const choice = await waitForChoice(id, own.signal);
        if (choice === "continue" || !live()) return;
        setView(id, { phase: "running", startedAt: Date.now(), foldMessages: 0, chunks: 0, chunk: 0, tokens: 0, error: "" });
      }
    }
  } finally {
    args.signal.removeEventListener("abort", forward);
    if (c.activeAbort === own) c.activeAbort = null;
    if (controllers.get(id) === c && myEpoch === c.epoch) setView(id, { phase: "idle", error: "" });
  }
}
