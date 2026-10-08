import { useSyncExternalStore } from "react";
import * as store from "./doc-store";
import type { OutlineItem } from "./doc-store";
import { getApiKey } from "./session";
import { getDocs, setDocs, type DocEntry } from "./sessions";

/**
 * Client side of documents (server: documents.py, /kb/extract and the
 * `documents` field of /chat).
 *
 * The BROWSER owns a chat's documents, the same pattern as the compaction
 * summary. Attaching a file or a long paste asks the server once to extract,
 * clean and chunk it; the result is kept in IndexedDB (doc-store.ts) with a
 * small chip entry in the session store. Every chat request then carries the
 * chat's documents and the server picks the passages for each question, so it
 * does not matter which server instance answers, or whether it restarted.
 */

/** A paste longer than this becomes an attachment instead of filling the box. */
export const PASTE_ATTACH_CHARS = 3000;
/** About 80 pages of text per chat; the server enforces a little more than this. */
export const MAX_CHARS_PER_CHAT = 400_000;

export type ChatDocument = {
  id: string;
  name: string;
  chunks: string[];
  overlaps: number[];
  outline: OutlineItem[];
};

type Extracted = {
  doc_id: string;
  name: string;
  chars: number;
  chunks: string[];
  overlaps: number[];
  outline: OutlineItem[];
  warnings: string[];
  pages?: number;
};

export type AttachResult =
  | { ok: true; duplicate: boolean; entry: DocEntry; warnings: string[]; pages?: number }
  | { ok: false; error: string };

// ---------------------------------------------------------------------------
// Small pure helpers (also unit tested)
// ---------------------------------------------------------------------------

export function shouldAttachPaste(text: string, threshold: number = PASTE_ATTACH_CHARS): boolean {
  return text.trim().length > threshold;
}

export function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

export function pasteName(text: string): string {
  const n = wordCount(text);
  return `Pasted text, ${n.toLocaleString("en-US")} ${n === 1 ? "word" : "words"}`;
}

export function formatSize(chars: number): string {
  if (chars < 1000) return `${chars} chars`;
  return `${chars < 10_000 ? (chars / 1000).toFixed(1) : Math.round(chars / 1000)}k chars`;
}

/** Whether `adding` more characters still fits the per-chat limit. */
export function fitsLimit(existing: readonly DocEntry[], adding: number): boolean {
  return existing.reduce((n, d) => n + (d.chars ?? 0), 0) + adding <= MAX_CHARS_PER_CHAT;
}

function describe(name: string, r: AttachResult): string {
  if (!r.ok) return `${name}: ${r.error}`;
  if (r.duplicate) return `${name} is already attached.`;
  const pages = r.pages ? `${r.pages} page${r.pages === 1 ? "" : "s"}, ` : "";
  return [`Added ${name} (${pages}${formatSize(r.entry.chars ?? 0)}).`, ...r.warnings].join(" ");
}

// ---------------------------------------------------------------------------
// UI state per chat: the composer lock, the status line, the Undo for a paste
// ---------------------------------------------------------------------------

type Ui = { reading: string | null; status: string; lastPaste: { id: string; text: string } | null };

const EMPTY: Ui = { reading: null, status: "", lastPaste: null };
const uis = new Map<string, Ui>();
const readers = new Map<string, number>();
const listeners = new Set<() => void>();

const getUi = (id: string): Ui => uis.get(id) ?? EMPTY;

function patchUi(id: string, patch: Partial<Ui>) {
  uis.set(id, { ...getUi(id), ...patch });
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useDocUi(sessionId: string): Ui {
  return useSyncExternalStore(subscribe, () => getUi(sessionId));
}

/** Current state outside React (tests and scripts). */
export function getDocUi(sessionId: string): Ui {
  return getUi(sessionId);
}

export function setDocStatus(sessionId: string, status: string) {
  patchUi(sessionId, { status });
}

/** Called when a message is sent: the paste Undo and the last status are stale by then. */
export function noteMessageSent(sessionId: string) {
  if (getUi(sessionId) !== EMPTY) patchUi(sessionId, { status: "", lastPaste: null });
}

async function withReading<T>(sessionId: string, label: string, fn: () => Promise<T>): Promise<T> {
  readers.set(sessionId, (readers.get(sessionId) ?? 0) + 1);
  patchUi(sessionId, { reading: label, status: "" });
  try {
    return await fn();
  } finally {
    const left = (readers.get(sessionId) ?? 1) - 1;
    readers.set(sessionId, left);
    if (left <= 0) patchUi(sessionId, { reading: null });
  }
}

// ---------------------------------------------------------------------------
// Talking to the server
// ---------------------------------------------------------------------------

async function errorText(res: Response): Promise<string> {
  if (res.status === 401) return "Wrong API key.";
  if (res.status === 429 || res.status === 503) return "The server is busy or still waking up. Try again in a few seconds.";
  const raw = await res.text().catch(() => "");
  try {
    const body = JSON.parse(raw);
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // not JSON: fall through to the raw text
  }
  return raw || `Request failed (${res.status})`;
}

async function postExtract(init: RequestInit): Promise<Extracted> {
  let res: Response;
  try {
    res = await fetch("/kb/extract", {
      method: "POST",
      ...init,
      headers: { ...(init.headers as Record<string, string> | undefined), "x-api-key": getApiKey() },
    });
  } catch {
    throw new Error("Couldn't reach the server. Check your connection and try again.");
  }
  if (!res.ok) throw new Error(await errorText(res));
  return (await res.json()) as Extracted;
}

const messageOf = (err: unknown) => (err instanceof Error ? err.message : String(err));

/** Stores an extracted document and adds its chip. The same text is the same document (same id): a no-op. */
async function keep(sessionId: string, ex: Extracted, kind: "file" | "paste"): Promise<AttachResult> {
  const existing = getDocs(sessionId);
  const known = existing.find((d) => d.id === ex.doc_id);
  if (known) return { ok: true, duplicate: true, entry: known, warnings: [] };
  if (!fitsLimit(existing, ex.chars)) {
    return {
      ok: false,
      error: `Adding this would go over the limit of about 80 pages (${MAX_CHARS_PER_CHAT.toLocaleString("en-US")} characters) per chat. Remove a document first.`,
    };
  }
  try {
    await store.putDoc(sessionId, {
      id: ex.doc_id,
      name: ex.name,
      chunks: ex.chunks,
      overlaps: ex.overlaps,
      outline: ex.outline,
      chars: ex.chars,
      kind,
      addedAt: Date.now(),
    });
  } catch {
    return { ok: false, error: "Couldn't store the document in this browser (storage may be full or blocked)." };
  }
  const entry: DocEntry = { id: ex.doc_id, name: ex.name, chunks: ex.chunks.length, chars: ex.chars, kind };
  setDocs(sessionId, [...getDocs(sessionId), entry]);
  const warnings = [...(ex.warnings ?? [])];
  if (!(await store.isPersistent())) {
    warnings.push("This browser isn't saving documents between visits (private mode?), so it forgets them when the tab closes.");
  }
  return { ok: true, duplicate: false, entry, warnings, pages: ex.pages };
}

// ---------------------------------------------------------------------------
// Attaching and removing
// ---------------------------------------------------------------------------

export function attachFile(sessionId: string, file: File): Promise<AttachResult> {
  return withReading(sessionId, `Reading ${file.name}\u2026`, async () => {
    let result: AttachResult;
    try {
      const form = new FormData();
      form.append("file", file);
      const ex = await postExtract({ body: form });
      result = await keep(sessionId, { ...ex, name: ex.name || file.name }, "file");
    } catch (err) {
      result = { ok: false, error: messageOf(err) };
    }
    setDocStatus(sessionId, describe(file.name, result));
    return result;
  });
}

/** A long paste becomes a document. On failure the caller puts the text back in the box. */
export function attachPastedText(sessionId: string, text: string): Promise<AttachResult> {
  const name = pasteName(text);
  return withReading(sessionId, "Reading pasted text\u2026", async () => {
    let result: AttachResult;
    try {
      const ex = await postExtract({
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, name }),
      });
      result = await keep(sessionId, { ...ex, name }, "paste");
    } catch (err) {
      result = { ok: false, error: messageOf(err) };
    }
    setDocStatus(sessionId, describe(name, result));
    if (result.ok && !result.duplicate) patchUi(sessionId, { lastPaste: { id: result.entry.id as string, text } });
    return result;
  });
}

export async function removeDocument(sessionId: string, docId: string): Promise<void> {
  await store.deleteDoc(sessionId, docId).catch(() => undefined);
  setDocs(sessionId, getDocs(sessionId).filter((d) => d.id !== docId));
  if (getUi(sessionId).lastPaste?.id === docId) patchUi(sessionId, { lastPaste: null });
}

export async function clearDocuments(sessionId: string): Promise<void> {
  await store.deleteSessionDocs(sessionId).catch(() => undefined);
  setDocs(sessionId, []);
  patchUi(sessionId, { status: "", lastPaste: null });
}

/** Undo for the last paste: removes its attachment and returns the text to put back in the box. */
export async function takeBackPaste(sessionId: string): Promise<string | null> {
  const paste = getUi(sessionId).lastPaste;
  if (!paste) return null;
  await removeDocument(sessionId, paste.id);
  patchUi(sessionId, { lastPaste: null, status: "" });
  return paste.text;
}

// ---------------------------------------------------------------------------
// What a chat request carries
// ---------------------------------------------------------------------------

/**
 * The chat's documents for a request. Always an array: an empty one tells the
 * server "nothing is attached" (it must not fall back to a stale in-memory
 * index), while an older client that omits the field keeps the old behaviour.
 */
export async function loadDocumentsPayload(sessionId: string): Promise<ChatDocument[]> {
  const chips = getDocs(sessionId);
  if (chips.length === 0) return [];
  let stored: Awaited<ReturnType<typeof store.getSessionDocs>>;
  try {
    stored = await store.getSessionDocs(sessionId);
  } catch {
    return [];
  }
  const byId = new Map(stored.map((d) => [d.id, d]));
  const payload: ChatDocument[] = [];
  for (const chip of chips) {
    const doc = chip.id ? byId.get(chip.id) : undefined;
    if (doc) payload.push({ id: doc.id, name: doc.name, chunks: doc.chunks, overlaps: doc.overlaps, outline: doc.outline });
  }
  return payload;
}

/**
 * On opening a chat: drop chips whose stored text is gone (cleared site data,
 * private browsing) so a chip never promises a document the model can't see.
 * Returns the names that were dropped.
 */
export async function reconcileDocuments(sessionId: string): Promise<string[]> {
  const chips = getDocs(sessionId);
  if (chips.length === 0) return [];
  let stored: Awaited<ReturnType<typeof store.getSessionDocs>>;
  try {
    stored = await store.getSessionDocs(sessionId);
  } catch {
    return [];
  }
  const have = new Set(stored.map((d) => d.id));
  const lost = chips.filter((c) => !c.id || !have.has(c.id));
  if (lost.length === 0) return [];
  setDocs(sessionId, chips.filter((c) => c.id && have.has(c.id)));
  return lost.map((c) => c.name);
}
