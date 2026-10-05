import { useEffect, useRef, useState } from "react";
import { getApiKey } from "./session";
import { setAiTitle, useSessionStore } from "./sessions";

const MAX_ATTEMPTS = 3;
const RETRY_DELAY_MS = 8_000;
const REQUEST_TIMEOUT_MS = 60_000;
const MAX_TITLE_CHARS = 60;

function cleanTitle(raw: unknown): string | null {
  if (typeof raw !== "string") return null;
  const title = raw
    .replace(/\s+/g, " ")
    .replace(/^["'“”‘’`\s]+|["'“”‘’`.\s]+$/g, "")
    .slice(0, MAX_TITLE_CHARS)
    .trim();
  return title || null;
}

/** Asks the server to name a chat from its first question. Resolves to null
 * when the model is busy or anything goes wrong — the caller keeps the
 * fallback title in that case. */
export async function generateTitle(question: string): Promise<string | null> {
  try {
    const res = await fetch("/chat/title", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": getApiKey() },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    if (!res.ok) return null;
    const data: { title?: unknown } = await res.json();
    return cleanTitle(data.title);
  } catch {
    return null;
  }
}

/**
 * Names chats in the background, one request at a time.
 *
 * It only runs right after a reply has finished in this page load (so the
 * server instance is known to be awake — a title request must never be what
 * wakes a scaled-to-zero model) and never while any chat is using the model
 * (a reply or a compaction), since the model serves one request at a time.
 * Chats that miss out (model busy) are retried a few times, then keep their
 * fallback title.
 */
export function useAutoTitles() {
  const { sessions, busy, turnTick } = useSessionStore();
  const attempts = useRef(new Map<string, number>());
  const inflight = useRef(new Set<string>());
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    if (turnTick === 0 || busy.length > 0 || inflight.current.size > 0) return;
    const next = [...sessions]
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .find(
        (s) =>
          s.titleSource === "fallback" &&
          s.replied &&
          s.question &&
          (attempts.current.get(s.id) ?? 0) < MAX_ATTEMPTS,
      );
    if (!next) return;

    const used = (attempts.current.get(next.id) ?? 0) + 1;
    attempts.current.set(next.id, used);
    inflight.current.add(next.id);
    void generateTitle(next.question).then((title) => {
      inflight.current.delete(next.id);
      if (title) {
        setAiTitle(next.id, title);
      } else {
        window.setTimeout(() => setRetry((n) => n + 1), used < MAX_ATTEMPTS ? RETRY_DELAY_MS : 0);
      }
    });
  }, [sessions, busy, turnTick, retry]);
}
