import { AssistantRuntimeProvider, useLocalRuntime } from "@assistant-ui/react";
import type { ThreadMessage } from "@assistant-ui/react";
import { useEffect, useMemo, useState } from "react";
import { Composer } from "@/components/composer";
import { Thread } from "@/components/thread";
import { reconcileSummary, toHistory, useCompactionView } from "@/lib/compaction";
import { createGemmaChatAdapter } from "@/lib/gemma-adapter";
import { SessionIdContext } from "@/lib/session-context";
import {
  loadMessages,
  noteTurnFinished,
  noteUserMessage,
  saveMessages,
  setAwaitingChoice,
  setRunning,
  textOf,
  touch,
} from "@/lib/sessions";

/** Streaming updates arrive many times a second; write them out at most this often. */
const SAVE_EVERY_MS = 1000;

/**
 * One chat: its own runtime, persisted to its own localStorage key.
 *
 * App keeps a ChatSession mounted while its reply is generating, even if the
 * user has switched to another chat (hidden, not unmounted), so the reply
 * keeps streaming in the background. Unmounting a running session therefore
 * only happens when the chat is deleted, and then the request is cancelled.
 */
export function ChatSession({
  sessionId,
  blockedBy,
}: {
  sessionId: string;
  /** Title of another chat that is mid-reply, if sending here must wait. */
  blockedBy: string | null;
}) {
  const adapter = useMemo(() => createGemmaChatAdapter(sessionId), [sessionId]);
  // useLocalRuntime only reads initialMessages on mount, so load once.
  const [initialMessages] = useState(() => {
    const messages = loadMessages(sessionId);
    // A stored summary only makes sense next to the messages it was made from;
    // if those were truncated or cleared, drop it.
    reconcileSummary(sessionId, toHistory(messages));
    return messages;
  });
  const runtime = useLocalRuntime(adapter, { initialMessages });

  // A failed compaction leaves this chat "running" while it waits for the user
  // to pick Retry or Continue anyway. The model is idle then, so other chats
  // must not be held back by it.
  const compactionPhase = useCompactionView(sessionId).phase;
  useEffect(() => {
    setAwaitingChoice(sessionId, compactionPhase === "error");
    return () => setAwaitingChoice(sessionId, false);
  }, [sessionId, compactionPhase]);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    let latest: readonly ThreadMessage[] | null = null;
    let lastCount = initialMessages.length;
    let wasRunning = false;

    const flush = () => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (latest) {
        saveMessages(sessionId, latest);
        latest = null;
      }
    };

    const onChange = () => {
      const { messages, isRunning } = runtime.thread.getState();

      const firstUser = messages.find((m) => m.role === "user");
      if (firstUser) noteUserMessage(sessionId, textOf(firstUser));
      if (messages.length !== lastCount) {
        lastCount = messages.length;
        touch(sessionId);
      }

      latest = messages;
      if (!timer) timer = setTimeout(flush, SAVE_EVERY_MS);

      if (isRunning !== wasRunning) {
        wasRunning = isRunning;
        setRunning(sessionId, isRunning);
        if (!isRunning) {
          flush();
          noteTurnFinished(sessionId);
        }
      }
    };

    const unsubscribe = runtime.thread.subscribe(onChange);
    window.addEventListener("pagehide", flush);
    return () => {
      unsubscribe();
      window.removeEventListener("pagehide", flush);
      if (runtime.thread.getState().isRunning) runtime.thread.cancelRun();
      flush();
      setRunning(sessionId, false);
    };
  }, [runtime, sessionId, initialMessages]);

  return (
    <SessionIdContext.Provider value={sessionId}>
      <AssistantRuntimeProvider runtime={runtime}>
        <Thread />
        <Composer sessionId={sessionId} blockedBy={blockedBy} />
      </AssistantRuntimeProvider>
    </SessionIdContext.Provider>
  );
}
