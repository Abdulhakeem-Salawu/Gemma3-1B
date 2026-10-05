import type {
  ChatModelAdapter,
  ReasoningMessagePart,
  TextMessagePart,
  ThreadAssistantMessagePart,
  ToolCallMessagePart,
} from "@assistant-ui/react";
import { compactConversation, summaryPayload, toHistory, type ContextInfo } from "./compaction";
import { getApiKey, getSessionId } from "./session";

/**
 * gemma-agent's SSE event shapes, exactly as app.py's stream_reply() emits
 * them. thinking/token carry incremental text; tool/tool_result are paired
 * by tool_call_id (a hex uuid minted server-side per call).
 */
type GemmaEvent =
  | { type: "thinking"; text: string }
  | { type: "thinking_done" }
  | { type: "tool"; tool_call_id: string; name: string; args: unknown }
  | { type: "tool_result"; tool_call_id: string; result: string }
  | { type: "token"; text: string }
  | { type: "done"; context?: ContextInfo }
  | { type: "error"; text: string };

async function* parseSseFrames(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<GemmaEvent> {
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
        if (!frame.startsWith("data: ")) continue;
        yield JSON.parse(frame.slice(6)) as GemmaEvent;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export const gemmaChatAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const history = toHistory(messages);

    // After a compaction the server gets the summary plus the full raw list;
    // `summary_covers` says how many leading messages the summary replaces.
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": getApiKey() },
      body: JSON.stringify({ messages: history, session_id: getSessionId(), ...summaryPayload(history) }),
      signal: abortSignal,
    });

    if (!res.ok || !res.body) {
      const detail = await res.text().catch(() => "");
      const message =
        res.status === 401
          ? "Wrong API key."
          : res.status === 429
            ? "The model is busy or still waking up. Try again in a few seconds."
            : detail || `Request failed (${res.status})`;
      throw new Error(message);
    }

    const parts: ThreadAssistantMessagePart[] = [];
    let reasoningIdx: number | null = null;
    let textIdx: number | null = null;
    const toolIndexById = new Map<string, number>();

    for await (const evt of parseSseFrames(res.body)) {
      switch (evt.type) {
        case "thinking": {
          if (reasoningIdx === null) {
            parts.push({ type: "reasoning", text: "" });
            reasoningIdx = parts.length - 1;
          }
          const prev = parts[reasoningIdx] as ReasoningMessagePart;
          parts[reasoningIdx] = { ...prev, text: prev.text + evt.text };
          break;
        }
        case "thinking_done": {
          reasoningIdx = null; // seals it — a later hop's thinking starts a fresh block
          break;
        }
        case "tool": {
          parts.push({
            type: "tool-call",
            toolCallId: evt.tool_call_id,
            toolName: evt.name,
            args: (evt.args ?? {}) as ToolCallMessagePart["args"],
            argsText: JSON.stringify(evt.args ?? {}),
          });
          toolIndexById.set(evt.tool_call_id, parts.length - 1);
          break;
        }
        case "tool_result": {
          const idx = toolIndexById.get(evt.tool_call_id);
          if (idx !== undefined) {
            const prev = parts[idx] as ToolCallMessagePart;
            parts[idx] = { ...prev, result: evt.result };
          }
          break;
        }
        case "token": {
          if (textIdx === null) {
            parts.push({ type: "text", text: "" });
            textIdx = parts.length - 1;
          }
          const prev = parts[textIdx] as TextMessagePart;
          parts[textIdx] = { ...prev, text: prev.text + evt.text };
          break;
        }
        case "error":
          throw new Error(evt.text);
        case "done": {
          yield { content: [...parts] };
          // The context is nearly full: condense the older messages NOW, while
          // the thread still counts as running, so the composer stays locked
          // until the summary is stored. Never throws; on failure the user
          // picks Retry or Continue anyway.
          if (evt.context?.needs_compaction) {
            const answer = toHistory([{ role: "assistant", content: parts }]);
            await compactConversation({
              history: [...history, ...answer],
              threadMessages: messages,
              apiKey: getApiKey(),
              sessionId: getSessionId(),
              signal: abortSignal,
            });
          }
          return;
        }
      }
      yield { content: [...parts] };
    }
  },
};
