import type {
  ChatModelAdapter,
  ReasoningMessagePart,
  TextMessagePart,
  ThreadAssistantMessagePart,
  ThreadMessage,
  ToolCallMessagePart,
} from "@assistant-ui/react";
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
  | { type: "done" }
  | { type: "error"; text: string };

function extractText(message: ThreadMessage): string {
  return message.content
    .filter((p): p is TextMessagePart => p.type === "text")
    .map((p) => p.text)
    .join("\n\n");
}

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
    const history = messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({ role: m.role as "user" | "assistant", content: extractText(m) }))
      .filter((m) => m.content.trim().length > 0);

    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": getApiKey() },
      body: JSON.stringify({ messages: history, session_id: getSessionId() }),
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
        case "done":
          yield { content: [...parts] };
          return;
      }
      yield { content: [...parts] };
    }
  },
};
