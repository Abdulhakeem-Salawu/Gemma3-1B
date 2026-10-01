import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import type {
  ReasoningMessagePartProps,
  ToolCallMessagePartProps,
} from "@assistant-ui/react";
import remarkGfm from "remark-gfm";
import { ChevronDownIcon, WrenchIcon } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

export function AssistantText() {
  return (
    <MarkdownTextPrimitive
      remarkPlugins={[remarkGfm]}
      className={cn(
        "prose prose-sm max-w-none leading-relaxed",
        "prose-headings:font-semibold prose-headings:text-[var(--fg)]",
        "prose-p:my-2 prose-p:text-[var(--fg)]",
        "prose-strong:text-[var(--fg)] prose-strong:font-semibold",
        "prose-a:text-[var(--accent)] prose-a:no-underline hover:prose-a:underline",
        "prose-code:rounded prose-code:bg-[var(--bg-soft)] prose-code:px-1 prose-code:py-0.5",
        "prose-code:before:content-none prose-code:after:content-none",
        "prose-code:font-mono prose-code:text-[0.85em]",
        "prose-pre:rounded-lg prose-pre:bg-[var(--bg-soft)] prose-pre:border prose-pre:border-[var(--line)]",
        "prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5",
        "prose-blockquote:border-l-[var(--accent)] prose-blockquote:text-[var(--fg-muted)]",
      )}
    />
  );
}

/** Collapsible "Thought for a moment" disclosure — quiet by default, the
 * answer is always the visual lead, not the reasoning. */
export function Reasoning({ text, status }: ReasoningMessagePartProps) {
  const [open, setOpen] = useState(false);
  const streaming = status?.type === "running";

  return (
    <div className="mb-2 text-sm">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 text-[var(--fg-muted)] hover:text-[var(--fg)] transition-colors"
      >
        <ChevronDownIcon
          className={cn("size-3.5 transition-transform", open && "rotate-180")}
        />
        <span className={cn(streaming && "animate-pulse")}>
          {streaming ? "Thinking…" : "Thought for a moment"}
        </span>
      </button>
      {open && (
        <div className="mt-1.5 border-l-2 border-[var(--line)] pl-3 italic text-[var(--fg-muted)]">
          {text}
        </div>
      )}
    </div>
  );
}

/** Fallback renderer for any tool call gemma-agent makes — a small quiet
 * status line, not a full interactive card (these are simple read-only
 * lookups, not actions worth a rich UI). */
export function ToolFallback({ toolName, args, result }: ToolCallMessagePartProps) {
  const argsSummary = Object.entries((args as Record<string, unknown>) ?? {})
    .map(([k, v]) => `${k}: ${String(v)}`)
    .join(", ");
  return (
    <div className="mb-2 flex items-start gap-1.5 rounded-md border border-[var(--line)] bg-[var(--bg-soft)] px-2.5 py-1.5 text-xs text-[var(--fg-muted)]">
      <WrenchIcon className="mt-0.5 size-3 shrink-0" />
      <div>
        <span className="font-medium text-[var(--fg)]">{toolName}</span>
        {argsSummary && <span>({argsSummary})</span>}
        {result !== undefined && (
          <div className="mt-1 line-clamp-3 text-[var(--fg-muted)]">
            {String(result)}
          </div>
        )}
      </div>
    </div>
  );
}
