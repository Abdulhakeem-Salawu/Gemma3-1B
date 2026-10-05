import {
  ActionBarPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAuiState,
} from "@assistant-ui/react";
import { useMessageError } from "@assistant-ui/core/react";
import { ArrowDownIcon, CheckIcon, CopyIcon, Sparkles } from "lucide-react";
import { AssistantText, Reasoning, ToolFallback } from "./message-parts";
import { PendingIndicator } from "./pending-indicator";
import { useCompactionView } from "@/lib/compaction";
import { cn } from "@/lib/utils";

/** A quiet marker where older messages were condensed into a summary. The
 * transcript above it stays fully readable; only what the model receives
 * changed. Rendered before the first message the summary does not cover. */
function CompactionDivider() {
  const { dividerIndex } = useCompactionView();
  const index = useAuiState((s) => s.message.index);
  if (dividerIndex === null || index !== dividerIndex) return null;
  return (
    <div
      role="separator"
      className="mx-auto flex w-full max-w-[720px] items-center gap-3 px-4 py-2 text-[11px] text-[var(--fg-muted)]"
    >
      <span className="h-px flex-1 bg-[var(--line)]" />
      <span>Earlier messages condensed</span>
      <span className="h-px flex-1 bg-[var(--line)]" />
    </div>
  );
}

function UserMessage() {
  return (
    <>
      <CompactionDivider />
      <MessagePrimitive.Root className="mx-auto flex w-full max-w-[720px] justify-end px-4 py-1.5">
        <div className="max-w-[85%] rounded-2xl bg-[var(--bubble-user)] px-4 py-2.5 text-[15px] leading-relaxed">
          <MessagePrimitive.Parts />
        </div>
      </MessagePrimitive.Root>
    </>
  );
}

function MessageErrorText() {
  const text = useMessageError();
  return <>{text}</>;
}

function AssistantMessage() {
  return (
    <>
      <CompactionDivider />
      <MessagePrimitive.Root className="group mx-auto w-full max-w-[720px] px-4 py-3">
        <div className="flex gap-3">
          <div className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full bg-[var(--accent)] text-[var(--accent-contrast)]">
            <Sparkles className="size-3.5" />
          </div>
          <div className="min-w-0 flex-1 text-[15px]">
            <PendingIndicator />
            <MessagePrimitive.Parts
              components={{
                Text: AssistantText,
                Reasoning: Reasoning,
                tools: { Fallback: ToolFallback },
              }}
            />
            <MessagePrimitive.Error>
              <div className="mt-1 rounded-md border border-[var(--danger)]/30 bg-[var(--danger)]/10 px-3 py-2 text-sm text-[var(--danger)]">
                <MessageErrorText />
              </div>
            </MessagePrimitive.Error>
            <ActionBarPrimitive.Root
              hideWhenRunning
              className="mt-1.5 flex gap-2 opacity-0 transition-opacity group-hover:opacity-100"
            >
              <ActionBarPrimitive.Copy className="group/copy flex items-center gap-1 rounded text-xs text-[var(--fg-muted)] hover:text-[var(--fg)]">
                <CopyIcon className="size-3.5 group-data-[copied=true]/copy:hidden" />
                <CheckIcon className="hidden size-3.5 group-data-[copied=true]/copy:block" />
              </ActionBarPrimitive.Copy>
            </ActionBarPrimitive.Root>
          </div>
        </div>
      </MessagePrimitive.Root>
    </>
  );
}

function EmptyState() {
  return (
    <div className="mx-auto flex h-full max-w-[520px] flex-col items-center justify-center px-4 text-center">
      <div className="mb-4 flex size-12 items-center justify-center rounded-full bg-[var(--accent)] text-[var(--accent-contrast)]">
        <Sparkles className="size-6" />
      </div>
      <h1 className="text-xl font-semibold text-[var(--fg)]">gemma-agent</h1>
      <p className="mt-1.5 text-sm text-[var(--fg-muted)]">
        Ask a question, look something up, or attach a document to ground the
        conversation in it.
      </p>
    </div>
  );
}

export function Thread() {
  return (
    <ThreadPrimitive.Root className="relative flex h-full flex-1 flex-col overflow-hidden">
      <ThreadPrimitive.Viewport
        autoScroll
        className={cn(
          "flex-1 overflow-y-auto scroll-smooth pt-4 pb-2",
          "[scrollbar-gutter:stable]",
        )}
      >
        <ThreadPrimitive.Empty>
          <EmptyState />
        </ThreadPrimitive.Empty>
        <ThreadPrimitive.Messages
          components={{ UserMessage, AssistantMessage }}
        />
      </ThreadPrimitive.Viewport>

      <ThreadPrimitive.ScrollToBottom
        className={cn(
          "absolute bottom-2 left-1/2 -translate-x-1/2 rounded-full border border-[var(--line)]",
          "bg-[var(--bg)] p-2 shadow-sm hover:bg-[var(--bg-soft)]",
          "disabled:pointer-events-none disabled:opacity-0",
        )}
      >
        <ArrowDownIcon className="size-4 text-[var(--fg-muted)]" />
      </ThreadPrimitive.ScrollToBottom>
    </ThreadPrimitive.Root>
  );
}
