import { ComposerPrimitive, ThreadPrimitive, useAui } from "@assistant-ui/react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";
import type { ClipboardEvent } from "react";
import { useCompactionView } from "@/lib/compaction";
import { attachPastedText, shouldAttachPaste, useDocUi } from "@/lib/documents";
import { CompactionBanner } from "./compaction-banner";
import { DocumentsBar } from "./documents-bar";

export function Composer({ sessionId, blockedBy }: { sessionId: string; blockedBy: string | null }) {
  // No new question is accepted while older messages are being condensed.
  const condensing = useCompactionView().phase !== "idle";
  // The model serves one reply at a time: hold sending while another chat is mid-reply.
  const blocked = blockedBy !== null;
  // Nothing can be sent while a file or a long paste is still being read, so a question never races it.
  const reading = useDocUi(sessionId).reading !== null;
  const busy = condensing || blocked || reading;
  const aui = useAui();

  // A long paste becomes an attachment (chunked and searchable) instead of filling the box. Anything
  // typed stays; if it can't be attached the text goes back in the box, never lost.
  function onPaste(e: ClipboardEvent<HTMLTextAreaElement>) {
    const text = e.clipboardData.getData("text/plain");
    if (!shouldAttachPaste(text)) return;
    e.preventDefault();
    void attachPastedText(sessionId, text).then((result) => {
      if (!result.ok) {
        const box = aui.composer();
        box.setText(box.getState().text + text);
      }
    });
  }
  return (
    <div className="border-t border-[var(--line)] bg-[var(--bg)] pb-[env(safe-area-inset-bottom,0px)]">
      <CompactionBanner />
      <DocumentsBar sessionId={sessionId} disabled={busy} />
      <ComposerPrimitive.Root className="mx-auto flex w-full max-w-[720px] items-end gap-2 px-4 py-3">
        <ComposerPrimitive.Input
          rows={1}
          autoFocus
          disabled={busy}
          onPaste={onPaste}
          placeholder={
            reading
              ? "Reading the attachment\u2026"
              : condensing
                ? "Condensing the conversation\u2026"
                : blocked
                  ? `Waiting for \u201c${blockedBy}\u201d to finish replying\u2026`
                  : "Ask something\u2026"
          }
          className="max-h-40 flex-1 resize-none rounded-2xl border border-[var(--line)] bg-[var(--bg-soft)] px-4 py-2.5 text-[15px] leading-relaxed text-[var(--fg)] placeholder:text-[var(--fg-muted)] focus:border-[var(--accent)] focus:outline-none disabled:opacity-60"
        />
        <ThreadPrimitive.If running={false}>
          <ComposerPrimitive.Send
            aria-label="Send message"
            disabled={blocked || reading}
            className="flex size-9 shrink-0 items-center justify-center rounded-full bg-[var(--accent)] text-[var(--accent-contrast)] transition-opacity hover:bg-[var(--accent-hover)] disabled:opacity-40"
          >
            <ArrowUpIcon className="size-4" />
          </ComposerPrimitive.Send>
        </ThreadPrimitive.If>
        <ThreadPrimitive.If running>
          <ComposerPrimitive.Cancel
            aria-label="Stop reply"
            className="flex size-9 shrink-0 items-center justify-center rounded-full bg-[var(--accent)] text-[var(--accent-contrast)] hover:bg-[var(--accent-hover)]"
          >
            <SquareIcon className="size-3.5 fill-current" />
          </ComposerPrimitive.Cancel>
        </ThreadPrimitive.If>
      </ComposerPrimitive.Root>
      <p className="mx-auto max-w-[720px] px-4 pb-2 text-center text-[11px] text-[var(--fg-muted)]">
        gemma-agent can make mistakes. First reply after a quiet spell can take a minute or two.
      </p>
    </div>
  );
}
