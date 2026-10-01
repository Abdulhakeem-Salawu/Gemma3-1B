import { ComposerPrimitive, ThreadPrimitive } from "@assistant-ui/react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";
import { DocumentsBar } from "./documents-bar";

export function Composer() {
  return (
    <div className="border-t border-[var(--line)] bg-[var(--bg)] pb-[env(safe-area-inset-bottom,0px)]">
      <DocumentsBar />
      <ComposerPrimitive.Root className="mx-auto flex w-full max-w-[720px] items-end gap-2 px-4 py-3">
        <ComposerPrimitive.Input
          rows={1}
          autoFocus
          placeholder="Ask something…"
          className="max-h-40 flex-1 resize-none rounded-2xl border border-[var(--line)] bg-[var(--bg-soft)] px-4 py-2.5 text-[15px] leading-relaxed text-[var(--fg)] placeholder:text-[var(--fg-muted)] focus:border-[var(--accent)] focus:outline-none"
        />
        <ThreadPrimitive.If running={false}>
          <ComposerPrimitive.Send
            className="flex size-9 shrink-0 items-center justify-center rounded-full bg-[var(--accent)] text-[var(--accent-contrast)] transition-opacity hover:bg-[var(--accent-hover)] disabled:opacity-40"
          >
            <ArrowUpIcon className="size-4" />
          </ComposerPrimitive.Send>
        </ThreadPrimitive.If>
        <ThreadPrimitive.If running>
          <ComposerPrimitive.Cancel
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
