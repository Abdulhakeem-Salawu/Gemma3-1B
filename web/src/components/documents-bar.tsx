import { useAui } from "@assistant-ui/react";
import { PaperclipIcon, XIcon } from "lucide-react";
import { useEffect, useRef } from "react";
import {
  attachFile,
  clearDocuments,
  formatSize,
  reconcileDocuments,
  removeDocument,
  setDocStatus,
  takeBackPaste,
  useDocUi,
} from "@/lib/documents";
import { useSessionDocs } from "@/lib/sessions";

/**
 * Attachments bar above the composer: one chip per document (file or long
 * paste) with its size and a remove button, a status line, and the paperclip.
 * Documents belong to the chat and ground every question in it, so this is not
 * assistant-ui's per-message attachment. The text itself lives in IndexedDB
 * (lib/doc-store); the chips live in the session store.
 */
export function DocumentsBar({ sessionId, disabled }: { sessionId: string; disabled?: boolean }) {
  const aui = useAui();
  const docs = useSessionDocs(sessionId);
  const ui = useDocUi(sessionId);
  const reading = ui.reading !== null;
  const fileInput = useRef<HTMLInputElement>(null);

  // Chips whose stored text is gone (site data cleared, private browsing) are dropped, and said so.
  useEffect(() => {
    void reconcileDocuments(sessionId).then((lost) => {
      if (lost.length > 0) {
        setDocStatus(
          sessionId,
          `${lost.join(", ")} ${lost.length === 1 ? "is" : "are"} no longer stored in this browser and ${
            lost.length === 1 ? "was" : "were"
          } removed. Attach ${lost.length === 1 ? "it" : "them"} again if you need ${lost.length === 1 ? "it" : "them"}.`,
        );
      }
    });
  }, [sessionId]);

  async function undoPaste() {
    const text = await takeBackPaste(sessionId);
    if (text) {
      const box = aui.composer();
      box.setText(box.getState().text + text);
    }
  }

  const showBar = docs.length > 0 || reading || ui.status;

  return (
    <div className="mx-auto w-full max-w-[720px] px-4">
      {showBar && (
        <div className="mb-1.5 flex flex-wrap items-center gap-1.5 pt-2.5 text-xs text-[var(--fg-muted)]">
          {docs.map((d) => (
            <span
              key={d.id ?? d.name}
              className="inline-flex items-center gap-1 rounded-full border border-[var(--line)] bg-[var(--bg-soft)] py-0.5 pl-2.5 pr-1"
            >
              <span className="max-w-[16rem] truncate">{d.name}</span>
              <span className="text-[var(--accent)]">{formatSize(d.chars ?? 0)}</span>
              <button
                type="button"
                aria-label={`Remove ${d.name}`}
                title="Remove"
                disabled={reading}
                onClick={() => d.id && void removeDocument(sessionId, d.id)}
                className="flex size-4 items-center justify-center rounded-full hover:bg-[var(--line)] hover:text-[var(--fg)] disabled:opacity-40"
              >
                <XIcon className="size-3" />
              </button>
            </span>
          ))}
          {reading && <span role="status">{ui.reading}</span>}
          {!reading && ui.status && <span role="status">{ui.status}</span>}
          {!reading && ui.lastPaste && (
            <button
              type="button"
              onClick={() => void undoPaste()}
              className="rounded border border-[var(--line)] px-2 py-0.5 hover:text-[var(--fg)]"
            >
              Undo
            </button>
          )}
          {docs.length > 1 && (
            <button
              type="button"
              disabled={reading}
              onClick={() => void clearDocuments(sessionId)}
              className="ml-auto rounded border border-[var(--line)] px-2 py-0.5 hover:text-[var(--fg)] disabled:opacity-40"
            >
              Clear all
            </button>
          )}
        </div>
      )}
      <div className="flex items-center gap-1 pt-1">
        <input
          ref={fileInput}
          type="file"
          accept=".pdf,.docx,.txt,.md"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) void attachFile(sessionId, file);
          }}
        />
        <button
          type="button"
          title="Attach a PDF, DOCX, TXT or MD file for the model to use as context. Long pasted text is attached automatically."
          aria-label="Attach a file"
          disabled={disabled || reading}
          onClick={() => fileInput.current?.click()}
          className="flex size-8 items-center justify-center rounded-full text-[var(--fg-muted)] hover:bg-[var(--bg-soft)] hover:text-[var(--fg)] disabled:opacity-40"
        >
          <PaperclipIcon className="size-4" />
        </button>
      </div>
    </div>
  );
}
