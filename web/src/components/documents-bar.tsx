import { PaperclipIcon } from "lucide-react";
import { useRef, useState } from "react";
import { getApiKey } from "@/lib/session";
import { getDocs, setDocs, useSessionDocs } from "@/lib/sessions";

/**
 * Upload/clear bar for session-scoped RAG documents, rendered directly above
 * the composer's input row. Kept as a standalone component rather than
 * assistant-ui's per-message AttachmentAdapter: uploads here persist for the
 * whole chat and ground every subsequent question, not just one message.
 * The chips live in the session store, one list per chat.
 */
export function DocumentsBar({ sessionId, disabled }: { sessionId: string; disabled?: boolean }) {
  const docs = useSessionDocs(sessionId);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  async function upload(file: File) {
    setBusy(true);
    setStatus(`Uploading ${file.name}…`);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`/kb/upload?session_id=${encodeURIComponent(sessionId)}`, {
        method: "POST",
        headers: { "x-api-key": getApiKey() },
        body: form,
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.detail || `Upload failed (${res.status})`);
      setDocs(sessionId, [...getDocs(sessionId), { name: body.filename, chunks: body.total_chunks }]);
      setStatus(`${file.name}: added ${body.chunks_added} chunk(s)`);
    } catch (err) {
      setStatus(`${file.name}: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  async function clearAll() {
    setStatus("Clearing…");
    try {
      await fetch("/kb/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-api-key": getApiKey() },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch {
      // best effort — clear the client list regardless
    }
    setDocs(sessionId, []);
    setStatus("");
  }

  return (
    <div className="mx-auto w-full max-w-[720px] px-4">
      {(docs.length > 0 || status) && (
        <div className="mb-1.5 flex flex-wrap items-center gap-1.5 pt-2.5 text-xs text-[var(--fg-muted)]">
          {docs.map((d, i) => (
            <span
              key={`${d.name}-${i}`}
              className="rounded-full border border-[var(--line)] bg-[var(--bg-soft)] px-2.5 py-0.5"
            >
              {d.name} <span className="text-[var(--accent)]">{d.chunks}</span>
            </span>
          ))}
          {status && <span>{status}</span>}
          {docs.length > 0 && (
            <button
              type="button"
              onClick={clearAll}
              className="ml-auto rounded border border-[var(--line)] px-2 py-0.5 hover:text-[var(--fg)]"
            >
              Clear
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
            if (file) void upload(file);
          }}
        />
        <button
          type="button"
          title="Attach a PDF, DOCX, TXT or MD file for the model to use as context"
          disabled={disabled || busy}
          onClick={() => fileInput.current?.click()}
          className="flex size-8 items-center justify-center rounded-full text-[var(--fg-muted)] hover:bg-[var(--bg-soft)] hover:text-[var(--fg)] disabled:opacity-40"
        >
          <PaperclipIcon className="size-4" />
        </button>
      </div>
    </div>
  );
}
