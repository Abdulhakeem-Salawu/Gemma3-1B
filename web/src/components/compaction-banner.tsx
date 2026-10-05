import { LoaderCircleIcon, TriangleAlertIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { retryCompaction, skipCompaction, useCompactionView } from "@/lib/compaction";
import { useSessionId } from "@/lib/session-context";

function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function RunningBody({ startedAt }: { startedAt: number }) {
  const view = useCompactionView();
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const step = view.chunks > 1 ? `Part ${view.chunk} of ${view.chunks} · ` : "";
  const stage =
    view.tokens > 0
      ? "Writing the summary…"
      : view.chunks > 0
        ? "Reading the earlier messages…"
        : "Getting ready…";

  return (
    <div role="status" aria-live="polite" className="flex items-start gap-2.5">
      <LoaderCircleIcon
        aria-hidden="true"
        className="mt-0.5 size-4 shrink-0 animate-spin text-[var(--accent)] motion-reduce:animate-none"
      />
      <div className="min-w-0">
        <div className="text-[13px] text-[var(--fg)]">
          Condensing our conversation so I can keep going… this can take a couple of minutes.
        </div>
        <div className="mt-0.5 text-xs text-[var(--fg-muted)]">
          {step}
          {stage} · {formatElapsed(now - startedAt)}
        </div>
      </div>
    </div>
  );
}

/** Shown above the composer while older messages are being condensed (the
 * input is locked meanwhile), or when that failed and the user has to choose
 * between trying again and carrying on without it. */
export function CompactionBanner() {
  const sessionId = useSessionId();
  const view = useCompactionView();
  if (view.phase === "idle") return null;

  return (
    <div className="mx-auto w-full max-w-[720px] px-4 pt-2.5">
      <div className="rounded-xl border border-[var(--line)] bg-[var(--bg-soft)] px-3.5 py-2.5">
        {view.phase === "running" ? (
          <RunningBody startedAt={view.startedAt} />
        ) : (
          <div role="alert" className="flex items-start gap-2.5">
            <TriangleAlertIcon aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-[var(--danger)]" />
            <div className="min-w-0 flex-1">
              <div className="text-[13px] text-[var(--fg)]">
                Couldn’t condense the conversation. {view.error}
              </div>
              <div className="mt-0.5 text-xs text-[var(--fg-muted)]">
                You can keep chatting; very old messages may then be shortened instead.
              </div>
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  onClick={() => retryCompaction(sessionId)}
                  className="rounded-md bg-[var(--accent)] px-2.5 py-1 text-xs text-[var(--accent-contrast)] hover:bg-[var(--accent-hover)]"
                >
                  Retry
                </button>
                <button
                  type="button"
                  onClick={() => skipCompaction(sessionId)}
                  className="rounded-md border border-[var(--line)] px-2.5 py-1 text-xs text-[var(--fg-muted)] hover:text-[var(--fg)]"
                >
                  Continue anyway
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
