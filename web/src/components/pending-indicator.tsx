import { useEffect, useState } from "react";
import { useAuiState } from "@assistant-ui/react";
import { LoaderCircleIcon } from "lucide-react";

/**
const MESSAGES = [
  "Thinking…",
  "Reading your question…",
  "Working through it…",
  "Putting the pieces together…",
  "Still working, this one takes a moment…",
]; */

const MESSAGES = [
  "Reading your message…",
  "Getting the full picture…",
  "Figuring out the best approach…",
  "Working on it for you…",
  "Crafting your answer…",
  "Almost there…",
  "Finalizing everything now…"
];

const INTERVAL_MS = 7000;

/** Spinner + status line. Mounts when the wait begins and unmounts when the
 * first content arrives, so every request restarts from the first message.
 * Advances every INTERVAL_MS and holds on the last message. */
function PendingBody() {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (index >= MESSAGES.length - 1) return;
    const id = setTimeout(() => setIndex((i) => i + 1), INTERVAL_MS);
    return () => clearTimeout(id);
  }, [index]);

  return (
    <div
      role="status"
      aria-live="polite"
      className="flex h-6 items-center gap-2 text-sm text-[var(--fg-muted)]"
    >
      <LoaderCircleIcon
        aria-hidden="true"
        className="size-4 shrink-0 animate-spin text-[var(--accent)] motion-reduce:animate-none"
      />
      {/* key restarts the fade-in each time the text changes */}
      <span key={index} className="animate-fade-in motion-reduce:animate-none">
        {MESSAGES[index]}
      </span>
    </div>
  );
}

/** Rendered inside an assistant message. Visible only while the thread is
 * running, this is the newest message, and nothing has streamed into it yet
 * (no thinking text, tool call, or answer token). Gating on the thread
 * running also stops a stale, half-saved message from a reload spinning. */
export function PendingIndicator() {
  const waiting = useAuiState(
    (s) => s.thread.isRunning && s.message.isLast && s.message.content.length === 0,
  );
  return waiting ? <PendingBody /> : null;
}
