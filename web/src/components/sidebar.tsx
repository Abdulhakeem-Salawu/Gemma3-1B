import { LoaderCircleIcon, PlusIcon, Trash2Icon, TriangleAlertIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import { deleteSession, newChat, selectSession, useSessionStore } from "@/lib/sessions";

/**
 * Chat list. A fixed slide-over drawer below the md breakpoint (opened from
 * the header), a static column above it.
 */
export function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { sessions, activeId, running, awaiting, busy, storageFull } = useSessionStore();
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const sorted = useMemo(() => [...sessions].sort((a, b) => b.updatedAt - a.updatedAt), [sessions]);
  const onBlankDraft = !sessions.some((s) => s.id === activeId);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <>
      {open && (
        <button
          type="button"
          aria-label="Close chat list"
          onClick={onClose}
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
        />
      )}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-[var(--line)] bg-[var(--bg)]",
          "pt-[env(safe-area-inset-top,0px)] pb-[env(safe-area-inset-bottom,0px)] transition-transform",
          "md:static md:z-auto md:w-64 md:shrink-0 md:translate-x-0 md:pt-0 md:pb-0",
          open ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="p-2">
          <button
            type="button"
            disabled={onBlankDraft}
            onClick={() => {
              newChat();
              onClose();
            }}
            className="flex w-full items-center gap-2 rounded-md border border-[var(--line)] px-3 py-2 text-sm text-[var(--fg)] hover:bg-[var(--bg-soft)] disabled:opacity-40"
          >
            <PlusIcon className="size-4" />
            New chat
          </button>
        </div>

        <nav aria-label="Chats" className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
          {sorted.length === 0 && (
            <p className="px-2 py-3 text-xs text-[var(--fg-muted)]">
              No saved chats yet. A chat appears here once you send your first message.
            </p>
          )}
          <ul className="flex flex-col gap-0.5">
            {sorted.map((s) => (
              <li key={s.id}>
                {confirmId === s.id ? (
                  <div className="flex items-center gap-2 rounded-md bg-[var(--bg-soft)] px-2 py-1.5 text-sm">
                    <span className="min-w-0 flex-1 truncate text-[var(--fg-muted)]">
                      {running.includes(s.id) ? "Delete and stop reply?" : "Delete this chat?"}
                    </span>
                    <button
                      type="button"
                      autoFocus
                      onClick={() => {
                        setConfirmId(null);
                        deleteSession(s.id);
                      }}
                      className="rounded px-1.5 py-0.5 text-xs font-medium text-[var(--danger)] hover:bg-[var(--danger)]/10"
                    >
                      Delete
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirmId(null)}
                      className="rounded px-1.5 py-0.5 text-xs text-[var(--fg-muted)] hover:text-[var(--fg)]"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <div
                    className={cn(
                      "group flex items-center rounded-md",
                      s.id === activeId ? "bg-[var(--bg-soft)]" : "hover:bg-[var(--bg-soft)]",
                    )}
                  >
                    <button
                      type="button"
                      title={s.title}
                      aria-current={s.id === activeId ? "true" : undefined}
                      onClick={() => {
                        selectSession(s.id);
                        onClose();
                      }}
                      className="flex min-w-0 flex-1 items-center gap-2 px-2 py-2 text-left text-sm text-[var(--fg)]"
                    >
                      {busy.includes(s.id) && (
                        <LoaderCircleIcon
                          aria-label="Replying"
                          className="size-3.5 shrink-0 animate-spin text-[var(--accent)] motion-reduce:animate-none"
                        />
                      )}
                      {awaiting.includes(s.id) && (
                        <TriangleAlertIcon
                          aria-label="Needs your input"
                          className="size-3.5 shrink-0 text-[var(--danger)]"
                        />
                      )}
                      <span className="truncate">{s.title}</span>
                    </button>
                    <button
                      type="button"
                      aria-label={`Delete chat: ${s.title}`}
                      onClick={() => setConfirmId(s.id)}
                      className="mr-1 flex size-7 shrink-0 items-center justify-center rounded text-[var(--fg-muted)] hover:text-[var(--danger)] focus-visible:opacity-100 md:opacity-0 md:group-hover:opacity-100"
                    >
                      <Trash2Icon className="size-3.5" />
                    </button>
                  </div>
                )}
              </li>
            ))}
          </ul>
        </nav>

        {storageFull && (
          <p className="border-t border-[var(--line)] px-3 py-2 text-xs text-[var(--danger)]">
            Browser storage is full, so new messages aren&apos;t being saved. Delete old chats to free space.
          </p>
        )}
      </aside>
    </>
  );
}
