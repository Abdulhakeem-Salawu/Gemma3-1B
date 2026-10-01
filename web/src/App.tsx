import { AssistantRuntimeProvider, useLocalRuntime } from "@assistant-ui/react";
import type { ThreadMessageLike } from "@assistant-ui/react";
import { MoonIcon, SunIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { Composer } from "@/components/composer";
import { clearStoredDocs } from "@/components/documents-bar";
import { Thread } from "@/components/thread";
import { gemmaChatAdapter } from "@/lib/gemma-adapter";
import { checkServerRestarted, resetSessionId } from "@/lib/session";

const HISTORY_STORAGE = "agent_history_v2";
const THEME_STORAGE = "agent_theme";

function loadHistory(): ThreadMessageLike[] {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_STORAGE) ?? "[]");
  } catch {
    return [];
  }
}

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark" | null>(
    () => (localStorage.getItem(THEME_STORAGE) as "light" | "dark" | null) ?? null,
  );
  useEffect(() => {
    if (theme) {
      document.documentElement.setAttribute("data-theme", theme);
      localStorage.setItem(THEME_STORAGE, theme);
    } else {
      document.documentElement.removeAttribute("data-theme");
      localStorage.removeItem(THEME_STORAGE);
    }
  }, [theme]);
  return [theme, setTheme] as const;
}

function Header({
  hasHistory,
  onNewChat,
  theme,
  onToggleTheme,
}: {
  hasHistory: boolean;
  onNewChat: () => void;
  theme: "light" | "dark" | null;
  onToggleTheme: () => void;
}) {
  const prefersDark =
    typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  const isDark = theme === "dark" || (theme === null && prefersDark);
  return (
    <header className="flex items-center gap-2 border-b border-[var(--line)] px-4 py-2.5">
      <span className="text-sm font-medium text-[var(--fg)]">gemma-agent</span>
      <div className="flex-1" />
      <button
        type="button"
        onClick={onToggleTheme}
        aria-label="Toggle theme"
        className="flex size-8 items-center justify-center rounded-full text-[var(--fg-muted)] hover:bg-[var(--bg-soft)] hover:text-[var(--fg)]"
      >
        {isDark ? <SunIcon className="size-4" /> : <MoonIcon className="size-4" />}
      </button>
      <button
        type="button"
        onClick={onNewChat}
        disabled={!hasHistory}
        className="rounded-md border border-[var(--line)] px-2.5 py-1 text-xs text-[var(--fg-muted)] hover:text-[var(--fg)] disabled:opacity-40"
      >
        New chat
      </button>
    </header>
  );
}

/**
 * Owns the actual runtime + its persistence subscription. Remounted (via the
 * `key` its parent assigns) whenever "New chat" starts a fresh session —
 * useLocalRuntime only reads initialMessages on mount, so a genuinely new
 * mount is what clearing history requires, not just re-rendering in place.
 */
function ChatSession({
  initialMessages,
  theme,
  onToggleTheme,
  onNewChat,
}: {
  initialMessages: ThreadMessageLike[];
  theme: "light" | "dark" | null;
  onToggleTheme: () => void;
  onNewChat: () => void;
}) {
  const runtime = useLocalRuntime(gemmaChatAdapter, { initialMessages });

  useEffect(() => {
    return runtime.thread.subscribe(() => {
      const messages = runtime.thread.getState().messages;
      localStorage.setItem(HISTORY_STORAGE, JSON.stringify(messages));
    });
  }, [runtime]);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-full flex-col">
        <Header
          hasHistory={initialMessages.length > 0}
          onNewChat={onNewChat}
          theme={theme}
          onToggleTheme={onToggleTheme}
        />
        <Thread />
        <Composer />
      </div>
    </AssistantRuntimeProvider>
  );
}

export default function App() {
  const [theme, setTheme] = useTheme();
  const [generation, setGeneration] = useState(0);
  const [initialMessages, setInitialMessages] = useState<ThreadMessageLike[]>(loadHistory);

  useEffect(() => {
    void checkServerRestarted().then((restarted) => {
      if (restarted) clearStoredDocs();
    });
  }, []);

  function handleNewChat() {
    if (
      initialMessages.length > 0 &&
      !window.confirm("Start a new chat? This clears the conversation and any attached documents.")
    ) {
      return;
    }
    localStorage.removeItem(HISTORY_STORAGE);
    clearStoredDocs();
    resetSessionId();
    setInitialMessages([]);
    setGeneration((g) => g + 1);
  }

  return (
    <ChatSession
      key={generation}
      initialMessages={initialMessages}
      theme={theme}
      onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
      onNewChat={handleNewChat}
    />
  );
}
