import { MenuIcon, MoonIcon, PlusIcon, SunIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ChatSession } from "@/components/chat-session";
import { Sidebar } from "@/components/sidebar";
import { newChat, useSessionStore } from "@/lib/sessions";
import { useAutoTitles } from "@/lib/titles";
import { cn } from "@/lib/utils";

const THEME_STORAGE = "agent_theme";

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
  title,
  onOpenSidebar,
  onNewChat,
  theme,
  onToggleTheme,
}: {
  title: string;
  onOpenSidebar: () => void;
  onNewChat: () => void;
  theme: "light" | "dark" | null;
  onToggleTheme: () => void;
}) {
  const prefersDark =
    typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  const isDark = theme === "dark" || (theme === null && prefersDark);
  const iconButton =
    "flex size-8 shrink-0 items-center justify-center rounded-full text-[var(--fg-muted)] hover:bg-[var(--bg-soft)] hover:text-[var(--fg)]";
  return (
    <header className="flex items-center gap-2 border-b border-[var(--line)] px-4 py-2.5">
      <button type="button" onClick={onOpenSidebar} aria-label="Open chat list" className={cn(iconButton, "md:hidden")}>
        <MenuIcon className="size-4" />
      </button>
      <span className="min-w-0 flex-1 truncate text-sm font-medium text-[var(--fg)]">{title}</span>
      <button type="button" onClick={onToggleTheme} aria-label="Toggle theme" className={iconButton}>
        {isDark ? <SunIcon className="size-4" /> : <MoonIcon className="size-4" />}
      </button>
      <button type="button" onClick={onNewChat} aria-label="New chat" className={cn(iconButton, "md:hidden")}>
        <PlusIcon className="size-4" />
      </button>
    </header>
  );
}

export default function App() {
  const [theme, setTheme] = useTheme();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { sessions, activeId, running, busy } = useSessionStore();
  useAutoTitles();

  const closeDrawer = useCallback(() => setDrawerOpen(false), []);

  // The open chat, plus any chat still generating a reply: those stay mounted
  // (hidden) so their stream keeps running while you look at another chat.
  const mountedIds = useMemo(
    () => [activeId, ...running.filter((id) => id !== activeId)],
    [activeId, running],
  );

  const titleOf = (id: string) => sessions.find((s) => s.id === id)?.title;
  // The model serves one request at a time, so sending is held while another chat is using it.
  const otherRunning = busy.find((id) => id !== activeId);
  const blockedBy = otherRunning ? (titleOf(otherRunning) ?? "another chat") : null;

  return (
    <div className="flex h-full">
      <Sidebar open={drawerOpen} onClose={closeDrawer} />
      <div className="flex min-w-0 flex-1 flex-col">
        <Header
          title={titleOf(activeId) ?? "gemma-agent"}
          onOpenSidebar={() => setDrawerOpen(true)}
          onNewChat={newChat}
          theme={theme}
          onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
        />
        <div className="relative min-h-0 flex-1">
          {mountedIds.map((id) => (
            <div key={id} className={cn("absolute inset-0 flex-col", id === activeId ? "flex" : "hidden")}>
              <ChatSession sessionId={id} blockedBy={id === activeId ? blockedBy : null} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
