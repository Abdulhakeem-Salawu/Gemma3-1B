const API_KEY_STORAGE = "agent_api_key";
const SESSION_ID_STORAGE = "agent_session_id";
const INSTANCE_ID_STORAGE = "agent_instance_id";

function newId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

/** Prompts once, then remembers. Blank is a valid answer (no auth configured). */
export function getApiKey(): string {
  const stored = localStorage.getItem(API_KEY_STORAGE);
  if (stored !== null) return stored;
  const entered = window.prompt("API key for this agent (leave blank if none set):") ?? "";
  localStorage.setItem(API_KEY_STORAGE, entered);
  return entered;
}

export function getSessionId(): string {
  let id = localStorage.getItem(SESSION_ID_STORAGE);
  if (!id) {
    id = newId();
    localStorage.setItem(SESSION_ID_STORAGE, id);
  }
  return id;
}

export function resetSessionId(): string {
  const id = newId();
  localStorage.setItem(SESSION_ID_STORAGE, id);
  return id;
}

/**
 * The server keeps uploaded documents in memory only — a restart (scale to
 * zero after being idle, a crash, a new deploy) loses them even though chat
 * history survives in the browser. /session/init returns an id that changes
 * every time the server process starts; comparing it against what we saw
 * last time tells us whether a restart happened since. Returns true the
 * first time a mismatch is observed (and never again for that mismatch).
 */
export async function checkServerRestarted(): Promise<boolean> {
  try {
    const res = await fetch("/session/init");
    const data: { instance_id: string } = await res.json();
    const lastSeen = localStorage.getItem(INSTANCE_ID_STORAGE);
    localStorage.setItem(INSTANCE_ID_STORAGE, data.instance_id);
    return Boolean(lastSeen && lastSeen !== data.instance_id);
  } catch {
    return false; // non-fatal — treat "can't tell" the same as "no restart"
  }
}
