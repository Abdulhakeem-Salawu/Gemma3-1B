const API_KEY_STORAGE = "agent_api_key";

export function newId(): string {
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
