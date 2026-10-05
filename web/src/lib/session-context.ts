import { createContext, useContext } from "react";

/** Id of the chat a component is rendered inside (provided by ChatSession),
 * so deeply nested pieces — the transcript divider, the compaction banner —
 * can read that chat's state without prop drilling. */
export const SessionIdContext = createContext<string>("");

export function useSessionId(): string {
  return useContext(SessionIdContext);
}
