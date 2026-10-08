/**
 * Where a chat's documents live: IndexedDB in this browser (it survives reloads
 * and server restarts and handles hundreds of KB per document, unlike
 * localStorage's ~5 MB). The server is stateless about documents: every chat
 * request carries them (see documents.ts), so any instance can answer.
 *
 * If IndexedDB is unavailable (some private-browsing modes) documents are kept
 * in memory for this page load only; documents.ts notices the loss on the next
 * load and tells the user rather than pretending they are still there.
 */

export type OutlineItem = { t: string; c: number };

export type StoredDocument = {
  id: string;
  name: string;
  chunks: string[];
  overlaps: number[];
  outline: OutlineItem[];
  chars: number;
  kind: "file" | "paste";
  addedAt: number;
};

type Row = { key: string; sessionId: string; doc: StoredDocument };

const DB_NAME = "gemma-agent-docs";
const STORE = "docs";
const memory = new Map<string, Row>();

let dbPromise: Promise<IDBDatabase | null> | null = null;

function openDb(): Promise<IDBDatabase | null> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve) => {
      if (typeof indexedDB === "undefined") return resolve(null);
      try {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => {
          const store = req.result.createObjectStore(STORE, { keyPath: "key" });
          store.createIndex("session", "sessionId");
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
        req.onblocked = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
  }
  return dbPromise;
}

const keyOf = (sessionId: string, docId: string) => `${sessionId}:${docId}`;

function done(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("IndexedDB transaction failed"));
    tx.onabort = () => reject(tx.error ?? new Error("IndexedDB transaction aborted"));
  });
}

function request<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB request failed"));
  });
}

/** True when documents are really persisted (false: memory fallback). */
export async function isPersistent(): Promise<boolean> {
  return (await openDb()) !== null;
}

export async function putDoc(sessionId: string, doc: StoredDocument): Promise<void> {
  const row: Row = { key: keyOf(sessionId, doc.id), sessionId, doc };
  const db = await openDb();
  if (!db) {
    memory.set(row.key, row);
    return;
  }
  const tx = db.transaction(STORE, "readwrite");
  tx.objectStore(STORE).put(row);
  await done(tx);
}

export async function getSessionDocs(sessionId: string): Promise<StoredDocument[]> {
  const db = await openDb();
  if (!db) return [...memory.values()].filter((r) => r.sessionId === sessionId).map((r) => r.doc);
  const rows = await request<Row[]>(db.transaction(STORE).objectStore(STORE).index("session").getAll(sessionId));
  return rows.map((r) => r.doc);
}

export async function deleteDoc(sessionId: string, docId: string): Promise<void> {
  const db = await openDb();
  if (!db) {
    memory.delete(keyOf(sessionId, docId));
    return;
  }
  const tx = db.transaction(STORE, "readwrite");
  tx.objectStore(STORE).delete(keyOf(sessionId, docId));
  await done(tx);
}

export async function deleteSessionDocs(sessionId: string): Promise<void> {
  const db = await openDb();
  if (!db) {
    for (const [key, row] of memory) if (row.sessionId === sessionId) memory.delete(key);
    return;
  }
  const tx = db.transaction(STORE, "readwrite");
  const store = tx.objectStore(STORE);
  const keys = store.index("session").getAllKeys(sessionId);
  // Deleting right in the callback (not after an await) keeps it inside the transaction.
  keys.onsuccess = () => {
    for (const key of keys.result) store.delete(key);
  };
  await done(tx);
}
