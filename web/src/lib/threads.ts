import type { ThreadSummary } from "../types";

const KEY = "warren.threads";

/** History lives in localStorage: the service has no thread index, only
 *  `GET /threads/{id}`, so the client is the only thing that knows which ids
 *  belong to this user. */
export function loadThreads(): ThreadSummary[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (t): t is ThreadSummary =>
        typeof t?.id === "string" && typeof t?.title === "string",
    );
  } catch {
    return [];
  }
}

export function saveThreads(threads: ThreadSummary[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(threads.slice(0, 50)));
  } catch {
    /* private mode / quota — history is a convenience, not state we own */
  }
}

export function titleFor(question: string): string {
  const clean = question.replace(/\s+/g, " ").trim();
  return clean.length > 48 ? `${clean.slice(0, 48)}…` : clean;
}
