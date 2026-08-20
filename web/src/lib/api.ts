import type { Health, QueryResponse, ThreadResponse } from "../types";

const BASE = "/api";

/** The service answers with `{detail: ...}` on every handled failure, so read
 *  that before falling back to the status line — "at capacity", "database
 *  unavailable" and the recursion message are all worth showing verbatim. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body; keep the status line */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function query(
  question: string,
  threadId: string | null,
  signal?: AbortSignal,
): Promise<QueryResponse> {
  return request<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // An empty thread_id is rejected by the API on purpose — send null.
    body: JSON.stringify({ question, thread_id: threadId || null }),
    signal,
  });
}

export function readThread(threadId: string): Promise<ThreadResponse> {
  return request<ThreadResponse>(`/threads/${encodeURIComponent(threadId)}`);
}

export function health(): Promise<Health> {
  return request<Health>("/health");
}
