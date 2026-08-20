import type {
  Health,
  QueryResponse,
  StreamEvent,
  ThreadResponse,
} from "../types";

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

/** Read the error body the same way `request` does, so a failure before the
 *  stream opens reads identically whichever call the caller made. */
async function detailOf(response: Response): Promise<string> {
  let detail = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    /* non-JSON error body; keep the status line */
  }
  return detail;
}

/** One turn, reported as it happens.
 *
 *  `fetch` rather than `EventSource`: the question goes in a POST body, and
 *  EventSource can only issue a GET. That costs us reconnection, which we do
 *  not want anyway — a run is not idempotent and re-billing it on a dropped
 *  connection is the one thing a retry must not do here.
 *
 *  Failures arrive two ways, and the split is deliberate. Anything the server
 *  knew before the stream opened (no key, at capacity, a blank question) is a
 *  status code and throws. Anything that went wrong mid-run arrives as an
 *  `error` event, because the status line was already sent — so callers must
 *  handle both, and `App` does. */
export async function streamQuery(
  question: string,
  threadId: string | null,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${BASE}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: threadId || null }),
    signal,
  });
  if (!response.ok) throw new Error(await detailOf(response));
  if (!response.body) throw new Error("the server sent no response body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line. A chunk can split a frame
    // anywhere, so only whole frames are consumed and the remainder stays
    // buffered — cutting a frame in half is how a streaming client ends up
    // dropping the token that happened to straddle a packet boundary.
    let split: number;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      for (const line of frame.split("\n")) {
        if (line.startsWith("data:")) {
          onEvent(JSON.parse(line.slice(5).trim()) as StreamEvent);
        }
      }
    }
  }
}
