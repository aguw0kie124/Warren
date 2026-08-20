export type CitationType = "filing" | "news" | "web";

export interface Citation {
  type: CitationType;
  label: string;
  source_url: string;
}

/** The four classes E1's router dispatches on. Only `research` uses tools. */
export type Route = "research" | "simple" | "advisory" | "clarify";

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  thread_id: string;
  route: Route;
}

export interface ThreadResponse {
  thread_id: string;
  messages: { role: string; content: string }[];
  citations: Citation[];
}

export interface Health {
  status: string;
  database: boolean;
  keys: Record<string, boolean>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  /** Only the citations this answer added — the API returns the whole thread's. */
  citations?: Citation[];
  route?: Route;
  error?: boolean;
  /** The progress log for this turn, kept after it finishes so the answer can
   *  still be checked against what was actually read. */
  steps?: Step[];
  /** True while the turn is still streaming — drives the live status line. */
  streaming?: boolean;
}

export interface ThreadSummary {
  id: string;
  title: string;
  updatedAt: number;
}

/** One line in the progress log — a tool call, described in prose by
 *  `app/tools.describe_tool_call`. Never a tool name: the point is that a
 *  reader can check the agent against what it says it is doing. */
export type Step = string;

/** What `POST /query/stream` sends, one JSON object per SSE frame.
 *
 *  `reset` is the one that matters and the one easiest to ignore. Tokens stream
 *  from a model turn before anyone knows whether that turn was the answer or a
 *  preamble to a tool call; when it turns out to have been a preamble, the
 *  server withdraws it. A client that drops `reset` shows "Let me check the
 *  filings." glued to the front of the answer — wrong, but plausible enough to
 *  survive review. */
export type StreamEvent =
  | { type: "start"; thread_id: string }
  | { type: "route"; route: Route }
  | { type: "step"; label: Step }
  | { type: "sources"; citations: Citation[] }
  | { type: "token"; text: string }
  | { type: "reset" }
  | {
      type: "done";
      answer: string;
      citations: Citation[];
      thread_id: string;
      route: Route;
    }
  | { type: "error"; detail: string };
