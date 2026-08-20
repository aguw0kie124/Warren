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
}

export interface ThreadSummary {
  id: string;
  title: string;
  updatedAt: number;
}
