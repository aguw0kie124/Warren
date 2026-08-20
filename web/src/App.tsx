import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Composer } from "./components/Composer";
import { EmptyState } from "./components/EmptyState";
import { Message } from "./components/Message";
import { Sidebar } from "./components/Sidebar";
import * as api from "./lib/api";
import { marketStatus } from "./lib/market";
import { loadThreads, saveThreads, titleFor } from "./lib/threads";
import type { ChatMessage, Citation, ThreadSummary } from "./types";

const key = (citation: Citation) =>
  `${citation.type}|${citation.label}|${citation.source_url}`;

let counter = 0;
const nextId = () => `m${++counter}`;

export default function App() {
  const [threads, setThreads] = useState<ThreadSummary[]>(() => loadThreads());
  const [threadId, setThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(() => marketStatus());

  const scroller = useRef<HTMLDivElement>(null);
  // The whole thread's citations, so a turn can show only what it added.
  const seen = useRef<Set<string>>(new Set());

  useEffect(() => saveThreads(threads), [threads]);

  useEffect(() => {
    const id = setInterval(() => setStatus(marketStatus()), 60_000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  const startNew = useCallback(() => {
    setThreadId(null);
    setMessages([]);
    setInput("");
    seen.current = new Set();
  }, []);

  const openThread = useCallback(
    async (id: string) => {
      if (busy || id === threadId) return;
      setThreadId(id);
      setInput("");
      seen.current = new Set();
      setMessages([
        { id: nextId(), role: "assistant", text: "Loading conversation…" },
      ]);
      try {
        const thread = await api.readThread(id);
        // Tool messages are in the response too; a UI filters on role, and an
        // `ai` turn whose content is only tool calls flattens to empty text.
        const restored: ChatMessage[] = thread.messages
          .filter((m) => m.role === "human" || m.role === "ai")
          .filter((m) => m.content.trim())
          .map((m) => ({
            id: nextId(),
            role: m.role === "human" ? "user" : "assistant",
            text: m.content,
          }));
        // The service holds citations per thread, not per turn, so a restored
        // conversation shows them once, on the last answer.
        const last = restored.filter((m) => m.role === "assistant").at(-1);
        if (last) last.citations = thread.citations;
        thread.citations.forEach((c) => seen.current.add(key(c)));
        setMessages(restored);
      } catch (error) {
        setMessages([
          {
            id: nextId(),
            role: "assistant",
            text: `Could not load that session: ${(error as Error).message}`,
            error: true,
          },
        ]);
      }
    },
    [busy, threadId],
  );

  const send = useCallback(
    async (raw: string) => {
      const question = raw.trim();
      if (!question || busy) return;

      const id = nextId();
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "user", text: question },
        { id, role: "assistant", text: "", streaming: true, steps: [] },
      ]);
      setInput("");
      setBusy(true);

      // Only this turn's new sources belong under this turn, and the server
      // sends the thread's whole list — so what counts as "already shown" is
      // frozen here, before the turn adds to it.
      const already = new Set(seen.current);
      const patch = (change: Partial<ChatMessage>) =>
        setMessages((prev) =>
          prev.map((m) => (m.id === id ? { ...m, ...change } : m)),
        );

      try {
        await api.streamQuery(question, threadId, (event) => {
          switch (event.type) {
            case "start":
              setThreadId(event.thread_id);
              setThreads((prev) => {
                const existing = prev.find((t) => t.id === event.thread_id);
                return [
                  {
                    id: event.thread_id,
                    title: existing?.title ?? titleFor(question),
                    updatedAt: Date.now(),
                  },
                  ...prev.filter((t) => t.id !== event.thread_id),
                ];
              });
              break;

            case "route":
              patch({ route: event.route });
              break;

            case "step":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === id ? { ...m, steps: [...(m.steps ?? []), event.label] } : m,
                ),
              );
              break;

            case "sources": {
              const fresh = event.citations.filter((c) => !already.has(key(c)));
              fresh.forEach((c) => seen.current.add(key(c)));
              if (fresh.length) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === id
                      ? { ...m, citations: [...(m.citations ?? []), ...fresh] }
                      : m,
                  ),
                );
              }
              break;
            }

            case "token":
              setMessages((prev) =>
                prev.map((m) => (m.id === id ? { ...m, text: m.text + event.text } : m)),
              );
              break;

            case "reset":
              // That model turn was a preamble to a tool call, not the answer.
              patch({ text: "" });
              break;

            case "done":
              // The server's own final read, which is what `POST /query` would
              // have returned. Preferred over the accumulated tokens so a
              // dropped frame cannot leave a subtly truncated answer standing.
              patch({
                text: event.answer,
                route: event.route,
                streaming: false,
              });
              break;

            case "error":
              patch({ text: event.detail, error: true, streaming: false });
              break;
          }
        });
      } catch (error) {
        patch({
          text: (error as Error).message,
          error: true,
          streaming: false,
        });
      } finally {
        // Belt and braces: a stream that ends without `done` or `error` — a
        // dropped connection — must still stop rendering as in-flight.
        patch({ streaming: false });
        setBusy(false);
      }
    },
    [busy, threadId],
  );

  const title = useMemo(
    () => threads.find((t) => t.id === threadId)?.title ?? "New research",
    [threads, threadId],
  );

  return (
    <div className="app">
      <Sidebar
        threads={threads}
        activeId={threadId}
        onNew={startNew}
        onSelect={openThread}
      />

      <main className="main">
        <header className="topbar">
          <div className="session">{title}</div>
          <div className="market">
            <span className={`dot${status.open ? " open" : ""}`} />
            {status.label}
          </div>
        </header>

        <div className="scroll" ref={scroller}>
          <div className="column">
            {messages.length === 0 && !busy ? (
              <EmptyState onPick={send} />
            ) : (
              messages.map((message) => (
                <Message key={message.id} message={message} />
              ))
            )}
          </div>
        </div>

        <Composer
          value={input}
          onChange={setInput}
          onSend={() => send(input)}
          disabled={busy}
        />
      </main>
    </div>
  );
}
