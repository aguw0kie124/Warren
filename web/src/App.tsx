import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Composer } from "./components/Composer";
import { EmptyState } from "./components/EmptyState";
import { Message } from "./components/Message";
import { Sidebar } from "./components/Sidebar";
import { Thinking } from "./components/Thinking";
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

      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "user", text: question },
      ]);
      setInput("");
      setBusy(true);

      try {
        const result = await api.query(question, threadId);
        const fresh = result.citations.filter((c) => !seen.current.has(key(c)));
        fresh.forEach((c) => seen.current.add(key(c)));

        setThreadId(result.thread_id);
        setThreads((prev) => {
          const rest = prev.filter((t) => t.id !== result.thread_id);
          const existing = prev.find((t) => t.id === result.thread_id);
          return [
            {
              id: result.thread_id,
              title: existing?.title ?? titleFor(question),
              updatedAt: Date.now(),
            },
            ...rest,
          ];
        });
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "assistant",
            text: result.answer,
            citations: fresh,
            route: result.route,
          },
        ]);
      } catch (error) {
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "assistant",
            text: (error as Error).message,
            error: true,
          },
        ]);
      } finally {
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
            {busy && <Thinking />}
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
