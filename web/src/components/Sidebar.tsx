import type { ThreadSummary } from "../types";

export function Sidebar({
  threads,
  activeId,
  onNew,
  onSelect,
}: {
  threads: ThreadSummary[];
  activeId: string | null;
  onNew: () => void;
  onSelect: (id: string) => void;
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="mark">W</div>
        <div className="brand-name">Warren</div>
      </div>

      <button className="new-chat" onClick={onNew}>
        <span className="plus">+</span> New research
      </button>

      <div className="section-label">History</div>

      <nav className="history">
        {threads.length === 0 ? (
          <div className="history-empty">Your sessions appear here.</div>
        ) : (
          threads.map((thread) => (
            <button
              key={thread.id}
              className={`history-item${thread.id === activeId ? " active" : ""}`}
              onClick={() => onSelect(thread.id)}
              title={thread.title}
            >
              {thread.title}
            </button>
          ))
        )}
      </nav>

      <div className="sidebar-foot">
        Grounded in SEC filings, market data and the web.
      </div>
    </aside>
  );
}
