const PROMPTS = [
  "What are Apple's key risk factors this year?",
  "How did Microsoft's revenue and margins trend?",
  "What's the latest news on NVIDIA?",
  "Summarize Tesla's most recent 10-Q.",
];

export function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="empty">
      <div className="mark large">W</div>
      <div>
        <h1>Ask Warren about the markets</h1>
        <p>Filings, fundamentals, news and the web — answered with sources.</p>
      </div>
      <div className="prompts">
        {PROMPTS.map((text) => (
          <button key={text} className="prompt" onClick={() => onPick(text)}>
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}
