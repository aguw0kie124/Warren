import type { Citation, CitationType } from "../types";

/** `type` is the point of the citation model — an audited filing, a news
 *  article and a web page are not equally authoritative, so the chip says
 *  which it is rather than showing a bare link. */
const KIND: Record<CitationType, { label: string; className: string }> = {
  filing: { label: "SEC", className: "chip-filing" },
  news: { label: "News", className: "chip-news" },
  web: { label: "Web", className: "chip-web" },
};

export function Sources({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <div className="sources">
      <div className="sources-label">Sources</div>
      <div className="source-list">
        {citations.map((citation, i) => {
          const kind = KIND[citation.type] ?? KIND.web;
          return (
            <a
              key={`${citation.type}-${citation.label}-${citation.source_url}`}
              className="source"
              href={citation.source_url}
              target="_blank"
              rel="noreferrer"
            >
              <span className="source-n">{i + 1}</span>
              <span className="source-name">{citation.label}</span>
              <span className={`source-kind ${kind.className}`}>
                {kind.label}
              </span>
            </a>
          );
        })}
      </div>
    </div>
  );
}
