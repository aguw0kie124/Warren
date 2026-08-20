import { useState } from "react";
import type { Citation, CitationType } from "../types";

/** `type` is the point of the citation model — an audited filing, a news
 *  article and a web page are not equally authoritative, so the chip says
 *  which it is rather than showing a bare link. */
const KIND: Record<CitationType, { label: string; className: string }> = {
  filing: { label: "SEC", className: "chip-filing" },
  news: { label: "News", className: "chip-news" },
  web: { label: "Web", className: "chip-web" },
};

/** How many sources show before the list collapses.
 *
 *  A research turn can gather fifteen or more — one news call alone used to
 *  return ten — and a list that long stops being read at all, which defeats
 *  the point of assembling it. Four is roughly what fits on one line, and the
 *  rest are one click away rather than hidden: nothing is dropped from the
 *  response, this is only what shows first. */
const SHOWN = 4;

export function Sources({ citations }: { citations: Citation[] }) {
  const [expanded, setExpanded] = useState(false);
  if (!citations.length) return null;

  const hidden = citations.length - SHOWN;
  const visible = expanded ? citations : citations.slice(0, SHOWN);

  return (
    <div className="sources">
      <div className="sources-label">
        {citations.length} source{citations.length === 1 ? "" : "s"}
      </div>
      <div className="source-list">
        {visible.map((citation, i) => {
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
        {hidden > 0 && (
          <button
            className="source more"
            onClick={() => setExpanded((was) => !was)}
          >
            {expanded ? "Show fewer" : `+${hidden} more`}
          </button>
        )}
      </div>
    </div>
  );
}
