import type { JSX } from "react";
import type { Citation } from "../types";

/** The model writes prose with light markdown and `[n]` markers that index the
 *  citation list. Rendering that as a plain string loses the markers' meaning,
 *  and pulling in a markdown library for three constructs is more surface than
 *  the answers use. Bold, inline code, bullets, and `[n]` — nothing else. */

const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\[\d+\])/g;

function inline(text: string, citations: Citation[], keyBase: string) {
  return text.split(INLINE).map((part, i) => {
    const key = `${keyBase}-${i}`;
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 1) {
      return <code key={key}>{part.slice(1, -1)}</code>;
    }
    const marker = /^\[(\d+)\]$/.exec(part);
    if (marker) {
      const citation = citations[Number(marker[1]) - 1];
      if (!citation) return <span key={key}>{part}</span>;
      return (
        <a
          key={key}
          className="marker"
          href={citation.source_url}
          target="_blank"
          rel="noreferrer"
          title={citation.label}
        >
          {marker[1]}
        </a>
      );
    }
    return <span key={key}>{part}</span>;
  });
}

export function AnswerText({
  text,
  citations = [],
}: {
  text: string;
  citations?: Citation[];
}) {
  const blocks: JSX.Element[] = [];
  let bullets: string[] = [];

  const flush = () => {
    if (!bullets.length) return;
    const items = bullets;
    bullets = [];
    blocks.push(
      <ul key={`ul-${blocks.length}`}>
        {items.map((item, i) => (
          <li key={i}>{inline(item, citations, `li-${blocks.length}-${i}`)}</li>
        ))}
      </ul>,
    );
  };

  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) {
      flush();
      continue;
    }
    const bullet = /^[-*•]\s+(.*)$/.exec(line) ?? /^\d+\.\s+(.*)$/.exec(line);
    if (bullet) {
      bullets.push(bullet[1]);
      continue;
    }
    flush();
    blocks.push(
      <p key={`p-${blocks.length}`}>
        {inline(line, citations, `p-${blocks.length}`)}
      </p>,
    );
  }
  flush();

  return <div className="answer">{blocks}</div>;
}
