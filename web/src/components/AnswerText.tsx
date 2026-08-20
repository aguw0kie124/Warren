import type { Element, ElementContent, Root, Text } from "hast";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation } from "../types";

/** The model writes markdown. It used to be rendered by a hand-written parser
 *  that understood bold, inline code and bullets — which was honest about what
 *  the answers contained at the time, and became wrong the moment the system
 *  prompt started asking for tables and headings. A period-over-period
 *  comparison is the case that matters: rendered by the old parser, a markdown
 *  table arrived as a column of pipe-separated lines, which is worse than no
 *  table at all because it is still readable enough to ship.
 *
 *  What the old parser did that a markdown library does not is `[n]` citation
 *  markers, so that survives as a rehype plugin below. */

const MARKER = /\[(\d+)\]/g;

/** `[n]` in the answer text becomes a link to citation n.
 *
 *  A rehype plugin rather than a pass over the raw string, because the raw
 *  string is markdown: rewriting `[3]` there would corrupt link syntax and
 *  anything inside a fenced code block. Working on the parsed tree means the
 *  substitution only ever sees text that was going to be rendered as text.
 *
 *  An unresolvable marker is left as plain text — the same choice the previous
 *  renderer made. A number the reader can see and ignore beats a link that
 *  goes somewhere arbitrary, and in a system whose whole claim is that every
 *  citation resolves, a wrong link is the more expensive failure. */
function citationMarkers(citations: Citation[]) {
  return () => (tree: Root) => {
    walk(tree);

    function walk(node: Root | Element): void {
      const out: ElementContent[] = [];
      let replaced = false;

      for (const child of node.children as ElementContent[]) {
        if (child.type === "element") {
          // Never inside code: `[0]` in a snippet is an array index.
          if (child.tagName !== "code" && child.tagName !== "pre") walk(child);
          out.push(child);
          continue;
        }
        if (child.type !== "text") {
          out.push(child);
          continue;
        }

        const pieces = split(child);
        if (pieces) {
          replaced = true;
          out.push(...pieces);
        } else {
          out.push(child);
        }
      }

      if (replaced) node.children = out;
    }

    function split(node: Text): ElementContent[] | null {
      MARKER.lastIndex = 0;
      if (!MARKER.test(node.value)) return null;
      MARKER.lastIndex = 0;

      const pieces: ElementContent[] = [];
      let cursor = 0;
      let match: RegExpExecArray | null;

      while ((match = MARKER.exec(node.value)) !== null) {
        const citation = citations[Number(match[1]) - 1];
        if (!citation) continue;
        if (match.index > cursor) {
          pieces.push({ type: "text", value: node.value.slice(cursor, match.index) });
        }
        pieces.push({
          type: "element",
          tagName: "a",
          properties: {
            className: ["marker"],
            href: citation.source_url,
            title: citation.label,
          },
          children: [{ type: "text", value: match[1] }],
        });
        cursor = match.index + match[0].length;
      }

      if (!pieces.length) return null;
      if (cursor < node.value.length) {
        pieces.push({ type: "text", value: node.value.slice(cursor) });
      }
      return pieces;
    }
  };
}

export function AnswerText({
  text,
  citations = [],
}: {
  text: string;
  citations?: Citation[];
}) {
  return (
    <div className="answer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[citationMarkers(citations)]}
        components={{
          // Every link leaves the app — sources are on sec.gov and the
          // publishers' own sites, never here.
          a: ({ node, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer" />
          ),
          // A financial table is wide by nature. Scrolling it inside its own
          // box keeps the answer column from scrolling sideways as a whole,
          // which on a narrow window makes the prose unreadable too.
          table: ({ node, ...props }) => (
            <div className="table-wrap">
              <table {...props} />
            </div>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
