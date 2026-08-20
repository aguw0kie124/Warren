import type { ChatMessage } from "../types";
import { AnswerText } from "./AnswerText";
import { RouteNote } from "./RouteNote";
import { Sources } from "./Sources";
import { Steps } from "./Steps";

/** Sources render above the answer, not below it.
 *
 *  During a run they are also the first thing that exists — the agent has read
 *  its filings well before it has written a sentence — so putting them first
 *  means the evidence is on screen while the answer is still arriving, and the
 *  reader can see what it rests on before they read a word of it. Below the
 *  answer they were a footnote to a claim already made. */
export function Message({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return <div className="bubble-user">{message.text}</div>;
  }
  if (message.error) {
    return (
      <div className="turn">
        <div className="mark small">W</div>
        <div className="turn-body">
          <div className="error">{message.text}</div>
        </div>
      </div>
    );
  }

  const citations = message.citations ?? [];
  const steps = message.steps ?? [];
  const streaming = Boolean(message.streaming);

  // Three named phases, derived from what has arrived rather than announced by
  // the server: sources exist once something has been read, tokens exist once
  // the answer is being written. A `reset` clears the text, which correctly
  // drops the label back to Searching for the next round of tool calls.
  const phase = message.text
    ? "Writing"
    : citations.length
      ? "Reading"
      : "Searching";

  return (
    <div className="turn">
      <div className={`mark small${streaming ? " pulsing" : ""}`}>W</div>
      <div className="turn-body">
        <Steps steps={steps} phase={phase} streaming={streaming} />
        <Sources citations={citations} />
        {message.text && (
          <AnswerText text={message.text} citations={citations} />
        )}
        {!streaming && (
          <RouteNote route={message.route} hasCitations={citations.length > 0} />
        )}
      </div>
    </div>
  );
}
