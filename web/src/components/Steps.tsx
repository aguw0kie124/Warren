import { useState } from "react";
import type { Step } from "../types";

/** The progress log: what the agent read, in the order it read it.
 *
 *  While a turn is running this is the answer to "is it stuck or is it
 *  working" — a question the old elapsed-seconds counter could not answer,
 *  because a spinner at 40s looks the same whether the agent is reading a 10-K
 *  or has hung. Each line is prose written by `describe_tool_call`, so it names
 *  the company and the document rather than a function and its arguments.
 *
 *  It collapses once the turn finishes rather than disappearing. The finished
 *  log is what lets someone check an answer against its method — "it never
 *  read the filings" is visible here and nowhere else — but it is not what
 *  they came to read, so it stays one click away. */
export function Steps({
  steps,
  phase,
  streaming,
}: {
  steps: Step[];
  phase: string;
  streaming: boolean;
}) {
  const [open, setOpen] = useState(false);
  if (!steps.length && !streaming) return null;

  if (streaming) {
    return (
      <div className="steps live">
        <div className="phase">
          <span>{phase}</span>
          <span className="dots" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
        </div>
        {steps.map((step, i) => (
          <div className="step" key={`${step}-${i}`}>
            <span className="step-dot" aria-hidden="true" />
            {step}
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="steps">
      <button
        className="steps-toggle"
        onClick={() => setOpen((was) => !was)}
        aria-expanded={open}
      >
        <span className={`caret${open ? " open" : ""}`} aria-hidden="true">
          ›
        </span>
        Completed {steps.length} step{steps.length === 1 ? "" : "s"}
      </button>
      {open &&
        steps.map((step, i) => (
          <div className="step" key={`${step}-${i}`}>
            <span className="step-dot done" aria-hidden="true" />
            {step}
          </div>
        ))}
    </div>
  );
}
