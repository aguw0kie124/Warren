import type { ChatMessage } from "../types";
import { AnswerText } from "./AnswerText";
import { RouteNote } from "./RouteNote";
import { Sources } from "./Sources";

export function Message({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return <div className="bubble-user">{message.text}</div>;
  }
  const citations = message.citations ?? [];
  return (
    <div className="turn">
      <div className="mark small">W</div>
      <div className="turn-body">
        {message.error ? (
          <div className="error">{message.text}</div>
        ) : (
          <>
            <AnswerText text={message.text} citations={citations} />
            <RouteNote route={message.route} hasCitations={citations.length > 0} />
            <Sources citations={citations} />
          </>
        )}
      </div>
    </div>
  );
}
