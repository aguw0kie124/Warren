import { useEffect, useState } from "react";

/** A run is 20–60s, so a spinner alone reads as a hang. The elapsed count is
 *  the cheapest honest signal that work is still happening. */
export function Thinking() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="thinking">
      <div className="mark small pulsing">W</div>
      <span>Researching</span>
      <span className="dots" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
      {seconds > 3 && <span className="elapsed">{seconds}s</span>}
    </div>
  );
}
