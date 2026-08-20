import type { Route } from "../types";

/** A `research` answer with no citations found nothing; a `simple`, `advisory`
 *  or `clarify` answer has none by design. The API reports `route` precisely so
 *  a client can tell those apart, and collapsing them is the failure this whole
 *  system exists to avoid. */
const NOTES: Partial<Record<Route, string>> = {
  simple: "Answered from general knowledge — no sources consulted.",
  advisory: "Warren does not make buy, sell or hold recommendations.",
  clarify: "Needs more detail before it can be researched.",
};

export function RouteNote({
  route,
  hasCitations,
}: {
  route?: Route;
  hasCitations: boolean;
}) {
  if (!route) return null;
  if (route === "research") {
    return hasCitations ? null : (
      <div className="route-note">No sources found for this question.</div>
    );
  }
  const note = NOTES[route];
  return note ? <div className="route-note">{note}</div> : null;
}
