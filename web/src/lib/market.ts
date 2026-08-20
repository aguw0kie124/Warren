/** US equity regular session, 09:30–16:00 America/New_York, weekdays.
 *  Holidays are not modelled — the dot is ambience, not a trading signal, and
 *  a wrong-but-confident holiday answer is worse than a coarse one. */
export function marketStatus(now: Date = new Date()): {
  open: boolean;
  label: string;
} {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    hour: "2-digit",
    minute: "2-digit",
    weekday: "short",
    hour12: false,
  }).formatToParts(now);
  const get = (type: string) => parts.find((p) => p.type === type)?.value ?? "";
  const weekday = get("weekday");
  const minutes = Number(get("hour")) * 60 + Number(get("minute"));
  const weekend = weekday === "Sat" || weekday === "Sun";
  const open = !weekend && minutes >= 9 * 60 + 30 && minutes < 16 * 60;
  return { open, label: open ? "Markets open" : "Markets closed" };
}
