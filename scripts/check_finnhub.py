"""B1 verification: live market data, metrics, and news from Finnhub.

    python scripts/check_finnhub.py                  # AAPL, all four endpoints
    python scripts/check_finnhub.py --symbol TSLA
    python scripts/check_finnhub.py --skip-bogus     # save two API calls

What you are checking by eye, because none of it is assertable offline:

  1. The quote matches a real quote in your browser (allow for free-tier delay).
  2. The metrics are recognisable — a plausible P/E, margins in percent.
  3. The headlines are recent and real, and the URLs open.
  4. The default date range kicks in when you pass no dates.
  5. A bogus symbol fails cleanly instead of returning a $0.00 quote that the
     agent would happily narrate as fact.
"""

import argparse
import logging
from datetime import date, timedelta

from app.finnhub import (
    DEFAULT_NEWS_DAYS,
    FinnhubError,
    close_client,
    get_basic_financials,
    get_company_news,
    get_company_profile,
    get_quote,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

BOGUS_SYMBOL = "ZZQQNOTREAL"


def rule(title: str) -> None:
    print(f"\n{'─' * 70}\n{title}\n{'─' * 70}")


def check_quote(symbol: str) -> None:
    rule(f"/quote · {symbol}")
    q = get_quote(symbol)
    move = f"{q.change:+.2f} ({q.percent_change:+.2f}%)" if q.change is not None else "—"
    print(f"  price        {q.current_price:,.2f}  {move}")
    print(f"  day range    {q.low:,.2f} – {q.high:,.2f}   (open {q.open:,.2f})")
    print(f"  prev close   {q.previous_close:,.2f}")
    print(f"  as of        {q.as_of}")
    print("\n  ✔ compare against a real quote in your browser.")


def check_financials(symbol: str) -> None:
    rule(f"/stock/metric · {symbol}")
    fin = get_basic_financials(symbol)
    present = fin.present()
    for name, value in present.items():
        print(f"  {name:<26} {value:>12,.2f}")
    missing = sorted(set(fin.model_dump(exclude={"symbol"})) - set(present))
    print(f"\n  {len(present)} metric(s) present.")
    if missing:
        # Absence is normal (Finnhub lacks some metrics per filer). All of them
        # missing would have raised inside get_basic_financials.
        print(f"  not reported by Finnhub: {', '.join(missing)}")


def check_news(symbol: str) -> None:
    rule(f"/company-news · {symbol} · default range")
    items = get_company_news(symbol)
    expected_start = date.today() - timedelta(days=DEFAULT_NEWS_DAYS)
    print(f"  default window should start ≈ {expected_start} (trailing "
          f"{DEFAULT_NEWS_DAYS} days)\n")

    if not items:
        print("  no articles in the window — quiet week, or a thin-coverage ticker.")
        return

    for item in items:
        print(f"  {item.published_date}  {item.source:<18} {item.headline[:70]}")
        print(f"              {item.url}")
    oldest = min(item.published_date for item in items)
    print(f"\n  {len(items)} article(s); oldest {oldest}.")
    print("  ✔ open a URL or two — they must resolve to the real article.")


def check_profile(symbol: str) -> None:
    rule(f"/stock/profile2 · {symbol}")
    p = get_company_profile(symbol)
    cap = f"{p.market_cap_millions:,.0f}M {p.currency}" if p.market_cap_millions else "—"
    print(f"  {p.name} ({p.symbol}) · {p.exchange}")
    print(f"  industry     {p.industry}")
    print(f"  market cap   {cap}")
    print(f"  ipo          {p.ipo or '—'}   country {p.country}")
    print(f"  website      {p.website}")


def check_bogus() -> None:
    rule(f"failure mode · {BOGUS_SYMBOL}")
    # Finnhub answers 200 with an empty or all-zero body for symbols it does not
    # know. If any of these *returned* instead of raising, the agent would be
    # reporting fabricated numbers with total confidence.
    for label, call in (
        ("get_quote", lambda: get_quote(BOGUS_SYMBOL)),
        ("get_basic_financials", lambda: get_basic_financials(BOGUS_SYMBOL)),
        ("get_company_profile", lambda: get_company_profile(BOGUS_SYMBOL)),
        ("get_company_news", lambda: get_company_news(BOGUS_SYMBOL)),
    ):
        try:
            result = call()
        except FinnhubError as exc:
            print(f"  ✔ {label:<22} raised {type(exc).__name__}: {exc}")
        else:
            print(f"  ✘ {label:<22} RETURNED {result!r} — should have raised")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--skip-bogus", action="store_true")
    args = parser.parse_args()

    try:
        check_quote(args.symbol)
        check_financials(args.symbol)
        check_news(args.symbol)
        check_profile(args.symbol)
        if not args.skip_bogus:
            check_bogus()
    except FinnhubError as exc:
        # A missing key or a plan restriction is a setup problem, not a stack
        # trace worth reading.
        print(f"\n✘ {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
    finally:
        close_client()

    print("\nB1 checks complete.\n")


if __name__ == "__main__":
    main()
