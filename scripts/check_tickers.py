"""A1 verification: resolve real tickers and confirm a bogus one fails cleanly.

    python scripts/check_tickers.py

Hits sec.gov on first run, then uses the local cache for a week.
"""

import logging

from app.edgar import MissingUserAgentError, close_client
from app.tickers import UnknownTickerError, get_index, resolve_ticker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

KNOWN = ["AAPL", "TSLA", "MSFT", "aapl"]  # lowercase to check normalization
BOGUS = ["ZZZZ", "NOTAREALTICKER"]

# Independently known-correct CIKs, so this is a real assertion rather than
# "whatever EDGAR returned looks plausible".
EXPECTED_CIK = {
    "AAPL": "0000320193",
    "TSLA": "0001318605",
    "MSFT": "0000789019",
}


def main() -> None:
    index = get_index()
    print(f"\nticker map loaded: {len(index):,} companies\n")

    print("=== known tickers ===")
    failures = 0
    for symbol in KNOWN:
        company = resolve_ticker(symbol)
        expected = EXPECTED_CIK.get(symbol.upper())
        ok = expected is None or company.cik == expected
        failures += not ok
        mark = "ok " if ok else "BAD"
        print(f"  [{mark}] {symbol:<16} -> {company.cik}  {company.name}")
        if not ok:
            print(f"          expected CIK {expected}")

    print("\n=== bogus tickers (should raise) ===")
    for symbol in BOGUS:
        try:
            company = resolve_ticker(symbol)
        except UnknownTickerError as exc:
            print(f"  [ok ] {symbol:<16} -> UnknownTickerError: {exc}")
        else:
            failures += 1
            print(f"  [BAD] {symbol:<16} -> unexpectedly resolved to {company}")

    print("\n=== CIK padding ===")
    apple = resolve_ticker("AAPL")
    padded_ok = len(apple.cik) == 10 and apple.cik.startswith("0")
    failures += not padded_ok
    print(f"  [{'ok ' if padded_ok else 'BAD'}] padded  {apple.cik} (len {len(apple.cik)})")
    print(f"  [ok ] unpadded {apple.cik_int} (for Archives URLs in A3)")

    print("\nA1:", "PASS" if failures == 0 else f"FAIL ({failures} problem(s))")


if __name__ == "__main__":
    try:
        main()
    except MissingUserAgentError as exc:
        print(f"\nEDGAR_USER_AGENT problem:\n  {exc}\n")
        raise SystemExit(1) from None
    finally:
        close_client()
