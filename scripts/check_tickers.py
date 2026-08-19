"""A1 verification: resolve real tickers and confirm a bogus one fails cleanly.

    python scripts/check_tickers.py

Hits sec.gov on first run, then uses the local cache for a week.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import exit_code, note, rule, summary, verdict  # noqa: E402

from app.sec_http import MissingUserAgentError, close_client  # noqa: E402
from app.tickers import UnknownTickerError, get_index, resolve_ticker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

KNOWN = ["AAPL", "SPCX", "MSFT", "aapl"]  # lowercase to check normalization
BOGUS = ["NOTREAL", "RETARD"]

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

    rule("known tickers")
    for symbol in KNOWN:
        company = resolve_ticker(symbol)
        expected = EXPECTED_CIK.get(symbol.upper())
        ok = expected is None or company.cik == expected
        verdict(ok, f"{symbol} -> {company.cik}  {company.name}")
        if not ok:
            print(f"          expected CIK {expected}")

    rule("bogus tickers (should raise)")
    for symbol in BOGUS:
        try:
            company = resolve_ticker(symbol)
        except UnknownTickerError as exc:
            verdict(True, f"{symbol} -> UnknownTickerError: {exc}")
        else:
            verdict(False, f"{symbol} -> unexpectedly resolved to {company}")

    rule("CIK padding")
    apple = resolve_ticker("AAPL")
    padded_ok = len(apple.cik) == 10 and apple.cik.startswith("0")
    verdict(padded_ok, f"padded  {apple.cik} (len {len(apple.cik)})")
    note(f"unpadded {apple.cik_int} (for Archives URLs in A3)")

    summary()


if __name__ == "__main__":
    try:
        main()
    except MissingUserAgentError as exc:
        print(f"\nEDGAR_USER_AGENT problem:\n  {exc}\n")
        raise SystemExit(1) from None
    finally:
        close_client()

    raise SystemExit(exit_code())
