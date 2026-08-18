"""A5 verification: chunk cached filings and eyeball chunks + metadata.

    python scripts/check_chunker.py                        # every ticker in data/raw/
    python scripts/check_chunker.py --ticker AAPL --show 3
    python scripts/check_chunker.py --ticker META --fetch 2   # new ticker, downloads

Filing HTML comes from the data/raw/ cache, but filing *metadata* is pulled
fresh from EDGAR's submissions API so source_url is the real citation URL
rather than something reconstructed from a filename. That's one small JSON
request per ticker; cached filings are never re-downloaded, and uncached ones
are skipped entirely unless --fetch asks for them.
"""

import argparse
import logging
import re
from collections import Counter

from app.chunker import chunk_filing
from app.config import settings
from app.edgar import fetch_filing, list_filings, local_path
from app.parser import SECTION_UNKNOWN, parse_sections
from app.sec_http import close_client
from app.tickers import resolve_ticker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

NAME_RE = re.compile(r"^(?P<ticker>[^_]+)_(?P<form>10-[KQ])_(?P<fy>\d{4})_")

# Fields that must never be empty on a stored chunk — a blank one here becomes
# an unciteable answer later, which is far harder to diagnose.
REQUIRED = (
    "accession_number", "ticker", "company_name", "form_type",
    "fiscal_year", "filing_date", "source_url", "section", "content",
)


def cached_tickers() -> list[str]:
    seen = {m["ticker"] for p in settings.raw_dir.glob("*.html") if (m := NAME_RE.match(p.name))}
    return sorted(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", action="append", help="repeatable; default = all cached")
    ap.add_argument("--show", type=int, default=1, help="full chunks to print per filing")
    ap.add_argument(
        "--fetch",
        type=int,
        default=0,
        metavar="N",
        help="download up to N uncached filings per ticker (default 0 = offline)",
    )
    args = ap.parse_args()

    tickers = args.ticker or cached_tickers()
    if not tickers:
        print(f"No filings in {settings.raw_dir}. Run scripts/check_edgar.py first.")
        return

    problems = 0
    processed = 0
    try:
        for ticker in tickers:
            company = resolve_ticker(ticker)
            filings = list_filings(company, form_types=("10-K", "10-Q"), years=None, limit=20)

            cached = [f for f in filings if local_path(f).exists()]
            # Without --fetch this stays offline, so a ticker with nothing on
            # disk yields nothing to check — which is reported, never passed.
            missing = [f for f in filings if not local_path(f).exists()][: args.fetch]
            if not cached and not missing:
                print(f"\n{ticker}: no cached filings. Re-run with --fetch 2 to download some.")
            filings = cached + missing

            for filing in filings:
                processed += 1
                print(f"\n=== {filing.ticker} {filing.form_type} FY{filing.fiscal_year} "
                      f"({filing.accession_number}) ===")

                sections = parse_sections(fetch_filing(filing), filing.form_type)
                chunks = chunk_filing(filing, sections)

                if not chunks:
                    problems += 1
                    print("  [BAD] no chunks produced")
                    continue

                if SECTION_UNKNOWN in sections:
                    print("  [warn] parser fell back to whole-document")

                per_section = Counter(c.section for c in chunks)
                for section, count in per_section.items():
                    print(f"  {section:<24} {count:>4} chunks")

                lengths = [len(c.content) for c in chunks]
                print(f"  total {len(chunks)} chunks | chars min {min(lengths):,} "
                      f"avg {sum(lengths) // len(lengths):,} max {max(lengths):,}")

                # chunk_index must be a gapless 0..n-1 run, or ON CONFLICT
                # idempotency in A6 silently stops matching on re-ingest.
                indices = [c.chunk_index for c in chunks]
                if indices != list(range(len(chunks))):
                    problems += 1
                    print("  [BAD] chunk_index is not a gapless 0..n-1 sequence")

                blanks = {
                    field
                    for c in chunks
                    for field in REQUIRED
                    if getattr(c, field) in (None, "")
                }
                if blanks:
                    problems += 1
                    print(f"  [BAD] empty metadata fields: {sorted(blanks)}")
                else:
                    print("  [ok ] every metadata field populated")

                for chunk in chunks[: args.show]:
                    print(f"\n  --- chunk {chunk.chunk_index} ({chunk.section}) ---")
                    print(f"  {chunk.citation}")
                    print(f"  {chunk.source_url}")
                    print(f"  fiscal_year={chunk.fiscal_year} ticker={chunk.ticker}")
                    print("  " + chunk.content[:600].replace("\n", "\n  "))
    finally:
        close_client()

    if not processed:
        print("\nA5: NO DATA — nothing was checked, which is not a pass.")
    else:
        verdict = "PASS" if problems == 0 else f"FAIL ({problems} problem(s))"
        print(f"\nA5: {verdict} ({processed} filing(s) checked)")


if __name__ == "__main__":
    main()
