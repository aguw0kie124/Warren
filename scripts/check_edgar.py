"""A2 + A3 verification: list a company's filings, then download one.

    python scripts/check_edgar.py            # AAPL
    python scripts/check_edgar.py --ticker TSLA
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import exit_code, rule, summary, verdict  # noqa: E402

from app.edgar import fetch_filing, list_filings, local_path  # noqa: E402
from app.sec_http import close_client  # noqa: E402
from app.tickers import resolve_ticker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--years", type=int, default=2)
    args = ap.parse_args()

    company = resolve_ticker(args.ticker)
    print(f"\n{company.name}  (CIK {company.cik})")

    # --- A2: annual filings, sanity-checkable against reality ---
    rule("last 5 10-K filings (A2)")
    annual = list_filings(company, form_types=("10-K",), years=None, limit=5)
    for f in annual:
        print(
            f"  {f.filing_date}  FY{f.fiscal_year}  {f.accession_number}  "
            f"period_end={f.report_date}  {f.primary_document}"
        )

    # --- A2: the actual ingestion window ---
    rule(f"10-K + 10-Q in the last {args.years} years (the A6 ingest set)")
    window = list_filings(company, years=args.years)
    for f in window:
        print(f"  {f.filing_date}  {f.form_type:<5} FY{f.fiscal_year}  {f.accession_number}")
    print(f"  -> {len(window)} filings")

    if not verdict(bool(window), "filings found (needed to continue to A3)"):
        summary()
        return

    # --- A2: amendments must not leak in ---
    forms = {f.form_type for f in window}
    verdict(forms <= {"10-K", "10-Q"}, f"form types are exactly 10-K/10-Q: {sorted(forms)}")

    # --- A3: download the most recent 10-K ---
    target = annual[0] if annual else window[0]
    rule(f"fetching {target.form_type} {target.accession_number} (A3)")
    html = fetch_filing(target)
    path = local_path(target)

    print(f"  saved to   {path}")
    print(f"  size       {len(html):,} chars ({len(html) / 1_048_576:.1f} MB)")
    verdict("<" in html[:2000].lower(), "content looks like HTML")
    # a real 10-K is megabytes, not kilobytes
    verdict(len(html) > 100_000, "content is substantial (>100k chars)")
    print(f"\n  open in browser:  file://{path}")
    print(f"  source URL:       {target.source_url}")

    summary()


if __name__ == "__main__":
    try:
        main()
    finally:
        close_client()

    raise SystemExit(exit_code())
