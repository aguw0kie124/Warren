"""F1: backfill XBRL company facts — one HTTP call per filer, full history.

    python scripts/backfill_facts.py --ticker AAPL --ticker MSFT
    python scripts/backfill_facts.py --tickers-file data/universe_500.txt
    python scripts/backfill_facts.py --tickers-file ... --force

Cheap in a way the text corpus is not, which is what makes coverage
asymmetric. One `companyfacts` request returns every fact a company has ever
reported — 2009 to today — against ~25 seconds of parse-and-embed for a single
filing. So fundamentals cover hundreds of companies while filing *text* covers
tens, and a question about Nvidia's margins is answerable even though nothing
of its 10-K has been ingested.

Costs no money. It is SEC-rate-limited at 3 req/s by `app/sec_http.py`, so
wall-clock is roughly one second per company plus parse time.

**Resumable and safe to re-run.** Filers already present are skipped unless
`--force`, and even `--force` converges rather than duplicating: the upsert
keeps whichever figure was filed later, so a restatement supersedes and a
re-run changes nothing.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.db import close_pool, get_conn, init_schema  # noqa: E402
from app.sec_http import close_client  # noqa: E402
from app.store import existing_fact_ciks, upsert_facts  # noqa: E402
from app.tickers import try_resolve_ticker  # noqa: E402
from app.xbrl import fetch_company_facts, parse_facts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backfill_facts")


def read_universe(path: Path) -> list[str]:
    """One ticker per line. Blank lines and `#` comments ignored, so the list
    can carry a note about why a name is on it."""
    tickers = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tickers.append(line.upper())
    return tickers


def backfill_one(ticker: str, already: set[str], force: bool) -> tuple[int, str]:
    """One company. Returns (facts written, status) — never raises.

    **Per-ticker isolation is the point.** A 500-name universe will contain a
    delisted symbol, a filer with no XBRL, and a transient 500 from SEC. Any
    one of those aborting the batch would mean re-running thirty minutes of
    work to get past a name nobody cared about.
    """
    company = try_resolve_ticker(ticker)
    if company is None:
        return 0, "unknown ticker"
    if company.cik in already and not force:
        return 0, "skipped (already present)"

    facts = parse_facts(fetch_company_facts(company.cik), ticker)
    if not facts:
        # Real and not an error: a company that has never filed with XBRL, or
        # files only forms this module skips.
        return 0, "no usable facts"

    with get_conn() as conn:
        written = upsert_facts(conn, facts)
    return written, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", action="append", default=[], help="repeatable")
    ap.add_argument("--tickers-file", type=Path, help="one ticker per line")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch filers already present")
    args = ap.parse_args()

    tickers = [t.upper() for t in args.ticker]
    if args.tickers_file:
        tickers += read_universe(args.tickers_file)
    # De-duplicated, order preserved, so a file plus flags cannot fetch twice.
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise SystemExit(2)

    init_schema()
    try:
        with get_conn() as conn:
            already = existing_fact_ciks(conn)

        total = 0
        problems: list[tuple[str, str]] = []
        for i, ticker in enumerate(tickers, start=1):
            try:
                written, status = backfill_one(ticker, already, args.force)
            except Exception as exc:  # noqa: BLE001 - isolation is the job here
                written, status = 0, f"{type(exc).__name__}: {exc}"

            total += written
            if status not in ("ok", "skipped (already present)"):
                problems.append((ticker, status))
            logger.info("[%d/%d] %-6s %6d fact(s)  %s",
                        i, len(tickers), ticker, written, status)

        print(f"\nbackfilled {total:,} fact(s) across {len(tickers)} ticker(s)")
        if problems:
            print(f"\n{len(problems)} ticker(s) had problems — the batch continued:")
            for ticker, status in problems:
                print(f"  {ticker:<8} {status}")
    finally:
        close_client()
        close_pool()


if __name__ == "__main__":
    main()
