"""A6: ingest SEC filings into Postgres — resolve, fetch, parse, chunk, embed, store.

    python scripts/ingest.py --ticker AAPL
    python scripts/ingest.py --ticker AAPL --ticker MSFT --years 2
    python scripts/ingest.py --ticker AAPL --force      # re-ingest what's there

Safe to re-run: filings already in the ingestion ledger are skipped, and even
--force converges on the same rows rather than duplicating them.
"""

import argparse
import logging

from app.chunker import chunk_filing
from app.db import close_pool, get_conn, init_schema
from app.edgar import fetch_filing, list_filings
from app.embeddings import embed_documents
from app.parser import SECTION_UNKNOWN, parse_sections
from app.sec_http import close_client
from app.store import existing_accessions, upsert_chunks, upsert_filing
from app.tickers import resolve_ticker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ingest")


def ingest_ticker(ticker: str, forms: tuple[str, ...], years: int | None,
                  limit: int, force: bool) -> tuple[int, int]:
    """Ingest one company. Returns (filings written, chunks written)."""
    company = resolve_ticker(ticker)
    filings = list_filings(company, form_types=forms, years=years, limit=limit)
    if not filings:
        logger.warning("%s: no filings matched", ticker)
        return 0, 0

    with get_conn() as conn:
        already = set() if force else existing_accessions(
            conn, [f.accession_number for f in filings]
        )

    pending = [f for f in filings if f.accession_number not in already]
    logger.info(
        "%s: %d filing(s) matched, %d already ingested, %d to do",
        ticker, len(filings), len(already), len(pending),
    )

    filings_written = chunks_written = 0
    for filing in pending:
        label = f"{filing.ticker} {filing.form_type} FY{filing.fiscal_year}"

        sections = parse_sections(fetch_filing(filing), filing.form_type)
        if SECTION_UNKNOWN in sections:
            logger.warning("%s: parser fell back to whole-document", label)

        chunks = chunk_filing(filing, sections)
        if not chunks:
            # No chunks means no retrievable content, so writing the filing row
            # would put an entry in the ledger that suppresses future retries.
            logger.error("%s: produced no chunks, skipping", label)
            continue

        vectors = embed_documents([c.content for c in chunks])

        # One transaction per filing: a crash mid-run leaves whole filings
        # ingested or absent, never a ledger entry with no chunks behind it.
        with get_conn() as conn:
            upsert_filing(conn, filing)
            written = upsert_chunks(conn, chunks, vectors)

        filings_written += 1
        chunks_written += written
        logger.info("%s: stored %d chunks", label, written)

    return filings_written, chunks_written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", action="append", required=True, help="repeatable")
    ap.add_argument("--forms", default="10-K,10-Q")
    ap.add_argument("--years", type=int, default=None, help="lookback window")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-ingest already-stored filings")
    args = ap.parse_args()

    forms = tuple(f.strip() for f in args.forms.split(",") if f.strip())

    init_schema()
    totals = [0, 0]
    try:
        for ticker in args.ticker:
            filings, chunks = ingest_ticker(
                ticker, forms, args.years, args.limit, args.force
            )
            totals[0] += filings
            totals[1] += chunks
    finally:
        close_client()
        close_pool()

    print(f"\ningested {totals[0]} filing(s), {totals[1]} chunk(s)")


if __name__ == "__main__":
    main()
