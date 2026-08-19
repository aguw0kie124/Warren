"""Writing filings and chunks to Postgres, idempotently.

Every write here is safe to repeat. That is the load-bearing property of the
whole ingestion path: re-running an ingest must converge on the same rows
rather than accumulating duplicates, because a corpus that silently doubles
degrades retrieval in a way no downstream test would obviously catch.
"""

import logging
from collections.abc import Sequence

from pgvector import Vector
from psycopg import Connection

from app.chunker import Chunk
from app.xbrl import Fact

logger = logging.getLogger(__name__)

Embedding = Sequence[float]

_UPSERT_FILING = """
INSERT INTO filings (
    accession_number, cik, ticker, company_name,
    form_type, fiscal_year, filing_date, source_url
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (accession_number) DO UPDATE SET
    cik          = EXCLUDED.cik,
    ticker       = EXCLUDED.ticker,
    company_name = EXCLUDED.company_name,
    form_type    = EXCLUDED.form_type,
    fiscal_year  = EXCLUDED.fiscal_year,
    filing_date  = EXCLUDED.filing_date,
    source_url   = EXCLUDED.source_url,
    ingested_at  = now()
"""

_UPSERT_CHUNK = """
INSERT INTO chunks (accession_number, chunk_index, section, content, embedding)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (accession_number, chunk_index) DO UPDATE SET
    section   = EXCLUDED.section,
    content   = EXCLUDED.content,
    embedding = EXCLUDED.embedding
"""


# F1 · The conflict target names the columns of the UNIQUE NULLS NOT DISTINCT
# constraint, so instant facts (period_start IS NULL) collide with themselves
# rather than inserting again on every backfill.
#
# **The WHERE clause is the restatement rule.** A later filing that revises a
# period supersedes the earlier one; an earlier filing arriving out of order
# (a backfill re-run, a resumed job) leaves the newer figure alone. Without it,
# whichever run happened last would win, which is not the same thing.
_UPSERT_FACT = """
INSERT INTO company_facts (
    cik, ticker, concept, unit, period_start, period_end,
    period_type, calendar_year, value, form, accession_number, filed_date
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (cik, concept, unit, period_start, period_end) DO UPDATE SET
    ticker           = EXCLUDED.ticker,
    period_type      = EXCLUDED.period_type,
    calendar_year    = EXCLUDED.calendar_year,
    value            = EXCLUDED.value,
    form             = EXCLUDED.form,
    accession_number = EXCLUDED.accession_number,
    filed_date       = EXCLUDED.filed_date
WHERE EXCLUDED.filed_date > company_facts.filed_date
"""


def upsert_facts(conn: Connection, facts: Sequence[Fact]) -> int:
    """Write XBRL facts. Returns the number offered, not the number changed.

    Offered rather than changed on purpose: a re-run legitimately updates
    nothing, and `executemany` cannot report per-row conflict outcomes anyway.
    The idempotency claim this supports is about *row counts in the table*,
    which is what `scripts/check_data.py` measures.
    """
    if not facts:
        return 0

    conn.cursor().executemany(
        _UPSERT_FACT,
        [
            (
                f.cik, f.ticker, f.concept, f.unit, f.period_start, f.period_end,
                f.period_type, f.calendar_year, f.value, f.form,
                f.accession_number, f.filed_date,
            )
            for f in facts
        ],
    )
    return len(facts)


def existing_accessions(conn: Connection, accessions: Sequence[str]) -> set[str]:
    """Which of these filings are already ingested — the ingestion ledger."""
    if not accessions:
        return set()
    rows = conn.execute(
        "SELECT accession_number FROM filings WHERE accession_number = ANY(%s)",
        (list(accessions),),
    ).fetchall()
    return {row[0] for row in rows}


def upsert_filing(conn: Connection, filing) -> None:
    conn.execute(
        _UPSERT_FILING,
        (
            filing.accession_number,
            filing.cik,
            filing.ticker,
            filing.company_name,
            filing.form_type,
            filing.fiscal_year,
            filing.filing_date,
            filing.source_url,
        ),
    )


def upsert_chunks(
    conn: Connection,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Embedding],
) -> int:
    """Write chunks with their vectors. Returns the number written."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"{len(chunks)} chunks but {len(embeddings)} embeddings — refusing to "
            f"write, since the mismatch would pair text with the wrong vector"
        )
    if not chunks:
        return 0

    accession = chunks[0].accession_number
    if any(c.accession_number != accession for c in chunks):
        raise ValueError("upsert_chunks expects chunks from a single filing")

    conn.cursor().executemany(
        _UPSERT_CHUNK,
        [
            (c.accession_number, c.chunk_index, c.section, c.content, Vector(vec))
            for c, vec in zip(chunks, embeddings, strict=True)
        ],
    )

    # A re-parse that yields fewer chunks than last time would otherwise leave
    # the tail of the old run orphaned — still indexed, still retrievable, and
    # attributed to a filing it no longer reflects.
    stale = conn.execute(
        "DELETE FROM chunks WHERE accession_number = %s AND chunk_index >= %s",
        (accession, len(chunks)),
    ).rowcount
    if stale:
        logger.info("removed %d stale chunk(s) from %s", stale, accession)

    return len(chunks)
