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
