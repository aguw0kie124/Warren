"""Retrieval over the filing corpus.

A7 is dense (vector) search; A8 adds the hybrid BM25 + RRF variant alongside it.

Citations are assembled here, in code, from what the database already knows.
They are never asked of a model: an LLM handed an accession number will
cheerfully reformat it, and a citation that doesn't resolve is worse than no
citation, because it looks audited.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from pgvector import Vector

from app.db import get_conn
from app.embeddings import embed_query

logger = logging.getLogger(__name__)

DEFAULT_K = 6

# Ordered to match the SELECT below — the two must move together.
_COLUMNS = """
    c.accession_number, c.chunk_index, c.section, c.content,
    f.ticker, f.company_name, f.form_type, f.fiscal_year,
    f.filing_date, f.source_url
"""


@dataclass(frozen=True)
class Result:
    """One retrieved chunk, carrying a citation that resolves to real EDGAR."""

    accession_number: str
    chunk_index: int
    section: str
    content: str
    ticker: str
    company_name: str
    form_type: str
    fiscal_year: int
    filing_date: date
    source_url: str
    score: float  # cosine similarity in [-1, 1]; higher is better

    @property
    def citation(self) -> str:
        return (
            f"[{self.company_name} {self.form_type}, {self.section}, "
            f"filed {self.filing_date.isoformat()}]"
        )


def _filters(
    ticker: str | None,
    section: str | None,
    form_type: str | None,
    fiscal_year: int | None,
) -> tuple[list[str], list[object]]:
    """Build the shared WHERE fragments and their parameters.

    Returned as fragments rather than interpolated SQL so both search paths
    filter identically — an A8 hybrid query that filtered differently from A7
    would make the two impossible to compare, which is the whole point of A8.
    """
    clauses: list[str] = []
    params: list[object] = []

    if ticker:
        # Tickers are stored uppercase; callers (and later, an LLM) will not be
        # reliable about that.
        clauses.append("f.ticker = %s")
        params.append(ticker.upper())
    if section:
        clauses.append("c.section = %s")
        params.append(section)
    if form_type:
        clauses.append("f.form_type = %s")
        params.append(form_type.upper())
    if fiscal_year:
        clauses.append("f.fiscal_year = %s")
        params.append(fiscal_year)

    return clauses, params


def _where(clauses: Sequence[str]) -> str:
    return f"WHERE {' AND '.join(clauses)}" if clauses else ""


def _to_results(rows: Sequence[tuple]) -> list[Result]:
    return [
        Result(
            accession_number=row[0],
            chunk_index=row[1],
            section=row[2],
            content=row[3],
            ticker=row[4],
            company_name=row[5],
            form_type=row[6],
            fiscal_year=row[7],
            filing_date=row[8],
            source_url=row[9],
            score=row[10],
        )
        for row in rows
    ]


def search(
    query: str,
    ticker: str | None = None,
    section: str | None = None,
    form_type: str | None = None,
    fiscal_year: int | None = None,
    k: int = DEFAULT_K,
) -> list[Result]:
    """Dense vector search: the k chunks closest to the query in embedding space.

    Ordering is by cosine distance (`<=>`), which the HNSW index serves
    directly. Score is reported as similarity (1 - distance) so that larger is
    better, matching every other ranking signal in the system.
    """
    vector = Vector(embed_query(query))
    clauses, params = _filters(ticker, section, form_type, fiscal_year)

    sql = f"""
        SELECT {_COLUMNS}, 1 - (c.embedding <=> %s) AS score
        FROM chunks c
        JOIN filings f USING (accession_number)
        {_where(clauses)}
        ORDER BY c.embedding <=> %s
        LIMIT %s
    """

    with get_conn() as conn:
        rows = conn.execute(sql, [vector, *params, vector, k]).fetchall()

    logger.debug("dense search %r -> %d result(s)", query, len(rows))
    return _to_results(rows)
