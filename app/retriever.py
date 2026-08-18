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

# Candidates each ranker contributes before fusion. Deeper than k on purpose: a
# chunk that one ranker puts 30th and the other puts 2nd is exactly the result
# hybrid exists to surface, and it never appears if both lists stop at 6.
CANDIDATES = 50

# plainto_tsquery ANDs every term, so a natural-language question demands that
# all of its words appear in one chunk: "What are the main business risks?"
# matched 0 of 806 chunks that way, leaving the sparse half of hybrid search
# contributing nothing. OR-ing the lexemes lets ts_rank_cd rank by how well a
# chunk covers the query, which is what it is for, instead of the matcher
# silently doing the filtering.
TSQUERY = "replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery"

# The standard RRF damping constant. Large relative to the ranks that matter, so
# no single ranker's #1 can dominate on its own — agreement between the two
# rankers outweighs a strong opinion from either.
RRF_K = 60

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
    score: float  # dense: cosine similarity. hybrid: fused RRF score.
    # Which ranker found this, and where. None means that ranker missed it
    # entirely — the asymmetry is the whole diagnostic value of hybrid search.
    dense_rank: int | None = None
    sparse_rank: int | None = None

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
            dense_rank=row[11] if len(row) > 11 else None,
            sparse_rank=row[12] if len(row) > 12 else None,
        )
        for row in rows
    ]


def rrf_score(dense_rank: int | None, sparse_rank: int | None, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion: the canonical definition of the SQL below.

    Only rank *position* is used. Cosine similarity and ts_rank_cd are on
    incomparable scales, so blending the scores themselves would need a
    normalization that is fragile and has to be retuned per dataset; rank
    sidesteps that entirely. A rank of None contributes nothing, which is what
    lets a chunk found by only one ranker still place.
    """
    dense = 1.0 / (k + dense_rank) if dense_rank else 0.0
    sparse = 1.0 / (k + sparse_rank) if sparse_rank else 0.0
    return dense + sparse


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


def hybrid_search(
    query: str,
    ticker: str | None = None,
    section: str | None = None,
    form_type: str | None = None,
    fiscal_year: int | None = None,
    k: int = DEFAULT_K,
    candidates: int = CANDIDATES,
) -> list[Result]:
    """Dense + BM25 retrieval fused by Reciprocal Rank Fusion, in one query.

    Filings are dense with exact-match tokens — 'material weakness', 'going
    concern', a proper noun — that embeddings blur into their neighbourhood.
    BM25 catches those literally; dense catches paraphrase. Neither alone is
    sufficient over this corpus.

    Both rankings are computed server-side in a single statement, so there is no
    second index to keep in sync and no round trip between the two halves.
    """
    vector = Vector(embed_query(query))
    clauses, params = _filters(ticker, section, form_type, fiscal_year)

    dense_where = _where(clauses)
    # The sparse side always carries its tsquery match, so its filters are
    # appended to that rather than forming the WHERE on their own.
    sparse_where = " AND ".join(["c.content_tsv @@ tsq.q", *clauses])

    sql = f"""
        WITH tsq AS (
            -- Hoisted into its own CTE because a cast expression cannot be a
            -- FROM item, and both the match and the ranking need it.
            SELECT {TSQUERY} AS q
        ),
        dense AS (
            SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s) AS rank
            FROM chunks c
            JOIN filings f USING (accession_number)
            {dense_where}
            ORDER BY c.embedding <=> %s
            LIMIT %s
        ),
        sparse AS (
            SELECT c.id,
                   ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.content_tsv, tsq.q) DESC) AS rank
            FROM chunks c
            JOIN filings f USING (accession_number)
            CROSS JOIN tsq
            WHERE {sparse_where}
            ORDER BY ts_rank_cd(c.content_tsv, tsq.q) DESC
            LIMIT %s
        ),
        fused AS (
            -- FULL OUTER JOIN, not INNER: a chunk only one ranker found still
            -- scores. Requiring agreement would discard precisely the
            -- exact-phrase hits that dense search misses.
            SELECT
                id,
                COALESCE(1.0 / (%s + dense.rank), 0)
                  + COALESCE(1.0 / (%s + sparse.rank), 0) AS score,
                dense.rank  AS dense_rank,
                sparse.rank AS sparse_rank
            FROM dense
            FULL OUTER JOIN sparse USING (id)
            ORDER BY score DESC
            LIMIT %s
        )
        SELECT {_COLUMNS}, fused.score, fused.dense_rank, fused.sparse_rank
        FROM fused
        JOIN chunks c ON c.id = fused.id
        JOIN filings f USING (accession_number)
        ORDER BY fused.score DESC
    """

    # Positional params must follow the order the fragments appear above:
    # tsquery, dense (vector, filters, vector, limit), sparse (filters, limit),
    # then the two RRF constants and the final k.
    args = [
        query,
        vector, *params, vector, candidates,
        *params, candidates,
        RRF_K, RRF_K, k,
    ]

    with get_conn() as conn:
        rows = conn.execute(sql, args).fetchall()

    logger.debug("hybrid search %r -> %d result(s)", query, len(rows))
    return _to_results(rows)
