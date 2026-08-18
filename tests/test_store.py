"""Offline tests for the store's guard rails.

These cover the checks that fire *before* any database work, so they need no
Postgres. Idempotency itself is verified by re-running scripts/ingest.py, since
it is a property of the SQL rather than of Python.
"""

from datetime import date

import pytest

from app.chunker import Chunk
from app.store import upsert_chunks


def chunk(index: int, accession: str = "0000320193-25-000079") -> Chunk:
    return Chunk(
        accession_number=accession,
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type="10-K",
        fiscal_year=2025,
        filing_date=date(2025, 10, 31),
        source_url="https://www.sec.gov/Archives/edgar/data/320193/x.htm",
        section="Item 1A Risk Factors",
        chunk_index=index,
        content=f"Risk factor {index}.",
    )


def test_mismatched_embedding_count_is_refused():
    # Silent zip() truncation here would pair text with another chunk's vector,
    # which retrieval would surface as confidently wrong citations.
    with pytest.raises(ValueError, match="refusing to write"):
        upsert_chunks(None, [chunk(0), chunk(1)], [[0.1] * 768])


def test_chunks_from_multiple_filings_are_refused():
    # The stale-chunk cleanup keys off a single accession number, so a mixed
    # batch would delete another filing's chunks.
    with pytest.raises(ValueError, match="single filing"):
        upsert_chunks(None, [chunk(0), chunk(1, "0000320193-26-000013")], [[0.1] * 768] * 2)


def test_empty_batch_writes_nothing():
    assert upsert_chunks(None, [], []) == 0
