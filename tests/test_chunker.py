"""Offline tests for chunking.

The tokenizer is loaded from the local HF cache, so these run without network
once the model has been pulled once. Text is synthetic: what's under test is
boundary and metadata behaviour, not prose quality, which is what
scripts/check_chunker.py is for.
"""

from datetime import date
from functools import lru_cache

import pytest

from app.chunker import MIN_CHUNK_CHARS, Chunk, chunk_filing, split_section
from app.config import settings
from app.edgar import Filing

FILING = Filing(
    accession_number="0000320193-25-000079",
    cik="0000320193",
    ticker="AAPL",
    company_name="Apple Inc.",
    form_type="10-K",
    filing_date=date(2025, 10, 31),
    report_date=date(2025, 9, 27),
    primary_document="aapl-20250927.htm",
    fiscal_year_end_month=9,
)


def prose(sentence: str, repeat: int) -> str:
    """Paragraph-separated filler, matching how the parser joins blocks."""
    return "\n".join(f"{sentence} Paragraph {i}." for i in range(repeat))


LONG = prose("Our business is subject to macroeconomic and supply chain risk.", 120)
SHORT = "There have been no material changes to our risk factors."


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(settings.embedding_model)


def token_count(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def test_long_section_splits_into_multiple_chunks():
    parts = split_section(LONG)
    assert len(parts) > 1


def test_chunks_respect_the_token_budget():
    # The splitter's guarantee is the reason chunk_tokens can be trusted as a
    # retrieval-quality knob rather than a hopeful suggestion.
    for part in split_section(LONG):
        assert token_count(part) <= settings.chunk_tokens


def test_short_section_survives_as_one_chunk():
    # A 10-Q Item 1A saying only "no material changes" is a real answer, and
    # must not be filtered away as a split artifact.
    assert len(SHORT) >= MIN_CHUNK_CHARS
    assert split_section(SHORT) == [SHORT]


def test_chunks_never_cross_a_section_boundary():
    sections = {"Item 1A Risk Factors": LONG, "Item 7 MD&A": prose("Net sales grew.", 120)}
    chunks = chunk_filing(FILING, sections)

    for chunk in chunks:
        assert chunk.content in sections[chunk.section]


def test_chunk_index_is_gapless_across_the_whole_filing():
    # Per-section numbering would collide on UNIQUE (accession_number,
    # chunk_index) and silently break A6's ON CONFLICT idempotency.
    sections = {"Item 1 Business": LONG, "Item 1A Risk Factors": LONG}
    chunks = chunk_filing(FILING, sections)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_sections_are_emitted_in_document_order():
    sections = {"Item 1 Business": LONG, "Item 1A Risk Factors": LONG}
    chunks = chunk_filing(FILING, sections)

    seen = list(dict.fromkeys(c.section for c in chunks))
    assert seen == ["Item 1 Business", "Item 1A Risk Factors"]


@pytest.mark.parametrize("field", [f.name for f in Chunk.__dataclass_fields__.values()])
def test_every_metadata_field_is_populated(field):
    chunks = chunk_filing(FILING, {"Item 1A Risk Factors": LONG})

    for chunk in chunks:
        assert getattr(chunk, field) not in (None, "")


def test_metadata_is_carried_from_the_filing():
    chunk = chunk_filing(FILING, {"Item 1A Risk Factors": LONG})[0]

    assert chunk.accession_number == FILING.accession_number
    assert chunk.ticker == "AAPL"
    assert chunk.company_name == "Apple Inc."
    assert chunk.form_type == "10-K"
    assert chunk.filing_date == date(2025, 10, 31)
    # Sept fiscal year end: a Sept-2025 period is FY2025, not FY2026.
    assert chunk.fiscal_year == 2025
    assert chunk.source_url == FILING.source_url


def test_citation_reads_as_a_source_line():
    chunk = chunk_filing(FILING, {"Item 1A Risk Factors": LONG})[0]

    assert chunk.citation == "[Apple Inc. 10-K, Item 1A Risk Factors, filed 2025-10-31]"


def test_empty_section_produces_no_chunks():
    assert chunk_filing(FILING, {"Item 1A Risk Factors": ""}) == []


def test_fragment_shorter_than_the_floor_is_dropped():
    assert split_section("Too short.") == []
