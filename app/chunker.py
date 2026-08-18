"""Parsed sections -> embeddable chunks carrying full citation metadata.

Chunking never crosses a section boundary. That is the whole reason a citation
can name the section it came from: a chunk spanning the end of Item 1A and the
start of Item 1B has no honest answer to "which section is this?".
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.edgar import Filing

logger = logging.getLogger(__name__)

# Separators in descending order of how much we'd rather split there. Leading
# "\n" (not "\n\n") because parser.html_to_lines joins blocks with a single
# newline, so one newline *is* the paragraph break here.
SEPARATORS = ["\n", ". ", " ", ""]

# Chunks shorter than this are dropped as split artifacts — a trailing few words
# left over from the previous chunk embeds to noise. Deliberately low: a 10-Q
# Item 1A saying only "no material changes since our last Annual Report" is a
# legitimate ~150-char section, not garbage, and must survive.
MIN_CHUNK_CHARS = 50


@dataclass(frozen=True)
class Chunk:
    """One embeddable passage plus everything needed to cite it.

    Every field is populated at construction — a chunk that reaches the store
    without knowing its own provenance can never be cited, and the failure
    surfaces only at answer time when a citation comes out blank.
    """

    accession_number: str
    ticker: str
    company_name: str
    form_type: str
    fiscal_year: int
    filing_date: date
    source_url: str
    section: str
    chunk_index: int
    content: str

    @property
    def citation(self) -> str:
        """The human-readable half of a citation; the URL is the other half."""
        return (
            f"[{self.company_name} {self.form_type}, {self.section}, "
            f"filed {self.filing_date.isoformat()}]"
        )


@lru_cache(maxsize=1)
def _splitter() -> RecursiveCharacterTextSplitter:
    """Token-counting splitter, built once per process.

    Length is measured with the embedding model's own tokenizer rather than a
    character-count approximation: filings are dense with numbers, tickers and
    legal boilerplate that tokenize far denser than the ~4-chars-per-token rule
    of thumb assumes, so a character budget silently overshoots.
    """
    # Imported lazily — transformers pulls in torch, and nothing that merely
    # imports this module (tests collecting, --help) should pay that cost.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        separators=SEPARATORS,
    )


def split_section(text: str) -> list[str]:
    """Split one section's text into token-bounded passages."""
    parts = _splitter().split_text(text)
    return [p for part in parts if len(p := part.strip()) >= MIN_CHUNK_CHARS]


def chunk_filing(filing: Filing, sections: Mapping[str, str]) -> list[Chunk]:
    """Turn parsed sections into chunks, in document order.

    `chunk_index` runs across the whole filing rather than restarting per
    section, because `UNIQUE (accession_number, chunk_index)` is what makes
    re-ingestion idempotent — per-section numbering would collide on it.
    """
    chunks: list[Chunk] = []

    for section, text in sections.items():
        pieces = split_section(text)
        if not pieces:
            logger.warning(
                "section %s of %s produced no chunks (%d chars)",
                section,
                filing.accession_number,
                len(text),
            )
            continue

        for piece in pieces:
            chunks.append(
                Chunk(
                    accession_number=filing.accession_number,
                    ticker=filing.ticker,
                    company_name=filing.company_name,
                    form_type=filing.form_type,
                    fiscal_year=filing.fiscal_year,
                    filing_date=filing.filing_date,
                    source_url=filing.source_url,
                    section=section,
                    chunk_index=len(chunks),
                    content=piece,
                )
            )

        logger.debug("%s: %d chunks", section, len(pieces))

    logger.info(
        "%s %s: %d chunks across %d sections",
        filing.ticker,
        filing.form_type,
        len(chunks),
        len(sections),
    )
    return chunks
