"""Split a filing's HTML into named sections.

This is the most fragile step in the pipeline: EDGAR markup varies by filer and
by year, and there is no reliable structural marker for "section heading" — a
10-K is one flat document where headings are just bold spans.

Three different things in a filing say "Item 1A":

    1. the table of contents      <a href="#...">Item 1A.</a>
    2. the actual heading         <span style="font-weight:700">Item 1A. Risk Factors</span>
    3. prose cross-references     "...as discussed in Part I, Item 1A of this Form 10-K..."

Two rules separate them, both structural rather than filer-specific:

    * A heading line is SHORT. Cross-references live inside long sentences.
    * Take the LAST qualifying match. A table of contents always precedes the
      body it points at.

Plus a title check ("Item 1A" must be near the words "risk factor"), which is
what disambiguates 10-Q Part I Item 2 (MD&A) from Part II Item 2 (Unregistered
Sales) — both are "Item 2".
"""

import logging
import re
import unicodedata
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

logger = logging.getLogger(__name__)

# Modern filings are inline XBRL, so bs4's HTML parser warns about XML. Parsing
# them as HTML is exactly what we want — the XBRL tags are metadata we discard.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# A real heading is a handful of words. Anything longer is a sentence that
# happens to mention an item number.
MAX_HEADING_CHARS = 120

# Below this, we almost certainly captured a table-of-contents artifact rather
# than a real section body.
MIN_SECTION_CHARS = 500

SECTION_UNKNOWN = "unknown"

# Typographic punctuation -> ASCII. NFKC does not do this.
_PUNCTUATION_FOLD = str.maketrans(
    {
        "‘": "'", "’": "'",  # single quotes
        "“": '"', "”": '"',  # double quotes
        "–": "-", "—": "-",  # en/em dash
        "…": "...",
    }
)


@dataclass(frozen=True)
class SectionSpec:
    """A section worth extracting."""

    key: str  # stored on every chunk; shows up in citations
    number: str  # '1A', '7', ...
    title_keywords: tuple[str, ...]  # at least one must appear near the heading
    min_chars: int = MIN_SECTION_CHARS


# Deliberately apostrophe-free: filers write "Management's" with a curly U+2019,
# an ASCII quote, or both in the same document. Matching on the words after it
# sidesteps the whole problem.
_MDA_TITLE = ("discussion and analysis",)

# Sections we ingest, per form. Ordering here doesn't matter; document position
# is resolved from the text itself.
SECTIONS_10K = (
    SectionSpec("Item 1 Business", "1", ("business",)),
    SectionSpec("Item 1A Risk Factors", "1A", ("risk factor",)),
    SectionSpec("Item 7 MD&A", "7", _MDA_TITLE),
)

SECTIONS_10Q = (
    # Part I Item 2. The title check is load-bearing: Part II also has an
    # "Item 2" (Unregistered Sales), and without it the last-match rule would
    # pick the wrong one.
    SectionSpec("Item 2 MD&A", "2", _MDA_TITLE),
    # 10-Q risk factors are updates to the 10-K's and often just say "no
    # material changes since our last 10-K" — a couple of hundred characters,
    # which is a legitimate answer rather than a parse failure.
    SectionSpec("Item 1A Risk Factors", "1A", ("risk factor",), min_chars=150),
)

SECTIONS_BY_FORM = {"10-K": SECTIONS_10K, "10-Q": SECTIONS_10Q}

# Every item number that can start a section, used only to find where the
# section we want ENDS. Titles are irrelevant for that.
BOUNDARY_ITEMS = (
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
    "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16",
)


def html_to_lines(html: str) -> list[str]:
    """Flatten filing HTML into clean, non-empty text lines.

    Newline-separated (not space) so each block-level element lands on its own
    line — that separation is what makes a heading distinguishable from the
    prose around it.
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()
    # ix:header holds hidden XBRL metadata — thousands of lines of noise.
    for tag in soup.find_all(re.compile(r"^ix:header$", re.I)):
        tag.decompose()

    text = soup.get_text("\n")
    # NFKC folds &#160; (non-breaking space) and friends into plain spaces, but
    # leaves typographic punctuation alone — so fold that separately. Filers mix
    # curly and ASCII quotes freely (Tesla uses both in one document), and it
    # also gives the BM25 index in A8 consistent tokens.
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PUNCTUATION_FOLD)

    lines = []
    for raw in text.split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        if line:
            lines.append(line)
    return lines


def _item_pattern(number: str) -> re.Pattern[str]:
    """Match a line that STARTS with this item number.

    The trailing \\b stops 'Item 1' from matching 'Item 1A'.
    """
    return re.compile(rf"^item\s*{re.escape(number)}\b", re.IGNORECASE)


def _title_follows(lines: list[str], i: int, keywords: tuple[str, ...]) -> bool:
    """Whether the section title appears at or just after line i.

    Checked two ways because filers split headings across nested elements, and
    the split can land mid-word: Microsoft yields 'ITEM 1A. RIS' + 'K FACTORS'.
    Joining with a space finds normal splits; joining with nothing finds
    mid-word ones.

    Only short lines count. A title is a few words; a body paragraph is not.
    Without this, a running page header ('Item 1A' atop every page of the risk
    factors) qualifies whenever the paragraph beneath it happens to contain the
    words "risk factors" — which is common in that very section.
    """
    window = [ln for ln in lines[i : i + 3] if len(ln) <= MAX_HEADING_CHARS]
    spaced = " ".join(window).lower()
    glued = "".join(window).lower()
    return any(kw in spaced or kw in glued for kw in keywords)


def _find_anchor(
    lines: list[str],
    number: str,
    title_keywords: tuple[str, ...] = (),
) -> int | None:
    """Line index where this item's section begins, or None.

    Returns the LAST qualifying match, since a table of contents always
    precedes the body it points at.
    """
    pattern = _item_pattern(number)
    match_index = None

    for i, line in enumerate(lines):
        if len(line) > MAX_HEADING_CHARS or not pattern.match(line):
            continue
        if title_keywords and not _title_follows(lines, i, title_keywords):
            continue
        match_index = i

    return match_index


def _item_marks(lines: list[str]) -> list[tuple[int, str]]:
    """Every (line index, item number) that could start a section.

    Includes running page headers and stray references — filtering happens at
    the point of use.
    """
    marks: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if len(line) > MAX_HEADING_CHARS:
            continue
        for number in BOUNDARY_ITEMS:
            if _item_pattern(number).match(line):
                marks.append((i, number))
                break  # '1A' and '1' are mutually exclusive, so first wins
    return marks


def _section_end(marks: list[tuple[int, str]], start: int, number: str) -> int | None:
    """Where the section starting at `start` ends: the first mark for a
    DIFFERENT item.

    Excluding the section's own item number is what makes this survive running
    page headers. Microsoft prints 'Item 1A' at the top of all ~15 pages of its
    risk factors; treating those as boundaries would truncate the section to
    its final page.
    """
    return next((i for i, num in marks if i > start and num != number), None)


def parse_sections(html: str, form_type: str) -> dict[str, str]:
    """Extract the named sections we ingest from a filing.

    Degrades rather than failing: if nothing is confidently identified, returns
    the whole document under SECTION_UNKNOWN so ingestion still proceeds with
    reduced citation precision.
    """
    specs = SECTIONS_BY_FORM.get(form_type)
    if not specs:
        logger.warning("no section spec for form %r; treating as unknown", form_type)
        specs = ()

    lines = html_to_lines(html)
    if not lines:
        return {}

    marks = _item_marks(lines)

    sections: dict[str, str] = {}
    for spec in specs:
        start = _find_anchor(lines, spec.number, spec.title_keywords)
        if start is None:
            logger.warning("section not found: %s (%s)", spec.key, form_type)
            continue

        end = _section_end(marks, start, spec.number) or len(lines)
        body = "\n".join(lines[start:end]).strip()

        if len(body) < spec.min_chars:
            logger.warning(
                "section %s too short (%d chars, expected >=%d) — likely a "
                "table-of-contents artifact, skipping",
                spec.key,
                len(body),
                spec.min_chars,
            )
            continue

        sections[spec.key] = body
        logger.debug("%s: %d chars (lines %d-%d)", spec.key, len(body), start, end)

    if not sections:
        logger.warning(
            "no sections identified in this %s; falling back to whole document",
            form_type,
        )
        return {SECTION_UNKNOWN: "\n".join(lines)}

    return sections
