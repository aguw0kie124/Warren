"""Offline tests for section parsing.

Uses a hand-built fixture rather than a real 1.4 MB filing, so each trap the
parser has to survive is visible and isolated. Every case here was found in an
actual filing — see the comment on each.
"""

from app.parser import SECTION_UNKNOWN, html_to_lines, parse_sections


def body(text: str, repeat: int = 12) -> str:
    """Filler long enough to clear MIN_SECTION_CHARS (500)."""
    return f"<div>{(text + ' ') * repeat}</div>"


# Apple/Tesla shape: a linked table of contents, then bold headings. Includes a
# prose cross-reference and a curly apostrophe in the MD&A title (Tesla mixes
# curly and ASCII quotes within one document).
TEN_K = f"""
<div>
  <div><a href="#a">Item 1.</a></div><div>Business</div><div>3</div>
  <div><a href="#b">Item 1A.</a></div><div>Risk Factors</div><div>12</div>
  <div><a href="#c">Item 7.</a></div><div>Management's Discussion and Analysis</div><div>30</div>

  <div>Item 1. Business</div>
  {body("The Company designs and sells consumer electronics.")}

  <div>Item 1A. Risk Factors</div>
  {body("Our business is subject to macroeconomic risk and supply chain disruption.")}
  <div>Item 1A</div>
  {body("Adverse outcomes could affect results; these risk factors are not exhaustive.")}

  <div>Item 1B. Unresolved Staff Comments</div>
  <div>None.</div>

  <div>Item 7. Management’s Discussion and Analysis of Financial Condition</div>
  {body("Total net sales increased 2% during 2024 compared to 2023.")}

  <div>Item 8. Financial Statements</div>
  {body("Consolidated statements of operations.")}
</div>
"""


def test_extracts_the_expected_10k_sections():
    sections = parse_sections(TEN_K, "10-K")
    assert set(sections) == {
        "Item 1 Business",
        "Item 1A Risk Factors",
        "Item 7 MD&A",
    }


def test_body_heading_wins_over_table_of_contents():
    """The TOC lists every item before the body does. Taking the first match
    would capture a page number instead of the section."""
    risks = parse_sections(TEN_K, "10-K")["Item 1A Risk Factors"]
    assert "macroeconomic risk" in risks
    assert "12" not in risks.split("\n")[:2]  # the TOC page number


def test_running_page_headers_do_not_truncate_a_section():
    """Microsoft repeats a bare 'Item 1A' atop all ~15 pages of its risk
    factors. Treating those as section boundaries cut the section to its last
    page (81k chars -> 5k)."""
    risks = parse_sections(TEN_K, "10-K")["Item 1A Risk Factors"]
    assert "supply chain disruption" in risks  # before the page header
    assert "not exhaustive" in risks  # after it


def test_section_stops_at_the_next_item():
    risks = parse_sections(TEN_K, "10-K")["Item 1A Risk Factors"]
    assert "Unresolved Staff Comments" not in risks
    assert "Total net sales" not in risks


def test_mda_title_matches_through_a_curly_apostrophe():
    """Filers write Management's with U+2019, ASCII ', or both. Matching on
    'discussion and analysis' sidesteps the apostrophe entirely."""
    mda = parse_sections(TEN_K, "10-K")["Item 7 MD&A"]
    assert "Total net sales increased" in mda


def test_typographic_punctuation_is_folded_to_ascii():
    lines = html_to_lines("<div>Management’s “view” – 2024…</div>")
    assert lines == ["Management's \"view\" - 2024..."]


# Microsoft splits its heading across nested elements, and the split lands
# mid-word: 'ITEM 1A. RIS' + 'K FACTORS'.
SPLIT_HEADING = f"""
<div>
  <div><a href="#b">Item 1A.</a></div><div>Risk Factors</div><div>14</div>
  <div>Item 1A</div><div><span>ITEM 1A. RIS</span><span>K FACTORS</span></div>
  {body("Our operations are subject to various risks and uncertainties.")}
  <div>Item 1B. Unresolved Staff Comments</div><div>None.</div>
</div>
"""


def test_heading_split_mid_word_is_still_found():
    sections = parse_sections(SPLIT_HEADING, "10-K")
    assert "risks and uncertainties" in sections["Item 1A Risk Factors"]


# A 10-Q has an Item 2 in BOTH parts. Only the title distinguishes them, and
# the Part II one comes last — so the last-match rule alone picks wrong.
TEN_Q = f"""
<div>
  <div>PART I</div>
  <div>Item 2. Management's Discussion and Analysis of Financial Condition</div>
  {body("Quarterly net sales rose 5% year over year.")}
  <div>Item 3. Quantitative and Qualitative Disclosures</div>
  {body("Interest rate risk is unchanged.")}

  <div>PART II</div>
  <div>Item 1A. Risk Factors</div>
  <div>Our operations and financial results are subject to various risks and
  uncertainties, including the factors discussed in Part I, Item 1A, Risk Factors,
  in our Annual Report on Form 10-K for the year ended December 31, 2025, which
  could adversely affect our business, financial condition and future results.
  There have been no material changes to those risk factors.</div>
  <div>Item 2. Unregistered Sales of Equity Securities</div>
  {body("The Company repurchased shares during the quarter.")}
</div>
"""


def test_10q_item_2_resolves_to_mda_not_unregistered_sales():
    mda = parse_sections(TEN_Q, "10-Q")["Item 2 MD&A"]
    assert "Quarterly net sales" in mda
    assert "repurchased shares" not in mda


def test_10q_risk_factors_may_legitimately_be_short():
    """10-Q risk factors are updates and often just point back at the 10-K.
    That is an answer, not a parse failure, so the floor is lower here."""
    risks = parse_sections(TEN_Q, "10-Q")["Item 1A Risk Factors"]
    assert "no material changes" in risks
    assert "repurchased shares" not in risks  # stops at Part II Item 2


def test_falls_back_to_whole_document_when_nothing_is_recognized():
    """A parsing miss should cost retrieval precision, not the whole run."""
    html = "<div>" + body("Some filing with no recognizable item headings.") + "</div>"
    sections = parse_sections(html, "10-K")
    assert set(sections) == {SECTION_UNKNOWN}
    assert "no recognizable item headings" in sections[SECTION_UNKNOWN]


def test_toc_only_document_does_not_produce_a_stub_section():
    """If only the table of contents matches, the section would be a couple of
    hundred characters of page numbers — worse than absent."""
    html = """
    <div><a href="#b">Item 1A.</a></div><div>Risk Factors</div><div>12</div>
    <div><a href="#c">Item 7.</a></div><div>Discussion and Analysis</div><div>30</div>
    """
    assert set(parse_sections(html, "10-K")) == {SECTION_UNKNOWN}


def test_unknown_form_type_degrades_instead_of_raising():
    sections = parse_sections(TEN_K, "8-K")
    assert set(sections) == {SECTION_UNKNOWN}
