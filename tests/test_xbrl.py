"""Offline tests for the XBRL fact parser.

No network. The payload shape is small and regular, so what is worth testing
here is the handful of rules that turn it into rows — and every one of them
fails *silently* if it is wrong, producing a plausible number rather than an
error:

- **The `fy` trap.** A 10-K restates its prior years, stamping the filing's
  fiscal year onto facts from earlier periods. Reading it would label 2023's
  revenue as FY2025 — the single most dangerous bug available in this module,
  because the figure is real and only the year is wrong.
- **Span classification.** Six- and nine-month cumulative facts look exactly
  like quarters once they are in a table. They have to be dropped, not
  mislabelled.
- **Deduplication.** The same period is reported by every filing that restates
  it. Keeping the wrong one means showing a superseded figure as current.
- **Decimal conversion.** `Decimal(0.1)` is not `Decimal("0.1")`, and the
  difference reaches a column whose entire job is exactness.
"""

from datetime import date
from decimal import Decimal

import pytest

from app.xbrl import Fact, parse_facts


def fact_row(val, end, start=None, form="10-K", filed="2025-10-31",
             accn="0000320193-25-000079", fy=2025, fp="FY"):
    """One row as companyfacts returns it — including `fy`/`fp`, which are
    present in the payload and must be ignored."""
    row = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp,
           "form": form, "filed": filed}
    if start:
        row["start"] = start
    return row


def payload(concept_units, cik=320193):
    return {"cik": cik, "facts": {"us-gaap": {
        concept: {"units": units} for concept, units in concept_units.items()
    }}}


# --- the fy trap ------------------------------------------------------------


def test_period_comes_from_start_and_end_not_fy():
    """The FY2025 10-K restates three years. All three rows carry fy=2025;
    each must keep its own period."""
    facts = parse_facts(payload({"Revenues": {"USD": [
        fact_row(383_285_000_000, "2023-09-30", "2022-09-25"),
        fact_row(391_035_000_000, "2024-09-28", "2023-10-01"),
        fact_row(416_161_000_000, "2025-09-27", "2024-09-29"),
    ]}}), "AAPL")

    by_year = {f.calendar_year: f.value for f in facts}
    assert by_year[2023] == Decimal("383285000000")
    assert by_year[2024] == Decimal("391035000000")
    assert by_year[2025] == Decimal("416161000000")


def test_calendar_year_is_the_period_end_year():
    facts = parse_facts(payload({"Revenues": {"USD": [
        fact_row(1, "2023-09-30", "2022-09-25", fy=2025),
    ]}}), "AAPL")

    assert facts[0].calendar_year == 2023


# --- span classification ----------------------------------------------------


@pytest.mark.parametrize("start,end,expected", [
    ("2024-09-29", "2025-09-27", "annual"),      # 363 days, 52-week year
    ("2024-10-01", "2025-09-30", "annual"),      # 364
    ("2025-06-29", "2025-09-27", "quarterly"),   # 90
    (None,         "2025-09-27", "instant"),     # balance-sheet item
])
def test_recognised_spans(start, end, expected):
    facts = parse_facts(payload({"X": {"USD": [fact_row(1, end, start)]}}), "AAPL")

    assert facts[0].period_type == expected


@pytest.mark.parametrize("start,end", [
    ("2025-03-30", "2025-09-27"),   # ~181 days, six-month cumulative
    ("2024-12-29", "2025-09-27"),   # ~272 days, nine-month cumulative
])
def test_cumulative_spans_are_dropped_not_mislabelled(start, end):
    """The reason the two windows have a gap between them. A nine-month figure
    sitting in a quarterly series is indistinguishable from a very good
    quarter."""
    assert parse_facts(payload({"X": {"USD": [fact_row(1, end, start)]}}), "AAPL") == []


# --- deduplication ----------------------------------------------------------


def test_the_latest_filing_wins():
    """The same period, reported twice, once restated."""
    facts = parse_facts(payload({"Revenues": {"USD": [
        fact_row(100, "2023-09-30", "2022-09-25", filed="2023-11-03", accn="a-23"),
        fact_row(111, "2023-09-30", "2022-09-25", filed="2025-10-31", accn="a-25"),
    ]}}), "AAPL")

    assert len(facts) == 1
    assert facts[0].value == Decimal("111")
    assert facts[0].accession_number == "a-25"


def test_the_latest_filing_wins_regardless_of_payload_order():
    facts = parse_facts(payload({"Revenues": {"USD": [
        fact_row(111, "2023-09-30", "2022-09-25", filed="2025-10-31", accn="a-25"),
        fact_row(100, "2023-09-30", "2022-09-25", filed="2023-11-03", accn="a-23"),
    ]}}), "AAPL")

    assert facts[0].value == Decimal("111")


def test_a_same_day_tie_resolves_by_accession_not_dict_order():
    """Deterministic across runs, which dict iteration order would not be."""
    facts = parse_facts(payload({"Revenues": {"USD": [
        fact_row(1, "2023-09-30", "2022-09-25", filed="2025-10-31", accn="a-1"),
        fact_row(2, "2023-09-30", "2022-09-25", filed="2025-10-31", accn="a-2"),
    ]}}), "AAPL")

    assert facts[0].accession_number == "a-2"


def test_different_units_of_one_concept_are_different_facts():
    facts = parse_facts(payload({"EarningsPerShareDiluted": {
        "USD/shares": [fact_row(6.08, "2025-09-27", "2024-09-29")],
        "USD":        [fact_row(1.00, "2025-09-27", "2024-09-29")],
    }}), "AAPL")

    assert {f.unit for f in facts} == {"USD/shares", "USD"}


# --- filtering --------------------------------------------------------------


def test_only_periodic_report_forms_are_kept():
    """8-K facts are earnings-release figures. Mixing them in would put
    unaudited numbers beside audited ones with nothing to tell them apart."""
    facts = parse_facts(payload({"Revenues": {"USD": [
        fact_row(1, "2025-09-27", "2024-09-29", form="10-K"),
        fact_row(2, "2025-06-28", "2025-03-30", form="10-Q"),
        fact_row(3, "2025-01-01", "2024-01-01", form="8-K"),
        fact_row(4, "2025-01-02", "2024-01-02", form="S-1"),
    ]}}), "AAPL")

    assert {f.form for f in facts} == {"10-K", "10-Q"}


def test_a_fact_without_an_accession_is_dropped():
    """No accession means no citation, which is the main reason to keep it."""
    facts = parse_facts(payload({"X": {"USD": [
        fact_row(1, "2025-09-27", "2024-09-29", accn=""),
    ]}}), "AAPL")

    assert facts == []


def test_only_the_us_gaap_taxonomy_is_read():
    doc = {"cik": 320193, "facts": {
        "us-gaap": {"Revenues": {"units": {"USD": [fact_row(1, "2025-09-27", "2024-09-29")]}}},
        "dei": {"EntityCommonStockSharesOutstanding": {
            "units": {"shares": [fact_row(2, "2025-09-27")]}}},
    }}

    facts = parse_facts(doc, "AAPL")

    assert [f.concept for f in facts] == ["Revenues"]


# --- values and identity ----------------------------------------------------


def test_float_values_convert_without_binary_rounding():
    facts = parse_facts(payload({"EarningsPerShareDiluted": {
        "USD/shares": [fact_row(6.08, "2025-09-27", "2024-09-29")]}}), "AAPL")

    assert facts[0].value == Decimal("6.08")


def test_cik_is_zero_padded_and_ticker_uppercased():
    """The padded form is what EDGAR URLs and the filings table both use."""
    facts = parse_facts(payload({"X": {"USD": [
        fact_row(1, "2025-09-27", "2024-09-29")]}}, cik=320193), "aapl")

    assert facts[0].cik == "0000320193"
    assert facts[0].ticker == "AAPL"


def test_an_empty_payload_yields_nothing():
    assert parse_facts({"cik": 1, "facts": {}}, "AAPL") == []


# --- how a period is described ----------------------------------------------


def test_periods_are_labelled_by_end_date_never_by_fiscal_year():
    """`calendar_year` is an approximation of the filer's own label; an end
    date is not. Anything a reader or a model sees uses the end date."""
    annual = Fact("0000320193", "AAPL", "Revenues", "USD", date(2024, 9, 29),
                  date(2025, 9, 27), "annual", 2025, Decimal("1"), "10-K",
                  "a", date(2025, 10, 31))
    instant = Fact("0000320193", "AAPL", "Assets", "USD", None,
                   date(2025, 9, 27), "instant", 2025, Decimal("1"), "10-K",
                   "a", date(2025, 10, 31))

    assert annual.label == "annual period ending 2025-09-27"
    assert instant.label == "as of 2025-09-27"
    assert "FY" not in annual.label and "2025-09-27" in annual.label
