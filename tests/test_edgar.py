"""Offline tests for EDGAR filing parsing. No network."""

from datetime import date

from app.edgar import (
    Filing,
    filings_from_recent,
    parse_fiscal_year_end_month,
)
from app.tickers import Company

APPLE = Company(ticker="AAPL", cik="0000320193", name="Apple Inc.")

# Column-oriented, exactly as the submissions API returns it.
RECENT = {
    "accessionNumber": [
        "0000320193-25-000079",
        "0000320193-26-000006",
        "0000320193-24-000123",
        "0000320193-99-000001",  # unusable: no primaryDocument
    ],
    "form": ["10-K", "10-Q", "10-K", "10-K"],
    "filingDate": ["2025-10-31", "2026-01-30", "2024-11-01", "1999-01-01"],
    "reportDate": ["2025-09-27", "2025-12-27", "2024-09-28", "1998-12-31"],
    "primaryDocument": ["aapl-20250927.htm", "aapl-20251227.htm", "aapl-20240928.htm", ""],
}


def test_skips_entries_without_a_primary_document():
    """No document means no fetchable URL, so the record is useless."""
    filings = filings_from_recent(RECENT, APPLE)
    assert len(filings) == 3
    assert all(f.primary_document for f in filings)


def test_maps_company_metadata_onto_every_filing():
    filing = filings_from_recent(RECENT, APPLE)[0]
    assert filing.ticker == "AAPL"
    assert filing.cik == "0000320193"
    assert filing.company_name == "Apple Inc."
    assert filing.filing_date == date(2025, 10, 31)
    assert filing.report_date == date(2025, 9, 27)


def test_source_url_uses_unpadded_cik_and_undashed_accession():
    """Archives paths reject the padded CIK and the dashed accession number."""
    filing = filings_from_recent(RECENT, APPLE)[0]
    assert filing.source_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019325000079/aapl-20250927.htm"
    )


def test_parse_fiscal_year_end_month():
    assert parse_fiscal_year_end_month("0927") == 9  # Apple, September
    assert parse_fiscal_year_end_month("1231") == 12  # December majority
    assert parse_fiscal_year_end_month("0630") == 6  # Microsoft, June
    assert parse_fiscal_year_end_month(None) is None
    assert parse_fiscal_year_end_month("bogus") is None
    assert parse_fiscal_year_end_month("1331") is None  # month 13


def _filing(report: date, fye_month: int | None) -> Filing:
    return Filing(
        accession_number="x",
        cik="0000320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type="10-Q",
        filing_date=report,
        report_date=report,
        primary_document="d.htm",
        fiscal_year_end_month=fye_month,
    )


def test_fiscal_year_rolls_forward_past_the_fiscal_year_end():
    """Apple's FY ends in September, so its Oct-Dec quarter is Q1 of the NEXT
    fiscal year — the case that makes calendar-year labelling wrong."""
    assert _filing(date(2025, 12, 27), 9).fiscal_year == 2026  # Q1 FY2026
    assert _filing(date(2026, 3, 28), 9).fiscal_year == 2026  # Q2 FY2026
    assert _filing(date(2025, 9, 27), 9).fiscal_year == 2025  # FY2025 10-K


def test_fiscal_year_for_june_filers():
    """Microsoft: FY2026 begins in July 2025."""
    assert _filing(date(2025, 6, 30), 6).fiscal_year == 2025
    assert _filing(date(2025, 9, 30), 6).fiscal_year == 2026


def test_fiscal_year_for_december_filers_is_the_calendar_year():
    assert _filing(date(2024, 12, 31), 12).fiscal_year == 2024
    assert _filing(date(2025, 3, 31), 12).fiscal_year == 2025


def test_fiscal_year_falls_back_to_calendar_year_when_fye_unknown():
    assert _filing(date(2024, 12, 31), None).fiscal_year == 2024


def test_fiscal_year_falls_back_to_filing_date_without_report_date():
    filing = Filing(
        accession_number="x",
        cik="0000320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type="10-K",
        filing_date=date(2024, 11, 1),
        report_date=None,
        primary_document="d.htm",
        fiscal_year_end_month=9,
    )
    assert filing.fiscal_year == 2025  # Nov > Sep, so next FY
