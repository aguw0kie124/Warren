"""EDGAR domain operations: what a company filed, and fetching those documents.

A2 lists filings from the submissions API; A3 downloads a filing's primary
document. HTTP concerns (User-Agent, rate limiting, retry) live in
app/sec_http.py.
"""

import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.config import settings
from app.sec_http import sec_get
from app.tickers import Company

logger = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"

DEFAULT_FORM_TYPES = ("10-K", "10-Q")


# ---------------------------------------------------------------------------
# A2: list a company's filings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Filing:
    """One SEC filing, carrying everything needed to fetch it and to cite it."""

    accession_number: str  # dashed form, e.g. '0000320193-24-000123'
    cik: str  # zero-padded 10-digit
    ticker: str
    company_name: str
    form_type: str
    filing_date: date
    report_date: date | None  # period covered; drives fiscal_year
    primary_document: str  # e.g. 'aapl-20240928.htm'
    fiscal_year_end_month: int | None = None  # from the filer's fiscalYearEnd

    @property
    def accession_plain(self) -> str:
        """Accession number without dashes, as Archives paths use it."""
        return self.accession_number.replace("-", "")

    @property
    def fiscal_year(self) -> int:
        """The fiscal year *as the company labels it*.

        Not simply the calendar year of the period end. Apple's FY ends in
        September, so its Oct-Dec quarter (period ending Dec 2025) is Q1 of
        FY2026, not FY2025. Any period ending after the fiscal year-end month
        belongs to the following fiscal year.

        Falls back to calendar year when fiscalYearEnd is unavailable, which is
        correct for the December-FYE majority anyway.
        """
        period = self.report_date or self.filing_date
        fye = self.fiscal_year_end_month
        if fye is None or fye == 12:
            return period.year
        return period.year + 1 if period.month > fye else period.year

    @property
    def source_url(self) -> str:
        """Canonical public URL for the document — becomes the citation link."""
        return ARCHIVES_URL.format(
            cik_int=int(self.cik),  # Archives uses the UNPADDED cik
            accession=self.accession_plain,
            document=self.primary_document,
        )


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _at(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def parse_fiscal_year_end_month(value: str | None) -> int | None:
    """EDGAR reports fiscalYearEnd as MMDD ('0927' = September 27)."""
    if not value or len(value) != 4 or not value.isdigit():
        return None
    month = int(value[:2])
    return month if 1 <= month <= 12 else None


def filings_from_recent(
    recent: dict[str, list[Any]],
    company: Company,
    fiscal_year_end_month: int | None = None,
) -> list[Filing]:
    """Convert the submissions API's parallel-array block into Filing objects.

    Pure (no network) so it can be unit tested. EDGAR returns column-oriented
    data — every key maps to a list, and index i across all of them describes
    one filing.
    """
    accessions = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    documents = recent.get("primaryDocument", [])

    filings: list[Filing] = []
    for i, accession in enumerate(accessions):
        filing_date = _parse_date(_at(filing_dates, i))
        document = _at(documents, i) or ""
        form = _at(forms, i) or ""

        # Without a filing date or a primary document we can neither order nor
        # fetch it; skip rather than emit an unusable record.
        if not filing_date or not document:
            logger.debug("skipping %s: missing filingDate or primaryDocument", accession)
            continue

        filings.append(
            Filing(
                accession_number=accession,
                cik=company.cik,
                ticker=company.ticker,
                company_name=company.name,
                form_type=form,
                filing_date=filing_date,
                report_date=_parse_date(_at(report_dates, i)),
                primary_document=document,
                fiscal_year_end_month=fiscal_year_end_month,
            )
        )
    return filings


def list_filings(
    company: Company,
    form_types: tuple[str, ...] = DEFAULT_FORM_TYPES,
    years: int | None = 2,
    limit: int | None = None,
) -> list[Filing]:
    """List a company's filings, most recent first.

    form_types matches exactly, so '10-K' excludes amendments ('10-K/A') and
    notifications ('NT 10-K') — those have different structure and would break
    the A4 parser.

    Only the submissions API's `recent` block is read (up to 1000 filings),
    which covers many years of 10-K/10-Q for any normal filer. Older filings
    live in paginated `files` entries; unnecessary at Phase 1 scope.
    """
    url = SUBMISSIONS_URL.format(cik=company.cik)
    payload = sec_get(url, accept="application/json").json()

    recent = payload.get("filings", {}).get("recent", {})
    fye_month = parse_fiscal_year_end_month(payload.get("fiscalYearEnd"))
    filings = filings_from_recent(recent, company, fye_month)

    if form_types:
        filings = [f for f in filings if f.form_type in form_types]

    if years is not None:
        cutoff = date.today() - timedelta(days=365 * years)
        filings = [f for f in filings if f.filing_date >= cutoff]

    filings.sort(key=lambda f: f.filing_date, reverse=True)

    if limit is not None:
        filings = filings[:limit]

    logger.info(
        "%s: %d filings (forms=%s, years=%s)",
        company.ticker,
        len(filings),
        ",".join(form_types) if form_types else "all",
        years,
    )
    return filings


# ---------------------------------------------------------------------------
# A3: fetch a filing document
# ---------------------------------------------------------------------------

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


def local_path(filing: Filing) -> Path:
    """Where this filing's raw HTML is cached on disk."""
    settings.ensure_dirs()
    name = _UNSAFE_FILENAME.sub(
        "_",
        f"{filing.ticker}_{filing.form_type}_{filing.fiscal_year}"
        f"_{filing.accession_number}.html",
    )
    return settings.raw_dir / name


def fetch_filing(filing: Filing, force: bool = False) -> str:
    """Download a filing's primary document, caching it under data/raw/.

    Keeping the raw HTML on disk matters for A4: EDGAR markup varies by filer
    and year, so the section parser needs many iterations against real files.
    Re-hitting SEC for each of those would be slow and rude.
    """
    path = local_path(filing)

    if path.exists() and not force:
        logger.debug("using cached %s", path.name)
        return path.read_text(encoding="utf-8", errors="replace")

    logger.info(
        "fetching %s %s (%s)", filing.ticker, filing.form_type, filing.accession_number
    )
    text = sec_get(filing.source_url, accept="text/html").text

    path.write_text(text, encoding="utf-8")
    logger.info("saved %s (%.1f MB)", path.name, len(text) / 1_048_576)
    return text
