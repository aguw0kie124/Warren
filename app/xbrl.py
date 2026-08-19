"""F1 · XBRL company facts — the audited numbers, as data.

Every filing this project already downloads is inline XBRL: each figure in the
financial statements is wrapped in a tag naming its us-gaap concept, its unit
and its period. The AAPL FY2025 10-K carries 969 of them. `app/parser.py` then
calls `soup.get_text()`, which keeps `416,161` and throws the tag away — which
is precisely why numbers reach the chunk corpus as prose, and why the segment
table retrieves as `Americas / $ / 178,353 / 7 / %`.

This module takes the same audited figures from SEC's `companyfacts` API,
where they arrive already extracted and spanning every filing a company has
made — 2009 to today. Three things follow that RAG cannot do:

- a **time series**, where a single filing shows only two or three comparative
  years;
- **cross-company comparison** by SQL rather than four retrievals of four
  differently-worded tables;
- a **stronger citation**, because each fact carries the accession number it
  was reported in, which names the exact filing rather than a chunk of one.

Layered like `app/edgar.py`: `fetch_company_facts` does the HTTP, `parse_facts`
is pure and takes the payload, so the parsing rules below are testable offline
against a hand-built fixture rather than against the network.
"""

import logging
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from app.sec_http import sec_get

logger = logging.getLogger(__name__)

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# us-gaap only. The `dei` taxonomy in the same payload carries entity facts
# (shares outstanding, public float) that a market-cap calculation would want;
# nothing in F needs them, so they are skipped rather than stored on spec.
TAXONOMY = "us-gaap"

# Matching the text corpus. companyfacts also carries facts first reported on
# 8-K and S-1; 8-K is deferred, and mixing forms here would put earnings-release
# figures next to audited ones with nothing distinguishing them but a column.
FORMS = ("10-K", "10-Q")

# **Period classification is by span, never by the payload's `fp` field.**
# A 52/53-week fiscal year is 364 or 371 days, so the annual window has to be
# wider than "365". The gap between the windows is the point: filers emit
# six- and nine-month cumulative facts (~180 and ~270 days) that would
# otherwise land in a quarterly series looking exactly like quarters.
ANNUAL_DAYS = (300, 400)
QUARTER_DAYS = (60, 100)


@dataclass(frozen=True)
class Fact:
    """One reported figure, with the period it covers and the filing it came from."""

    cik: str
    ticker: str
    concept: str
    unit: str
    period_start: date | None      # None for instant facts
    period_end: date
    period_type: str               # 'annual' | 'quarterly' | 'instant'
    calendar_year: int
    value: Decimal
    form: str
    accession_number: str
    filed_date: date

    @property
    def label(self) -> str:
        """How a period is named to a reader — always by its end date.

        Never "FY2023". The payload's own fiscal-year field belongs to the
        filing, not the fact (see `parse_facts`), and this project's own
        `calendar_year` is an approximation of the filer's label. An end date
        is the one description that is always exactly true.
        """
        if self.period_type == "instant":
            return f"as of {self.period_end.isoformat()}"
        return f"{self.period_type} period ending {self.period_end.isoformat()}"


def fetch_company_facts(cik: str) -> dict:
    """Every XBRL fact SEC holds for one filer, as raw JSON.

    One request covering the company's whole filing history — ~3.8 MB and
    ~25,000 facts for a large filer. Goes through `sec_http.sec_get`, so the
    required User-Agent and the 3 req/s limit are already applied and there is
    no second rate limiter to keep in step.
    """
    url = COMPANY_FACTS_URL.format(cik=cik)
    payload = sec_get(url, accept="application/json").json()
    logger.debug("companyfacts %s: %d taxonomies", cik, len(payload.get("facts", {})))
    return payload


def _period_type(start: date | None, end: date) -> str | None:
    """'annual' | 'quarterly' | 'instant', or None for a span to discard."""
    if start is None:
        return "instant"
    days = (end - start).days
    if ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]:
        return "annual"
    if QUARTER_DAYS[0] <= days <= QUARTER_DAYS[1]:
        return "quarterly"
    return None


def _as_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def parse_facts(payload: dict, ticker: str) -> list[Fact]:
    """Flatten a companyfacts payload into deduplicated `Fact`s.

    **The payload's `fy` and `fp` describe the filing, not the fact, and using
    them is the trap this function exists to avoid.** A 10-K restates its two
    prior years, so AAPL's FY2025 filing stamps `fy=2025` onto facts whose
    periods are 2022-23, 2023-24 and 2024-25. Reading `fy` as the fact's year
    labels 2023's revenue as FY2025 — a wrong number that looks entirely
    plausible. Everything here is derived from `start` and `end` instead.

    **Deduplication keeps the latest filing.** The same period is reported
    again by every filing that restates it, so one company yields ~25,000 raw
    facts and ~12,000 distinct ones. Latest-filed wins, so a restatement
    supersedes the original — and the tie-break is the accession number, so
    two filings on one day resolve the same way on every run rather than by
    dict ordering.
    """
    cik = str(payload.get("cik", "")).zfill(10)
    concepts = payload.get("facts", {}).get(TAXONOMY, {})

    # key -> (sort key, Fact). Built as a dict rather than filtered afterwards
    # so the raw list never has to be held twice.
    best: dict[tuple, tuple[tuple, Fact]] = {}
    skipped_span = 0

    for concept, body in concepts.items():
        for unit, rows in (body.get("units") or {}).items():
            for row in rows:
                if row.get("form") not in FORMS:
                    continue
                end = _as_date(row.get("end"))
                if end is None or row.get("val") is None:
                    continue
                start = _as_date(row.get("start"))
                period_type = _period_type(start, end)
                if period_type is None:
                    skipped_span += 1
                    continue

                filed = _as_date(row.get("filed"))
                accession = row.get("accn") or ""
                if filed is None or not accession:
                    # Without these there is no citation and no way to resolve a
                    # restatement, which are the two reasons to keep the row.
                    continue

                key = (concept, unit, start, end)
                rank = (filed, accession)
                if key in best and best[key][0] >= rank:
                    continue

                best[key] = (
                    rank,
                    Fact(
                        cik=cik,
                        ticker=ticker.upper(),
                        concept=concept,
                        unit=unit,
                        period_start=start,
                        period_end=end,
                        period_type=period_type,
                        calendar_year=end.year,
                        # str() first: some values arrive as floats, and
                        # Decimal(float) carries the binary rounding into a
                        # column whose whole job is to be exact.
                        value=Decimal(str(row["val"])),
                        form=row["form"],
                        accession_number=accession,
                        filed_date=filed,
                    ),
                )

    facts = [fact for _, fact in best.values()]
    logger.debug(
        "parsed %d fact(s) for %s from %d concept(s); %d skipped on span",
        len(facts), ticker, len(concepts), skipped_span,
    )
    return facts
