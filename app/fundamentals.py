"""F2 · Reading `company_facts` as financial statements.

`app/xbrl.py` fetches and parses; this reads back and arranges. The split
mirrors `edgar.py` → `retriever.py`: one module talks to SEC, another answers
questions over what was stored.

What it adds is the **statement mapping** — turning us-gaap tags into the lines
a person recognises. That mapping is the fiddliest thing in this module and the
most likely to be wrong on a company nobody has tried yet, because filers tag
the same line differently: Apple reports revenue as
`RevenueFromContractWithCustomerExcludingAssessedTax`, others as `Revenues`,
older filings as `SalesRevenueNet`. Each line therefore carries an ordered list
of candidate tags and takes the first with data.

**Which tag matched is reported, not hidden.** A line silently falling through
to a second-choice tag is how a plausible wrong number gets into an answer, so
`StatementRow.concept` records what actually resolved and the gate prints it.

**An empty line has two very different causes, and they must not be conflated.**
Measured across 187 backfilled companies: "Gross profit" resolves for 47% and
"Dividends paid" for 80% — but Accenture genuinely has no gross-profit line,
Allstate has no R&D, and Amazon pays no dividend. Those are *correct* absences,
and showing them as blanks is right. A genuine mapping gap looks different: the
company reports the line in its filings under a tag no candidate names, the way
NVIDIA's capital expenditure moved from `PaymentsToAcquirePropertyPlantAndEquipment`
to `PaymentsToAcquireProductiveAssets` after FY2020 and silently emptied five
years of the row. So the gate asserts only the lines that are genuinely
universal — revenue, total assets, net income, operating cash flow — and prints
the rest for a human to read.

**Nothing here is derived.** AMD and AbbVie never tag `Liabilities`; total
liabilities could be computed as `LiabilitiesAndStockholdersEquity` minus
equity, and that arithmetic would be correct — but it would put a number in a
statement that appears in no filing, in a system whose entire claim is that
every figure is reported and traceable to one. The line stays blank instead.
"""

import logging
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from app.db import get_conn

logger = logging.getLogger(__name__)

# (display label, candidate us-gaap tags in preference order)
StatementLine = tuple[str, tuple[str, ...]]

INCOME_STATEMENT: tuple[StatementLine, ...] = (
    ("Revenue", ("RevenueFromContractWithCustomerExcludingAssessedTax",
                 "Revenues", "SalesRevenueNet")),
    ("Cost of revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold",
                         "CostOfGoodsSold")),
    ("Gross profit", ("GrossProfit",)),
    ("Research & development", ("ResearchAndDevelopmentExpense",)),
    ("Selling, general & admin", ("SellingGeneralAndAdministrativeExpense",
                                  "GeneralAndAdministrativeExpense")),
    ("Operating income", ("OperatingIncomeLoss",)),
    ("Pre-tax income", (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    )),
    ("Income tax", ("IncomeTaxExpenseBenefit",)),
    ("Net income", ("NetIncomeLoss",)),
    ("Diluted EPS", ("EarningsPerShareDiluted",)),
)

BALANCE_SHEET: tuple[StatementLine, ...] = (
    ("Cash & equivalents", ("CashAndCashEquivalentsAtCarryingValue",)),
    ("Short-term investments", ("ShortTermInvestments",
                                "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
                                "MarketableSecuritiesCurrent",
                                "OtherShortTermInvestments")),
    ("Total current assets", ("AssetsCurrent",)),
    ("Total assets", ("Assets",)),
    ("Total current liabilities", ("LiabilitiesCurrent",)),
    ("Long-term debt", ("LongTermDebtNoncurrent", "LongTermDebt",
                        "LongTermDebtAndCapitalLeaseObligations")),
    ("Total liabilities", ("Liabilities",)),
    ("Shareholders' equity", (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    )),
)

CASH_FLOW: tuple[StatementLine, ...] = (
    ("Operating cash flow", (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    )),
    # NVDA is the worked example for why these lists exist: it used
    # PaymentsToAcquirePropertyPlantAndEquipment until FY2020 and
    # PaymentsToAcquireProductiveAssets after, so a single-tag line would show
    # a company with no capital spending for five years rather than an error.
    ("Capital expenditure", ("PaymentsToAcquirePropertyPlantAndEquipment",
                             "PaymentsToAcquireProductiveAssets",
                             "PaymentsForCapitalImprovements",
                             "PaymentsToAcquireOtherPropertyPlantAndEquipment")),
    ("Investing cash flow", ("NetCashProvidedByUsedInInvestingActivities",)),
    ("Financing cash flow", ("NetCashProvidedByUsedInFinancingActivities",)),
    ("Dividends paid", ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends")),
    ("Share repurchases", ("PaymentsForRepurchaseOfCommonStock",)),
)

STATEMENTS: dict[str, tuple[StatementLine, ...]] = {
    "income": INCOME_STATEMENT,
    "balance": BALANCE_SHEET,
    "cash_flow": CASH_FLOW,
}

# **Balance-sheet facts are instants, not durations**, so they carry no period
# type of their own — a balance sheet is a photograph, an income statement is a
# film. The requested `period` therefore selects *which* instants: the ones
# dated at the annual or quarterly period ends.
INSTANT_STATEMENTS = ("balance",)

VALID_PERIODS = ("annual", "quarterly")

# A concept can be reported in more than one unit. Preference order, so a
# per-share figure is not mistaken for a dollar total.
_UNIT_PRIORITY = ("USD", "USD/shares", "shares", "pure")

MAX_PERIODS = 12


@dataclass(frozen=True)
class StatementRow:
    """One line, possibly assembled from more than one tag.

    `concepts` is every tag that contributed, in the order they were consulted.
    More than one is normal and correct: a filer that re-tags a line partway
    through its history reports the same economic quantity under two names, and
    NVIDIA does exactly this for both revenue and capital expenditure.
    """

    label: str
    concepts: tuple[str, ...]           # empty = nothing resolved
    unit: str | None
    values: dict[date, Decimal]

    @property
    def concept(self) -> str | None:
        """The primary tag — the first that contributed."""
        return self.concepts[0] if self.concepts else None

    @property
    def mixed(self) -> bool:
        """Whether this line drew on more than one tag. Worth surfacing: it is
        usually a re-tagging, but it is also what a wrong candidate list looks
        like."""
        return len(self.concepts) > 1


@dataclass(frozen=True)
class Statement:
    ticker: str
    # Carried because a citation URL is built from cik + accession, and an
    # accession's own prefix is the *filing agent's* CIK, not the company's —
    # Meta's filings begin 0001628280, which is its filer's id, not Meta's.
    cik: str
    statement: str
    period: str
    periods: list[date]                 # newest first
    rows: list[StatementRow]
    sources: dict[date, tuple[str, str, date]]   # period -> (accn, form, filed)

    @property
    def resolved(self) -> list[StatementRow]:
        return [row for row in self.rows if row.concept]

    @property
    def unresolved(self) -> list[str]:
        """Lines no candidate tag could fill — the mapping's own failure list."""
        return [row.label for row in self.rows if not row.concept]


# **A ticker names the *current* registrant, which is not always the company's
# whole history.** SEC's ticker file maps XOM to CIK 2115436, "ExxonMobil
# Holdings Corp" — a successor entity created in a reorganisation — not to CIK
# 34088, which holds decades of Exxon filings. Measured 2026-08-19: 3 of 386
# backfilled companies are in this position (XOM, HONA, CBRS), all with data
# beginning in 2024.
#
# Nothing here tries to follow the chain. SEC publishes `formerNames` on the
# submissions endpoint but no predecessor-CIK link, so resolving it would mean
# guessing at a name match — and a wrong guess attributes one company's numbers
# to another, which is worse than a short history. `get_financials` reports the
# absence instead.


# A 10-Q lands roughly every 90 days, so a ticker whose newest fact was filed
# longer ago than this has almost certainly filed something we do not hold.
#
# Derived from `filed_date` rather than from a `fetched_at` column, for the
# same reason periods are derived from `start`/`end` and never from the
# payload's own `fy`: a stored timestamp is a second source of truth that can
# disagree with the facts, and when it does it wins silently.
REFETCH_AFTER_DAYS = 100


def ensure_facts(ticker: str) -> bool:
    """Make sure this ticker's facts are present and not obviously behind SEC.

    **Nothing is backfilled.** One `companyfacts` request buys a company's
    entire filing history, so a stored universe would only ever be a subset of
    what is freely available — and gating on it reports "no coverage" for a
    company one free call would answer. What is already in `company_facts` is
    kept as a warm start; anything else is fetched the first time it is asked
    for and written back.

    Returns False when SEC genuinely has nothing: a filer with no XBRL, or a
    successor-CIK case like XOM (see the note above). **Never raises** — a
    fetch failure returns False so `get_financials` reports an absence rather
    than surfacing a traceback, per the tool-failures-are-text rule.

    Re-fetching is safe and needs no reconciliation: `store.upsert_facts`
    supersedes only on a later `filed_date`, so a restatement wins and an
    unchanged filing writes nothing.
    """
    # Imported here rather than at module scope: `app.tickers` and `app.xbrl`
    # both reach the network layer, and `app.store` imports `app.xbrl` for the
    # Fact type. Keeping them local preserves this module's property of being
    # importable, and unit-testable, without any of that being reachable.
    from app.store import upsert_facts
    from app.tickers import try_resolve_ticker
    from app.xbrl import fetch_company_facts, parse_facts

    ticker = ticker.upper()
    with get_conn() as conn:
        newest = conn.execute(
            "SELECT max(filed_date) FROM company_facts WHERE ticker = %s",
            (ticker,),
        ).fetchone()[0]

    if newest is not None and (date.today() - newest).days < REFETCH_AFTER_DAYS:
        return True

    # A ticker that genuinely stopped filing — delisted, acquired — re-fetches
    # on every call, because max(filed_date) never advances again. One SEC
    # request, and rare; a negative marker row would fix it and is not worth
    # building on speculation.
    try:
        company = try_resolve_ticker(ticker)
        if company is None:
            return newest is not None
        facts = parse_facts(fetch_company_facts(company.cik), ticker)
        if facts:
            with get_conn() as conn:
                upsert_facts(conn, facts)
            logger.info("fetched %d fact(s) for %s on demand", len(facts), ticker)
    except Exception as exc:  # noqa: BLE001 - reported as an absence, not raised
        logger.warning("companyfacts fetch failed for %s: %s", ticker, exc)
        # Stale beats nothing: if rows are already held, serve them.
        return newest is not None

    return newest is not None or bool(facts)


def _period_ends(conn, ticker: str, period: str, limit: int) -> list[date]:
    """The N most recent *completed* period ends of the requested duration type.

    **The `CURRENT_DATE` bound is not defensive tidying.** XBRL carries
    forward-dated duration facts that are assumptions rather than results —
    Nucor tags `DefinedBenefitPlanUltimateHealthCareCostTrendRate` over
    2027-01-01 to 2027-12-31, in a filing from 2011. Without this clause that
    date is Nucor's most recent "annual period", so a four-year income
    statement renders a blank 2027 column and silently drops a real year off
    the bottom. A period that has not ended cannot have a statement.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT period_end FROM company_facts
        WHERE ticker = %s AND period_type = %s AND period_end <= CURRENT_DATE
        ORDER BY period_end DESC LIMIT %s
        """,
        (ticker.upper(), period, limit),
    ).fetchall()
    return [r[0] for r in rows]


def _pick_unit(units: dict[str, dict[date, Decimal]]) -> str:
    """The unit to read a concept in, when it was reported in several."""
    return min(units, key=lambda u: (_UNIT_PRIORITY.index(u)
                                     if u in _UNIT_PRIORITY else 99))


def _resolve_line(
    label: str,
    candidates: tuple[str, ...],
    by_concept: dict[str, dict[str, dict[date, Decimal]]],
    periods: list[date],
) -> StatementRow:
    """Fill one line from its candidate tags, **period by period**.

    The obvious rule — take the first candidate that has any data — is wrong,
    and wrong in the way that matters here. NVIDIA reports revenue under
    `RevenueFromContractWithCustomerExcludingAssessedTax` for one early year
    and under `Revenues` for every year since; first-with-any-data locks onto
    the early tag and renders a company with four blank revenue years and a
    full gross-profit row beneath it. Filling per period instead lets a
    re-tagged line reassemble into the continuous series it actually is.

    Preference order still decides who wins a period both tags cover, and a
    candidate reported in a different unit from the one already chosen is
    skipped rather than mixed — that would put dollars and dollars-per-share
    in one row.
    """
    values: dict[date, Decimal] = {}
    used: list[str] = []
    unit: str | None = None

    for concept in candidates:
        units = by_concept.get(concept)
        if not units:
            continue
        concept_unit = _pick_unit(units)
        if unit is None:
            unit = concept_unit
        elif concept_unit != unit:
            continue

        contributed = False
        for period_end in periods:
            if period_end in values:
                continue
            value = units[concept_unit].get(period_end)
            if value is not None:
                values[period_end] = value
                contributed = True
        if contributed:
            used.append(concept)
        if len(values) == len(periods):
            break

    return StatementRow(label, tuple(used), unit if used else None, values)


def load_statement(
    ticker: str,
    statement: str = "income",
    period: str = "annual",
    limit: int = 5,
) -> Statement:
    """One statement, newest period first.

    Two queries rather than one per line: pull every candidate tag at once,
    then resolve the mapping in Python. A per-line query would be forty round
    trips for a statement, inside a tool call that already sits behind a model
    turn.
    """
    ticker = ticker.upper()
    ensure_facts(ticker)
    lines = STATEMENTS[statement]
    limit = max(1, min(limit, MAX_PERIODS))
    concepts = [c for _, candidates in lines for c in candidates]

    with get_conn() as conn:
        periods = _period_ends(conn, ticker, period, limit)
        if not periods:
            return Statement(ticker, "", statement, period, [], [], {})

        # An instant statement is dated at the duration periods' end dates, so
        # the two kinds of statement line up in one table.
        fact_type = "instant" if statement in INSTANT_STATEMENTS else period
        rows = conn.execute(
            """
            SELECT concept, unit, period_end, value, accession_number, form,
                   filed_date, cik
            FROM company_facts
            WHERE ticker = %s AND period_type = %s
              AND concept = ANY(%s) AND period_end = ANY(%s)
            """,
            (ticker, fact_type, concepts, periods),
        ).fetchall()

    # concept -> unit -> {period_end: value}
    by_concept: dict[str, dict[str, dict[date, Decimal]]] = {}
    sources: dict[date, tuple[str, str, date]] = {}
    cik = ""
    for concept, unit, period_end, value, accession, form, filed, row_cik in rows:
        cik = cik or row_cik
        by_concept.setdefault(concept, {}).setdefault(unit, {})[period_end] = value
        # One source per period. Whichever filing reported it is the citation;
        # the parser already kept only the latest, so there is no choice to make.
        sources.setdefault(period_end, (accession, form, filed))

    built = [
        _resolve_line(label, candidates, by_concept, periods)
        for label, candidates in lines
    ]

    statement_obj = Statement(ticker, cik, statement, period, periods, built, sources)
    if statement_obj.unresolved:
        logger.info(
            "%s %s: no tag resolved for %s",
            ticker, statement, ", ".join(statement_obj.unresolved),
        )
    return statement_obj


def load_concept(
    ticker: str, concept: str, period: str = "annual", limit: int = 5
) -> Statement:
    """One named us-gaap tag as a series — the escape hatch from the mapping.

    Exists because the curated statements cover the common lines and a real
    question sometimes wants a specific one (`DeferredRevenueCurrent`,
    `OperatingLeaseLiability`). Reuses the statement machinery so the output
    shape, the citations and the period labelling are identical.
    """
    ticker = ticker.upper()
    ensure_facts(ticker)
    with get_conn() as conn:
        periods = _period_ends(conn, ticker, period, max(1, min(limit, MAX_PERIODS)))
        if not periods:
            return Statement(ticker, "", concept, period, [], [], {})
        rows = conn.execute(
            """
            SELECT unit, period_end, value, accession_number, form, filed_date, cik
            FROM company_facts
            WHERE ticker = %s AND concept = %s
              AND period_end = ANY(%s) AND period_type IN (%s, 'instant')
            """,
            (ticker, concept, periods, period),
        ).fetchall()

    units: dict[str, dict[date, Decimal]] = {}
    sources: dict[date, tuple[str, str, date]] = {}
    cik = ""
    for unit, period_end, value, accession, form, filed, row_cik in rows:
        cik = cik or row_cik
        units.setdefault(unit, {})[period_end] = value
        sources.setdefault(period_end, (accession, form, filed))

    if not units:
        return Statement(ticker, "", concept, period, periods, [
            StatementRow(concept, (), None, {})
        ], {})

    unit = min(units, key=lambda u: (_UNIT_PRIORITY.index(u)
                                     if u in _UNIT_PRIORITY else 99))
    return Statement(
        ticker, cik, concept, period, periods,
        [StatementRow(concept, (concept,), unit, units[unit])], sources,
    )
