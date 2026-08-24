"""C1 · The tool layer — Modules A and B, exposed to a language model.

Deliberately thin. Every function here delegates to a plain function that
already exists and was verified on its own gate: `retriever.hybrid_search`
(A8), `app.prices` (B1), `app.websearch` (B2). Nothing new is computed. What
this file adds is three things a model needs and a plain function doesn't:

1. **Docstrings that route.** Six tools, four of which plausibly answer
   "what's going on with Apple." There is no router node in the graph — a
   tool-calling model *is* the router — so the docstrings are the routing
   logic, not commentary on it. Each says what its tool is for and when to
   prefer a sibling. If the agent picks the wrong source, this is the file to
   fix.

2. **Citations assembled in code.** Every tool that produces sources returns
   `Citation` objects alongside its text, tagged `filing` / `web`.
   They ride on the ToolMessage as an *artifact*, so the model never sees the
   objects and can never reformat a URL or invent an accession number — the
   retriever and the APIs already know the ground truth.

3. **A corpus-gap result distinct from an empty one.** See `search_filings`.

Failures are reported as text rather than raised. A tool that raises tells the
model "something broke"; a tool that says "Yahoo has no symbol APPL" tells it
what to do next.
"""

import logging
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app import fundamentals, prices, retriever, websearch
from app.db import get_conn
from app.edgar import filing_index_url
from app.parser import SECTIONS_10K, SECTIONS_10Q

logger = logging.getLogger(__name__)

# The exact section vocabulary the parser stores, so the model can be told the
# real values instead of guessing at "Risk Factors" and reading the resulting
# empty result as a fact about the company.
VALID_SECTIONS: tuple[str, ...] = tuple(
    dict.fromkeys(spec.key for spec in (*SECTIONS_10K, *SECTIONS_10Q))
)
VALID_FORM_TYPES: tuple[str, ...] = ("10-K", "10-Q")

# k is exposed because a comparison question legitimately wants more passages
# than a lookup does, but it is clamped: an unbounded k lets one tool call
# swallow the context window.
MAX_K = 12

_STALENESS_NOTE = (
    "Note: filings state the company's position as of their filing date only. "
    "For anything that happened since, use web_search with a `days` window."
)


class Citation(BaseModel):
    """One source, assembled in code from what the tool already knows.

    `type` is what lets an answer distinguish an audited SEC filing from a web
    page — two things that are not equally authoritative, and a reader deserves
    to see which is which.

    There used to be a third, `news`, produced only by a per-ticker Finnhub
    feed. That tool was measured against `web_search` in August 2026 and
    removed: its citation URLs had become `finnhub.io/api/news?id=...`
    redirects rather than article URLs, so the tag claimed a provenance the
    link did not deliver. A type that cannot be trusted to mean what it says is
    worse than one fewer distinction.
    """

    type: Literal["filing", "web"]
    label: str
    source_url: str


def _fail(message: str) -> tuple[str, list[Citation]]:
    """A failure the model should read and act on, not a traceback."""
    logger.info("tool reported failure: %s", message)
    return message, []


def _covered_tickers() -> set[str]:
    """Tickers with at least one ingested filing — the ingestion ledger.

    A plain SELECT against a table that already exists. This lives in the tool
    layer rather than the retriever because it is a question about *coverage*,
    not about relevance, and because it is the point where Phase 2's on-demand
    ingestion will hook in: the gap is the trigger.
    """
    with get_conn() as conn:
        return {row[0] for row in conn.execute("SELECT DISTINCT ticker FROM filings")}


# ---------------------------------------------------------------------------
# search_filings
# ---------------------------------------------------------------------------


class SearchFilingsArgs(BaseModel):
    query: str = Field(
        description="What to look for, in natural language or as an exact "
        "phrase. Exact financial terms work well — retrieval is hybrid, so "
        "'effective tax rate' matches literally as well as semantically."
    )
    ticker: str | None = Field(
        default=None,
        description="Restrict to one company, e.g. 'AAPL'. Almost always "
        "worth setting: without it, passages from other companies' filings "
        "can outrank the ones you want.",
    )
    section: str | None = Field(
        default=None,
        description="Restrict to one filing section. Must be exactly one of: "
        + "; ".join(VALID_SECTIONS)
        + ". Leave unset unless the question is clearly about one of them.",
    )
    form_type: str | None = Field(
        default=None,
        description="Restrict to '10-K' (annual) or '10-Q' (quarterly).",
    )
    fiscal_year: int | None = Field(
        default=None,
        description="Restrict to one fiscal year as the company defines it "
        "(Apple's fiscal year ends in September, so its Oct-Dec quarter is Q1 "
        "of the following fiscal year). Use two calls with different years to "
        "compare how a disclosure changed.",
    )
    k: int = Field(
        default=retriever.DEFAULT_K,
        description=f"How many passages to return (1-{MAX_K}).",
    )


@tool(args_schema=SearchFilingsArgs, response_format="content_and_artifact")
def search_filings(
    query: str,
    ticker: str | None = None,
    section: str | None = None,
    form_type: str | None = None,
    fiscal_year: int | None = None,
    k: int = retriever.DEFAULT_K,
) -> tuple[str, list[Citation]]:
    """Search SEC 10-K and 10-Q filings for what a company officially disclosed.

    This is the authoritative source: audited, filed with the SEC, and the
    company's own statement of its risks, business, and management's
    discussion of results. Prefer it for anything the company itself would
    have said — risk factors, segment descriptions, accounting policies,
    stated outlook, litigation, competitive positioning.

    Its one limitation is time. A filing is current only as of its filing
    date and says nothing about what happened afterwards, so for recent events
    or market reaction use `web_search` with a `days` window, and for the price
    itself `get_quote` or `get_price_history`.

    If the result reports that the company is NOT IN THE FILINGS CORPUS, say
    so plainly in your answer. That means no filings for it have been
    ingested — not that the company said nothing. Do not quietly answer from
    news or web results as if they were the company's own disclosures.
    """
    if not query.strip():
        return _fail("search_filings needs a non-empty query.")

    if section and section not in VALID_SECTIONS:
        return _fail(
            f"Unknown section {section!r}. Valid sections are: "
            f"{'; '.join(VALID_SECTIONS)}. Retry with one of those, or omit "
            f"`section` to search the whole filing."
        )
    if form_type and form_type.upper() not in VALID_FORM_TYPES:
        return _fail(
            f"Unknown form_type {form_type!r}. Valid values are "
            f"{' and '.join(VALID_FORM_TYPES)}."
        )

    covered = _covered_tickers()
    symbol = ticker.strip().upper() if ticker else None

    # The corpus gap. An uningested ticker retrieves nothing, which is
    # indistinguishable from a company that genuinely never discussed the
    # topic — and the failure that follows is not a missing answer but a
    # confident one built entirely on news, with no signal that the
    # authoritative source was absent rather than silent.
    if not covered:
        return _fail(
            "CORPUS GAP: no SEC filings have been ingested at all, so "
            "search_filings cannot answer anything right now. Say so rather "
            "than substituting news or web results for filings."
        )
    if symbol and symbol not in covered:
        return _fail(
            f"CORPUS GAP: {symbol} is not in the filings corpus — no filings "
            f"for it have been ingested, so nothing can be searched. This is "
            f"NOT evidence that {symbol} disclosed nothing. Tell the user the "
            f"filings for {symbol} are unavailable, and do not present news or "
            f"web results as the company's own disclosures. Companies "
            f"currently in the corpus: {', '.join(sorted(covered))}."
        )

    results = retriever.hybrid_search(
        query,
        ticker=symbol,
        section=section,
        form_type=form_type,
        fiscal_year=fiscal_year,
        k=max(1, min(k, MAX_K)),
    )

    filters = _describe_filters(
        ticker=symbol, section=section, form_type=form_type, fiscal_year=fiscal_year
    )
    if not results:
        return (
            f"No passages matched {query!r} in the ingested filings "
            f"({filters}). The corpus does cover this company — this is a "
            f"genuine miss, not missing data. Try a broader query, or drop a "
            f"filter.",
            [],
        )

    # No de-duplication here: the [n] markers below index into the citation
    # list position for position, and collapsing two chunks from the same
    # filing would silently break that alignment. Citations are de-duplicated
    # once, downstream, where the whole answer's list is assembled.
    citations = [
        Citation(
            type="filing",
            label=result.citation.strip("[]"),
            source_url=result.source_url,
        )
        for result in results
    ]

    blocks = [
        f"[{i}] {citation.label}\n{citation.source_url}\n{result.content}"
        for i, (result, citation) in enumerate(zip(results, citations), start=1)
    ]
    header = f"{len(results)} passage(s) for {query!r} ({filters}):"
    return "\n\n".join([header, *blocks, _STALENESS_NOTE]), citations


def _describe_filters(**filters) -> str:
    applied = ", ".join(f"{k}={v}" for k, v in filters.items() if v)
    return applied or "no filters"


# ---------------------------------------------------------------------------
# get_quote / get_key_stats / get_price_history
# ---------------------------------------------------------------------------


class SymbolArgs(BaseModel):
    symbol: str = Field(description="Stock ticker symbol, e.g. 'AAPL'.")


@tool(args_schema=SymbolArgs, response_format="content_and_artifact")
def get_quote(symbol: str) -> tuple[str, list[Citation]]:
    """Get the latest share price and the move from the previous session.

    Use for "what is X trading at", "how did X move today", or whenever an
    answer needs a current price. The price is the last daily close — during
    market hours that is the day's bar in progress, after the close it is the
    closing price — so the returned date is part of the answer. Say when the
    price is from rather than implying it is this instant.

    This is one point in time. For how the price has moved over a period —
    "since the last 10-K", "this year", "over five years" — call
    `get_price_history` instead. Never estimate a move from the day's range.
    """
    try:
        quote = prices.get_quote(symbol)
    except prices.UnknownSymbolError as exc:
        return _fail(str(exc))
    except (prices.PriceDataError, ValueError) as exc:
        return _fail(f"Could not fetch a quote for {symbol!r}: {exc}")

    change = "n/a" if quote.change is None else f"{quote.change:+.2f}"
    percent = "" if quote.percent_change is None else f" ({quote.percent_change:+.2f}%)"
    previous = "n/a" if quote.previous_close is None else f"{quote.previous_close:.2f}"

    text = (
        f"{quote.symbol} quote as of {quote.as_of}:\n"
        f"  price           {quote.price:.2f}\n"
        f"  change          {change}{percent}\n"
        f"  open            {quote.open:.2f}\n"
        f"  day high / low  {quote.high:.2f} / {quote.low:.2f}\n"
        f"  previous close  {previous}"
    )
    # No citation: a quote has no article behind it, and inventing a URL for
    # one would put an unverifiable source in a list whose whole value is that
    # every entry resolves.
    return text, []


@tool(args_schema=SymbolArgs, response_format="content_and_artifact")
def get_key_stats(symbol: str) -> tuple[str, list[Citation]]:
    """Get market-computed valuation ratios and the 52-week range for one ticker.

    Use for "is X expensive", "what is X's P/E", "how far is X off its highs",
    or to put a filing's stated outlook next to what the market currently
    prices in. Covers trailing and forward P/E, price/book, price/sales,
    EV/EBITDA, margins, return on equity, leverage, beta, market cap, dividend
    yield, and the 52-week high and low.

    These are the market-data provider's own figures: computed, point-in-time,
    unaudited, and recomputed every time the price moves. For a number the
    company itself reported, and for how it changed over time, use
    `get_financials` — those are audited and filed. For the company's own
    explanation of it, use `search_filings`.

    A metric the provider does not have is simply absent. A company with no
    earnings has no P/E, and that absence is a fact about the company — do not
    read it as a zero, and do not substitute the forward estimate for it.
    """
    try:
        stats = prices.get_key_stats(symbol)
    except prices.UnknownSymbolError as exc:
        return _fail(str(exc))
    except (prices.PriceDataError, ValueError) as exc:
        return _fail(f"Could not fetch key stats for {symbol!r}: {exc}")

    metrics = stats.present()
    if not metrics:
        return _fail(f"No market metrics are available for {stats.symbol}.")

    lines = "\n".join(f"  {name:<24} {value:,.2f}" for name, value in metrics.items())
    return (
        f"{stats.symbol} key metrics (provider-computed, not from the audited "
        f"filings):\n{lines}",
        [],
    )


class PriceHistoryArgs(BaseModel):
    symbol: str = Field(description="Stock ticker symbol, e.g. 'AAPL'.")
    period: str = Field(
        default=prices.DEFAULT_PERIOD,
        description="How far back to look: "
        + ", ".join(prices.VALID_PERIODS)
        + ". Pick the one that matches the question; default is 1y.",
    )


@tool(args_schema=PriceHistoryArgs, response_format="content_and_artifact")
def get_price_history(
    symbol: str, period: str = prices.DEFAULT_PERIOD
) -> tuple[str, list[Citation]]:
    """Get a summary of how one ticker's price moved over a period.

    Use for "how has X moved since the last 10-K", "how did X do this year",
    "how volatile is X". Returns the first and last close, the percentage
    change between them, the period high and low, annualised volatility, and
    about five evenly spaced closes that trace the shape of the move.

    It is a SUMMARY, not a series. There are no individual daily bars and no
    intraday data, so describe the shape of the move — do not quote a price on
    a day you were not given one for, and do not interpolate between the
    points you were given.

    Closes are adjusted for splits and dividends, so the percentage change is a
    total return and will not match a raw price chart across a split or a large
    dividend. Say "total return" where the difference matters.

    For today's price alone, `get_quote` is one call instead of a year of data.
    For what the company reported over the same years, `get_financials`.
    """
    try:
        summary = prices.get_price_history(symbol, period=period)
    except prices.UnknownSymbolError as exc:
        return _fail(str(exc))
    except (prices.PriceDataError, ValueError) as exc:
        return _fail(f"Could not fetch price history for {symbol!r}: {exc}")

    vol = (
        "n/a"
        if summary.realised_vol_pct is None
        else f"{summary.realised_vol_pct:.1f}% annualised"
    )
    path = "\n".join(f"    {day}  {close:,.2f}" for day, close in summary.anchors)

    text = (
        f"{summary.symbol} price over {summary.period} "
        f"({summary.start_date} to {summary.end_date}, {summary.sessions} sessions):\n"
        f"  start close     {summary.start_close:,.2f}\n"
        f"  latest close    {summary.end_close:,.2f}\n"
        f"  total return    {summary.percent_change:+.2f}%\n"
        f"  period high     {summary.high:,.2f}\n"
        f"  period low      {summary.low:,.2f}\n"
        f"  volatility      {vol}\n"
        f"  path (evenly spaced closes, NOT every session):\n{path}"
    )
    # No citation, for the same reason get_quote returns none.
    return text, []


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


class WebSearchArgs(BaseModel):
    query: str = Field(
        description="What to search for. Write it as a question or a phrase a "
        "financial journalist would use, not as keywords."
    )
    days: int | None = Field(
        default=None,
        description="Restrict to the last N days. Set it for 'what happened "
        "this week' questions; leave unset for durable context like analyst "
        "views or industry structure, where older sources are still good.",
    )


@tool(args_schema=WebSearchArgs, response_format="content_and_artifact")
def web_search(query: str, days: int | None = None) -> tuple[str, list[Citation]]:
    """Search the financial web — news, analyst commentary, and anything recent.

    **This is the news tool.** Use it for "what's the latest on Apple" as well
    as for analyst commentary, competitor and industry trends, macro and
    regulatory events, explanations of a market move, and comparisons spanning
    several companies. Results are restricted to a curated list of reputable
    finance and news domains, so what comes back is citable.

    Set `days` for anything time-scoped — "this week", "the latest", "since
    earnings" — which switches the search to recent news and is what makes a
    news question work. Leave it unset for durable context like strategy or
    industry structure, where a good analysis does not expire in a week. When
    asking about one company, name it AND its ticker: "Apple (AAPL) news".

    For what a company itself officially disclosed, use `search_filings`; for
    its reported numbers, `get_financials`.

    **A result that mentions neither the company nor its ticker is not evidence
    about that company.** This search always returns something plausible for
    any query, including for a company that does not exist — market-wide
    stories, sector commentary, an unrelated firm. Read each result and say
    plainly when nothing found is actually about the company asked for, rather
    than presenting adjacent coverage as though it were.
    """
    try:
        results = websearch.web_search(query, days=days)
    except websearch.MissingApiKeyError as exc:
        return _fail(str(exc))
    except (websearch.WebSearchError, ValueError) as exc:
        return _fail(f"Web search failed for {query!r}: {exc}")

    if not results:
        window = f" in the last {days} day(s)" if days else ""
        return (
            f"No results from the allowed finance sources for {query!r}"
            f"{window}. Try rephrasing, or widening the time window.",
            [],
        )

    citations = [
        Citation(
            type="web",
            label=f"{result.domain} — {result.title}".strip(" —"),
            source_url=result.url,
        )
        for result in results
    ]

    blocks = []
    for i, (result, citation) in enumerate(zip(results, citations), start=1):
        published = f"{result.published_date} · " if result.published_date else ""
        blocks.append(
            f"[{i}] {citation.label}\n{published}{result.url}\n{result.content}"
        )

    header = f"{len(results)} web result(s) for {query!r}:"
    return "\n\n".join([header, *blocks]), citations


# ---------------------------------------------------------------------------
# get_financials
# ---------------------------------------------------------------------------


class FinancialsArgs(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. 'AAPL'.")
    statement: str | None = Field(
        default="income",
        description="Which statement: 'income', 'balance', or 'cash_flow'. "
        "Ignored when `concept` is given.",
    )
    concept: str | None = Field(
        default=None,
        description="A specific us-gaap tag to pull as a series instead of a "
        "whole statement, e.g. 'DeferredRevenueCurrent'. Only use this when "
        "the question asks for a line the three statements do not carry.",
    )
    period: str = Field(
        default="annual",
        description="'annual' or 'quarterly'.",
    )
    limit: int = Field(
        default=5,
        description=f"How many periods, newest first (1-{fundamentals.MAX_PERIODS}).",
    )


def _format_value(value, unit: str) -> str:
    """One figure, scaled by its own unit.

    Per row, never per table: earnings per share arrives as `USD/shares` while
    everything around it is `USD`, so a single scale factor renders $6.08 as
    0.0 next to revenue in billions — which reads as a company that earned
    nothing rather than as a formatting bug.
    """
    number = float(value)
    # Per-share figures and ratios are read at their own magnitude — $4.90 is
    # $4.90, not 4.90 of anything. Counts and dollars both want abbreviating.
    if unit in ("USD/shares", "pure"):
        return f"{number:,.2f}"
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= scale:
            return f"{number / scale:,.2f}{suffix}"
    return f"{number:,.2f}"


def _render(statement: fundamentals.Statement) -> str:
    """The statement as a text table, periods as columns, newest first."""
    header = "".join(f"{p.isoformat():>16}" for p in statement.periods)
    lines = [f"{'':<28}{header}"]
    for row in statement.rows:
        if not row.concept:
            # Printed rather than dropped: a blank line is information — either
            # the company does not report it (Amazon pays no dividend) or the
            # tag mapping missed it. Silently omitting the row hides both.
            lines.append(f"{row.label:<28}" + "".join(f"{'—':>16}" for _ in statement.periods))
            continue
        cells = "".join(
            f"{_format_value(row.values[p], row.unit):>16}" if p in row.values
            else f"{'—':>16}"
            for p in statement.periods
        )
        lines.append(f"{row.label:<28}{cells}")
    return "\n".join(lines)


@tool(args_schema=FinancialsArgs, response_format="content_and_artifact")
def get_financials(
    ticker: str,
    statement: str | None = "income",
    concept: str | None = None,
    period: str = "annual",
    limit: int = 5,
) -> tuple[str, list[Citation]]:
    """Get a company's OWN REPORTED financial statements, as a multi-period series.

    Income statement, balance sheet, or cash flow, annual or quarterly, going
    back to 2009 — the figures the company filed with the SEC and its auditors
    signed. Use this for anything numeric the company itself reported: revenue,
    margins, earnings, debt, cash, capital spending, buybacks, and above all
    for **how any of them changed over time**, which is the one thing no single
    filing can show.

    Prefer this over `get_key_stats` whenever the question is about a
    reported figure or a trend. That tool returns a market-data provider's own
    computed ratios — P/E, beta, the 52-week range — which are useful for what
    the market currently thinks and are NOT audited. This one is what the
    company stated. When a question needs both ("is it expensive given how it
    has grown"), call both and say which number came from where.

    Coverage is wider than the filings corpus: essentially any US filer has
    full financial history here, while `search_filings` reports a corpus gap
    for the same company's risk factors. That is expected, not a
    contradiction — the two cover different things.

    Periods are labelled by their END DATE, not by a fiscal-year number,
    because fiscal years differ between companies and the end date is
    unambiguous. A blank cell means the company did not report that line.
    """
    symbol = ticker.strip().upper()
    # Fetches from SEC on a miss, so this is a real answer about the company
    # rather than about what happens to be stored. False means SEC itself has
    # nothing.
    if not fundamentals.ensure_facts(symbol):
        return _fail(
            f"NO FUNDAMENTALS: SEC publishes no XBRL financial data for {symbol}. "
            f"It may be a foreign private issuer (which files 20-F rather than "
            f"10-K), a recent listing, or a ticker that names a successor entity "
            f"holding none of the predecessor's filings. Say so rather than "
            f"estimating the numbers."
        )

    if period not in fundamentals.VALID_PERIODS:
        return _fail(
            f"Unknown period {period!r}. Valid values are "
            f"{' and '.join(fundamentals.VALID_PERIODS)}."
        )

    if concept:
        result = fundamentals.load_concept(symbol, concept.strip(), period, limit)
        if not result.resolved:
            return _fail(
                f"{symbol} reports no us-gaap concept named {concept!r}. Check the "
                f"tag name, or ask for a whole statement instead."
            )
    else:
        if statement not in fundamentals.STATEMENTS:
            return _fail(
                f"Unknown statement {statement!r}. Valid values are: "
                f"{', '.join(fundamentals.STATEMENTS)}."
            )
        result = fundamentals.load_statement(symbol, statement, period, limit)

    if not result.periods:
        return _fail(
            f"SEC holds no completed {period} periods for {symbol}, so no "
            f"statement can be built. A ticker can name a successor entity that "
            f"holds none of the predecessor's filing history."
        )

    # One citation per filing that reported these periods. Built from the
    # accession, so it resolves even for a company whose filing text was never
    # ingested — which is the normal case here.
    citations = [
        Citation(
            type="filing",
            label=f"{symbol} {form}, XBRL financial data, filed {filed.isoformat()}",
            source_url=filing_index_url(result.cik, accession),
        )
        for accession, form, filed in dict.fromkeys(result.sources.values())
    ]

    what = concept or f"{statement} statement"
    header = f"{symbol} {what}, {period}, as reported to the SEC:"
    footer = (
        "Figures are the company's own audited filings. Blank cells mean the "
        "company did not report that line."
    )
    return "\n".join([header, "", _render(result), "", footer]), citations


# The list C2 binds to the model. Adding a capability to this agent means
# appending one entry here — never touching the graph.
TOOLS = [
    search_filings,
    get_financials,
    get_quote,
    get_key_stats,
    get_price_history,
    web_search,
]


# ---------------------------------------------------------------------------
# Progress labels
# ---------------------------------------------------------------------------
#
# What the streaming endpoint shows a user while the loop runs. It lives here,
# beside the tools, rather than in app/agent.py or app/api.py, because it is
# keyed on tool names and argument names — so a new tool and its label are one
# file apart, not three. `_LABELS` is deliberately not exhaustive-by-assertion:
# an unmapped tool falls through to its own humanised name, which is worse
# writing but never a crash and never a blank line in the UI.
#
# **These are user-facing prose, not tool names.** "Reading AAPL risk factors"
# is a sentence someone can check the agent against; `search_filings(symbol=
# 'AAPL', section='risk_factors')` is an implementation detail leaking into a
# progress log, and a reader cannot tell whether it was the right call.

def _shorten(text: str, limit: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def describe_tool_call(name: str, args: dict | None = None) -> str:
    """A one-line, human-readable label for one tool call.

    Written for the person watching the answer being built, so it names the
    company and the thing being fetched rather than the function and its
    keyword arguments.
    """
    args = args or {}
    symbol = str(args.get("symbol") or args.get("ticker") or "").strip().upper()

    if name == "search_filings":
        # `section` is already prose — VALID_SECTIONS holds "Item 1A Risk
        # Factors", not a slug — so it needs no mapping table to stay readable.
        section = str(args.get("section") or "").strip()
        query = args.get("query")
        what = section or (f"“{_shorten(query, 48)}”" if query else "the filings")
        return f"Reading {symbol or 'SEC'} filings — {what}"

    if name == "get_financials":
        what = args.get("concept") or f"{args.get('statement') or 'income'} statement"
        return f"Pulling {symbol} {what}"

    if name == "get_quote":
        return f"Checking {symbol} price"

    if name == "get_key_stats":
        return f"Fetching {symbol} valuation metrics"

    if name == "get_price_history":
        return f"Charting {symbol} — {args.get('period') or '1y'} price move"

    if name == "web_search":
        return f"Searching the web — “{_shorten(args.get('query') or '', 48)}”"

    # An unmapped tool. Readable, if less specific than the six above.
    return name.replace("_", " ").capitalize()
