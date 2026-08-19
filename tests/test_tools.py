"""Offline tests for the tool layer.

Two properties are worth pinning here, and neither is about answer quality —
that is scripts/check_tools.py's job.

The first is *pass-through*: C1 is plumbing, so the arguments a tool receives
must reach the underlying function unchanged. If they don't, every later
oddity looks like a model problem when it is a wiring problem.

The second is the **corpus gap**, which is the one place C1 adds behaviour
rather than formatting. It fails invisibly: an uningested ticker and a company
that genuinely disclosed nothing both retrieve zero rows, and only the text of
the result tells them apart.
"""

from datetime import date, datetime, timezone

import pytest

from app import finnhub, retriever, tools, websearch
from app.tools import TOOLS, Citation, VALID_SECTIONS


def call(t, **kwargs):
    """Invoke a tool the way ToolNode does, returning the ToolMessage.

    Invoking with a plain dict returns only the content; a tool-call dict is
    what surfaces the artifact, which is where the citations live.
    """
    return t.invoke({"name": t.name, "args": kwargs, "id": "call-1", "type": "tool_call"})


def result(**overrides) -> retriever.Result:
    fields = dict(
        accession_number="0000320193-24-000123",
        chunk_index=0,
        section="Item 1A Risk Factors",
        content="Our business is subject to a variety of risks.",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type="10-K",
        fiscal_year=2025,
        filing_date=date(2024, 11, 1),
        source_url="https://www.sec.gov/Archives/edgar/data/320193/x.htm",
        score=0.5,
    )
    return retriever.Result(**{**fields, **overrides})


@pytest.fixture
def covered(monkeypatch):
    """Pretend AAPL and MSFT are ingested, without touching Postgres."""
    monkeypatch.setattr(tools, "_covered_tickers", lambda: {"AAPL", "MSFT"})


@pytest.fixture
def spy_hybrid(monkeypatch):
    """Record what search_filings passes down, and control what comes back."""
    captured: dict = {}
    returns: list = [result()]

    def fake(query, **kwargs):
        captured.clear()
        captured.update(query=query, **kwargs)
        return list(returns)

    monkeypatch.setattr(retriever, "hybrid_search", fake)
    return captured, returns


# --- the corpus gap ---------------------------------------------------------


def test_uningested_ticker_reports_a_corpus_gap(covered, spy_hybrid):
    captured, _ = spy_hybrid
    message = call(tools.search_filings, query="risk factors", ticker="NVDA")

    assert "CORPUS GAP" in message.content
    assert "NVDA" in message.content
    # The covered companies are named so the agent can offer an alternative.
    assert "AAPL" in message.content and "MSFT" in message.content
    assert message.artifact == []
    # And crucially: no search was run, because there was nothing to search.
    assert captured == {}


def test_empty_corpus_reports_a_gap_even_without_a_ticker(monkeypatch, spy_hybrid):
    monkeypatch.setattr(tools, "_covered_tickers", set)
    message = call(tools.search_filings, query="risk factors")
    assert "CORPUS GAP" in message.content


def test_covered_ticker_with_no_matches_is_not_a_corpus_gap(covered, spy_hybrid):
    _, returns = spy_hybrid
    returns.clear()

    message = call(tools.search_filings, query="lunar mining", ticker="AAPL")

    assert "CORPUS GAP" not in message.content
    assert "No passages matched" in message.content
    # The distinction the whole check exists for, stated for the model.
    assert "genuine miss, not missing data" in message.content
    assert message.artifact == []


# --- pass-through -----------------------------------------------------------


def test_filters_reach_the_retriever_unchanged(covered, spy_hybrid):
    captured, _ = spy_hybrid
    call(
        tools.search_filings,
        query="revenue growth",
        ticker="aapl",
        section="Item 7 MD&A",
        form_type="10-K",
        fiscal_year=2025,
    )

    assert captured == {
        "query": "revenue growth",
        "ticker": "AAPL",  # normalised for the model, which will not be reliable
        "section": "Item 7 MD&A",
        "form_type": "10-K",
        "fiscal_year": 2025,
        "k": retriever.DEFAULT_K,
    }


def test_k_is_clamped(covered, spy_hybrid):
    captured, _ = spy_hybrid
    call(tools.search_filings, query="risks", ticker="AAPL", k=500)
    assert captured["k"] == tools.MAX_K


def test_news_dates_reach_finnhub_unchanged(monkeypatch):
    captured: dict = {}

    def fake(symbol, from_date=None, to_date=None, limit=None):
        captured.update(symbol=symbol, from_date=from_date, to_date=to_date, limit=limit)
        return []

    monkeypatch.setattr(finnhub, "get_company_news", fake)
    call(tools.get_company_news, symbol="AAPL", from_date="2026-08-01", limit=3)

    assert captured == {
        "symbol": "AAPL", "from_date": "2026-08-01", "to_date": None, "limit": 3
    }


def test_days_reaches_web_search(monkeypatch):
    captured: dict = {}

    def fake(query, days=None):
        captured.update(query=query, days=days)
        return []

    monkeypatch.setattr(websearch, "web_search", fake)
    call(tools.web_search, query="analyst views on AI capex", days=7)

    assert captured == {"query": "analyst views on AI capex", "days": 7}


# --- citations --------------------------------------------------------------


def test_filing_citations_are_typed_and_aligned(covered, spy_hybrid):
    _, returns = spy_hybrid
    returns[:] = [result(chunk_index=i) for i in range(3)]

    message = call(tools.search_filings, query="risks", ticker="AAPL")

    assert len(message.artifact) == 3
    assert all(isinstance(c, Citation) and c.type == "filing" for c in message.artifact)
    # Marker [n] indexes into the artifact list position for position.
    for i, citation in enumerate(message.artifact, start=1):
        assert f"[{i}] {citation.label}" in message.content
    assert message.artifact[0].label == (
        "Apple Inc. 10-K, Item 1A Risk Factors, filed 2024-11-01"
    )
    assert message.content.count("Our business is subject to") == 3
    assert "as of their filing date" in message.content


def test_news_citations_are_typed(monkeypatch):
    article = finnhub.NewsItem(
        symbol="AAPL",
        headline="Apple services revenue beats",
        summary="Summary text.",
        source="Reuters",
        url="https://finnhub.io/api/news?id=abc",
        datetime=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(finnhub, "get_company_news", lambda *a, **k: [article])

    message = call(tools.get_company_news, symbol="AAPL")

    assert [c.type for c in message.artifact] == ["news"]
    assert message.artifact[0].label == "Reuters — Apple services revenue beats"
    assert message.artifact[0].source_url == article.url


def test_web_citations_are_typed(monkeypatch):
    hit = websearch.WebResult(
        title="Analysts on Apple's AI strategy",
        url="https://www.wsj.com/tech/apple-ai",
        content="Extracted article text.",
        score=0.9,
    )
    monkeypatch.setattr(websearch, "web_search", lambda *a, **k: [hit])

    message = call(tools.web_search, query="Apple AI strategy")

    assert [c.type for c in message.artifact] == ["web"]
    assert message.artifact[0].label == "wsj.com — Analysts on Apple's AI strategy"


def test_price_tools_produce_no_citations(monkeypatch):
    quote = finnhub.Quote(
        symbol="AAPL", c=226.01, d=1.23, dp=0.55, h=227.1, l=224.5, o=225.0,
        pc=224.78, t=datetime(2026, 8, 17, 20, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(finnhub, "get_quote", lambda s: quote)

    message = call(tools.get_quote, symbol="AAPL")

    assert message.artifact == []
    assert "226.01" in message.content
    assert "2026-08-17 20:00 UTC" in message.content


# --- failures are text, not tracebacks --------------------------------------


def test_unknown_symbol_is_reported_not_raised(monkeypatch):
    def boom(symbol):
        raise finnhub.UnknownSymbolError(symbol, "/quote")

    monkeypatch.setattr(finnhub, "get_quote", boom)
    message = call(tools.get_quote, symbol="APPL")

    assert "APPL" in message.content
    assert message.artifact == []
    # A raised tool error would set this; the point is that it does not.
    assert message.status == "success"


def test_unknown_section_lists_the_valid_ones(covered, spy_hybrid):
    captured, _ = spy_hybrid
    message = call(tools.search_filings, query="risks", ticker="AAPL",
                   section="Risk Factors")

    assert all(section in message.content for section in VALID_SECTIONS)
    assert captured == {}  # rejected before spending a search


def test_missing_api_key_is_reported(monkeypatch):
    def boom(query, days=None):
        raise websearch.MissingApiKeyError("TAVILY_API_KEY is not set.")

    monkeypatch.setattr(websearch, "web_search", boom)
    message = call(tools.web_search, query="anything")

    assert "TAVILY_API_KEY" in message.content
    assert message.artifact == []


# --- the tool surface itself ------------------------------------------------


def test_every_tool_is_registered_with_a_routing_docstring():
    """The whole surface, named rather than counted.

    Named because the number on its own says nothing about what changed, and
    because every addition here moves two things that were measured against
    the old list: the model's routing (C2's gate) and C3's cacheable prefix.
    Adding a tool should require editing this set, and editing it should mean
    re-running scripts/check_agent.py.
    """
    names = {t.name for t in TOOLS}
    assert names == {
        "search_filings", "get_financials", "get_quote", "get_basic_financials",
        "get_company_news", "web_search",
    }
    assert len(TOOLS) == len(names)   # no duplicate registration
    for t in TOOLS:
        # The docstring is the routing logic, so an empty one is a real defect.
        assert t.description and len(t.description) > 200
        assert t.response_format == "content_and_artifact"
