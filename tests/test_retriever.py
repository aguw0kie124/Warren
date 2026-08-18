"""Offline tests for retrieval plumbing.

Ranking quality needs a live corpus and a human reading the output — that's
scripts/check_retriever.py. What's testable offline is the part that silently
corrupts results without failing: filter construction, and row-to-Result
mapping. A8 adds the RRF fusion maths here.
"""

from datetime import date

from app.retriever import Result, _filters, _to_results, _where

ROW = (
    "0000320193-25-000079", 3, "Item 1A Risk Factors", "Our business is subject to risk.",
    "AAPL", "Apple Inc.", "10-K", 2025, date(2025, 10, 31),
    "https://www.sec.gov/Archives/edgar/data/320193/x.htm", 0.61,
)


def test_no_filters_produces_no_where_clause():
    clauses, params = _filters(None, None, None, None)
    assert clauses == []
    assert params == []
    assert _where(clauses) == ""


def test_ticker_is_uppercased():
    # Tickers are stored uppercase; a lowercase filter would silently match
    # nothing and look like "no relevant results" rather than a bug.
    _, params = _filters("aapl", None, None, None)
    assert params == ["AAPL"]


def test_form_type_is_uppercased():
    _, params = _filters(None, None, "10-k", None)
    assert params == ["10-K"]


def test_each_filter_contributes_one_clause_and_one_param():
    clauses, params = _filters("AAPL", "Item 7 MD&A", "10-K", 2025)
    assert len(clauses) == len(params) == 4
    assert params == ["AAPL", "Item 7 MD&A", "10-K", 2025]


def test_clauses_and_params_stay_in_step():
    # Order matters: params are positional, so a clause emitted without its
    # parameter (or vice versa) shifts every later value by one.
    clauses, params = _filters("AAPL", None, "10-Q", None)
    assert clauses == ["f.ticker = %s", "f.form_type = %s"]
    assert params == ["AAPL", "10-Q"]


def test_where_joins_with_and():
    assert _where(["a = %s", "b = %s"]) == "WHERE a = %s AND b = %s"


def test_row_maps_onto_result_fields_in_order():
    # The SELECT column order and this mapping have to move together; a
    # mismatch yields plausible-looking results with swapped metadata.
    result = _to_results([ROW])[0]
    assert result.ticker == "AAPL"
    assert result.section == "Item 1A Risk Factors"
    assert result.chunk_index == 3
    assert result.fiscal_year == 2025
    assert result.score == 0.61
    assert result.source_url.endswith("x.htm")


def test_citation_is_assembled_from_stored_metadata():
    assert _to_results([ROW])[0].citation == (
        "[Apple Inc. 10-K, Item 1A Risk Factors, filed 2025-10-31]"
    )


def test_empty_rows_map_to_empty_results():
    assert _to_results([]) == []


def test_result_is_immutable():
    # Results flow into citation assembly downstream; letting a caller mutate
    # one would let a citation drift from the row it came from.
    result = _to_results([ROW])[0]
    try:
        result.score = 1.0
    except Exception as exc:
        assert type(exc).__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("Result should be frozen")


def test_result_dataclass_exposes_every_citation_field():
    fields = set(Result.__dataclass_fields__)
    assert {"company_name", "form_type", "section", "filing_date", "source_url"} <= fields
