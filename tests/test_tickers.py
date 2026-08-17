"""Offline tests for the ticker index. No network — _index_from_payload is pure."""

import pytest

from app.tickers import Company, UnknownTickerError, _index_from_payload

# Shape mirrors EDGAR's company_tickers.json: positional keys, int cik_str.
PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "2": {"cik_str": 1318605, "ticker": "tsla", "title": "Tesla, Inc."},
}


def test_cik_is_zero_padded_to_ten_digits():
    """EDGAR's submissions API 404s on unpadded CIKs, so this is load-bearing."""
    index = _index_from_payload(PAYLOAD)
    assert index["AAPL"].cik == "0000320193"
    assert index["MSFT"].cik == "0000789019"
    assert all(len(c.cik) == 10 for c in index.values())


def test_tickers_are_normalized_to_uppercase():
    index = _index_from_payload(PAYLOAD)
    assert "TSLA" in index
    assert index["TSLA"].name == "Tesla, Inc."


def test_cik_int_strips_padding_for_archive_urls():
    """A3 builds Archives URLs, which use the unpadded form."""
    assert _index_from_payload(PAYLOAD)["AAPL"].cik_int == 320193


def test_entries_with_blank_ticker_are_skipped():
    index = _index_from_payload({"0": {"cik_str": 1, "ticker": "", "title": "Ghost"}})
    assert index == {}


def test_unknown_ticker_error_carries_the_symbol():
    err = UnknownTickerError("ZZZZ")
    assert err.ticker == "ZZZZ"
    assert "ZZZZ" in str(err)


def test_company_is_hashable_and_frozen():
    company = Company(ticker="AAPL", cik="0000320193", name="Apple Inc.")
    assert {company}  # hashable -> usable in sets/dict keys
    with pytest.raises(Exception):
        company.ticker = "MSFT"  # type: ignore[misc]
