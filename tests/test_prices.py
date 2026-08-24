"""Offline tests for app/prices.py.

Correctness against live Yahoo needs the network and belongs in
scripts/check_prices.py. What is testable offline is the part that matters most
and breaks most quietly: the silent-failure guards, the unit conversions, and
the arithmetic.

Everything funnels through `prices._fetch`, so that is the seam these tests
replace — the direct successor to the Finnhub suite's `_get(path, params)` fake.
Fixtures are trimmed copies of real responses captured 2026-08-20.
"""

from datetime import date

import pandas as pd
import pytest

from app import prices
from app.prices import (
    KeyStats,
    PriceDataError,
    UnknownSymbolError,
    anchor_indices,
    get_key_stats,
    get_price_history,
    get_quote,
    realised_vol_pct,
)


def frame(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """A daily-bar frame shaped like yfinance's: tz-aware DatetimeIndex named Date."""
    index = pd.DatetimeIndex(
        [pd.Timestamp(day, tz="America/New_York") for day, *_ in rows], name="Date"
    )
    return pd.DataFrame(
        {
            "Open": [r[1] for r in rows],
            "High": [r[2] for r in rows],
            "Low": [r[3] for r in rows],
            "Close": [r[4] for r in rows],
            "Volume": [1_000_000] * len(rows),
        },
        index=index,
    )


# yfinance's empty_df(): zero rows, but 'Close' IS among the columns — which is
# why the guard cannot test for the column alone.
EMPTY = pd.DataFrame(
    {c: [] for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")},
    index=pd.DatetimeIndex([], name="Date"),
)

# What a bogus ticker's `info` really is, once the 404 has been swallowed.
BOGUS_INFO = {"trailingPegRatio": None}

AAPL_INFO = {
    "symbol": "AAPL",
    "trailingPE": 36.44,
    "forwardPE": 33.27,
    "priceToBook": 43.12,
    "priceToSalesTrailing12Months": 9.92,
    "enterpriseToEbitda": 27.66,
    "returnOnEquity": 1.4875,
    "profitMargins": 0.2762,
    "revenueGrowth": 0.164,
    "debtToEquity": 78.445,
    "beta": 1.086,
    "marketCap": 4631900979200,
    "trailingEps": 8.71,
    "dividendYield": 0.34,
    "trailingAnnualDividendYield": 0.0033140802,
    "fiftyTwoWeekHigh": 344.57,
    "fiftyTwoWeekLow": 223.78,
}

FIVE_DAYS = [
    ("2026-08-14", 300.0, 305.0, 299.0, 304.0),
    ("2026-08-17", 304.0, 308.0, 303.0, 307.0),
    ("2026-08-18", 307.5, 311.5, 306.0, 310.0),
    ("2026-08-19", 310.1, 319.3, 310.0, 316.8),
    ("2026-08-20", 317.4, 320.3, 315.0, 320.0),
]


@pytest.fixture
def fake_fetch(monkeypatch):
    """Replace the network seam. An unregistered kind is a hard error, which is
    what makes "no fetch happened" a provable claim rather than an absence."""
    calls: list[tuple[str, str, dict]] = []
    responses: dict[str, object] = {}

    def _fake(kind, symbol, **params):
        calls.append((kind, symbol, params))
        if kind not in responses:
            raise AssertionError(f"unexpected {kind} fetch for {symbol}")
        return responses[kind]

    monkeypatch.setattr(prices, "_fetch", _fake)
    return type("FakeFetch", (), {"calls": calls, "responses": responses})()


# ---------------------------------------------------------------------------
# the silent-failure guards — the reason this module is shaped the way it is
# ---------------------------------------------------------------------------


def test_unknown_symbol_quote_raises_rather_than_reporting_zero(fake_fetch):
    fake_fetch.responses["history"] = EMPTY
    with pytest.raises(UnknownSymbolError, match="ZZQQ"):
        get_quote("ZZQQNOTREAL")


def test_unknown_symbol_history_raises(fake_fetch):
    fake_fetch.responses["history"] = EMPTY
    with pytest.raises(UnknownSymbolError):
        get_price_history("ZZQQNOTREAL")


def test_unknown_symbol_key_stats_raises_on_the_one_key_stub(fake_fetch):
    """The exact shape a bogus ticker leaves behind, verified against yfinance 1.6.0."""
    fake_fetch.responses["info"] = BOGUS_INFO
    with pytest.raises(UnknownSymbolError):
        get_key_stats("ZZQQNOTREAL")


def test_info_without_a_symbol_key_is_unknown(fake_fetch):
    """The structural check, so the guard survives Yahoo adding a second stub key."""
    fake_fetch.responses["info"] = {"trailingPegRatio": None, "someNewStub": None}
    with pytest.raises(UnknownSymbolError):
        get_key_stats("ZZQQNOTREAL")


def test_all_nan_closes_are_unknown_not_a_price(fake_fetch):
    fake_fetch.responses["history"] = frame(
        [("2026-08-19", float("nan"), float("nan"), float("nan"), float("nan"))]
    )
    with pytest.raises(UnknownSymbolError):
        get_quote("HALTED")


def test_renamed_info_keys_raise_shape_change_not_unknown_symbol(fake_fetch):
    """A rename must be loud. Returning a hollow model would read as
    'this company has no metrics', which is a different and wrong claim."""
    fake_fetch.responses["info"] = {"symbol": "AAPL", "trailingPriceEarnings": 36.4}
    with pytest.raises(PriceDataError, match="renamed"):
        get_key_stats("AAPL")


# ---------------------------------------------------------------------------
# quote
# ---------------------------------------------------------------------------


def test_quote_reads_the_last_bar_and_the_move_from_the_one_before(fake_fetch):
    fake_fetch.responses["history"] = frame(FIVE_DAYS)
    quote = get_quote("aapl")

    assert quote.symbol == "AAPL"
    assert quote.price == 320.0
    assert quote.previous_close == 316.8
    assert quote.change == pytest.approx(3.2)
    assert quote.percent_change == pytest.approx(3.2 / 316.8 * 100)
    assert quote.open == 317.4
    assert quote.session_date == date(2026, 8, 20)
    assert "2026-08-20" in quote.as_of


def test_quote_asks_for_five_days_not_one(fake_fetch):
    """One day has no previous close, and is empty on a holiday Monday for a
    real ticker — which the guard would misreport as an unknown symbol."""
    fake_fetch.responses["history"] = frame(FIVE_DAYS)
    get_quote("AAPL")
    _, _, params = fake_fetch.calls[0]
    assert params["period"] == "5d"
    assert params["interval"] == "1d"
    assert params["auto_adjust"] is True


def test_single_session_quote_has_no_change(fake_fetch):
    fake_fetch.responses["history"] = frame(FIVE_DAYS[-1:])
    quote = get_quote("AAPL")
    assert quote.price == 320.0
    assert quote.change is None
    assert quote.percent_change is None
    assert quote.previous_close is None


def test_quote_is_frozen(fake_fetch):
    fake_fetch.responses["history"] = frame(FIVE_DAYS)
    quote = get_quote("AAPL")
    with pytest.raises(Exception):
        quote.price = 1.0


# ---------------------------------------------------------------------------
# key stats
# ---------------------------------------------------------------------------


def test_key_stats_maps_yahoo_keys(fake_fetch):
    fake_fetch.responses["info"] = AAPL_INFO
    stats = get_key_stats("AAPL")
    assert stats.pe_ratio == 36.44
    assert stats.market_cap == 4631900979200
    assert stats.week_52_high == 344.57
    assert stats.eps_ttm == 8.71


def test_fractional_percentages_are_scaled_once(fake_fetch):
    """Yahoo sends these three as fractions and everything else pre-scaled."""
    fake_fetch.responses["info"] = AAPL_INFO
    stats = get_key_stats("AAPL")
    assert stats.return_on_equity_pct == pytest.approx(148.75)
    assert stats.profit_margin_pct == pytest.approx(27.62)
    assert stats.revenue_growth_yoy_pct == pytest.approx(16.4)


def test_dividend_yield_is_taken_as_a_percent_never_the_fraction(fake_fetch):
    """`dividendYield` is already a percent (AAPL 0.34) while
    `trailingAnnualDividendYield` is a fraction (0.0033). Sharing a fallback
    tuple across them would be a silent 100x error."""
    fake_fetch.responses["info"] = AAPL_INFO
    assert get_key_stats("AAPL").dividend_yield_pct == 0.34


def test_a_non_payer_has_no_dividend_yield_rather_than_zero(fake_fetch):
    """Yahoo omits dividendYield but sends trailingAnnualDividendYield == 0.0.
    Reading the latter would render 'pays no dividend' as a 0.00% yield."""
    fake_fetch.responses["info"] = {
        "symbol": "RIVN", "trailingEps": -2.59, "beta": 1.612,
        "trailingAnnualDividendYield": 0.0,
    }
    stats = get_key_stats("RIVN")
    assert stats.dividend_yield_pct is None
    assert "dividend_yield_pct" not in stats.present()


def test_a_company_without_earnings_has_no_pe_and_no_forward_pe(fake_fetch):
    """Yahoo omits trailingPE on negative EPS but still sends a negative
    forwardPE, which renders as a number a reader takes for a cheap multiple."""
    fake_fetch.responses["info"] = {
        "symbol": "RIVN", "forwardPE": -8.98, "trailingEps": -2.59, "beta": 1.612,
    }
    stats = get_key_stats("RIVN")
    assert stats.pe_ratio is None
    assert stats.forward_pe is None
    assert stats.eps_ttm == -2.59
    assert "pe_ratio" not in stats.present()


def test_present_drops_nones_and_excludes_symbol(fake_fetch):
    fake_fetch.responses["info"] = {"symbol": "X", "beta": 1.1}
    present = get_key_stats("X").present()
    assert present == {"beta": 1.1}
    assert None not in present.values()


def test_booleans_are_not_read_as_numbers():
    assert prices._pick({"beta": True}, "beta", ("beta",)) is None


def test_nan_metrics_are_absent_not_reported():
    assert prices._pick({"beta": float("nan")}, "beta", ("beta",)) is None


# ---------------------------------------------------------------------------
# price history
# ---------------------------------------------------------------------------


def test_percent_change_is_first_to_last_close(fake_fetch):
    fake_fetch.responses["history"] = frame(FIVE_DAYS)
    summary = get_price_history("AAPL", "1mo")
    assert summary.start_close == 304.0
    assert summary.end_close == 320.0
    assert summary.percent_change == pytest.approx((320.0 - 304.0) / 304.0 * 100)
    assert summary.high == 320.0
    assert summary.low == 304.0
    assert summary.sessions == 5
    assert summary.start_date == date(2026, 8, 14)


def test_anchors_pin_the_endpoints_exactly(fake_fetch):
    fake_fetch.responses["history"] = frame(FIVE_DAYS)
    summary = get_price_history("AAPL")
    assert summary.anchors[0][1] == summary.start_close
    assert summary.anchors[-1][1] == summary.end_close
    assert [d for d, _ in summary.anchors] == sorted(d for d, _ in summary.anchors)


def test_an_invalid_period_is_rejected_before_any_fetch(fake_fetch):
    with pytest.raises(ValueError, match="period must be one of"):
        get_price_history("AAPL", "7d")
    assert fake_fetch.calls == []


@pytest.mark.parametrize("n,expected", [(1, 1), (3, 3), (4, 4), (5, 5), (250, 5)])
def test_anchor_count_degrades_without_duplicating(n, expected):
    idx = anchor_indices(n)
    assert len(idx) == expected
    assert len(set(idx)) == len(idx)
    assert idx[0] == 0 and idx[-1] == n - 1


def test_realised_vol_matches_a_hand_computed_value():
    # Alternating +10% / -10% log-ish moves; checked against the formula by hand.
    closes = [100.0, 110.0, 99.0, 108.9, 98.01]
    returns = [0.0953101798, -0.1053605157, 0.0953101798, -0.1053605157]
    mean = sum(returns) / 4
    variance = sum((r - mean) ** 2 for r in returns) / 3
    expected = variance**0.5 * (252**0.5) * 100
    assert realised_vol_pct(closes) == pytest.approx(expected, rel=1e-9)


def test_realised_vol_is_none_when_undefined():
    assert realised_vol_pct([100.0, 101.0]) is None      # one return, no sample stdev
    assert realised_vol_pct([100.0, 0.0, 101.0]) is None  # non-positive close is corrupt
