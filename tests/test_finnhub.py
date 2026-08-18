"""Offline tests for the Finnhub client.

Whether a quote is *correct* needs the live API and a browser to compare
against — that's scripts/check_finnhub.py. What's testable offline is the part
that fails silently: Finnhub signals "no such symbol" with a 200 and an empty
body, so every guard against turning that into a real-looking number is pinned
here, along with the date defaulting and the payload mapping.

Payloads are trimmed copies of real responses.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from app import finnhub
from app.finnhub import (
    BasicFinancials,
    CompanyProfile,
    FinnhubError,
    Quote,
    UnknownSymbolError,
    _as_date,
    _pick,
    _symbol,
    get_basic_financials,
    get_company_news,
    get_company_profile,
    get_quote,
)

QUOTE_PAYLOAD = {
    "c": 226.8, "d": 1.53, "dp": 0.6791, "h": 227.29,
    "l": 223.02, "o": 224.99, "pc": 225.27, "t": 1727467200,
}

# What an unknown symbol actually returns: HTTP 200, everything zeroed.
EMPTY_QUOTE_PAYLOAD = {"c": 0, "d": None, "dp": None, "h": 0, "l": 0, "o": 0, "pc": 0, "t": 0}

METRIC_PAYLOAD = {
    "metric": {
        "52WeekHigh": 260.1,
        "52WeekLow": 164.08,
        "beta": 1.24,
        "currentRatioQuarterly": 0.87,
        "grossMarginTTM": 46.21,
        "netProfitMarginTTM": 26.44,
        "operatingMarginTTM": 31.51,
        "peTTM": 34.6,
        "psTTM": 8.9,
        "roeTTM": 157.4,
        "totalDebt/totalEquityQuarterly": 1.87,
    },
    "series": {},
    "symbol": "AAPL",
}

NEWS_PAYLOAD = [
    {
        "category": "company", "datetime": 1727380800, "headline": "Older story",
        "id": 1, "image": "", "related": "AAPL", "source": "Reuters",
        "summary": "…", "url": "https://reuters.com/older",
    },
    {
        "category": "company", "datetime": 1727467200, "headline": "Newer story",
        "id": 2, "image": "", "related": "AAPL", "source": "CNBC",
        "summary": "…", "url": "https://cnbc.com/newer",
    },
]

PROFILE_PAYLOAD = {
    "country": "US", "currency": "USD", "exchange": "NASDAQ NMS - GLOBAL MARKET",
    "finnhubIndustry": "Technology", "ipo": "1980-12-12", "logo": "https://…",
    "marketCapitalization": 3_430_000.0, "name": "Apple Inc",
    "shareOutstanding": 15117.0, "ticker": "AAPL", "weburl": "https://www.apple.com/",
}


@pytest.fixture
def fake_get(monkeypatch):
    """Replace the HTTP layer, recording every call so callers can assert on
    the request Finnhub would have received."""
    calls: list[tuple[str, dict]] = []
    responses: dict[str, object] = {}

    def _fake(path, params):
        calls.append((path, params))
        if path not in responses:
            raise AssertionError(f"unexpected call to {path}")
        return responses[path]

    monkeypatch.setattr(finnhub, "_get", _fake)
    return type("FakeGet", (), {"calls": calls, "responses": responses})()


# --- symbol and date normalisation ------------------------------------------


def test_symbol_is_uppercased_and_stripped():
    # Finnhub matches case-sensitively; an LLM will pass "aapl" sooner or later.
    assert _symbol(" aapl ") == "AAPL"


def test_empty_symbol_is_rejected():
    with pytest.raises(ValueError):
        _symbol("   ")


def test_as_date_accepts_iso_strings_and_dates():
    assert _as_date("2025-03-01", date(2020, 1, 1)) == date(2025, 3, 1)
    assert _as_date(date(2025, 3, 1), date(2020, 1, 1)) == date(2025, 3, 1)
    assert _as_date(None, date(2020, 1, 1)) == date(2020, 1, 1)


def test_as_date_rejects_garbage():
    with pytest.raises(ValueError):
        _as_date("last tuesday", date(2020, 1, 1))


# --- /quote ------------------------------------------------------------------


def test_quote_maps_single_letter_keys(fake_get):
    fake_get.responses["/quote"] = QUOTE_PAYLOAD
    quote = get_quote("aapl")

    assert quote.symbol == "AAPL"
    assert quote.current_price == 226.8
    assert quote.previous_close == 225.27
    assert quote.quoted_at == datetime(2024, 9, 27, 20, 0, tzinfo=timezone.utc)
    assert fake_get.calls == [("/quote", {"symbol": "AAPL"})]


def test_unknown_symbol_quote_raises_rather_than_reporting_zero(fake_get):
    # The whole point: this payload validates perfectly into a $0.00 quote
    # dated 1970, which an agent would report as a real price.
    fake_get.responses["/quote"] = EMPTY_QUOTE_PAYLOAD
    with pytest.raises(UnknownSymbolError):
        get_quote("ZZQQNOTREAL")


# --- /stock/metric -----------------------------------------------------------


def test_metrics_are_mapped_to_named_fields(fake_get):
    fake_get.responses["/stock/metric"] = METRIC_PAYLOAD
    fin = get_basic_financials("AAPL")

    assert fin.pe_ratio == 34.6
    assert fin.gross_margin_pct == 46.21
    assert fin.debt_to_equity == 1.87
    assert fin.week_52_high == 260.1


def test_absent_metrics_stay_none_and_are_dropped_from_present(fake_get):
    fake_get.responses["/stock/metric"] = METRIC_PAYLOAD
    fin = get_basic_financials("AAPL")

    assert fin.dividend_yield_pct is None
    assert "dividend_yield_pct" not in fin.present()
    assert fin.present()["pe_ratio"] == 34.6


def test_metric_key_fallback_prefers_the_first_available():
    keys = ("peTTM", "peNormalizedAnnual")
    assert _pick({"peTTM": 30.0, "peNormalizedAnnual": 28.0}, keys) == 30.0
    assert _pick({"peNormalizedAnnual": 28.0}, keys) == 28.0
    assert _pick({"somethingElse": 28.0}, keys) is None


def test_non_numeric_metric_is_ignored():
    # Finnhub occasionally returns null or a string where a number is expected.
    assert _pick({"beta": None}, ("beta",)) is None
    assert _pick({"beta": "n/a"}, ("beta",)) is None


def test_empty_metric_dict_means_unknown_symbol(fake_get):
    fake_get.responses["/stock/metric"] = {"metric": {}, "series": {}, "symbol": "ZZ"}
    with pytest.raises(UnknownSymbolError):
        get_basic_financials("ZZ")


def test_all_keys_renamed_is_an_error_not_an_empty_model(fake_get):
    # A shape change must fail loudly. Returning a model of Nones would read as
    # "this company reports no financials", which is a plausible-looking lie.
    fake_get.responses["/stock/metric"] = {"metric": {"peSomethingNew": 30.0}}
    with pytest.raises(FinnhubError, match="shape has probably changed"):
        get_basic_financials("AAPL")


# --- /company-news -----------------------------------------------------------


def test_news_defaults_to_the_trailing_week(fake_get):
    fake_get.responses["/company-news"] = NEWS_PAYLOAD
    get_company_news("AAPL")

    _, params = fake_get.calls[0]
    today = date.today()
    assert params["to"] == today.isoformat()
    assert params["from"] == (today - timedelta(days=finnhub.DEFAULT_NEWS_DAYS)).isoformat()


def test_explicit_dates_are_passed_through(fake_get):
    fake_get.responses["/company-news"] = NEWS_PAYLOAD
    get_company_news("AAPL", from_date="2025-01-01", to_date=date(2025, 1, 31))

    _, params = fake_get.calls[0]
    assert params["from"] == "2025-01-01"
    assert params["to"] == "2025-01-31"


def test_inverted_date_range_is_rejected(fake_get):
    with pytest.raises(ValueError, match="after"):
        get_company_news("AAPL", from_date="2025-02-01", to_date="2025-01-01")


def test_news_is_newest_first_and_capped(fake_get):
    fake_get.responses["/company-news"] = NEWS_PAYLOAD
    items = get_company_news("AAPL")
    assert [i.headline for i in items] == ["Newer story", "Older story"]

    assert len(get_company_news("AAPL", limit=1)) == 1


def test_articles_without_a_url_are_dropped(fake_get):
    # The URL *is* the citation; an item without one cannot be cited.
    fake_get.responses["/company-news"] = [
        {"datetime": 1727467200, "headline": "No link", "url": "", "source": "X"},
        *NEWS_PAYLOAD,
    ]
    assert all(item.url for item in get_company_news("AAPL"))


def test_empty_news_for_a_real_symbol_is_an_empty_list(fake_get):
    fake_get.responses["/company-news"] = []
    fake_get.responses["/stock/profile2"] = PROFILE_PAYLOAD
    assert get_company_news("AAPL") == []


def test_empty_news_for_an_unknown_symbol_raises(fake_get):
    # Otherwise "no recent news for APPL" reads as a fact about the company.
    fake_get.responses["/company-news"] = []
    fake_get.responses["/stock/profile2"] = {}
    with pytest.raises(UnknownSymbolError):
        get_company_news("APPL")


# --- /stock/profile2 ---------------------------------------------------------


def test_profile_maps_finnhub_names(fake_get):
    fake_get.responses["/stock/profile2"] = PROFILE_PAYLOAD
    profile = get_company_profile("AAPL")

    assert profile.name == "Apple Inc"
    assert profile.industry == "Technology"
    assert profile.market_cap_millions == 3_430_000.0
    assert profile.ipo == date(1980, 12, 12)


def test_blank_ipo_becomes_none():
    assert CompanyProfile(symbol="X", ipo="").ipo is None


def test_empty_profile_means_unknown_symbol(fake_get):
    fake_get.responses["/stock/profile2"] = {}
    with pytest.raises(UnknownSymbolError):
        get_company_profile("ZZQQNOTREAL")


# --- models ------------------------------------------------------------------


def test_models_ignore_fields_finnhub_adds():
    # Extra keys must not break parsing — only *missing* required ones should.
    quote = Quote(symbol="AAPL", **QUOTE_PAYLOAD, unexpected="new field")
    assert quote.current_price == 226.8


def test_present_excludes_the_symbol():
    fin = BasicFinancials(symbol="AAPL", pe_ratio=30.0)
    assert fin.present() == {"pe_ratio": 30.0}
