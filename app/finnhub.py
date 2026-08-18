"""B1 · Finnhub client — live market data, news, and company metrics.

Module A gives the agent deep but *stale* knowledge: a 10-K filed in November
says nothing about last week's price or headlines. This is the first half of
closing that gap (app/websearch.py is the second).

Every endpoint returns a Pydantic model rather than raw JSON. That is a
correctness choice, not tidiness: if Finnhub renames a field, a typed model
fails loudly here, whereas a raw dict quietly yields `None` and the agent
narrates a confidently wrong answer built on a missing key.

Transport concerns (API key, rate limiting, retry) are applied centrally in
`_get()`, the same way app/sec_http.py does for EDGAR, so no call site can
forget them.
"""

import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://finnhub.io/api/v1"

# Finnhub's free tier allows 60 calls/minute. One per second keeps us inside it
# without thinking about bursts. Duplicated rather than shared with
# app/sec_http.py on purpose: each external service owns its own limit, and a
# shared limiter would throttle EDGAR because Finnhub was busy.
REQUESTS_PER_SECOND = 1.0

# How far back get_company_news looks when the caller gives no dates. It must
# have a default: an LLM asked for dates it doesn't know will invent them.
DEFAULT_NEWS_DAYS = 7

# A single ticker can produce hundreds of articles in a week, which would
# swamp the agent's context for no gain. Newest first, then cut.
DEFAULT_NEWS_LIMIT = 10


class FinnhubError(RuntimeError):
    """Any failure talking to Finnhub."""


class MissingApiKeyError(FinnhubError):
    """FINNHUB_API_KEY is unset or still a placeholder."""


class PlanRestrictedError(FinnhubError):
    """The key is valid, but this endpoint needs a paid plan."""


class UnknownSymbolError(FinnhubError):
    """Finnhub has no data for this symbol.

    Its own signal for this is an HTTP 200 carrying an empty or all-zero body,
    never an error status — so every endpoint below has to detect it explicitly
    or the agent would report a real-looking $0.00 quote for a typo.
    """

    def __init__(self, symbol: str, endpoint: str) -> None:
        super().__init__(
            f"Finnhub returned no data for symbol {symbol!r} ({endpoint}). "
            "The symbol is probably wrong, or is not covered by the free tier."
        )
        self.symbol = symbol


class _RetryableStatus(FinnhubError):
    """Transient HTTP status worth retrying (429 / 5xx)."""


def _api_key() -> str:
    key = settings.finnhub_api_key.strip()
    if not key or key.lower().startswith("your"):
        raise MissingApiKeyError(
            "FINNHUB_API_KEY is not set. Get a free key at finnhub.io/register "
            "and put it in .env before making market-data requests."
        )
    return key


class _RateLimiter:
    """Spaces requests at least `min_interval` apart. Thread-safe, so it still
    holds if the agent ever fans out tool calls in parallel."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_allowed - now
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


_limiter = _RateLimiter(1.0 / REQUESTS_PER_SECOND)
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            base_url=BASE_URL,
            # Header rather than the `token=` query parameter Finnhub also
            # accepts: the key then never lands in a log line or an exception
            # message containing the URL.
            headers={"X-Finnhub-Token": _api_key()},
            timeout=httpx.Timeout(20.0, connect=10.0),
        )
    return _client


@retry(
    retry=retry_if_exception_type((_RetryableStatus, httpx.TransportError)),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _get(path: str, params: dict[str, Any]) -> Any:
    """GET a Finnhub endpoint with the API key, rate limiting, and retry."""
    _api_key()  # fail before spending a rate-limit slot
    _limiter.wait()

    logger.debug("GET %s %s", path, params)
    response = _get_client().get(path, params=params)

    if response.status_code == 429 or response.status_code >= 500:
        raise _RetryableStatus(
            f"Finnhub returned {response.status_code} for {path} (will retry)"
        )
    if response.status_code == 401:
        raise MissingApiKeyError(
            f"Finnhub rejected the API key (401) for {path}. Check FINNHUB_API_KEY."
        )
    if response.status_code == 403:
        # A valid key hitting a premium endpoint. Distinct from 401 because the
        # fix is completely different — no amount of checking the key helps,
        # and reporting this as a key problem sends you looking in the wrong
        # place. /stock/candle is the one that bites: free tier lost it in 2024.
        raise PlanRestrictedError(
            f"Finnhub returned 403 for {path}: the key is valid but this "
            "endpoint is not on your plan."
        )
    if response.status_code != 200:
        raise FinnhubError(f"Finnhub returned {response.status_code} for {path}")

    return response.json()


def _symbol(symbol: str) -> str:
    """Finnhub matches symbols case-sensitively; an LLM will not be reliable."""
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("symbol must not be empty")
    return cleaned


def _as_date(value: date | str | None, default: date) -> date:
    if value is None:
        return default
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"expected an ISO date (YYYY-MM-DD), got {value!r}") from exc


# ---------------------------------------------------------------------------
# /quote
# ---------------------------------------------------------------------------


class Quote(BaseModel):
    """A real-time-ish quote. Finnhub's single-letter keys, given names."""

    symbol: str
    current_price: float = Field(alias="c")
    change: float | None = Field(default=None, alias="d")
    percent_change: float | None = Field(default=None, alias="dp")
    high: float = Field(alias="h")
    low: float = Field(alias="l")
    open: float = Field(alias="o")
    previous_close: float = Field(alias="pc")
    quoted_at: datetime = Field(alias="t")

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @property
    def as_of(self) -> str:
        return self.quoted_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def get_quote(symbol: str) -> Quote:
    """Current price and today's move for one symbol.

    Delayed on the free tier, and stale outside market hours — `quoted_at` is
    returned so a caller can say *when* rather than implying "now".
    """
    sym = _symbol(symbol)
    payload = _get("/quote", {"symbol": sym})

    # An unknown symbol comes back as a 200 with every field zeroed, so the
    # zero timestamp is the only reliable tell. Checked before validation:
    # otherwise this parses cleanly into a $0.00 quote dated 1970.
    if not payload or not payload.get("t"):
        raise UnknownSymbolError(sym, "/quote")

    return Quote(symbol=sym, **payload)


# ---------------------------------------------------------------------------
# /stock/metric
# ---------------------------------------------------------------------------

# Friendly field -> the Finnhub metric keys that can carry it, best first.
#
# Finnhub reports the same concept under several keys depending on what it has
# for a given filer (TTM vs annual, basic vs diluted), and omits the ones it
# lacks. Taking the first key present keeps a company with no TTM data from
# looking like it has no P/E at all.
_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "pe_ratio": ("peTTM", "peNormalizedAnnual", "peBasicExclExtraTTM"),
    "price_to_sales": ("psTTM", "psAnnual"),
    "price_to_book": ("pbQuarterly", "pbAnnual"),
    "ev_to_ebitda": ("currentEv/freeCashFlowTTM", "enterpriseValue/EBITDATTM"),
    "gross_margin_pct": ("grossMarginTTM", "grossMarginAnnual"),
    "operating_margin_pct": ("operatingMarginTTM", "operatingMarginAnnual"),
    "net_margin_pct": ("netProfitMarginTTM", "netProfitMarginAnnual"),
    "return_on_equity_pct": ("roeTTM", "roeRfy"),
    "return_on_assets_pct": ("roaTTM", "roaRfy"),
    "revenue_growth_yoy_pct": ("revenueGrowthTTMYoy", "revenueGrowthQuarterlyYoy"),
    "eps_growth_yoy_pct": ("epsGrowthTTMYoy", "epsGrowthQuarterlyYoy"),
    "debt_to_equity": (
        "totalDebt/totalEquityQuarterly",
        "totalDebt/totalEquityAnnual",
    ),
    "current_ratio": ("currentRatioQuarterly", "currentRatioAnnual"),
    "dividend_yield_pct": (
        "dividendYieldIndicatedAnnual",
        "currentDividendYieldTTM",
    ),
    "beta": ("beta",),
    "week_52_high": ("52WeekHigh",),
    "week_52_low": ("52WeekLow",),
}


class BasicFinancials(BaseModel):
    """Valuation ratios and margins for one symbol.

    Every field is optional because Finnhub genuinely lacks some metrics for
    some filers — that is data absence, not breakage. Breakage is *all* of them
    being absent at once, which get_basic_financials raises on.
    """

    symbol: str
    pe_ratio: float | None = None
    price_to_sales: float | None = None
    price_to_book: float | None = None
    ev_to_ebitda: float | None = None
    gross_margin_pct: float | None = None
    operating_margin_pct: float | None = None
    net_margin_pct: float | None = None
    return_on_equity_pct: float | None = None
    return_on_assets_pct: float | None = None
    revenue_growth_yoy_pct: float | None = None
    eps_growth_yoy_pct: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    dividend_yield_pct: float | None = None
    beta: float | None = None
    week_52_high: float | None = None
    week_52_low: float | None = None

    def present(self) -> dict[str, float]:
        """Only the metrics Finnhub actually had.

        What a model should be shown: a wall of `None`s invites it to comment
        on the absences instead of the numbers.
        """
        return {
            name: value
            for name, value in self.model_dump(exclude={"symbol"}).items()
            if value is not None
        }


def _pick(metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def get_basic_financials(symbol: str) -> BasicFinancials:
    """Key valuation ratios, margins, and growth rates for one symbol.

    Point-in-time figures from Finnhub's own computation — not from the filings
    ingested in Module A, and not audited.
    """
    sym = _symbol(symbol)
    payload = _get("/stock/metric", {"symbol": sym, "metric": "all"})

    metrics = (payload or {}).get("metric") or {}
    if not metrics:
        raise UnknownSymbolError(sym, "/stock/metric")

    values = {name: _pick(metrics, keys) for name, keys in _METRIC_KEYS.items()}

    # Finnhub had metrics but none we recognise: that is a renamed-key shape
    # change, and it must fail here rather than return a hollow model that
    # reads as "this company has no financials".
    if not any(value is not None for value in values.values()):
        raise FinnhubError(
            f"Finnhub returned {len(metrics)} metrics for {sym} but none of the "
            f"{len(_METRIC_KEYS)} expected keys. The API shape has probably "
            f"changed — update _METRIC_KEYS. Sample keys: {sorted(metrics)[:8]}"
        )

    return BasicFinancials(symbol=sym, **values)


# ---------------------------------------------------------------------------
# /company-news
# ---------------------------------------------------------------------------


class NewsItem(BaseModel):
    """One article. `url` is the citation — it is never synthesised."""

    symbol: str
    headline: str
    summary: str = ""
    source: str = ""
    url: str
    published_at: datetime = Field(alias="datetime")
    category: str = ""

    # Finnhub also sends id/image/related, which nothing downstream needs.
    model_config = {"populate_by_name": True, "extra": "ignore"}

    @property
    def published_date(self) -> date:
        return self.published_at.date()


def get_company_news(
    symbol: str,
    from_date: date | str | None = None,
    to_date: date | str | None = None,
    limit: int = DEFAULT_NEWS_LIMIT,
) -> list[NewsItem]:
    """Recent headlines for one ticker, newest first.

    Dates default to the trailing week so the caller can pass just a symbol.
    An empty list is a real answer (a quiet week), not an error — but an empty
    list for a symbol Finnhub has never heard of is a typo, so that case is
    checked and raised.
    """
    sym = _symbol(symbol)
    today = date.today()
    start = _as_date(from_date, today - timedelta(days=DEFAULT_NEWS_DAYS))
    end = _as_date(to_date, today)
    if start > end:
        raise ValueError(f"from_date {start} is after to_date {end}")

    payload = _get(
        "/company-news",
        {"symbol": sym, "from": start.isoformat(), "to": end.isoformat()},
    )

    if not payload:
        # Ambiguous on its own: no news, or no such company. One extra call
        # resolves it, and only on the path where we already have no data to
        # lose. Without this the agent cheerfully reports "no recent news for
        # APPL" as though the silence were meaningful.
        _assert_symbol_exists(sym, "/company-news")
        logger.info("no %s news between %s and %s", sym, start, end)
        return []

    items = [
        NewsItem(symbol=sym, **article)
        for article in payload
        if article.get("url") and article.get("headline")
    ]
    items.sort(key=lambda item: item.published_at, reverse=True)
    return items[:limit]


# ---------------------------------------------------------------------------
# /stock/profile2
# ---------------------------------------------------------------------------


class CompanyProfile(BaseModel):
    """Mostly static reference data: who this company is."""

    symbol: str
    name: str = ""
    exchange: str = ""
    industry: str = Field(default="", alias="finnhubIndustry")
    country: str = ""
    currency: str = ""
    ipo: date | None = None
    # Finnhub reports this in millions of `currency`.
    market_cap_millions: float | None = Field(default=None, alias="marketCapitalization")
    shares_outstanding_millions: float | None = Field(
        default=None, alias="shareOutstanding"
    )
    website: str = Field(default="", alias="weburl")

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @field_validator("ipo", mode="before")
    @classmethod
    def _blank_ipo_is_none(cls, value: object) -> object | None:
        """Finnhub sends "" for filers with no IPO date, which is not a date."""
        return value or None


def get_company_profile(symbol: str) -> CompanyProfile:
    """Sector/industry, exchange, and market cap for one symbol."""
    sym = _symbol(symbol)
    payload = _get("/stock/profile2", {"symbol": sym})
    if not payload:
        raise UnknownSymbolError(sym, "/stock/profile2")
    return CompanyProfile(symbol=sym, **payload)


def _assert_symbol_exists(symbol: str, endpoint: str) -> None:
    """Raise UnknownSymbolError if Finnhub has no profile for this symbol."""
    if not _get("/stock/profile2", {"symbol": symbol}):
        raise UnknownSymbolError(symbol, endpoint)


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
