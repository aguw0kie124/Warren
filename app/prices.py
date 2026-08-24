"""B1 · Market data — quotes, key stats and price-history summaries, via yfinance.

Replaces the Finnhub client. Structured the same way it was: one lazily-created
session, one central fetch carrying rate limiting and retry so no call site can
forget them, project-owned models rather than the provider's shapes, and errors
raised here but converted to text by the tool layer above.

**The fabrication guard is why this file is shaped like its predecessor.**
Finnhub reported "no such symbol" as an HTTP 200 with an all-zero body, which
validated into a $0.00 quote dated 1970. yfinance has the identical failure
shape and also never raises. Measured against yfinance 1.6.0 on 2026-08-20:

    yf.Ticker("ZZQQNOTREAL").info      -> {'trailingPegRatio': None}
    yf.Ticker("ZZQQNOTREAL").history() -> empty DataFrame, but 'Close' IS in
                                          .columns, so `.empty` is the only tell

The `info` stub happens because the quoteSummary 404 is swallowed (yfinance's
`debug.hide_exceptions` defaults to True), the v7 quote returns an empty list so
nothing is injected, and `_fetch_complementary` then writes that one key
unconditionally. Every function below checks the empty shape BEFORE building a
model.

yfinance also logs an ERROR line for every unknown ticker. **That line is not
the signal** — `UnknownSymbolError` is. Resist setting
`yf.config.debug.hide_exceptions = False` to make history() raise instead: it is
a process-global any importer can flip back, and it would give the guard two
mechanisms that diverge between the offline fixtures and production, leaving the
one the tests pin dead in the real path.

**Models are Pydantic and frozen, not the `@dataclass(frozen=True)` sketched in
docs/phase-2-plan.md §3.** The immutability is honoured; the type is the
project's. `Ticker.info` is an untyped 180-key scraped dict with no contract, so
loud failure matters more here than at Finnhub, not less; these models are built
from a DataFrame, so `np.float64` and `NaN` reach the constructor and only
Pydantic gives us a place to reject them (NaN is the DataFrame-native form of
the 1970 quote); and every other model in this project is Pydantic.

**Consolidation risk, on the record.** Quotes, ratios and history now share one
unofficial provider, so a Yahoo change takes out all market data at once — the
earlier design split providers precisely to avoid that. The mitigation is that
everything returned here is a project model, keeping a provider swap a one-file
change.

**Not traced.** app/tracing.py's rule is to instrument a function when the tool
span above it hides a *decision* or a cost, not merely because it is on the
path. `get_quote` and `get_price_history` are one request each; `get_key_stats`
is three sequential requests inside one attribute access, which is a cost
already visible as the parent span's duration, not a hidden decision. The
trigger to revisit is a *branch*: if `get_quote` ever falls back from history to
info, or `get_key_stats` picks between two sources for one metric, that choice
is invisible to the tool span and earns a span the moment it exists.
"""

import logging
import math
import threading
import time
from datetime import date
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, field_validator
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Yahoo publishes no documented limit and yfinance spaces nothing itself. Its
# own `config.network.retries` covers transient network errors only and
# explicitly excludes the 429 that YFRateLimitError signals, so both the spacing
# and the backoff below are ours.
REQUESTS_PER_SECOND = 2.0

DEFAULT_PERIOD = "1y"
# Yahoo's own vocabulary, minus the intraday windows: this module deals in daily
# bars only. Validated here rather than at Yahoo, whose error names a resampled
# interval and is unreadable.
VALID_PERIODS = ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")

ANCHOR_POINTS = 5
TRADING_DAYS_PER_YEAR = 252

# A long weekend plus a market holiday. Used by the gate, not enforced here — a
# stale quote is still a real quote, and it carries its own date.
MAX_QUOTE_STALENESS_DAYS = 5


class PriceDataError(RuntimeError):
    """Any failure fetching market data."""


class UnknownSymbolError(PriceDataError):
    """Yahoo has no data for this symbol.

    Signalled by an empty DataFrame or a one-key `info` stub, never by an
    exception — so every entry point has to detect it explicitly or the agent
    reports a real-looking quote for a typo.
    """

    def __init__(self, symbol: str, endpoint: str) -> None:
        super().__init__(
            f"Yahoo returned no data for symbol {symbol!r} ({endpoint}). "
            "The symbol is probably wrong, or is not listed."
        )
        self.symbol = symbol


class _RetryableStatus(PriceDataError):
    """Rate limited. Worth retrying after a wait."""


class _RateLimiter:
    """Spaces requests at least `min_interval` apart. Thread-safe, so it holds
    when the API fans several runs out across the threadpool.

    Copied from the Finnhub client rather than shared, for the reason that one
    stated: each external service owns its own limit, and a shared limiter would
    throttle EDGAR because Yahoo was busy. This is the third copy; app/sec_http.py
    has the second. They collapse into one class with three instances when
    app/ratelimit.py lands.
    """

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
_session: Any = None
_session_lock = threading.Lock()


def _get_session() -> Any:
    """One curl_cffi session for the process, so the cookie and crumb are reused.

    curl_cffi is yfinance's supported backend and its TLS fingerprint is what
    keeps Yahoo from rate-limiting; the plain-requests fallback is documented as
    blockable. Built here rather than left to yfinance's implicit global so the
    lifetime is ours and `close_client()` means something.
    """
    global _session
    with _session_lock:
        if _session is None:
            from yfinance._http import new_session

            _session = new_session()
        return _session


_Kind = Literal["history", "info"]


@retry(
    retry=retry_if_exception_type(_RetryableStatus),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _fetch(kind: _Kind, symbol: str, **params: Any) -> Any:
    """The one place this module talks to Yahoo.

    The Finnhub client's seam was `_get(path, params)`. yfinance has no HTTP
    layer we can reach, so the seam is drawn one level up and dispatches on
    `kind` instead of on a path. Everything above this line is pure and testable
    offline; everything below it is network, and tests replace exactly this
    function.

    It returns yfinance's own types — a DataFrame or a dict — deliberately.
    Pushing the extraction below the seam would put the part most likely to break
    on a yfinance change (column names, key names) on the untested side.
    """
    # Imported lazily so that importing this module — in tests, or in tools.py —
    # costs neither the dependency nor the ~1s pandas import.
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError

    _limiter.wait()
    logger.debug("yfinance %s %s %s", kind, symbol, params)
    ticker = yf.Ticker(symbol, session=_get_session())
    try:
        if kind == "history":
            return ticker.history(**params)
        if kind == "info":
            return ticker.info
    except YFRateLimitError as exc:
        raise _RetryableStatus(f"Yahoo rate-limited {symbol} ({kind}); will retry") from exc
    except Exception as exc:
        raise PriceDataError(f"yfinance failed for {symbol} ({kind}): {exc}") from exc
    raise ValueError(f"unknown fetch kind {kind!r}")


def _symbol(symbol: str) -> str:
    """Yahoo matches symbols case-insensitively; an LLM will not be consistent."""
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("symbol must not be empty")
    return cleaned


def _closes(df: Any, symbol: str, endpoint: str):
    """The empty-shape guard, and the only way a caller gets at the prices.

    Three checks, not one. An unknown ticker yields yfinance's `empty_df()`,
    which is zero rows but *does* carry a 'Close' column — so testing for the
    column alone passes. A real ticker on a market holiday can yield rows whose
    closes are all NaN.
    """
    if df is None or getattr(df, "empty", True) or "Close" not in df.columns:
        raise UnknownSymbolError(symbol, endpoint)
    closes = df["Close"].dropna()
    if closes.empty:
        raise UnknownSymbolError(symbol, endpoint)
    return closes


class _Model(BaseModel):
    model_config = {"frozen": True}

    @field_validator("*")
    @classmethod
    def _reject_nan(cls, value: Any) -> Any:
        """NaN is what the 1970 quote looks like once the data is a DataFrame.

        It passes a bare `float` annotation, formats as "nan", and reads to a
        model as a number it can narrate. Rejected everywhere rather than at the
        two places it is expected.
        """
        if isinstance(value, float) and math.isnan(value):
            raise ValueError("value is NaN")
        return value


# ---------------------------------------------------------------------------
# quote
# ---------------------------------------------------------------------------


class Quote(_Model):
    """The last daily bar, plus the move from the one before it."""

    symbol: str
    price: float
    change: float | None = None
    percent_change: float | None = None
    open: float
    high: float
    low: float
    previous_close: float | None = None
    session_date: date

    @property
    def as_of(self) -> str:
        return f"{self.session_date.isoformat()} (last daily bar, delayed)"


def get_quote(symbol: str) -> Quote:
    """Latest close and the move from the previous session.

    A five-day window rather than one day, for two reasons: a one-day window has
    no previous close to compare against, and on a holiday Monday it comes back
    empty for a perfectly real ticker — which the guard would then misreport as
    an unknown symbol.

    Prices are split- and dividend-adjusted (`auto_adjust`), the one convention
    in this module. The last bar's adjustment ratio is 1.0 by construction, so
    `price` is the traded price either way; only `previous_close` differs, and
    only across an ex-dividend date. Raw closes would render a split inside the
    window as a -75% move, which is the fabrication failure in a different
    costume.

    `session_date` is a date, not a minute. A daily bar is timestamped at
    exchange-local midnight, so there is no trade time to report without a second
    request for history metadata — and a date is the honest unit anyway.
    """
    sym = _symbol(symbol)
    df = _fetch("history", sym, period="5d", interval="1d", auto_adjust=True, actions=False)
    closes = _closes(df, sym, "history")

    last = df.loc[closes.index[-1]]
    price = float(closes.iloc[-1])
    previous = float(closes.iloc[-2]) if len(closes) >= 2 else None

    return Quote(
        symbol=sym,
        price=price,
        change=None if previous is None else price - previous,
        percent_change=None if not previous else (price - previous) / previous * 100.0,
        open=float(last["Open"]),
        high=float(last["High"]),
        low=float(last["Low"]),
        previous_close=previous,
        session_date=closes.index[-1].date(),
    )


# ---------------------------------------------------------------------------
# key stats
# ---------------------------------------------------------------------------

# Friendly field -> the yfinance `info` keys that can carry it, best first.
#
# Verified against AAPL / KO / RIVN on 2026-08-20. Two findings are baked in:
#
#   * `dividendYield` is already a PERCENT (AAPL 0.34, KO 2.35), while
#     `trailingAnnualDividendYield` is a FRACTION (0.0033, 0.0230). They are
#     therefore NOT interchangeable and must never share a fallback tuple — a
#     "first key present wins" lookup across them is a silent 100x error. The
#     fraction is also 0.0 rather than absent for a non-payer (RIVN), which
#     would render "pays no dividend" as a 0.00% yield: an absence dressed as a
#     number, which is the thing this project refuses to do.
#   * Yahoo omits `trailingPE` entirely for a company with negative trailing
#     EPS (RIVN), which is exactly the behaviour we want and needs no help.
_INFO_KEYS: dict[str, tuple[str, ...]] = {
    "pe_ratio": ("trailingPE",),
    "forward_pe": ("forwardPE",),
    "price_to_book": ("priceToBook",),
    "price_to_sales": ("priceToSalesTrailing12Months",),
    "ev_to_ebitda": ("enterpriseToEbitda",),
    "return_on_equity_pct": ("returnOnEquity",),
    "profit_margin_pct": ("profitMargins",),
    "revenue_growth_yoy_pct": ("revenueGrowth",),
    "debt_to_equity": ("debtToEquity",),
    "beta": ("beta",),
    "market_cap": ("marketCap",),
    "eps_ttm": ("trailingEps",),
    "dividend_yield_pct": ("dividendYield",),
    "week_52_high": ("fiftyTwoWeekHigh",),
    "week_52_low": ("fiftyTwoWeekLow",),
}

# Yahoo reports these three as fractions; every other percentage field it sends
# is already scaled. Converted on the way in so the model names the unit.
_AS_FRACTION = frozenset({"return_on_equity_pct", "profit_margin_pct", "revenue_growth_yoy_pct"})

# A price/earnings ratio is undefined when earnings are negative. Yahoo omits
# `trailingPE` in that case but still sends a negative `forwardPE` (RIVN:
# -8.98), which renders as a number a reader would take for a cheap multiple.
_POSITIVE_ONLY = frozenset({"pe_ratio", "forward_pe", "price_to_sales", "price_to_book"})

# What a bogus ticker leaves behind once the 404 has been swallowed. Verified
# 2026-08-20; `"symbol" not in info` is the structural reason it happens and is
# checked too, so the guard survives Yahoo adding a second always-present stub.
_INFO_STUB_KEYS = frozenset({"trailingPegRatio"})


class KeyStats(_Model):
    """Market-computed valuation ratios for one symbol.

    Every field is optional because Yahoo genuinely lacks some metrics for some
    filers — that is data absence, not breakage. Breakage is *all* of them being
    absent at once, which `get_key_stats` raises on.
    """

    symbol: str
    pe_ratio: float | None = None
    forward_pe: float | None = None
    price_to_book: float | None = None
    price_to_sales: float | None = None
    ev_to_ebitda: float | None = None
    return_on_equity_pct: float | None = None
    profit_margin_pct: float | None = None
    revenue_growth_yoy_pct: float | None = None
    debt_to_equity: float | None = None
    beta: float | None = None
    market_cap: float | None = None
    eps_ttm: float | None = None
    dividend_yield_pct: float | None = None
    week_52_high: float | None = None
    week_52_low: float | None = None

    def present(self) -> dict[str, float]:
        """Only the metrics Yahoo actually had.

        What a model should be shown: a wall of `None`s invites it to comment on
        the absences instead of the numbers.
        """
        return {
            name: value
            for name, value in self.model_dump(exclude={"symbol"}).items()
            if value is not None
        }


def _pick(info: dict[str, Any], name: str, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = info.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isnan(value):
            continue
        if name in _POSITIVE_ONLY and value <= 0:
            continue
        return value * 100.0 if name in _AS_FRACTION else value
    return None


def get_key_stats(symbol: str) -> KeyStats:
    """Valuation ratios, margins and the 52-week range for one symbol.

    Yahoo's own computed figures — point-in-time, unaudited, and recomputed
    every time the price moves. Not the audited numbers the company filed; those
    are app/fundamentals.py.
    """
    sym = _symbol(symbol)
    info = _fetch("info", sym)

    if not info or set(info) <= _INFO_STUB_KEYS or "symbol" not in info:
        raise UnknownSymbolError(sym, "info")

    values = {name: _pick(info, name, keys) for name, keys in _INFO_KEYS.items()}

    # Yahoo had an info payload but none of the keys we recognise: that is a
    # renamed-key shape change, and it must fail here rather than return a
    # hollow model that reads as "this company has no metrics".
    if not any(value is not None for value in values.values()):
        raise PriceDataError(
            f"yfinance returned {len(info)} info keys for {sym} but none of the "
            f"{len(_INFO_KEYS)} expected ones. Yahoo has probably renamed them — "
            f"update _INFO_KEYS. Sample keys: {sorted(info)[:8]}"
        )

    return KeyStats(symbol=sym, **values)


# ---------------------------------------------------------------------------
# price history
# ---------------------------------------------------------------------------


class PriceSummary(_Model):
    """What a period did, in a dozen lines rather than 250 rows."""

    symbol: str
    period: str
    start_date: date
    end_date: date
    start_close: float
    end_close: float
    percent_change: float
    high: float
    low: float
    realised_vol_pct: float | None = None
    sessions: int
    # (date, close). Deliberately a plain pair — it is rendered and never
    # queried, so a model class would be ceremony.
    anchors: list[tuple[date, float]]


def realised_vol_pct(closes: list[float]) -> float | None:
    """Annualised standard deviation of daily log returns, in percent.

    Log returns rather than simple ones, so the measure is symmetric in
    direction and additive in time. `ddof=1` because this is a sample — numpy's
    default of 0 understates it, which is the difference between a defined
    quantity and an arbitrary one. 252 is the US trading-day convention and is a
    named constant so a non-US listing can be reasoned about rather than
    silently mis-scaled.

    `None` below three sessions (two returns are the minimum for a sample
    stdev), and `None` if any close is non-positive — that is corrupt data, not
    volatility.
    """
    prices = np.asarray(closes, dtype=float)
    if prices.size < 3 or (prices <= 0).any():
        return None
    returns = np.diff(np.log(prices))
    return float(np.std(returns, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)


def anchor_indices(n: int, k: int = ANCHOR_POINTS) -> list[int]:
    """`k` evenly spaced positions across `n` sessions, endpoints exact.

    Spaced by index rather than by calendar date, so every anchor lands on a real
    session — no interpolation and no invented price on a holiday. i=0 and i=k-1
    map to 0 and n-1 exactly, so the first and last anchors *are* the period's
    first and last close and a gate can assert identity rather than tolerance.
    The set collapses duplicates, so a four-session period yields four anchors
    rather than five with a repeat.

    Deliberately not the k biggest moves, and not local extrema: the point is to
    let the model say "flat until March, then up". Unevenly spaced points let it
    say something shaped instead of something true.
    """
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def get_price_history(symbol: str, period: str = DEFAULT_PERIOD) -> PriceSummary:
    """How one symbol's price moved over a period, as a summary.

    Never returns bars. A year of daily OHLCV is ~250 rows that would spend the
    context guard's whole budget on data the model cannot reason over, and there
    is no chart to feed.
    """
    sym = _symbol(symbol)
    if period not in VALID_PERIODS:
        raise ValueError(f"period must be one of {', '.join(VALID_PERIODS)}, got {period!r}")

    df = _fetch("history", sym, period=period, interval="1d", auto_adjust=True, actions=False)
    closes = _closes(df, sym, "history")

    values = [float(v) for v in closes.to_numpy()]
    dates = [stamp.date() for stamp in closes.index]
    start, end = values[0], values[-1]

    return PriceSummary(
        symbol=sym,
        period=period,
        start_date=dates[0],
        end_date=dates[-1],
        start_close=start,
        end_close=end,
        percent_change=(end - start) / start * 100.0 if start else 0.0,
        high=max(values),
        low=min(values),
        realised_vol_pct=realised_vol_pct(values),
        sessions=len(values),
        anchors=[(dates[i], values[i]) for i in anchor_indices(len(values))],
    )


def close_client() -> None:
    global _session
    with _session_lock:
        if _session is not None:
            _session.close()
            _session = None
