"""Ticker -> CIK resolution.

SEC identifies companies by CIK, not ticker, so every EDGAR call downstream
needs this translation first. The mapping comes from EDGAR's public
company_tickers.json and is cached on disk — it changes rarely (new listings,
ticker changes), so re-fetching on every run would be pure waste.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.edgar import sec_get

logger = logging.getLogger(__name__)

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CACHE_FILENAME = "company_tickers.json"
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # a week


class UnknownTickerError(LookupError):
    """Ticker is not in EDGAR's mapping."""

    def __init__(self, ticker: str) -> None:
        super().__init__(
            f"{ticker!r} is not in EDGAR's company_tickers.json. Check the "
            "symbol, or refresh the cache if it is a recent listing: "
            "resolve_ticker(..., force_refresh=True)"
        )
        self.ticker = ticker


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: str  # zero-padded to 10 digits, as EDGAR's APIs expect
    name: str

    @property
    def cik_int(self) -> int:
        """Unpadded CIK, used in Archives URLs (A3)."""
        return int(self.cik)


def _index_from_payload(payload: dict[str, Any]) -> dict[str, Company]:
    """Build the ticker index from the raw JSON payload.

    Kept pure (no network, no disk) so it can be unit tested directly.

    EDGAR's shape is positional, not keyed by ticker:
        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    """
    index: dict[str, Company] = {}
    for entry in payload.values():
        ticker = str(entry["ticker"]).strip().upper()
        if not ticker:
            continue
        index[ticker] = Company(
            ticker=ticker,
            # cik_str arrives as an int with leading zeros stripped; EDGAR's
            # submissions API requires the 10-digit padded form.
            cik=f"{int(entry['cik_str']):010d}",
            name=str(entry["title"]).strip(),
        )
    return index


def _cache_path():
    settings.ensure_dirs()
    return settings.cache_dir / CACHE_FILENAME


def _load_payload(force_refresh: bool = False) -> dict[str, Any]:
    path = _cache_path()

    if not force_refresh and path.exists():
        age = time.time() - path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            logger.debug("using cached ticker map (%.1f days old)", age / 86400)
            return json.loads(path.read_text())

    logger.info("fetching %s", COMPANY_TICKERS_URL)
    payload = sec_get(COMPANY_TICKERS_URL, accept="application/json").json()
    path.write_text(json.dumps(payload))
    logger.info("cached %d companies -> %s", len(payload), path)
    return payload


_index: dict[str, Company] | None = None


def get_index(force_refresh: bool = False) -> dict[str, Company]:
    """The full ticker -> Company mapping, memoized per process."""
    global _index
    if _index is None or force_refresh:
        _index = _index_from_payload(_load_payload(force_refresh))
    return _index


def resolve_ticker(ticker: str, force_refresh: bool = False) -> Company:
    """Resolve a ticker to its Company record.

    Raises UnknownTickerError if the symbol is not listed.
    """
    key = ticker.strip().upper()
    try:
        return get_index(force_refresh)[key]
    except KeyError:
        raise UnknownTickerError(key) from None


def try_resolve_ticker(ticker: str) -> Company | None:
    """Non-raising variant, for callers iterating over a watchlist."""
    try:
        return resolve_ticker(ticker)
    except UnknownTickerError:
        return None
