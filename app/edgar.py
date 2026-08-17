"""SEC EDGAR HTTP access.

Every request to sec.gov / data.sec.gov goes through `sec_get()` so the two
things SEC enforces are applied centrally and can never be forgotten at a call
site:

  1. A descriptive User-Agent identifying who is making the request. Requests
     without one are refused.
  2. Rate limiting. SEC's published ceiling is 10 req/sec; we stay well under.

A1 adds only this shared foundation. A2/A3 add filing listing and document
fetching on top of it.
"""

import logging
import threading
import time

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

logger = logging.getLogger(__name__)

# SEC allows 10 req/sec. Stay conservative — being throttled or blocked costs
# far more than the handful of seconds this saves.
REQUESTS_PER_SECOND = 3.0
_MIN_INTERVAL = 1.0 / REQUESTS_PER_SECOND

_PLACEHOLDER_MARKERS = ("your name", "your@email.com", "example.com")


class SecError(RuntimeError):
    """Any failure talking to SEC EDGAR."""


class MissingUserAgentError(SecError):
    """EDGAR_USER_AGENT is unset or still holding the .env.example placeholder."""


def _user_agent() -> str:
    ua = settings.edgar_user_agent.strip()
    if not ua:
        raise MissingUserAgentError(
            "EDGAR_USER_AGENT is not set. SEC requires a descriptive User-Agent "
            'identifying who you are, e.g. "Jane Doe jane@example.org". '
            "Set it in .env before making SEC requests."
        )
    lowered = ua.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        raise MissingUserAgentError(
            f"EDGAR_USER_AGENT still looks like the placeholder ({ua!r}). "
            "Replace it with your real name and email — SEC uses it to contact "
            "you if your requests cause problems."
        )
    if "@" not in ua:
        raise MissingUserAgentError(
            f"EDGAR_USER_AGENT ({ua!r}) has no email address. SEC expects a "
            'contact, e.g. "Jane Doe jane@example.org".'
        )
    return ua


class _RateLimiter:
    """Spaces requests at least _MIN_INTERVAL apart. Thread-safe so it still
    holds if ingestion is ever parallelized."""

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


_limiter = _RateLimiter(_MIN_INTERVAL)
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            headers={
                "User-Agent": _user_agent(),
                # SEC asks for these alongside the UA.
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        )
    return _client


class _RetryableStatus(SecError):
    """Transient HTTP status worth retrying (429 / 5xx)."""


@retry(
    retry=retry_if_exception_type((_RetryableStatus, httpx.TransportError)),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def sec_get(url: str, *, accept: str | None = None) -> httpx.Response:
    """GET a SEC URL with the required User-Agent, rate limiting, and retry.

    Raises SecError on non-retryable failures (403, 404, ...).
    """
    _limiter.wait()
    headers = {"Accept": accept} if accept else None

    logger.debug("GET %s", url)
    response = _get_client().get(url, headers=headers)

    if response.status_code == 429 or response.status_code >= 500:
        raise _RetryableStatus(
            f"SEC returned {response.status_code} for {url} (will retry)"
        )
    if response.status_code == 403:
        raise SecError(
            f"SEC refused the request (403) for {url}. This almost always means "
            "the User-Agent is missing, malformed, or you are being rate limited. "
            f"Current EDGAR_USER_AGENT: {settings.edgar_user_agent!r}"
        )
    if response.status_code != 200:
        raise SecError(f"SEC returned {response.status_code} for {url}")

    return response


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
