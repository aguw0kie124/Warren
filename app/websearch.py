"""B2 · Web search — open-web financial context, via Tavily.

The second half of closing Module A's staleness gap. Finnhub answers questions
scoped to one ticker; this answers everything else — analyst commentary,
competitor and industry moves, macro events, anything with no CIK and no
symbol.

Tavily rather than a raw search API because it returns extracted article text,
not HTML to scrape. Named for the capability, not the vendor: swapping the
provider should touch this file and nothing else, so `WebResult` is our shape
rather than Tavily's.

**The domain allowlist is the point of this module.** An unfiltered web search
on a ticker question is a magnet for SEO spam and pump-and-dump content, which
is exactly what a research assistant must not cite. One list does more for
answer trustworthiness than any amount of prompting.
"""

import logging
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

# Edit freely — this is meant to be tuned once you see what the agent actually
# pulls back. Wire services and papers of record first, then the primary
# sources (SEC, exchanges) that are authoritative by construction.
FINANCE_DOMAINS: list[str] = [
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "barrons.com",
    "marketwatch.com",
    "economist.com",
    "apnews.com",
    "nytimes.com",
    "axios.com",
    "theinformation.com",
    "sec.gov",
    "federalreserve.gov",
    "nyse.com",
    "investor.gov",
    # Added after the B2 gate. Each was checked against the promotional query
    # below and returned nothing for it — that test, not reputation, is what
    # earns a place here.
    "morningstar.com",
    "fortune.com",
    "spglobal.com",
    "investopedia.com",
    "theguardian.com",
    "washingtonpost.com",
    "semianalysis.com",
    "stratechery.com",
]

# Rejected by the same test, recorded so they don't get re-proposed:
# fool.com took 4 of 8 slots on "best stocks to buy now for huge gains"
# (listicles are Motley Fool's business model), and businessinsider.com,
# forbes.com (contributor network) and benzinga.com placed on it too.

# Removed after the B2 gate, not on principle: nasdaq.com is on-topic but
# syndicates Motley Fool / InvestorPlace promotional content under its own
# domain, so it answered "best stocks to buy now for huge gains" with five
# listicles ("...for HUGE Gains in the Shroom Boom"). A domain that launders
# spam through a reputable hostname is worse than one that is obviously spam,
# because the allowlist is what the agent trusts instead of judging sources.

DEFAULT_MAX_RESULTS = 5

# Tavily is billed per request, not per result, so asking for more than we need
# is free — and dedup below can only shrink the list. Without headroom, five
# near-duplicate URLs return as one result.
OVERFETCH = 3
MAX_TAVILY_RESULTS = 20

# Stripped before comparing URLs for duplicates. Analytics and paywall-referrer
# parameters change per request while pointing at the same article.
TRACKING_PARAM_PREFIXES = ("utm_", "gaa_")
TRACKING_PARAMS = frozenset(
    {"mod", "ref", "src", "srnd", "ocid", "cmpid", "taid", "tsrc", "guccounter", "s"}
)


class WebSearchError(RuntimeError):
    """Any failure running a web search."""


class MissingApiKeyError(WebSearchError):
    """TAVILY_API_KEY is unset or still a placeholder."""


class WebResult(BaseModel):
    """One search hit, with the extracted text rather than raw HTML."""

    title: str
    url: str
    content: str
    # Tavily's own relevance score. Comparable within one response only — it is
    # not on the same scale as anything the retriever produces, which is why
    # these are never fused with filing results.
    score: float = 0.0
    published_date: str | None = None

    @property
    def domain(self) -> str:
        return _domain(self.url)


def _api_key() -> str:
    key = settings.tavily_api_key.strip()
    if not key or key.lower().startswith("your"):
        raise MissingApiKeyError(
            "TAVILY_API_KEY is not set. Get a free key at tavily.com and put it "
            "in .env before running web searches."
        )
    return key


def _domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _allowed(url: str, domains: list[str]) -> bool:
    """Suffix match, so ir.apple.com and www.reuters.com both pass."""
    host = _domain(url)
    return any(host == d or host.endswith("." + d) for d in domains)


def _canonical(url: str) -> str:
    """The identity of an article, for duplicate detection.

    Tracking parameters are dropped and everything else is kept: a bare
    `?id=1234` really does identify a different article on some sites, so
    stripping the whole query string would silently merge distinct sources.
    """
    parts = urlparse(url)
    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
        and not key.lower().startswith(TRACKING_PARAM_PREFIXES)
    ]
    return urlunparse(
        ("", _domain(url), parts.path.rstrip("/"), "", urlencode(sorted(kept)), "")
    )


def _client():
    """Built per call, not cached: the SDK object is cheap, and this keeps a
    key rotated in .env from needing a process restart."""
    # Imported lazily so that importing this module (in tests, or in tools.py)
    # doesn't require the dependency or a key to be present.
    try:
        from tavily import TavilyClient
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise WebSearchError(
            "tavily-python is not installed. Run: pip install -e '.[dev]'"
        ) from exc

    return TavilyClient(api_key=_api_key())


def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    include_domains: list[str] | None = None,
    days: int | None = None,
) -> list[WebResult]:
    """Search the financial web, restricted to the domain allowlist.

    `include_domains` defaults to FINANCE_DOMAINS. Pass an explicit list to
    narrow it, or `[]` to search the open web — which is deliberately awkward
    to do by accident.

    `days` limits results to the last N days, and switches Tavily to its news
    topic (the only one that honours it). Leave it None for no recency bound —
    the right default for "what do analysts think", the wrong one for "what
    happened this week".
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    domains = FINANCE_DOMAINS if include_domains is None else include_domains

    kwargs = {
        "query": query,
        "max_results": min(max_results * OVERFETCH, MAX_TAVILY_RESULTS),
        "search_depth": "basic",
    }
    if domains:
        kwargs["include_domains"] = domains
    if days is not None:
        # `days` is only honoured on the news topic — sent with the default
        # general topic it is accepted and ignored, which would make a recency
        # bound look applied when it wasn't.
        kwargs["days"] = days
        kwargs["topic"] = "news"

    try:
        payload = _client().search(**kwargs)
    except WebSearchError:
        raise
    except Exception as exc:  # the SDK raises its own exception hierarchy
        raise WebSearchError(f"Tavily search failed for {query!r}: {exc}") from exc

    return _parse(payload, domains)[:max_results]


def _parse(payload: dict, domains: list[str]) -> list[WebResult]:
    """Validate the response, re-check the allowlist, and drop duplicates.

    The allowlist check is not paranoia about Tavily: it is what makes the
    filter *verifiable*. A provider that quietly ignores `include_domains` — or
    a caller that fat-fingers the parameter name — would otherwise degrade the
    source quality with no visible symptom at all.

    Deduplication earns its place for the same reason. The B2 gate returned
    five "results" for one Apple query that were all the same MarketWatch live
    blog, distinguished only by `gaa_*` tracking parameters. The agent would
    have seen one source presented as five and read the repetition as
    corroboration.
    """
    raw = (payload or {}).get("results")
    if raw is None:
        raise WebSearchError(f"Tavily response had no 'results' key: {sorted(payload or {})}")

    results: list[WebResult] = []
    seen: set[str] = set()
    for item in raw:
        if not item.get("url"):
            continue
        result = WebResult(
            title=item.get("title") or "",
            url=item["url"],
            content=item.get("content") or "",
            score=item.get("score") or 0.0,
            published_date=item.get("published_date"),
        )
        if domains and not _allowed(result.url, domains):
            logger.warning(
                "dropping off-allowlist result from %s (%s)", result.domain, result.url
            )
            continue

        # Tavily returns results best-first, so the first copy is the one to
        # keep — it carries the highest score and usually the fullest extract.
        key = _canonical(result.url)
        if key in seen:
            logger.debug("dropping duplicate of %s", key)
            continue
        seen.add(key)
        results.append(result)

    return results
