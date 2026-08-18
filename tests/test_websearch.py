"""Offline tests for web search.

Result *quality* is a human judgement — that's scripts/check_websearch.py. What
matters here is the allowlist, because it is the module's only real safety
property and it fails invisibly: if it stops being applied, nothing errors, the
agent just starts citing worse sources.
"""

import pytest

from app import websearch
from app.websearch import (
    FINANCE_DOMAINS,
    WebResult,
    WebSearchError,
    _allowed,
    _domain,
    _parse,
    web_search,
)


def response(*urls: str) -> dict:
    return {
        "query": "q",
        "results": [
            {"title": f"Story {i}", "url": url, "content": "extracted text",
             "score": 0.9 - i / 100}
            for i, url in enumerate(urls)
        ],
    }


@pytest.fixture
def fake_tavily(monkeypatch):
    """Stand in for TavilyClient, recording the kwargs it was called with."""
    captured: dict = {}
    payload: dict = {"value": response("https://reuters.com/a")}

    class FakeClient:
        def search(self, **kwargs):
            captured.update(kwargs)
            return payload["value"]

    monkeypatch.setattr(websearch, "_client", lambda: FakeClient())
    return type("FakeTavily", (), {"captured": captured, "payload": payload})()


# --- domain matching ---------------------------------------------------------


def test_www_and_subdomains_match_their_domain():
    assert _domain("https://www.reuters.com/business/x") == "reuters.com"
    assert _allowed("https://www.reuters.com/x", ["reuters.com"])
    assert _allowed("https://ir.apple.com/news", ["apple.com"])


def test_unrelated_domains_do_not_match():
    assert not _allowed("https://stockpicks.example.com/x", FINANCE_DOMAINS)


def test_lookalike_domains_do_not_match():
    # Substring matching would let both of these through, and both are exactly
    # the kind of source the allowlist exists to keep out.
    assert not _allowed("https://notreuters.com/x", ["reuters.com"])
    assert not _allowed("https://reuters.com.spam.example/x", ["reuters.com"])


# --- parsing and filtering ---------------------------------------------------


def test_results_are_mapped_to_our_shape():
    results = _parse(response("https://reuters.com/a"), FINANCE_DOMAINS)
    assert len(results) == 1
    assert results[0].url == "https://reuters.com/a"
    assert results[0].content == "extracted text"
    assert results[0].domain == "reuters.com"


def test_off_allowlist_results_are_dropped_locally():
    # The provider is asked to filter, and the answer is checked anyway — that
    # local check is what makes the filter verifiable rather than assumed.
    payload = response("https://reuters.com/a", "https://pumpanddump.example/b")
    results = _parse(payload, FINANCE_DOMAINS)
    assert [r.domain for r in results] == ["reuters.com"]


def test_empty_allowlist_keeps_everything():
    payload = response("https://reuters.com/a", "https://anything.example/b")
    assert len(_parse(payload, [])) == 2


def test_results_without_a_url_are_dropped():
    payload = {"results": [{"title": "No link", "url": "", "content": "x"}]}
    assert _parse(payload, []) == []


def test_urls_differing_only_by_tracking_params_are_one_result():
    # The exact shape the B2 gate returned: one MarketWatch live blog, five
    # times, distinguished only by gaa_* parameters.
    base = "https://www.marketwatch.com/livecoverage/apple-earnings"
    payload = response(
        f"{base}?gaa_at=eafs&gaa_ts=111",
        f"{base}?gaa_at=eafs&gaa_ts=222",
        f"{base}/",
        f"{base}?utm_source=x",
    )
    assert len(_parse(payload, FINANCE_DOMAINS)) == 1


def test_distinct_paths_on_one_domain_are_kept():
    base = "https://www.marketwatch.com/livecoverage/apple-earnings"
    payload = response(base, f"{base}/card/services-bright-spot")
    assert len(_parse(payload, FINANCE_DOMAINS)) == 2


def test_meaningful_query_params_still_distinguish_articles():
    # Stripping the whole query string would merge these two into one.
    payload = response(
        "https://sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=320193",
        "https://sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=789019",
    )
    assert len(_parse(payload, FINANCE_DOMAINS)) == 2


def test_the_first_copy_of_a_duplicate_is_the_one_kept():
    # Tavily orders best-first; the first copy has the highest score.
    payload = response("https://reuters.com/a?utm_source=x", "https://reuters.com/a")
    kept = _parse(payload, FINANCE_DOMAINS)
    assert [r.url for r in kept] == ["https://reuters.com/a?utm_source=x"]


def test_listicle_farm_is_not_on_the_allowlist():
    # nasdaq.com syndicates promotional content under a reputable hostname,
    # which is the failure the allowlist is supposed to prevent.
    assert not _allowed("https://www.nasdaq.com/articles/3-psychedelic-stocks", FINANCE_DOMAINS)


def test_missing_results_key_is_an_error():
    # A changed response shape must not read as "the web had nothing to say".
    with pytest.raises(WebSearchError, match="no 'results' key"):
        _parse({"query": "q", "answer": "…"}, FINANCE_DOMAINS)


# --- the call itself ---------------------------------------------------------


def test_allowlist_is_applied_by_default(fake_tavily):
    web_search("apple services revenue")
    assert fake_tavily.captured["include_domains"] == FINANCE_DOMAINS


def test_caller_can_narrow_the_allowlist(fake_tavily):
    web_search("apple", include_domains=["sec.gov"])
    assert fake_tavily.captured["include_domains"] == ["sec.gov"]


def test_open_web_requires_an_explicit_empty_list(fake_tavily):
    fake_tavily.payload["value"] = response("https://anything.example/b")
    results = web_search("apple", include_domains=[])
    assert "include_domains" not in fake_tavily.captured
    assert len(results) == 1


def test_recency_bound_is_only_sent_when_asked(fake_tavily):
    web_search("apple")
    assert "days" not in fake_tavily.captured

    web_search("apple", days=7)
    assert fake_tavily.captured["days"] == 7
    # Without the news topic Tavily accepts `days` and ignores it, so the bound
    # would silently do nothing.
    assert fake_tavily.captured["topic"] == "news"


def test_more_results_are_requested_than_returned(fake_tavily):
    # Dedup can only shrink the list, and Tavily bills per request rather than
    # per result — so ask for headroom.
    web_search("apple", max_results=3)
    assert fake_tavily.captured["max_results"] > 3


def test_result_count_is_capped_at_max_results(fake_tavily):
    fake_tavily.payload["value"] = response(
        "https://reuters.com/a", "https://ft.com/b", "https://cnbc.com/c"
    )
    assert len(web_search("apple", max_results=2)) == 2


def test_empty_query_is_rejected():
    with pytest.raises(ValueError):
        web_search("   ")


def test_provider_failures_surface_as_websearch_error(monkeypatch):
    class Exploding:
        def search(self, **kwargs):
            raise RuntimeError("402 usage limit exceeded")

    monkeypatch.setattr(websearch, "_client", lambda: Exploding())
    with pytest.raises(WebSearchError, match="usage limit"):
        web_search("apple")


def test_result_domain_property_strips_www():
    assert WebResult(title="t", url="https://www.ft.com/x", content="c").domain == "ft.com"
