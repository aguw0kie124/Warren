# Live data

Two vendor clients feeding real-time information to the agent's tools. Kept separate from the tool layer itself (`app/tools.py`) so each is callable and testable on its own, without an `@tool` decorator in the way.

| File | Vendor | Named for |
|---|---|---|
| `app/finnhub.py` | Finnhub | the vendor — its models mirror Finnhub's payload shape and wouldn't survive swapping providers |
| `app/websearch.py` | Tavily | the capability — Tavily is one interchangeable provider of "web search returning extracted text" |

## Finnhub (`app/finnhub.py`)

Free tier. Functions: `get_quote(symbol)`, `get_basic_financials(symbol)`, `get_company_news(symbol, days=...)`, `get_company_profile(symbol)`.

- **Market data only** — quotes and basic financials. **No price history**: the free tier returns 403 on `/stock/candle`, so there's no `get_price_history` tool and no chart data in Phase 1.
- **"No such symbol" comes back as HTTP 200 with an empty or all-zero body**, never an error status — `{"c":0,...,"t":0}` validates perfectly into a $0.00 quote dated 1970. Every endpoint checks for the empty shape explicitly and raises `UnknownSymbolError`.
- **Rate-limited to 1 req/sec**, enforced by a lock-holding `_RateLimiter.wait()` — concurrent calls serialize rather than racing the API.
- **News quality is uneven**: the free-tier `url` is a redirect (`finnhub.io/api/news?id=...` → 302 to the publisher, not the publisher's own URL), and the per-ticker feed is thick with syndicated aggregators and market-wide articles that just mention the ticker. This is why `get_company_news` and `web_search` need separate, carefully-worded docstrings — see [agent.md](agent.md).

## Tavily (`app/websearch.py`)

`web_search(query, days=None)` — web search returning extracted article text (not raw HTML), domain-restricted.

- **Allowlisted to 24 reputable finance domains**, checked twice: sent as `include_domains` to Tavily *and* re-verified locally in `_parse()`. The local check exists because a filter that silently stops being applied produces no error — just quietly worse sources.
- **An allowlisted domain isn't automatically a good source** — `nasdaq.com` was tested off the list after it turned out to syndicate Motley Fool / InvestorPlace listicles under its own hostname for promotional queries.
- **Results deduplicate by canonical URL.** Tavily has returned the same article five times, differing only in tracking parameters (`gaa_*`) — five sources by count, one by substance, and repetition reads as corroboration to an agent. `web_search` over-fetches slightly to compensate for what dedup removes.
- **`days=N` sets `topic="news"`** — Tavily accepts and silently ignores `days` on the default general topic.

## Verifying it

```bash
.venv/bin/python scripts/check_finnhub.py     # quote matches reality; bogus symbol raises
.venv/bin/python scripts/check_websearch.py   # every URL on the allowlist; off-domain query proves the filter works
.venv/bin/python -m pytest tests/test_finnhub.py tests/test_websearch.py -q
```

Both need API keys (`FINNHUB_API_KEY`, `TAVILY_API_KEY`) — free tiers are enough for both.
