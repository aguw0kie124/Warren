# Phase 2 — what's out of scope, and why

Phase 1 (this codebase, today) is a running service that answers questions from filings, market data, and the web, with verifiable citations. It does **not** do the things below — not because they're hard, but because each was evaluated and deliberately deferred. Recorded here so "why doesn't it do X" has an answer, and so the seams Phase 1 left are the right ones to build on.

## Priority: on-demand ingestion of uncovered tickers

Right now, a ticker outside the four covered (AAPL, META, MSFT, TSLA) hits a corpus gap — `search_filings` says so honestly rather than returning nothing indistinguishable from silence. The natural fix: when a gap is reported, fetch and ingest *just that company's latest 10-K*, then answer from it.

Measured cost: ~20-30s (parse + embed a ~140-chunk 10-K), versus 1-2 minutes for a full four-filing ingest — the difference between a tolerable pause and an unusable one. Constraints for whoever builds this:
- **Say what's happening.** A silent 25s stall reads as a hang. The Postgres checkpointer (already built) is the prerequisite for the clean version — `interrupt()` can pause, surface a notice, and resume.
- **Only the latest 10-K** — backfilling history is a batch job, not something to do while a user waits.
- **Cache the outcome, including failure** — a filer whose HTML the parser can't handle shouldn't be re-attempted every turn.
- **Cap it** at one ingest per turn.

## Screening ("good tech stocks with strong fundamentals")

Deliberately **not** a request-path tool. Neither Finnhub's symbol list nor SEC's ticker registry carries a sector field, and scoring candidates one API call at a time doesn't scale: even a 100-ticker screen at Finnhub's rate limit is well past the 20-30s tolerance above.

The right shape is a **scheduled job filling a table**, not a live fan-out:
- SEC's bulk **XBRL frames API** returns every filer's value for one concept/period in a single request (e.g. `Assets/USD/CY2025Q1I` → 5,649 filers, one 754 KB call) — cheap to build a wide fundamentals table from.
- `screen_stocks` then becomes `WHERE sector = ... ORDER BY ... LIMIT` in SQL — milliseconds, criteria auditable, no reranker hiding the logic.
- A two-stage design for anything needing live pricing (P/E, market cap): filter wide in SQL over the precomputed table, then enrich only the top ~10 survivors with a live `get_quote`.

This is also why the current system refuses "good tech stocks to buy" outright (see [agent.md](agent.md)'s C5 section) rather than improvising an answer from `web_search` — no tool exists to do this responsibly yet, and a listicle dressed as an audited comparison is worse than a clear refusal.

## Evals and observability

- A golden Q&A set with expected citations, RAGAS-style faithfulness / relevance / precision scoring, and an LLM-judge for citation correctness ("does the cited section actually support the claim") — none of that exists yet. Current verification is the `scripts/check_*.py` gates plus human reading, which establish *what was exercised*, not a regression-tested quality bar.
- Tracing (LangSmith or OpenTelemetry) and per-session cost tracking — not wired in. Nearly free to add later (`LANGCHAIN_TRACING_V2=true`, no code change), which is exactly why it was safe to skip for now.

## Retrieval quality

Cross-encoder reranking, per-query-type RRF weighting, query expansion — none implemented. Current retrieval is hybrid dense+BM25 fused by a fixed RRF formula, which the gates show beating dense-only on exact-phrase queries, but there's no measurement of where it falls short.

## Data and infrastructure

- Ingestion is manual (`scripts/ingest.py` per ticker), not scheduled across a universe.
- Postgres runs in a single Docker Compose container, not managed/hosted.
- No containerization of the app itself, no CI/CD.
- **No auth, no rate limiting** — `POST /query` is open. The concurrency cap (`MAX_CONCURRENT_RUNS`) protects the process from resource exhaustion, not the API from abuse.

## Product surface

- **No price history** — Finnhub's free tier 403s on `/stock/candle`. `yfinance` is the planned fix, added as a *sibling* client (not a replacement for Finnhub) so a Yahoo scraping breakage costs charts, not quotes.
- **No streaming** — `POST /query` blocks until the full answer is ready. SSE was scoped out of Phase 1 specifically because there's no UI to stream into yet.
- **No 8-K support, no XBRL numeric time series** in the retrieval corpus — only 10-K/10-Q text.
- **One model provider, one pinned model.** `claude-haiku-4-5-20251001` only — no multi-provider fallback, no model selection per request.

## What was *pulled forward* into Phase 1, contrary to the original plan

Two things originally scoped for Phase 2 shipped early, because a downstream decision made them cheap:

- **Multi-turn sessions.** The Postgres checkpointer (needed later for on-demand ingestion's `interrupt()` pattern) made `thread_id`-based conversations nearly free to add to the API, so they're in `POST /query` today rather than waiting.
- **Concurrency load-testing.** Originally a "someday" item; became part of the API's own verification gate (`scripts/check_api.py`'s cold concurrent burst) once a real segfault showed that single-request gates couldn't be trusted to catch concurrency bugs.
