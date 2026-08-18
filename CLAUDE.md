# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An AI-driven financial research platform: a user asks a natural-language question about a company ("What are Apple's key risk factors this year?") and gets an answer grounded in real SEC EDGAR filings via RAG, plus live market data via tools, with citations resolving back to the exact filing and section.

**The project is mid-build and follows a strict incremental plan.** The full design and build order live in `docs/PLAN.md` — read it before making architectural changes, since most decisions here were made deliberately and have recorded rationale.

`docs/PLAN.md` is gitignored (local working copy). If it is missing after a fresh clone, the canonical original is at `~/.claude/plans/want-to-design-a-golden-barto.md`.

## Build status

Work proceeds module by module, and **each step stops for human verification before the next begins**. Do not run ahead of the current step.

| Module | Steps | Status |
|---|---|---|
| **A** — data pipeline (EDGAR → Postgres, retrieval) | A0–A8 | **complete** (A0–A8) |
| **B** — live data (Finnhub, Tavily) | B1–B2 | **complete** (both gates run live) |
| **C** — LangGraph agent | C1–C2 | not started |
| **D** — FastAPI service | D1 | not started |

Module A step map: A0 schema · A1 `tickers.py` · A2/A3 `edgar.py` (+ `sec_http.py`) · A4 `parser.py` · A5 `chunker.py` · A6 `embeddings.py`+`store.py`+`scripts/ingest.py` · A7 dense retrieval · A8 hybrid retrieval.

Module B step map: B1 `finnhub.py` · B2 `websearch.py`. Both need API keys in `.env` (`FINNHUB_API_KEY`, `TAVILY_API_KEY`).

Two things the B gates found that **C1 has to account for**: Finnhub's free-tier `url` is a `finnhub.io/api/news?id=…` redirect (302 to the publisher) rather than the publisher's own URL, and its per-ticker feed is thick with syndicated aggregators (Benzinga, ChartMill, SeekingAlpha) plus market-wide articles that merely mention the ticker. The tool docstring that routes between `get_company_news` and `web_search` is written against that reality, not against an idealized news feed.

Each step has a `scripts/check_*.py` gate: `check_db`, `check_tickers`, `check_edgar`, `check_parser`, `check_chunker`, `check_retriever`, `check_finnhub`, `check_websearch`.

Dependencies for unstarted modules are deliberately **not** in `pyproject.toml` — add them when the module is built, not before.

## Commands

```bash
# Postgres (pgvector). --wait blocks until the healthcheck passes.
docker compose up -d --wait
docker compose down          # add -v to also drop the data volume

# Environment. Must be Python 3.13 — see constraint below.
/usr/local/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# A0 gate: applies schema (twice, proving idempotency) and dumps structure.
.venv/bin/python scripts/check_db.py

# Tests
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/test_parser.py::test_name   # single test

# psql is NOT installed on the host; use the container's client
docker compose exec db psql -U postgres -d research
```

## Architecture

Data flows in one direction, and each stage is a separate file so it can be tested alone:

```
EDGAR API → tickers → edgar → parser → chunker → embeddings → store → Postgres
                                                                        ↓
                                                                    retriever
                                                                        ↓
Finnhub → finnhub.py ─────────────────────→ tools.py ←──────────────────┘
Tavily  → websearch.py ───────────────────────↑
                                                ↓
                                            agent.py  (LangGraph)
                                                ↓
                                             api.py
```

### Live-data clients are separate from the tool layer

The plan sketched Module B as "`tools.py`, part 1". It is instead `app/finnhub.py` and `app/websearch.py`, with `tools.py` (C1) left as a pure `@tool` adapter over them — the same layering split as `sec_http.py`. Two reasons: C1's gate is *"call each tool directly and confirm the output matches what B1/B2 produced"*, which needs the plain functions to still exist next to the decorated ones; and `@tool`-decorating in place would make the clients uncallable from `scripts/check_finnhub.py` and the unit tests.

`websearch.py` is named for the capability, not the vendor, because Tavily is one interchangeable provider of it; `finnhub.py` is vendor-named because its models mirror Finnhub's payload shape and would not survive a swap.

### Market data is Finnhub only, and price history is deliberately absent

Evaluated at the end of Module B and settled: **Finnhub for quotes and metrics, no second provider in Phase 1.**

Finnhub's free tier returns **403 on `/stock/candle`** — historical OHLCV moved behind the paid tier — so there is no price history in Phase 1 and no tool that asks for one. Two alternatives were checked before accepting that:

- **yfinance** — free, no account, daily bars going back years, and the natural fit for "how has the stock moved since this 10-K." Its cost is that it scrapes undocumented Yahoo endpoints: no contract, no published rate limit, and a history of breaking outright until a version bump lands. Its news feed is Yahoo's, which is the same aggregator tier as Finnhub's, so it would not fix the news-quality problem and would give C1 a third overlapping news source to route between.
- **Webull OpenAPI** — an official, documented, sandboxed API, unlike yfinance. Ruled out on access, not engineering: market data requires an active paid OpenAPI subscription (Nasdaq Basic L1 or TotalView L2, provisioned separately from any app subscription) under **non-display** exchange licensing, which is the category an AI agent's usage falls into. Real-time exchange data also carries redistribution restrictions that delayed and end-of-day data does not — relevant the moment this serves anyone else.

Revisit when there is a `get_price_history` tool to justify it; yfinance is then the right pick, added as a **sibling** of `finnhub.py` rather than a replacement, so a Yahoo breakage costs charts and not quotes. Phase 2 already parks it there.

### Storage is Postgres + pgvector, and that choice drives the retrieval design

Filings are full of exact-match tokens that dense embeddings handle badly (`material weakness`, `going concern`, `fiscal 2024`). Retrieval is therefore **hybrid**: dense vector similarity + BM25 full-text, fused with Reciprocal Rank Fusion in a **single SQL query** over one table. RRF is used rather than weighted score blending because cosine distance and `ts_rank_cd` are on incomparable scales.

This is why the project uses pgvector rather than Chroma — Chroma has no real BM25, so hybrid there would mean a second index kept in sync. Postgres also serves as the ingestion ledger and, later, the session/eval store.

### The agent is a 2-node ReAct loop, not a router graph

`agent_node ⇄ tool_node`, looping until the model stops emitting tool calls. There is deliberately **no router node and no synthesis node**: tool-calling models already route, and the final tool-call-free turn *is* the synthesis.

The load-bearing consequence: **tools are list entries, not graph nodes.** Adding a capability means appending one `@tool` function — never modifying the graph. Retrieval is itself a tool (`search_filings`), which is what lets multi-hop questions work (the agent just calls it twice with different filters).

With five tools, three of which plausibly answer "what's going on with Apple," **tool docstrings are the routing logic** — they must say what the tool is for *and* when to prefer a sibling. If the agent picks the wrong source, fix the docstring, not the graph.

### Citations are assembled in code, never by the LLM

Models mangle URLs and invent accession numbers, while the retriever already knows ground truth. Citation objects are built programmatically as tool results flow through, and carry a `type` of `filing` / `news` / `web` so an audited SEC filing is visually distinguishable from a news article.

### `search_filings` must report a corpus gap, not an empty result (C1)

Nothing ingests automatically. A ticker that was never run through `scripts/ingest.py` retrieves nothing — identical to a company that genuinely never discussed the topic. The agent cannot tell those apart from `[]`, and the failure is not a missing answer but a misleading one: `web_search` and `get_company_news` work for *any* ticker, so the agent falls back to them and presents a news-grounded answer as though the filings had been consulted.

C1 therefore checks the `filings` ledger before searching and returns a distinct corpus-gap result, with a docstring clause telling the model to say so rather than paper over it. That check is also the trigger point for Phase 2's on-demand ingestion (fetch just the latest 10-K, ~20–30s), which is why it belongs in the tool layer rather than the retriever.

## Constraints that will silently break things

**Embedding dimension is coupled across three places.** `google/embeddinggemma-300m` → 768 dims → `vector(768)` in `sql/schema.sql` → `settings.embedding_dim`. Changing the model means changing all three plus a full re-embed.

**EmbeddingGemma is a gated HF repo.** Before the first embedding run, accept the license at `huggingface.co/google/embeddinggemma-300m` and `huggingface-cli login` (or set `HF_TOKEN`). One-time download gate only — once cached, embedding runs offline with no per-request key.

**Query and document prompts differ.** EmbeddingGemma uses `task: search result | query: ...` for queries and `title: none | text: ...` for documents. Swapping them **degrades retrieval silently** — nothing errors, results just get worse. Expose only `embed_query()` / `embed_documents()`, each pinning its own template, so no caller can choose wrong.

**Chunk size is a quality choice, not a model limit.** The model accepts 2048 tokens; chunks target ~512 because large chunks dilute the vector — one spanning three separate risk factors averages into something that matches none of them well. ~512 is roughly one named risk factor, the unit users actually ask about. Measure with the model's own tokenizer, never a character-count approximation.

**SEC requires a descriptive `User-Agent`** (`EDGAR_USER_AGENT`, format `"Name email@example.com"`) or it blocks requests. Set it centrally in `app/edgar.py` and rate-limit to ~2–3 req/sec (SEC's ceiling is 10) — never per call site.

**Python must be 3.13**, pinned in `pyproject.toml`. The machine's default `python3` is 3.14, which lacks reliable torch/sentence-transformers wheels.

**Re-ingestion must stay idempotent.** `UNIQUE (accession_number, chunk_index)` + `ON CONFLICT ... DO UPDATE` is the mechanism; `scripts/ingest.py` additionally skips accession numbers already present in `filings`. The standing test is that running ingestion twice leaves row counts unchanged.

**`chunks.content_tsv` is a `GENERATED` column** — Postgres maintains it automatically. Never write to it, and note that hybrid search needs no extra ingestion work as a result.

**`sql/schema.sql` is applied on every startup**, so every statement in it must remain idempotent (`IF NOT EXISTS`, named indexes).

**Finnhub reports "no such symbol" as HTTP 200 with an empty or all-zero body**, never an error status. `{"c":0,...,"t":0}` validates perfectly into a $0.00 quote dated 1970, which an agent will narrate as fact. Every endpoint in `app/finnhub.py` therefore checks for the empty shape explicitly and raises `UnknownSymbolError` — a new endpoint must do the same or it becomes a fabrication path.

**The Tavily domain allowlist is checked twice** — sent as `include_domains` *and* re-verified locally in `_parse()`. Not paranoia about the provider: a filter that stops being applied produces no error at all, just quietly worse sources. The local check is what makes it observable. `web_search(..., days=N)` also sets `topic="news"`, because Tavily accepts and ignores `days` on the default general topic.

**Web results are deduplicated by canonical URL, and the search over-fetches to compensate.** Tavily returned the same MarketWatch live blog five times for one query, differing only in `gaa_*` tracking parameters — five sources by count, one by substance, and repetition reads as corroboration to an agent. `_canonical()` strips only known tracking parameters, never the whole query string, because `?id=` genuinely identifies distinct articles on some sites.

**An allowlisted domain can still launder spam.** `nasdaq.com` was dropped after the B2 gate: it is on-topic and reputable-looking but syndicates Motley Fool / InvestorPlace listicles under its own hostname. Judge a candidate domain by what it *returns* for a promotional query, not by its reputation — the allowlist is what the agent trusts in place of judging sources itself.

## Parsing is the fragile part

`app/parser.py` (A4) splits filing HTML into named sections and is the step most likely to need iteration — EDGAR HTML varies by filer and year. Two defenses are required, not optional:

1. **Degrade, don't crash** — if no sections are confidently detected, fall back to chunking the whole document with `section="unknown"` and log a warning.
2. **Guard against the table of contents** — "Item 1A." and "Item 7." also appear in the TOC. Match the *last* occurrence or require a minimum section length, or you will extract a one-line TOC entry as your risk factors.

Sections are form-aware: 10-K uses Item 1 / 1A / 7; 10-Q uses Part I Item 2 (MD&A) and Part II Item 1A. Chunking never crosses a section boundary — citation precision depends on it.

## Testing approach

Per-step manual verification is the **primary** gate, because most failure modes here are quality judgments (does this chunk actually read like a risk factor?) rather than assertions. Each step in the plan file has an explicit test to run before proceeding.

Unit tests cover only the deterministic pieces and are pinned to saved filing fixtures in `data/raw/` so they run offline: parser section boundaries, chunker metadata completeness, retriever filter behavior and RRF math.
