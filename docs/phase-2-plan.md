# Phase 2, replanned — stop growing the data layer, start measuring the agent

*Supersedes the original E–I roadmap. F1/F2 (XBRL fundamentals) shipped as planned and stays;
everything after it — price data, coverage, the rest of the module ordering — is replaced by
this document.*

## Context

This project exists to learn **agentic AI / graph engineering, RAG, evals, and observability**. Measured against that, the codebase has drifted:

| Learning goal | Code today | Status |
|---|---|---|
| Agentic AI / graph engineering | `agent.py` 818 + `router.py` 222 | built, gated |
| RAG | parser/chunker/embeddings/retriever/store ≈ 934 | built, **never measured** |
| Evals | — | **none** |
| Observability | `cost.py` 73, used by one gate | **none** |
| *Data layer (not a goal)* | `xbrl.py` + `fundamentals.py` + backfill + gate ≈ 990 | largest Phase 2 addition |

F1/F2 (XBRL fundamentals) shipped and gated 50/50, and it stays — it gives the agent a real numeric capability, which is what makes observability and evals worth building. But the *rest* of the planned data work is being cut down hard, and everything after it is a learning goal.

**The redirect that shapes this plan:** market data is fetched live, never stored. No `prices` table, no price ETL. yfinance owns everything price-shaped; Finnhub keeps news only. That deletes more code than it adds.

**Accepted tradeoff, on the record:** yfinance is an unofficial Yahoo scraper, and consolidating quote + history + ratios onto it makes a Yahoo breakage take out all market data at once. The earlier plan split providers specifically to avoid that. Decision taken deliberately; the mitigation is that `app/prices.py` normalises to project dataclasses, so swapping the provider back out is a one-file change.

**Intended outcome:** an agent whose retrieval quality and per-run cost are both *numbers you can see*, over a corpus that covers the S&P 500 rather than four tickers.

---

## What changed from the previous plan

| Item | Was | Now |
|---|---|---|
| `prices` table + bar ETL | F3 | **cut** — live yfinance calls, nothing stored |
| Quote provider | Finnhub | yfinance |
| `get_basic_financials` | Finnhub | yfinance |
| Finnhub's role | quotes + ratios + news + profile | **news only** |
| Text corpus | 25-ticker file | S&P 500 file, ~25 backfilled, rest on demand |
| 10-K Item 8 / notes sections | F4 | **deferred** — see below |
| Gate script duplication | opportunistic | **Step 0**, done up front |

**Why Item 8 and the notes are deferred rather than built:** it is parser work, and the parser is the most fragile thing in the system. F1/F2 already put the statements in as audited XBRL, which is strictly better than the same numbers as prose. The notes are genuinely narrative and genuinely belong in RAG — but Module H is about to establish whether retrieval works *at all*, and changing the corpus shape underneath that measurement is the wrong order. Revisit after H has a baseline.

---

## Step 0 — Tidy (no new capability) — **done, 2026-08-19**

1. **Commit F1/F2.** ✅ It was the entire uncommitted diff — `app/xbrl.py`, `app/fundamentals.py`, `scripts/backfill_facts.py`, `scripts/check_data.py`, `data/`, the two test files, plus edits to `app/edgar.py`, `app/store.py`, `app/tools.py`, `sql/schema.sql`, `CLAUDE.md`, `tests/test_agent.py`, `tests/test_tools.py`.
2. **Migrate 11 gates to [scripts/_gate.py](scripts/_gate.py).** ✅ All thirteen `check_*.py` gates now import the shared `rule` / `verdict` / `note` / `indent` / `summary` / `exit_code` primitives; no gate carries its own copy. Two side effects worth recording: `check_tools.py`'s five-tools/six-tools assertion and `check_api.py`'s three-field/four-field response contract had both drifted stale since F1/F2 and E1 shipped (neither gate had been re-run since) — fixed in passing. Several of the older gates (`check_tickers`, `check_edgar`, `check_parser`, `check_chunker`, `check_retriever`, `check_db`) never actually set a process exit code on failure before this — `verdict`/`summary`/`exit_code` fixes that as a side effect of the migration, not a separate task. All thirteen re-run clean.
3. **Replace `docs/phase-2-plan.md` with this plan** and refresh `CLAUDE.md`'s planned section. ✅

---

## Step 1 — Market data on yfinance (`app/prices.py`)

Add `yfinance` to `pyproject.toml`. New `app/prices.py`, a sibling of `app/finnhub.py` and structured the same way — module-level client, project dataclasses, errors returned as values.

```python
@dataclass(frozen=True) class Quote:        # price, change, pct, as_of
@dataclass(frozen=True) class KeyStats:     # pe, beta, 52w high/low, mkt cap, div yield
@dataclass(frozen=True) class PriceSummary: # start, end, pct_change, high, low,
                                            # realised vol, ~5 anchor points
def get_quote(symbol) -> Quote
def get_key_stats(symbol) -> KeyStats
def get_price_history(symbol, period) -> PriceSummary
```

**Carry over the fabrication guard.** Finnhub's defining trap is reporting "no such symbol" as HTTP 200 with an all-zero body, which validates into a $0.00 quote dated 1970. **yfinance has the identical failure shape** — an unknown ticker yields an empty DataFrame and a `Ticker.info` dict of `None`s, never an exception. Every function here checks the empty shape explicitly and raises `UnknownSymbolError`, exactly as `app/finnhub.py:451` does. A new endpoint that skips this becomes a fabrication path.

**`get_price_history` returns a summary, never raw bars.** 250 OHLCV rows would spend the context guard's whole budget on data the model cannot reason over, and there is no chart to feed.

**No citations** from any of these — same rule as today's `get_quote`. There is no article behind a price, and a fabricated URL in a list whose entire value is that every entry resolves is worse than no entry.

### Deletions in `app/finnhub.py`

Remove `Quote`, `get_quote`, `BasicFinancials`, `get_basic_financials`, `_pick`, `CompanyProfile`, `get_company_profile` (already unused by any tool). Keep the error classes, `_RateLimiter`, `_get`, `NewsItem`, `get_company_news`. Check whether `_assert_symbol_exists` is still reachable from the news path before deleting it. Roughly 250 of 461 lines go, and `tests/test_finnhub.py` shrinks with them.

### Tool layer

`app/tools.py`: repoint `get_quote` and `get_basic_financials` at `prices`; add `get_price_history`. Takes the agent from 6 tools to **7**.

**Two prompt sites must change together** or the system keeps denying a capability it now has:
- [app/agent.py:100](app/agent.py#L100) — *"**There is no price history**: nothing can answer 'how has the stock moved since the 10-K'"*
- [app/tools.py:258](app/tools.py#L258) — `get_quote`'s docstring repeats the same denial in its last paragraph.

`get_basic_financials`' docstring must keep routing against `get_financials`: one is *computed, unaudited, point-in-time* ratios; the other is the company's *own reported, audited* figures with history.

**Watch item:** the cacheable prefix is ~3798 tokens and Haiku's measured floor is 4096. A 7th tool will likely push it over, activating C3's `cache_control` breakpoint for the first time. That is a *benefit*, not a regression — and Step 3's ledger is what will show it. Do not pad toward it deliberately.

**Gate `scripts/check_prices.py`:** quote is fresh and non-zero for a real symbol; an unknown symbol raises rather than returning $0.00/1970; a 1y summary's `pct_change` matches its own first and last close by hand arithmetic; `get_key_stats` returns a plausible P/E for a profitable company and `None` rather than a guess for one without earnings.

---

## Step 2 — S&P 500 coverage and on-demand text ingestion

**Nothing is backfilled for fundamentals, and `scripts/backfill_facts.py` is gone.** One `companyfacts` call buys a company's whole financial history, so a stored universe was only ever a subset of what is free — and `app/tools.py` gated the tool on it, reporting "no coverage" for a company one call would answer. `fundamentals.ensure_facts()` now serves what is held, fetches on a miss and writes back; the ~387 companies already in `company_facts` stay as a warm start. `data/universe_500.txt` and `store.existing_fact_ciks` went with the script.

**`data/sp500.txt` is static and committed**, generated once from the Wikipedia constituents list with the date recorded in a header comment. Membership changes ~20 times a year; a runtime scrape would be a new failure path for no benefit. It now drives the **text** backfill only.

- **Text:** `scripts/ingest.py` gains `--tickers-file` and **per-ticker error isolation** — today the loop at [scripts/ingest.py:90](scripts/ingest.py#L90) has no `try/except`, so one `UnknownTickerError` aborts the batch. Backfill ~25 tickers (~25s per filing).
- **The asymmetry is the design and must stay legible.** One HTTP call buys a company's whole financial history; text costs ~25s of parse-and-embed per filing. A company can have full fundamentals and still report a corpus gap for its risk factors.

### On-demand ingestion

The corpus-gap branch at [app/tools.py:190](app/tools.py#L190) is the trigger — `_covered_tickers`' docstring already names itself as the hook. On a gap for a real ticker: ingest the latest 10-K, then retry the search.

- **One per turn**, enforced through a `config: RunnableConfig` parameter on `search_filings`. LangChain excludes it from the model-facing schema. A `contextvars` approach would look cleaner and **fail silently**, because `ToolNode` runs tools in its own threadpool where context does not propagate.
- **Cache the outcome, including failure:**

```sql
CREATE TABLE IF NOT EXISTS ingest_attempts (
    ticker       text PRIMARY KEY,
    status       text NOT NULL,   -- 'ok' | 'failed'
    reason       text,
    attempted_at timestamptz NOT NULL DEFAULT now()
);
```

  A `failed` row suppresses retries for 7 days. **Record `failed` when the parser falls back to whole-document** (`SECTION_UNKNOWN`, [app/parser.py:254](app/parser.py#L254)) — today [scripts/ingest.py:52](scripts/ingest.py#L52) only warns, and an undifferentiated blob is worse than a recorded failure: it poisons the ledger, suppresses future retries, and yields citations pointing at nothing specific.
- **Progress is a log line for now**, but written through `langgraph.config.get_stream_writer()` — one line, and streaming picks it up free if it ever ships.

**Scaling watch item, recorded not solved:** `retriever.hybrid_search`'s dense CTE ([app/retriever.py:217](app/retriever.py#L217)) filters by ticker *and* orders by HNSW distance. At ~3k chunks Postgres seq-scans and results stay exact. Past roughly 50k, HNSW-with-filter under-returns unless `hnsw.ef_search` is raised or pgvector 0.8's `hnsw.iterative_scan` is enabled.

**Gate — extend `scripts/check_data.py`:** an uncovered-but-real ticker ingests and answers; a repeat ingests nothing and is faster; a forced parse failure writes `ingest_attempts` and is not retried inside the window; a ticker with fundamentals but no filings answers a numeric question *and* reports a corpus gap for a risk-factor question in the same conversation.

**Regression, and it costs money:** full `scripts/check_agent.py` benchmark plus `scripts/check_router.py`. Seven tools, three of which plausibly answer "what's going on with Apple", makes tool docstrings the routing logic. Per the standing rule, **the first hypothesis for a routing failure is the model** — re-run on `claude-sonnet-5` before editing a docstring.

---

## Step 3 — Observability (`app/observability.py`)

A `runs` ledger in Postgres: `run_id`, `thread_id`, `turn_index`, `route`, model, input/output/cache tokens, `cost_usd`, `duration_ms`, `tool_calls`, `citation_count`, `status`. `app/cost.py` already exists (`PRICING`, `estimate_usd` returning `None` rather than a guess for an unpriced model) and gets extended rather than replaced. LangSmith tracing goes in `.env.example` only — LangChain reads those vars from the environment, and a `settings.langsmith_tracing` that nothing consumes is a setting that lies.

Two things to get right:

- **Slice the turn out of the thread.** `answer()` returns whole-thread messages, so counting tokens over `state["messages"]` on turn 2 re-counts turn 1. Read `len(thread_state(thread_id)["messages"])` before invoking and record only the tail.
- **Recording must never fail a run.** Wrap it in `try/except Exception` and log. A bookkeeping table that can 500 an answer the user already paid for is worse than no table.

`QueryResponse` gains `run_id` (it gained `route` in E).

**Gate `scripts/check_observability.py`:** two turns on one thread give `turn_index` 0 and 1; turn 1's tokens are not re-counted in turn 0's row; `cost_usd` matches hand arithmetic; `route` is recorded on first turns and NULL on continuations; a failing run records `status='error'`; a `record_run` monkeypatched to raise does not break `answer()`.

The query that motivates the module — and the one that answers *"our agent is expensive even on the lowest model"* with a number instead of a feeling:

```sql
SELECT route, count(*), avg(cost_usd)::numeric(10,6), avg(duration_ms)::int
FROM runs WHERE route IS NOT NULL GROUP BY route ORDER BY 2 DESC;
```

---

## Step 4 — Retrieval evals (`scripts/check_retrieval_quality.py`)

Retrieval has no number attached to it. Hybrid RRF beat dense-only on exact-phrase queries in the A8 gate, but nothing measures where it falls short.

**A golden set, deliberately without an LLM judge.** ~30 `(query, expected accession + section)` pairs, hand-labelled against the ingested corpus, scored on **recall@k and MRR**. Both are arithmetic — no model, no cost, runnable on every change.

**Seed it from [data/retrieval_a7.json](data/retrieval_a7.json)**, which already holds 8 queries × 3 results carrying `query`, `filters`, `accession_number` and `section` — exactly the required shape. It is a system *output* dump, not ground truth, so each row must be **hand-confirmed or corrected** before it counts as a label; treating what the retriever returned as what it should have returned would bake today's behaviour in as the target.

Then, measured against that baseline rather than adopted on faith:

- **Cross-encoder reranking** — a local model (`sentence-transformers` is already a dependency), so no API cost and no new key. Retrieve `CANDIDATES`, rerank, keep `k`.
- **`k` and `CANDIDATES` tuning** — both constants today ([app/retriever.py](app/retriever.py): `DEFAULT_K = 6` line 23, `CANDIDATES = 50` line 28).
- **Per-query-type RRF weighting** — exact-phrase questions should favour the sparse half, conceptual ones the dense half. `RRF_K = 60` (line 41) is currently uniform.

The gate prints recall@k and MRR for each configuration side by side, and **fails only if a change makes the baseline worse**. A reranker that costs 200ms and gains nothing should be visible as such and then removed.

---

## Still deferred

The ticker page (profile, chart, statement tables, peers) and its REST endpoints; SSE streaming; the React UI; 8-K and earnings surprises; 10-K Item 8 and the notes.

The seams are left open: `company_facts` is queried by ticker and period so a REST endpoint over it is a thin SELECT (behind `ensure_facts`, so it would cover any filer rather than the stored ones); `sic`/`sicDescription` arrive free in the submissions payload [app/edgar.py:162](app/edgar.py#L162) already fetches, so peers is one parse function away; and Step 2's ingest notice goes through `get_stream_writer()` so streaming picks it up for free.

---

## Verification

`.venv/bin/python -m pytest` must stay green throughout — it is offline and needs no DB, keys, or model.

| Step | Command | Costs money |
|---|---|---|
| 0 | every `scripts/check_*.py` still exits 0 | some do — confirmed clean 2026-08-19 |
| 1 | `scripts/check_prices.py` | no |
| 2 | `scripts/ingest.py --tickers-file data/sp500.txt --limit 2` (~25 min) | no |
| 2 | `scripts/check_data.py` | partly |
| 2 | `scripts/check_agent.py` + `scripts/check_router.py` — the 7-tool regression | **yes** |
| 3 | `scripts/check_observability.py` | yes |
| 4 | `scripts/check_retrieval_quality.py` | no |

**End to end after Step 2**, exercising the whole asymmetry in one conversation:

```bash
.venv/bin/python scripts/check_agent.py --ask \
  "How has Nvidia's gross margin trended over five years, and how has the stock moved this year?"
```

Audited fundamentals from XBRL with EDGAR-resolving citations, a live price summary from yfinance with none — and if the same thread then asks about risk factors, either an on-demand ingest or an honest corpus gap, never a news-grounded substitute.

**Each step stops for human verification before the next begins.**
