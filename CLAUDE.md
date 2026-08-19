# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

An AI-driven financial research platform. A user asks a natural-language question about a public company ("What are Apple's key risk factors this year?") and gets an answer grounded in real SEC EDGAR filings via RAG, plus live market data and web search via tools, with citations that resolve back to the exact filing and section.

## Current state

**Phase 1 is complete and gated live.** A running service answering real questions from filings, market data, and the web, with verifiable typed citations. **Phase 2 is in progress** — Module E has shipped.

| Module | What | Status |
|---|---|---|
| **A** | Data pipeline — EDGAR → Postgres, hybrid retrieval | complete (A0–A8) |
| **B** | Live data — Finnhub, Tavily | complete |
| **C** | LangGraph agent — ReAct loop | complete (C4c cancelled, C5's `lookup_company` cut) |
| **D** | FastAPI service — `POST /query`, `GET /health`, `GET /threads/{id}` | complete |
| **E** | Query router — 4-class pre-dispatch | **complete, gated live 2026-08-19** (34/34; benchmark and `--memory` re-run clean) |
| **F1/F2** | XBRL fundamentals + `get_financials` | **complete, gated 2026-08-19** (50/50; benchmark and router re-run clean) |

**Corpus, text:** 4 tickers (AAPL, META, MSFT, TSLA), 13 filings, 806 chunks.

**Corpus, numeric:** 386 companies, 3.6M XBRL facts, ~1.7 GB, fiscal 2009–2026. **Coverage is deliberately asymmetric** — one HTTP call buys a company's whole financial history, against ~25s of parse-and-embed per filing — so a company can have full fundamentals and still report a corpus gap for its risk factors. That is the design, not a contradiction; `fundamentals.covered_tickers()` and `tools._covered_tickers()` answer the two halves separately and must not be conflated.

**Corpus, text (detail):** Exactly **one 10-K per ticker** — the other nine are 10-Qs — so **no year-over-year comparison of filings is answerable for any covered company**. Write gate questions against what is ingested, not against what the ingestion policy says it collects.

**Model:** `claude-haiku-4-5-20251001`, pinned to the dated snapshot rather than the alias, so model-scoped gate findings stay attached to a fixed model. Tool routing is the most model-sensitive thing here — if a routing failure appears, **the first hypothesis is the model, not a docstring bug**. Re-run on `claude-sonnet-5` to separate the two before editing `app/tools.py`.

## Commands

```bash
# Postgres (pgvector). --wait blocks until the healthcheck passes.
docker compose up -d --wait
docker compose down          # add -v to also drop the data volume

# Environment. Must be Python 3.13 — see constraints below.
/usr/local/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# The service. Applies the schema, compiles the graph, warms the embedding
# model at startup — expect ~10s before it answers.
.venv/bin/python -m uvicorn app.api:app --reload

# Ingestion — filing text (slow, ~25s/filing) and fundamentals (one call/company)
.venv/bin/python scripts/ingest.py --ticker AAPL
.venv/bin/python scripts/backfill_facts.py --tickers-file data/universe_500.txt

# Tests — offline, no DB / keys / model needed
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/test_parser.py::test_name

# psql is NOT installed on the host; use the container's client
docker compose exec db psql -U postgres -d research
```

Every step has a gate: `scripts/check_{db,tickers,edgar,parser,chunker,retriever,finnhub,websearch,tools,agent,api,router,data}.py`, sharing primitives from `scripts/_gate.py`. `check_router.py` has `--classify-only` (cheap: labels, no answers) and `--cost` (prices the routed graph against `build_graph(router=False)`). `check_agent.py` carries three modes — no flag is the benchmark set, `--guard` is the context guard, `--memory` is the checkpointer; `--show-thread ID` is free. `check_agent.py` and `check_api.py` cost money.

## Architecture

Data flows one direction; each stage is a separate file so it can be tested alone.

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

### Storage and retrieval

Postgres + pgvector, one `chunks` table. Retrieval is **hybrid** — dense vector similarity + BM25 full-text, fused by Reciprocal Rank Fusion in a **single SQL query**. RRF rather than weighted score blending because cosine distance and `ts_rank_cd` are on incomparable scales. Filings are full of exact-match tokens (`material weakness`, `going concern`) that embeddings blur; BM25 catches those literally.

This is why pgvector rather than Chroma: Chroma has no real BM25, so hybrid there means a second index kept in sync.

### The graph is a router in front of a 2-node ReAct loop

```
START → router ─┬─ respond → END          (simple | advisory | clarify)
                └─ agent ⇄ tools → END    (research)
```

The loop is unchanged from C2: `agent ⇄ tool_node`, looping until the model stops emitting tool calls, with **no synthesis node** — the final tool-call-free turn *is* the synthesis.

**E1's router is pre-dispatch, not tool routing.** It chooses which prompt and which tool set run at all; `tools_condition` still routes *within* a research question. The C2 argument ("a tool-calling model already routes") is about the latter and still holds. See the E1 section below.

The load-bearing consequence: **tools are list entries, not graph nodes.** Adding a capability means appending one `@tool` function to `TOOLS` in `app/tools.py` — never modifying the graph. Retrieval is itself a tool (`search_filings`), which is what makes multi-hop questions work.

With five tools, three of which plausibly answer "what's going on with Apple," **tool docstrings are the routing logic.** If the agent picks the wrong source, fix the docstring, not the graph.

Key decisions:
- **State is three fields** — `messages` (reduced by `add_messages`), `citations` (reduced by `merge_citations`), and `route` (E1's dispatch decision, last write wins). Citations merge through a reducer so a run that stops early still carries the sources it gathered. The rule used to be two fields; a fourth should have to argue for itself the way `route` did.
- **`tool_node` wraps the prebuilt `ToolNode`**, it does not replace it. Its only job is reading `artifact` off each `ToolMessage`.
- **Citations de-duplicate on `(type, label, source_url)`, never URL alone.** Two sections of one 10-K share a URL and are genuinely different sources.
- **The system prompt is prepended per turn, not stored in state** — keeps the cacheable prefix byte-identical every turn.
- **The model is built lazily** (`get_llm()`), so importing `app.agent` needs no API key.
- **`RECURSION_LIMIT = 25`** — roughly a dozen tool calls.
- **Retry `agent`, `router` and `respond` — never `tools`.** The tools node converts provider failures to text on purpose, so retrying it would re-bill live Finnhub and Tavily calls.
- **Context guard** (`trim_tool_results`) truncates old tool-result *bodies* on the way into the model; it never drops messages, because dropping can orphan a `ToolMessage` from its `AIMessage` and Anthropic rejects that. Applied to the invoke payload, not to state, so it is idempotent and the checkpoint keeps the real conversation.
- **Checkpointer** is `PostgresSaver` over the pool `db.py` already owns. `setup()` runs on a short-lived autocommit connection because its migrations use `CREATE INDEX CONCURRENTLY`, which Postgres refuses inside a transaction. `_build_lock` is an `RLock` — a plain `Lock` deadlocks silently. `Citation` is registered with the serializer.
- **`answer()` returns the whole thread's messages and citations, not the turn's.** The reducers accumulate, and turn 2's answer genuinely rests on turn 1's passages.
- **A resumed turn usually makes no tool call at all**, and that is correct — the passages are still in the conversation. Follow-up turns are much cheaper than first turns.

### E1 · The query router

Four classes. Three of them terminate at `respond_node` — one model call, a small prompt, **no tools bound**, then END. Only `research` enters the loop.

| Route | Meaning |
|---|---|
| `research` | Needs filings, market data, news, or the web. **The default.** |
| `simple` | Answerable from general knowledge of a company's *identity* alone — who leads it, what it does, its industry, HQ, listing venue. |
| `advisory` | Asks for a pick, a ranking, a screen, or a buy/sell/hold judgment. |
| `clarify` | Cannot be acted on as written — no identifiable company, or two readings leading to different work. |

- **The asymmetry is the safety argument, and it is enforced twice** — in `ROUTER_PROMPT` and again in `classify()`, which maps anything unrecognised to `research`. A `research → advisory` slip is a visibly bad answer. A **`research → simple` slip is an unsourced answer to a question that needed evidence** — fluent, confident, citation-free — which is the exact failure the corpus-gap check exists to prevent, arriving through a different door. `check_router.py` fails that case by name, separately from ordinary misclassification. The reverse (`simple → research`) is tolerated and merely reported: it costs money and returns a correct sourced answer, which is the direction to be wrong in.
- **A turn on a thread with history is never classified.** Read standalone, *"which of those involve suppliers outside the US"* is textbook `clarify`; routing it there would send C4b's whole multi-turn capability into a clarification prompt. `router_node` checks the message count and returns `research` without a model call, so follow-ups also pay no classifier.
- **`route` is written on every path**, never left to a `.get()` default — the checkpoint holds the previous turn's value, and a node that returned nothing would let it stand and silently steer a later turn.
- **The classifier binds no tools, and that is the whole economy.** `with_structured_output` was tried first and removed: measured 2026-08-19, it cost **974 input tokens per call against 297 for the prompt and question alone** — ~677 tokens of schema and forced `tool_choice` wrapped around a single enum field, i.e. more overhead than the prompt it protected. The label now comes back as bare text and `parse_label()` reads it; a reply naming two routes, or none, falls to `research`. Pinned by a test, because "let's make this type-safe" would silently restore the 3.3x.
- **The advisory and refusal clauses moved out of `SYSTEM_PROMPT` into `ADVISORY_PROMPT`.** A question asking for a pick no longer reaches the research prompt, so carrying the refusal there would be dead text riding on every expensive turn. Side effect: the research prefix shrank from ~3.4k to ~3.16k tokens, moving *further* below the 4096 cache floor. Already inert, so nothing was lost — but **do not pad it back**, and see the cache constraint below.
- **Measured on the first live run:** 16/16 classifications correct including all four traps; the routed graph is **7.3x cheaper** than the pre-router loop on questions that need no evidence (`$0.0037` vs `$0.0271` over four). Research questions pay the classifier and save nothing — F's `runs` table is what will show the mix.

### F1/F2 · XBRL fundamentals — the audited numbers as data

Every filing is inline XBRL: each figure in the statements is wrapped in a tag naming its us-gaap concept, unit and period (the AAPL FY2025 10-K carries 969 of them). `parser.html_to_lines` calls `get_text()`, which keeps `416,161` and drops the tag — which is why numbers reach the chunk corpus as prose. `app/xbrl.py` takes the same figures from SEC's free `companyfacts` API, `app/fundamentals.py` arranges them into statements, and `get_financials` serves them.

Five findings, each of which produced a plausible wrong answer before it was fixed:

- **`fy` and `fp` describe the *filing*, not the fact.** A 10-K restates its prior years, so AAPL's FY2025 filing stamps `fy=2025` on facts covering 2022-23, 2023-24 and 2024-25. Reading it labels 2023's revenue as FY2025 — a real number under the wrong year. **Everything is derived from `start`/`end`.** `check_data.py` asserts the three figures that catch it.
- **Tag fallthrough must be per period, not per line.** NVIDIA reports revenue under `RevenueFromContractWithCustomerExcludingAssessedTax` for one early year and `Revenues` since; "first candidate with any data" rendered four blank revenue years beneath a full gross-profit row. `_resolve_line` fills period by period and records every tag that contributed, so a re-tagged line reassembles. Capital expenditure has the same break at FY2020.
- **Periods must be bounded by `CURRENT_DATE`.** XBRL carries forward-dated *assumptions* — Nucor tags a health-care trend rate over 2027 in a 2011 filing — and without the bound that was Nucor's most recent "annual period", rendering a blank 2027 column and dropping a real year.
- **A ticker names the current registrant, not the whole history.** SEC maps XOM to CIK 2115436 "ExxonMobil Holdings Corp", a successor entity, not CIK 34088 which holds decades of Exxon filings. 3 of 386 companies are in this position (XOM, HONA, CBRS). Nothing follows the chain — SEC publishes no predecessor link, and guessing by name would attribute one company's numbers to another. The tool reports the absence.
- **`company_tickers.json` is only partly size-ordered.** It is ordered by market cap for recent CIKs, but companies registered long ago fall into a block near index 7100+: Exxon (CIK 34088) at 7424, JPMorgan at 7147, McDonald's at 7177, while Chevron sits at 31. A top-N slice silently misses mega-caps *by age of registration* — 22 of 128 well-known large caps were absent. `data/universe_500.txt` carries an explicit supplement for that reason.

Two more rules worth keeping:

- **Nothing is derived.** AMD and AbbVie never tag `Liabilities`; total liabilities could be computed from `LiabilitiesAndStockholdersEquity` minus equity, and would be correct — but it would put a figure in a statement that appears in no filing. The line stays blank.
- **A blank line has two causes and they must not be conflated.** Accenture has no gross profit, Allstate no R&D, Amazon no dividend — correct absences. A mapping gap looks different, and the gate therefore asserts only the genuinely universal lines (revenue, total assets, net income, operating cash flow) and prints the rest for a human.

### Citations are assembled in code, never by the LLM

Models mangle URLs and invent accession numbers; the retriever already knows ground truth. Each tool is `@tool(response_format="content_and_artifact")` and returns `(text, list[Citation])`. The `Citation` objects ride on the `ToolMessage` **without entering the model's context**, tagged `filing` / `news` / `web` so an audited SEC filing is distinguishable from a news article.

- **The `[n]` markers in a tool's text index its artifact list position for position**, so there is deliberately **no de-duplication inside a tool call**. De-duplication happens once, downstream, where the whole answer's list is assembled.
- **XBRL citations are built from cik + accession**, not looked up. A fact names the filing it was reported in, but the corpus usually holds no `filings` row for it — under asymmetric coverage that is the normal case. `edgar.filing_index_url` builds the EDGAR index URL from ids alone, so a number cites an openable filing for a company whose text was never ingested. **The accession's own prefix is the filing agent's CIK, not the company's**, so the CIK is carried through `Statement`.
- **`get_quote` and `get_basic_financials` return no citations.** There is no article behind a quote, and a fabricated URL in a list whose entire value is that every entry resolves is worse than no entry.

### Tool failures are returned as text, never raised

A raised tool error tells the model only "something broke"; "Finnhub returned no data for symbol 'APPL'" tells it what to do next. Genuinely unexpected exceptions propagate — a bug should look like a bug.

### `search_filings` reports a corpus gap, not an empty result

Nothing ingests automatically. A ticker never run through `scripts/ingest.py` retrieves nothing — identical to a company that genuinely never discussed the topic. The failure is not a missing answer but a misleading one: `web_search` and `get_company_news` work for *any* ticker, so the agent falls back to them and presents a news-grounded answer as though the filings had been consulted. The tool checks the `filings` ledger before searching and returns a distinct corpus-gap result.

**A nonsense query does not produce an empty result** — hybrid search's dense half always returns nearest neighbours, so an ingested ticker with an absurd query returns *irrelevant* passages, not `[]`. The genuine empty path is reached through filters. **`section` is validated against the parser's own vocabulary**; left unvalidated, the model invents `"Risk Factors"`, gets nothing, and reports the silence as a fact.

### The API

`app/api.py` is thin — `POST /query` is a call to `agent.answer()` and a response model. What it owns is what only matters under concurrency:

- **Endpoints are plain `def`**, so Starlette runs them in the anyio threadpool. `graph.invoke()` is uninterruptible either way; **a client that disconnects mid-run does not cancel it and holds its slot to completion.**
- **`MAX_CONCURRENT_RUNS = 4` is arithmetic, not taste** — the psycopg pool is `max_size=8` and one request fans out to ~2 brief borrows per `search_filings`. `embeddings._encode_lock` and `finnhub._RateLimiter` are the other serialisation points. Per process; single-worker is the shipped configuration.
- **Over the cap is an immediate 503 with `Retry-After`, never a queue.** A run is 20–60s, so queuing is indistinguishable from a hang.
- **The lifespan builds everything; `get_runtime` only hands it out.**
- **Startup failure is asymmetric.** Postgres unreachable aborts startup; a missing `ANTHROPIC_API_KEY` does not — it is a per-request capability, and `/health` exists to report it.
- **An unknown `thread_id` is 404, not a 200 with `[]`** — silence read as a fact is the corpus-gap failure through a different door.
- **Three exception handlers and no catch-all.** `GraphRecursionError` is 500, not 429/503: a retry re-bills the identical loop, so a status implying "try again" costs money to obey.

### Parsing is the fragile part

`app/parser.py` splits filing HTML into named sections and is most likely to need iteration — EDGAR HTML varies by filer and year. Two defenses are required:

1. **Degrade, don't crash** — if no sections are confidently detected, chunk the whole document with `section="unknown"` and log a warning.
2. **Guard against the table of contents** — "Item 1A." also appears in the TOC. Match the *last* occurrence and require a minimum section length, or you extract a one-line TOC entry as your risk factors.

Sections are form-aware: 10-K uses Item 1 / 1A / 7; 10-Q uses Part I Item 2 and Part II Item 1A. Chunking never crosses a section boundary — citation precision depends on it.

## Constraints that will silently break things

- **Embedding dimension is coupled across three places.** `google/embeddinggemma-300m` → 768 dims → `vector(768)` in `sql/schema.sql` → `settings.embedding_dim`. Changing the model means changing all three plus a full re-embed.
- **EmbeddingGemma is a gated HF repo.** Accept the license and `huggingface-cli login` (or set `HF_TOKEN`) once before the first embedding run. One-time download gate only.
- **Query and document prompts differ.** `task: search result | query: ...` vs `title: none | text: ...`. Swapping them **degrades retrieval silently**. Only `embed_query()` / `embed_documents()` are exposed, each pinning its own template.
- **The embedding model is a process-wide singleton that must be locked twice, and getting it wrong crashes rather than raises.** `@lru_cache` caches the *result* but not the *call* — four threads built four models and four simultaneous Metal/MPS inits **segfaulted CPython**. Fixing only the load race still died (SIGTRAP) on concurrent `encode()`. Both `get_model()` (double-checked locking) and the encode path (`_encode_lock`) are required.
- **A gate cannot catch that class of bug.** The benchmark's first question warms the singleton before anything runs in parallel. When a gate passes, it establishes what it exercised, not that the step is correct.
- **Chunk size is a quality choice, not a model limit.** The model accepts 2048 tokens; chunks target ~512 because a chunk spanning three risk factors averages into a vector matching none of them. Measure with the model's own tokenizer, never a character count.
- **SEC requires a descriptive `User-Agent`** (`EDGAR_USER_AGENT`, `"Name email@example.com"`) or it blocks requests. Set centrally in `app/sec_http.py`, rate-limited to ~3 req/sec; never per call site.
- **Python must be 3.13.** The machine's default `python3` is 3.14, which lacks reliable torch/sentence-transformers wheels.
- **Re-ingestion must stay idempotent.** `UNIQUE (accession_number, chunk_index)` + `ON CONFLICT ... DO UPDATE`; `scripts/ingest.py` also skips accession numbers already in `filings`. The standing test is that running ingestion twice leaves row counts unchanged.
- **`chunks.content_tsv` is a `GENERATED` column** — Postgres maintains it. Never write to it; hybrid search needs no extra ingestion work as a result.
- **`sql/schema.sql` is applied on every startup**, so every statement must remain idempotent.
- **The prompt cache breakpoint is inert on the pinned model.** Anthropic ignores a `cache_control` breakpoint below a per-model floor and says nothing when it does. `claude-haiku-4-5-20251001`'s real floor is **4096** (measured; the docs say 2048), and the prefix is ~3.4k. It stays because it is correct and free. **Do not pad the system prompt to reach the threshold** — filler degrades the routing the prompt exists to control.
- **Finnhub reports "no such symbol" as HTTP 200 with an empty or all-zero body.** `{"c":0,...,"t":0}` validates into a $0.00 quote dated 1970, which an agent narrates as fact. Every endpoint checks the empty shape explicitly and raises `UnknownSymbolError` — a new endpoint must do the same or it becomes a fabrication path.
- **Finnhub's free tier 403s on `/stock/candle`**, so there is no price history anywhere in the system and no tool that asks for one.
- **The Tavily domain allowlist is checked twice** — sent as `include_domains` *and* re-verified locally. A filter that stops being applied produces no error, just quietly worse sources. Web results are deduplicated by canonical URL, stripping only known tracking parameters.
- **An allowlisted domain can still launder spam.** `nasdaq.com` was dropped for syndicating listicles under its own hostname. Judge a candidate domain by what it *returns* for a promotional query, not by its reputation.

## Testing approach

Per-step manual verification through the `check_*.py` gates is the **primary** mechanism, because most failure modes are quality judgments (does this chunk actually read like a risk factor?) rather than assertions. Unit tests cover only the deterministic pieces and are pinned to saved fixtures so they run offline: parser section boundaries, chunker metadata, retriever filter behaviour and RRF math, agent wiring via a scripted model, API wiring via a fake runtime.

## Planned — Phase 2

Full design, gates, and rationale in [docs/phase-2-plan.md](docs/phase-2-plan.md). Ordered; each module stops for human verification before the next begins.

- **F3/F4 — the rest of the data layer.** Price history via a `yfinance` sibling client and `get_price_history` (a summary, never raw bars); widening the text corpus with 10-K Item 8 and the notes; a 25-ticker text backfill; on-demand ingestion triggered by the corpus gap, with a failure cache so a filer the parser cannot handle is not re-attempted every turn.
- **G — Observability and cost accounting.** A `runs` ledger in Postgres (tokens, cache, dollars, latency, route, tool calls) and LangSmith tracing as an env toggle. `app/cost.py` already exists, pulled forward for E's gate.
- **H — Retrieval quality.** A retrieval-only golden set scored on recall@k and MRR — arithmetic, so no LLM judge and no per-run cost — then cross-encoder reranking, k tuning and per-query-type RRF weighting measured against that baseline rather than adopted on faith.

**Deferred, not cancelled:** the ticker page (profile, chart, statement tables, peers) and its REST endpoints; SSE streaming; the React UI; 8-K and earnings surprises. F leaves the seams — `company_facts` and `prices` are queried by ticker and period, so REST endpoints over them are thin SELECTs; `sic`/`sicDescription` arrive free in the submissions payload `app/edgar.py` already fetches, so peers is one parse function away.

Also planned alongside: `app/ratelimit.py` (one class, separate instances per service — a shared limiter would throttle EDGAR because Finnhub was busy). `scripts/_gate.py` is **done** — the shared `rule` / `verdict` / `note` / `indent` / `summary` / `exit_code` primitives, extracted when E made this the twelfth gate. New gates import it; the older eleven still carry their own copies and can be migrated opportunistically.

**Dependencies for unstarted modules are deliberately not in `pyproject.toml`** — add them when the module is built, not before.
