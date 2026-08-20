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
| **P2·2** | Retrieval evals on LangSmith — `app/evals.py`, `scripts/check_retrieval_quality.py` | **harness complete 2026-08-19**; golden set seeded but **0 rows confirmed**, so scores are provisional |

Phase 2 has since shipped two changes that add no capability: the thirteen gates were migrated onto `scripts/_gate.py` (`7df4e3f`), and fundamentals moved from a backfilled corpus to fetch-through (`03ce174`, +288/−815). `03ce174` edited `get_financials`' docstring, and docstrings are the routing logic — **re-gated 2026-08-19 and clean**: `check_router.py --classify-only` 16/16 including all four traps, `check_agent.py` benchmark all checks passed, no routing change.

**Corpus, text:** 4 tickers (AAPL, META, MSFT, TSLA), 13 filings, 806 chunks.

**Corpus, numeric:** ~387 companies and 3.8M XBRL facts held, fiscal 2009–2026 — but **nothing is backfilled and the number is not a coverage boundary.** `fundamentals.ensure_facts()` fetches any filer from SEC on first use and writes it back, so what is stored is a warm start, not the universe. **Coverage is still deliberately asymmetric** — one HTTP call buys a company's whole financial history, against ~25s of parse-and-embed per filing — so a company can have full fundamentals and still report a corpus gap for its risk factors. That remains the design; the asymmetry is now a difference in *latency* (~1.5s to fetch facts) rather than in which names are on a list.

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

# Ingestion — filing text only (slow, ~25s/filing)
.venv/bin/python scripts/ingest.py --ticker AAPL
# Fundamentals need no ingestion step — get_financials fetches on demand.

# Tests — offline, no DB / keys / model needed
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/test_parser.py::test_name

# psql is NOT installed on the host; use the container's client
docker compose exec db psql -U postgres -d research
```

Every step has a gate: `scripts/check_{db,tickers,edgar,parser,chunker,retriever,finnhub,websearch,tools,agent,api,router,data,retrieval_quality}.py`, sharing primitives from `scripts/_gate.py`. `check_router.py` has `--classify-only` (cheap: labels, no answers) and `--cost` (prices the routed graph against `build_graph(router=False)`). `check_agent.py` carries three modes — no flag is the benchmark set, `--guard` is the context guard, `--memory` is the checkpointer; `--show-thread ID` is free. `check_agent.py` and `check_api.py` cost money; `check_retrieval_quality.py` does not — it runs no model at all.

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
- **`advisory` declines the rating and then offers the work, ending on a question.** A bare refusal was the wrong shape for the traffic: most people do not arrive with a precisely scoped research question, so *"I can't do that"* sent them away from an answer that was one sentence of redirection from existing. The rule that a refusal must not list candidates is unchanged, but it is now scoped to companies the *assistant* introduces — echoing back a company the **user** named is not a recommendation, and the offer is useless without it. `check_router.py` enforces exactly that distinction (`introduced_names`) plus "ends by offering to research something instead", and carries the named-company advisory question (*"Apple stock analysis, need buy rating…"*) as a case, because that is the shape people actually type.
- **The advisory and refusal clauses moved out of `SYSTEM_PROMPT` into `ADVISORY_PROMPT`.** A question asking for a pick no longer reaches the research prompt, so carrying the refusal there would be dead text riding on every expensive turn. Side effect: the research prefix shrank from ~3.4k to ~3.16k tokens, moving *further* below the 4096 cache floor. Already inert, so nothing was lost — but **do not pad it back**, and see the cache constraint below.
- **Measured on the first live run:** 16/16 classifications correct including all four traps; the routed graph is **7.3x cheaper** than the pre-router loop on questions that need no evidence (`$0.0037` vs `$0.0271` over four). Research questions pay the classifier and save nothing — F's `runs` table is what will show the mix.
- **Re-litigated 2026-08-20 against a prompt-only alternative, and the router held.** The proposal: drop the classifier, let the tool-bound agent decide for itself whether to call tools, and resolve ambiguity by asking the user rather than a dedicated `clarify` state. A one-off 3-arm ablation (router as shipped / today's `SYSTEM_PROMPT` with no classifier as control / a strengthened prompt absorbing every clause E1 moved out) ran 35 labelled cases scored on observed behaviour, not route labels. Result: **quality was a tie** — zero unsourced research answers and identical tool-use correctness in every arm — and **cost was not**: the router was 90% cheaper over the set, with no traffic mix at which either prompt-only arm crosses over to cheaper. The one place the router did not win was latency (~13% slower than the strengthened prompt, since it is two serial model calls instead of one), and on research questions alone the classifier bought nothing — its entire value sits in the non-research two-thirds of the set, a mix this ablation chose rather than one read off production. No code changed as a result; the harness was disposable and was not kept.

### F1/F2 · XBRL fundamentals — the audited numbers as data

Every filing is inline XBRL: each figure in the statements is wrapped in a tag naming its us-gaap concept, unit and period (the AAPL FY2025 10-K carries 969 of them). `parser.html_to_lines` calls `get_text()`, which keeps `416,161` and drops the tag — which is why numbers reach the chunk corpus as prose. `app/xbrl.py` takes the same figures from SEC's free `companyfacts` API, `app/fundamentals.py` arranges them into statements, and `get_financials` serves them.

Five findings, each of which produced a plausible wrong answer before it was fixed:

- **`fy` and `fp` describe the *filing*, not the fact.** A 10-K restates its prior years, so AAPL's FY2025 filing stamps `fy=2025` on facts covering 2022-23, 2023-24 and 2024-25. Reading it labels 2023's revenue as FY2025 — a real number under the wrong year. **Everything is derived from `start`/`end`.** `check_data.py` asserts the three figures that catch it.
- **Tag fallthrough must be per period, not per line.** NVIDIA reports revenue under `RevenueFromContractWithCustomerExcludingAssessedTax` for one early year and `Revenues` since; "first candidate with any data" rendered four blank revenue years beneath a full gross-profit row. `_resolve_line` fills period by period and records every tag that contributed, so a re-tagged line reassembles. Capital expenditure has the same break at FY2020.
- **Periods must be bounded by `CURRENT_DATE`.** XBRL carries forward-dated *assumptions* — Nucor tags a health-care trend rate over 2027 in a 2011 filing — and without the bound that was Nucor's most recent "annual period", rendering a blank 2027 column and dropping a real year.
- **A ticker names the current registrant, not the whole history.** SEC maps XOM to CIK 2115436 "ExxonMobil Holdings Corp", a successor entity, not CIK 34088 which holds decades of Exxon filings. 3 of 386 companies are in this position (XOM, HONA, CBRS). Nothing follows the chain — SEC publishes no predecessor link, and guessing by name would attribute one company's numbers to another. The tool reports the absence.
- **`company_tickers.json` is only partly size-ordered.** It is ordered by market cap for recent CIKs, but companies registered long ago fall into a block near index 7100+: Exxon (CIK 34088) at 7424, JPMorgan at 7147, McDonald's at 7177, while Chevron sits at 31. A top-N slice silently misses mega-caps *by age of registration* — 22 of 128 well-known large caps were absent. Nothing depends on this any more (fundamentals are fetched on demand, so there is no universe file to slice), but it stays recorded: any future code that builds a ticker list from that file has the same trap waiting.

**Nothing is backfilled — `ensure_facts` fetches through.** One `companyfacts` call buys a company's entire filing history, so any stored universe is by definition a subset of what is free, and gating on it reports "no coverage" for a company one call would answer. `ensure_facts()` serves what is held, fetches on a miss, and writes back. Two consequences:

- **Staleness is derived from `max(filed_date)`, not from a `fetched_at` column.** A 10-Q lands every ~90 days, so `REFETCH_AFTER_DAYS = 100`. Same reasoning as periods coming from `start`/`end` rather than `fy`: a stored timestamp is a second source of truth that can disagree with the facts, and when it does it wins silently.
- **A re-fetch needs no reconciliation.** `store.upsert_facts` supersedes only on a later `filed_date`, so a restatement wins and an unchanged filing writes nothing. That is what makes fetch-through safe with no delete-first and no dirty-tracking.

Accepted cost: a ticker that genuinely stopped filing (delisted, acquired) re-fetches on every call, because `max(filed_date)` never advances again. One SEC request, rare; a negative-marker row would fix it and isn't worth building on speculation.

Two more rules worth keeping:

- **Nothing is derived.** AMD and AbbVie never tag `Liabilities`; total liabilities could be computed from `LiabilitiesAndStockholdersEquity` minus equity, and would be correct — but it would put a figure in a statement that appears in no filing. The line stays blank.
- **A blank line has two causes and they must not be conflated.** Accenture has no gross profit, Allstate no R&D, Amazon no dividend — correct absences. A mapping gap looks different, and the gate therefore asserts only the genuinely universal lines (revenue, total assets, net income, operating cash flow) and prints the rest for a human.

### P2·2 · Observability and evals both run on LangSmith

**There is no `runs` table and no cost ledger.** A Postgres ledger was built and then removed in favour of LangSmith, which already gives per-run traces, token counts, latency and cost without a schema to maintain or a bookkeeping write that could fail a paid-for answer.

- **Tracing is entirely env-driven.** LangChain reads `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` from the environment itself, so **no code turns it on and `app/config.py` has no setting for it** — a `settings.langsmith_tracing` that nothing consumes would be a setting that lies. Set the vars and every graph run, model call and tool call is traced; unset them and nothing is sent. See `.env.example`.
- **But `.env` is not the environment, and that made the above false for a while.** pydantic-settings reads `.env` into the `Settings` object and never touches `os.environ`, so every variable with a `Settings` field worked and the LangSmith ones — which only LangChain reads, straight from `os.environ` — did not. `LANGSMITH_TRACING=true` in `.env` left `tracing_is_enabled()` returning False with **no error and no warning, just zero traces**. `app/config.py` now calls `load_dotenv(PROJECT_ROOT / ".env", override=False)`. That adds no setting and names no LangSmith variable — it makes `.env` mean what the README already said it meant — and `override=False` keeps a shell-exported variable winning, so CI (real env vars, no `.env`) is unaffected.
- **`@traceable` fills in what a tool span hides — see `app/tracing.py`.** LangChain traces the graph, the model calls and the tool calls for free, but a `@tool` is one opaque span: `search_filings` reports its arguments and a block of prose, and the embedding, the fused SQL and any fetch-through to SEC happen invisibly inside it. Four functions are decorated — `retriever.search` / `hybrid_search` (as `run_type="retriever"`, shaped so LangSmith renders ranked documents keyed by `(accession_number, chunk_index)`, the same identity `app/evals.py` scores against), `embeddings.embed_query` / `embed_documents`, and `fundamentals.ensure_facts` (whose bool hides the ~1.5s warm/cold difference that *is* the remaining coverage asymmetry). Same env-driven contract: with tracing off, `@traceable` is a pass-through, measured at ~8µs against functions whose fastest member is a network round trip. **Payload shaping is load-bearing, not cosmetic** — `embed_documents` takes a batch of chunk texts and returns 768-float vectors, so traced naively one ingestion run ships the corpus to LangSmith a second time; `process_inputs`/`process_outputs` reduce both to shapes.
- **The first thing tracing found: `embed_query` was 6.61s of a 6.70s `search_filings`.** Not the SQL — the EmbeddingGemma singleton loading, because `get_model()` is called from inside `embed_query` and so lands in its span on the first search of a cold process. `app/api.py` already warms it (`api_warm_embeddings`); gate scripts did not, which meant any gate comparing two configurations charged the model load to whichever ran first. A gate that benchmarks more than one configuration needs to warm embeddings before timing either.
- **Retrieval is scored on recall@k and MRR, with no LLM judge.** Both are arithmetic, so the gate costs nothing and can run on every change. The maths is in `app/evals.py` and unit-tested offline; `scripts/check_retrieval_quality.py` owns LangSmith — it pushes the golden set as a dataset and runs `evaluate()` so scores get a history to compare a tuning change against. `--local` scores in-process with no account.
- **Both metrics exist because recall is blind to ordering.** A reranker that fixes ranking and nothing else moves MRR and leaves recall flat. A chunk is identified by `(accession_number, chunk_index)` — never by section, which many chunks of one filing share.

**The golden set is not ground truth yet, and this is the important part.** `data/retrieval_golden.json` was seeded from `data/retrieval_a7.json`, which is a dump of what the retriever *returned*. Two consequences, both recorded in the file itself:

- **Three of the eight seeds were read and rejected outright.** `material weakness in internal control` retrieved cybersecurity prose and `going concern` retrieved loss-contingency and tax prose — the retriever *missing*, not a label. `Item 7A quantitative and qualitative disclosures` names a section the parser does not extract at all. They are carried with `status="unlabelled"` and excluded from scoring, rather than counted as satisfied. Scoring an empty relevant set raises rather than returning 0.0, because "nothing was measured" is not "retrieved nothing relevant".
- **The seed is biased towards dense, so the dense/hybrid delta is currently meaningless.** A7 *is* the dense retriever — the dump's `score` values are cosine similarities, not the ~0.03 an RRF fusion produces. Measured at k=6 on the seed: **dense 1.000, hybrid 0.800** — which establishes only that hybrid ranks differently from the thing that wrote the labels. **Do not read that as dense beating hybrid.** Confirming rows is what makes the comparison mean anything, and until then every score prints as provisional.

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

- **`POST /query/stream` is the same run, reported differently — not a fifth capability.** Same graph, same nodes, same checkpoint; `.stream()` instead of `.invoke()`, translated into a small SSE vocabulary (`start`, `route`, `step`, `sources`, `token`, `reset`, `done`, `error`). `answer()` and `POST /query` are untouched, because every gate calls them and a streaming rewrite of a working path would be a change with no test behind it.
- **Its run slot is acquired by hand, never through `Depends(run_slot)`.** A generator dependency is torn down when the *response* completes, and a StreamingResponse completes the moment the handler returns — before the graph has run at all. Under `Depends` the cap would still exist and would silently stop capping. Acquired before the response is built (so "at capacity" is still a real 503), released in the generator's `finally`.
- **After the first frame, failures are `error` events, not status codes.** The status line is already sent, so the alternative is a truncated body — which a client reads as a short answer, not a failure. Only the two failures a client can act on (no model key, over the cap) stay status codes, because they are raised before the stream opens.
- **`reset` is the event that is easy to miss and expensive to drop.** Tokens stream from a model turn before anyone knows whether that turn was the answer or a preamble to a tool call; when it turns out to be a preamble, the server withdraws it. A client ignoring `reset` shows *"Let me check the filings."* glued to the front of every answer — wrong, but plausible enough to survive review.
- **`done` is read back off the checkpoint, not accumulated from tokens**, so a dropped frame cannot leave a subtly truncated answer standing. The token stream is a preview; the checkpoint is the record.
- **The stream applies `merge_citations`' dedup rule itself.** A node update carries only what that node produced, raw and undeduped, so a stream that forwarded it verbatim disagrees with `done` — sources appear during the run and then change count when it ends.

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
- **The prompt cache breakpoint is inert on the pinned model.** Anthropic ignores a `cache_control` breakpoint below a per-model floor and says nothing when it does. The floor for `claude-haiku-4-5-20251001` is **not 4096** — that was a round number picked from inside a measured bracket. Direct measurement on 2026-08-20: a **4,118-token prompt does not cache**, and a 4,569-token one does, so the real floor lies in **(4118, 4569]**. `check_agent.py`'s `CACHE_MINIMUM_TOKENS` is therefore the smallest size *observed* to cache (4569), not a guess at the boundary: the constant decides when a non-caching prompt is failed as a bug, so a value below the true floor reddens the gate with no bug behind it. The prefix is now ~4.1k after the formatting rules were added — over the old guess, under the real floor, still inert. **Do not pad the system prompt to reach the threshold** — filler degrades the routing the prompt exists to control, and ~450 tokens of it would be needed.
- **`get_company_news` returns 5 articles by default, not 10, and is clamped at `tools.MAX_NEWS_LIMIT`.** The binding cost is the *citation list*, not the context window: every article becomes a source, so an unclamped news call puts twenty-odd entries under an answer that leaned on two. A source list nobody can scan is indistinguishable from one that was never checked.
- **Finnhub reports "no such symbol" as HTTP 200 with an empty or all-zero body.** `{"c":0,...,"t":0}` validates into a $0.00 quote dated 1970, which an agent narrates as fact. Every endpoint checks the empty shape explicitly and raises `UnknownSymbolError` — a new endpoint must do the same or it becomes a fabrication path.
- **Finnhub's free tier 403s on `/stock/candle`**, so there is no price history anywhere in the system and no tool that asks for one.
- **The Tavily domain allowlist is checked twice** — sent as `include_domains` *and* re-verified locally. A filter that stops being applied produces no error, just quietly worse sources. Web results are deduplicated by canonical URL, stripping only known tracking parameters.
- **An allowlisted domain can still launder spam.** `nasdaq.com` was dropped for syndicating listicles under its own hostname. Judge a candidate domain by what it *returns* for a promotional query, not by its reputation.

## Testing approach

Per-step manual verification through the `check_*.py` gates is the **primary** mechanism, because most failure modes are quality judgments (does this chunk actually read like a risk factor?) rather than assertions. Unit tests cover only the deterministic pieces and are pinned to saved fixtures so they run offline: parser section boundaries, chunker metadata, retriever filter behaviour and RRF math, agent wiring via a scripted model, API wiring via a fake runtime.

## Planned — Phase 2

Full spec, gates, and rationale in [docs/phase-2-plan.md](docs/phase-2-plan.md), which **supersedes the original E–I roadmap**. This project exists to learn agentic AI, RAG, evals, and observability; the data layer (F1/F2 aside) had started to crowd those goals out, so the plan was cut to two new capabilities and a shrinking data layer.

**Build order is 1 → 2 → 3 → 4**, with **1 cancelled and 2's harness shipped** — the learning goals first, and that order is also what makes each measurement meaningful. LangSmith tracing replaces what the ledger would have shown when the 7th tool crosses the prompt-cache floor. Evals go in before the corpus grows, because measuring retrieval at ~806 chunks (where HNSW-with-filter is still exact) and then scaling shows a *delta*, where measuring afterwards gives one number and no way to judge it. **Each capability stops for human verification before the next begins.**

1. ~~**Run ledger**~~ — **cancelled.** Superseded by LangSmith tracing, which gives per-run tokens, latency and cost with no schema and no write that could fail an answer. See the P2·2 section above.
2. **Retrieval evals** — **harness shipped**, see P2·2 above. Outstanding: confirm rows in `data/retrieval_golden.json` (0 of 8 confirmed today), label or retire the three rejected queries, and grow the set towards ~30. Only then are cross-encoder reranking, `k`/`CANDIDATES` tuning and per-query-type RRF weighting measurable against a baseline rather than adopted on faith.
3. **Market data on yfinance** (`app/prices.py`). Replaces Finnhub for quotes and ratios and adds `get_price_history`; Finnhub keeps only `get_company_news`. Nothing stored — no `prices` table. Takes the agent to **7 tools**, so two prompt sites that currently deny price history (`app/agent.py:100` and `get_quote`'s docstring) must change together. **yfinance repeats Finnhub's defining trap** — an unknown ticker yields empty frames and `None`s, never an exception — so every function checks the empty shape and raises `UnknownSymbolError`. **Accepted tradeoff:** consolidating onto one unofficial scraper makes a Yahoo breakage take out all market data at once; mitigation is that a provider swap stays a one-file change.
4. **Text corpus coverage.** `data/sp500.txt` (static, committed, dated) drives a ~25-ticker text backfill; the rest arrive through on-demand ingestion off the corpus-gap branch in `app/tools.py` — one ingest per turn, outcomes including parser failures cached in an `ingest_attempts` table so a bad filer isn't retried for 7 days. **There is no facts backfill** — `ensure_facts` already covers every filer.

**Why 10-K Item 8 and the notes are deferred rather than built:** it is parser work, and the parser is the most fragile thing in the system. F1/F2 already put the statements in as audited XBRL — strictly better than the same numbers as prose. Changing the corpus shape right before capability 2 measures retrieval for the first time is the wrong order; revisit once there is a baseline.

**Shipped ahead of the plan, and out of order:** the React UI (`web/`, Vite + React 19) and SSE streaming — both were listed as deferred. They add no agent capability; they are how the existing one is presented, and the presentation was the thing making it hard to use. Answers render as real markdown (`react-markdown` + `remark-gfm`), so period-over-period figures arrive as tables rather than pipe-separated prose, and the run reports its progress in prose steps written by `tools.describe_tool_call` instead of a spinner that cannot say whether it is reading a 10-K or hung.

**Still deferred, not cancelled:** the ticker page (profile, chart, statement tables, peers) and its REST endpoints; 8-K and earnings surprises. The seams are left open — `company_facts` is queried by ticker and period, so a REST endpoint over it is a thin SELECT, and behind `ensure_facts` it covers any filer rather than the stored ones; `sic`/`sicDescription` arrive free in the submissions payload `app/edgar.py` already fetches, so peers is one parse function away; and the ingest notice goes through `get_stream_writer()`, so streaming picks it up free if it ever ships.

Also planned alongside: `app/ratelimit.py` (one class, separate instances per service — a shared limiter would throttle EDGAR because Finnhub was busy).

**Dependencies for unstarted modules are deliberately not in `pyproject.toml`** — add them when the module is built, not before.
