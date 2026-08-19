# Phase 2 — router, observability, streaming, coverage, UI

## Context

Phase 1 is a complete, gated service: `POST /query` answers a natural-language question from SEC filings (hybrid RAG), Finnhub market data, and Tavily web search, with typed citations that resolve. Four modules (A–D), all gated live.

It has five gaps that make it a demo rather than a product:

1. **Every question costs the same shape of run.** "Who is Tesla's CEO?" pays a full agent turn with five tool schemas and a ~1k-token system prompt, then tool calls, then another turn. There is no cheap path and no way to decline early.
2. **No cost visibility.** `agent.token_usage()` exists but is called only by `scripts/check_agent.py`. Nothing in the request path records tokens, dollars, or latency.
3. **No UI and no streaming.** `POST /query` blocks 20–60s. There is zero frontend code in the repo — no `package.json`, no `ui/`.
4. **Four tickers.** AAPL/META/MSFT/TSLA, 13 filings, 806 chunks. Everything else hits a corpus gap.
5. **No price history, and the prompt says so.** Finnhub's free tier 403s on `/stock/candle`.

**Intended outcome:** a Perplexity-Finance-style streaming UI over a cost-routed, instrumented agent covering ~25 backfilled tickers plus on-demand ingestion for everything else.

**Decisions taken (from scoping):**
- Full roadmap, ordered modules E–I, same "stop for human verification at each gate" discipline as Phase 1.
- **A query router at the head of the graph**, built first.
- Advisory answers **keep C5's rule** — explain the refusal, never name companies.
- The router is **skipped entirely on threads with history**.
- **~25-ticker backfill**, on-demand ingestion as the primary coverage path.
- **Observability/tracing only** — no golden-set eval suite this phase.

**One assumption, stated because it was a deliberate call.** Building the router before observability means its win is sized by design rather than by measured traffic — the opposite of the C4c/C5 precedent, where features were cancelled on evidence rather than built on spec. The mitigation is in E's gate: `build_graph(router=False)` keeps the two-node graph compilable, so the gate runs one question set through both shapes and prints the real token and dollar delta. That is the evidence, arriving at the gate instead of before it.

---

## Module ordering

| Module | What | Why here |
|---|---|---|
| **E** | Query router (4-class pre-dispatch + respond node) | The biggest structural change; land it while the graph is still three nodes. |
| **F** | Observability & cost accounting | Instruments the finished graph shape, and makes cost-by-route a one-line query. |
| **G** | Streaming (SSE) | Unblocks the UI *and* gives H's 25-second on-demand ingest a channel to say what it's doing. |
| **H** | Coverage: backfill, on-demand ingest, price history | Needs G's progress channel to ship the ingest stall honestly. |
| **I** | React UI | Consumes G's event stream and H's price data. The only module whose gate is a human. |

Dependencies for unstarted modules stay out of `pyproject.toml` — add them when the module is built.

---

## Module E — The query router

**Steps: E1 (router + respond nodes) · E2 (the `simple` branch's memory whitelist)**

### Why this does not contradict C2

`agent.md` records "there is deliberately no router node — a tool-calling model already routes." That argument is about routing *within* a research question, and it still holds: `tools_condition` stays exactly as it is.

This router does a different job — **pre-dispatch**. It chooses which prompt and which tool set run at all, and it adds a terminal path (clarification) the two-node loop cannot express. Adding it does not make tools into graph nodes, and the "adding a capability means appending one `@tool`" property survives untouched.

### The shape: three nodes, not five

Three of the four classes are the same mechanical thing — one cheap model call, a small prompt, no tools bound, then END. They differ only in which prompt fragment is used. So:

```
START → router ─┬─ respond → END          (simple | advisory | clarify)
                └─ agent ⇄ tools → END    (research)
```

`agent`, `tools`, `tools_condition`, `RETRY_POLICY`, the context guard, the checkpointer and the citation reducer are all unchanged.

### The four classes

| Route | Meaning | Terminal? |
|---|---|---|
| `research` | Needs filings, market data, news, or the web. **The default.** | no — ReAct loop |
| `simple` | Answerable from the memory whitelist alone (E2). | yes |
| `advisory` | Asks for a pick, a ranking, a screen, or a buy/sell/hold judgment. | yes |
| `clarify` | Cannot be acted on as written — no identifiable company, or two readings that lead to genuinely different work. | yes |

**The asymmetry rule is load-bearing and belongs in two places.** In the prompt: anything not clearly one of the other three is `research`. In code: an unknown or unparseable label maps to `research`, never to `respond`. A `research → simple` misclassification produces a fluent, zero-citation, memory-sourced answer to a question that needed filings — precisely the failure the corpus-gap check and the whole citation design exist to prevent. Wrong-and-expensive is recoverable; wrong-and-unsourced is not.

### New file `app/router.py`

Holds `ROUTER_PROMPT`, the three `RESPOND_PROMPTS` fragments, `classify()`, and `get_router_llm()`. Kept out of `app/agent.py` for the same reason `tools.py` is separate — `agent.py` stays the graph and nothing else.

- **`get_router_llm()`** is a second lazily-built singleton on the same pinned model, `.with_structured_output(Route)` where `Route` is a one-field enum model. **It is deliberately not tool-bound** — binding the five research tool schemas to the classifier would spend exactly the savings this module exists to produce.
- **`ROUTER_PROMPT` targets under ~300 tokens.** A classification call is then roughly 300 in / 10 out — against a research turn's ~3.4k prefix plus retrieved passages. That ratio is the whole economic argument, so the prompt staying small is a constraint, not a preference.
- **No `cache_control` on any of the four small prompts.** All sit far below any published minimum; a breakpoint there would be decoration.

### Changes to `app/agent.py`

- **`AgentState` gains a third field**, `route: str`. The two-field rule is documented as "anything more is state the graph doesn't need" — this is needed: the conditional edge reads it and the API reports it. Plain last-write-wins, no reducer.
- **`router_node`** writes `route` on *every* turn, and owns the skip rule:

  ```python
  def router_node(state):
      # A follow-up is not a fresh question. "Which of those involve suppliers
      # outside the US" is textbook `clarify` read standalone, and routing it
      # there would send C4b's multi-turn capability to a clarification prompt.
      # Continuations go to the loop, which is also the safe direction.
      if len(state["messages"]) > 1:
          return {"route": "research"}
      return {"route": classify(state["messages"][-1].content)}
  ```
  Writing `route` unconditionally also stops a stale value surviving in the checkpoint from an earlier turn.
- **`respond_node`** — one model call, prompt chosen by `route`, no tools, returns `{"messages": [response]}`. Same `RETRY_POLICY` as `agent`, same reasoning.
- **`build_graph(checkpointer=None, router: bool = True)`** — the flag exists so E's gate can A/B the two shapes and so the offline tests can still compile the pre-router graph. One parameter, and it carries the module's central evidence.
- **`answer()` returns `route`** alongside the existing four keys.

### The system prompt splits, and the research path gets shorter

The scope, refusal, and "do not name example companies" clauses are roughly 40% of `SYSTEM_PROMPT` today and ride on every expensive research turn. They move wholesale into `ADVISORY_PROMPT`.

**Advisory keeps C5's rule verbatim:** explain that no tool takes a criterion and returns companies, say it does not rank companies as investments, offer the answerable version — and **name no company**, because a refusal that lists candidates has performed the recommendation it declined, and in a system where every other answer carries citations a name in passing reads as sourced.

Honest side effect: the research prefix shrinks from ~3.4k toward ~2.6k tokens, moving *further* below the measured 4096-token cache floor on `claude-haiku-4-5-20251001`. The breakpoint is already inert there, so nothing is lost today, but this makes it less likely to activate on this model. It stays because it is correct and free. **Do not pad the prompt to reach the floor** — filler degrades the routing the prompt exists to control.

### E2 — the `simple` branch's memory whitelist

This is where the model may answer from its own knowledge, and **only** here. `SIMPLE_PROMPT` is the one prompt in the system carrying that permission; the research path never sees it. That is a structural guarantee where a clause in the shared prompt would have been a textual one — and it is the strongest argument for this module.

Permitted, defined by property rather than enumeration: **who leads a company, what it does in one sentence, what industry it is in, where it is headquartered, which exchange it lists on.** Excluded, explicitly: any number, date, price, metric, ratio, risk, event, filing reference, or quote.

Three clauses make it safe rather than merely narrow:

1. **A mandatory visible hedge on every sentence.** *"From general knowledge rather than a filing or live source: Tesla's CEO is Elon Musk. I can pull its latest 10-K or current market data if you want something sourced."*
2. **Recency defeats the whitelist.** A question implying change or currency — "who is the CEO *now*", "did X just replace its CFO", anything naming a date — is `research`, not `simple`. Leadership is the least stable member of this list and the model's knowledge is frozen at its snapshot; a confidently stated ex-CEO is the worst output this system can produce.
3. **A `simple` answer emits zero citations**, which needs no code and which Module I renders as an explicit unsourced band.

### Gate — `scripts/check_router.py`

House style: `rule()` / `verdict()` / `FAILURES` / exit `0|1|2` (`scripts/check_agent.py:191-200`).

1. A labelled set of ~16 questions (4 per class) classifies correctly; print a confusion matrix.
2. **The sharp assertion:** no `research` question is ever classified `simple` or `clarify`. A `research → advisory` slip is a failure; `research → simple` is a hard failure, called out by name.
3. `simple` answers carry zero citations and contain a hedge marker (new `HEDGE_MARKERS` tuple). `advisory` answers name no company — reuse the existing `PICK_NAMES` tuple (`scripts/check_agent.py:185`), and widen it if a run slips a pick past it. `clarify` answers end in a question.
4. **Skip-on-history:** a second turn on the same thread makes no router call (asserted by turn count and token accounting) and still resolves its antecedent.
5. **The cost delta.** Run the same `simple` + `advisory` questions through `build_graph(router=True)` and `build_graph(router=False)`; print total tokens and dollars for each. This is the measurement the "build router first" ordering would otherwise defer.
6. Full `scripts/check_agent.py` benchmark re-run — the research path's prompt changed, and tool routing is the most model-sensitive thing here. Q7 ("good tech stocks") now travels the router+respond path; its assertions are unchanged and become an end-to-end check that the refusal survived the move.

Offline, `tests/test_router.py` using the existing `ScriptedModel` pattern (`tests/test_agent.py:30`): unknown label falls to `research`; skip-on-history fires on message count; `respond_node` binds no tools; `route` reaches state and survives a checkpoint round-trip.

---

## Module F — Observability and cost accounting

**Steps: F1 (run ledger + cost) · F2 (tracing)**

### F1 — a `runs` ledger in Postgres

```sql
CREATE TABLE IF NOT EXISTS runs (
    run_id                uuid PRIMARY KEY,
    thread_id             text        NOT NULL,
    turn_index            int         NOT NULL,   -- 0-based, within the thread
    question              text        NOT NULL,
    route                 text,                   -- E's class; NULL on continuations
    model                 text        NOT NULL,
    status                text        NOT NULL,   -- 'ok' | 'error'
    error                 text,
    started_at            timestamptz NOT NULL DEFAULT now(),
    duration_ms           int,
    model_turns           int         NOT NULL DEFAULT 0,
    input_tokens          int         NOT NULL DEFAULT 0,
    output_tokens         int         NOT NULL DEFAULT 0,
    cache_read_tokens     int         NOT NULL DEFAULT 0,
    cache_creation_tokens int         NOT NULL DEFAULT 0,
    cost_usd              numeric(12,6),
    tool_calls            jsonb       NOT NULL DEFAULT '[]'::jsonb,
    citation_count        int         NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS runs_thread_idx  ON runs (thread_id, turn_index);
CREATE INDEX IF NOT EXISTS runs_started_idx ON runs (started_at DESC);
```

`route` is what turns "is the router paying for itself" into one query.

**New file `app/cost.py`** — pricing and dollar arithmetic in one place:

```python
# USD per million tokens. Cache write is 1.25x input (5m TTL), read is 0.10x.
PRICING = {
    "claude-haiku-4-5-20251001": Price(input=1.00, output=5.00,
                                       cache_write=1.25, cache_read=0.10),
    "claude-haiku-4-5":          ...,   # alias, same rates
    "claude-sonnet-5":           Price(3.00, 15.00, 3.75, 0.30),
    "claude-opus-5":             Price(5.00, 25.00, 6.25, 0.50),
}
```
An unpriced model records `cost_usd = NULL` and logs a warning — a made-up price is worse than a null.

**New file `app/observability.py`** — `record_run(...)`, built from `agent.token_usage()` (`app/agent.py:664`) and `agent.tool_calls_made()` (`:712`).

**Two things this must get right:**

- **Slice the turn out of the thread.** `answer()` returns *whole-thread* messages, so `token_usage(state["messages"])` on turn 2 re-counts turn 1. Before invoking, read `len(thread_state(thread_id)["messages"])` when a `thread_id` was passed, and record only `state["messages"][before:]`. One cheap `SELECT` per continuing turn, and the only honest way to get per-turn cost. `turn_index` comes from the same count.
- **Recording must never fail a run.** Wrap the whole call in `try/except Exception` and log. A bookkeeping table that can 500 an answer the user already paid for is strictly worse than no table.

Wire it into `agent.answer()` — *not* `app/api.py` — so gate runs are recorded too and G's SSE path gets it for free.

`answer()` gains `run_id`; `QueryResponse` (`app/api.py:91`) gains `run_id: str` and `route: str`. (`tests/test_api.py:370`'s "exactly three response fields" test asserts the current shape and must be updated deliberately, not incidentally.)

**Reading it** — documented SQL in `docs/architecture/observability.md`, no new endpoint:

```sql
SELECT route, count(*), avg(cost_usd)::numeric(10,6), avg(duration_ms)::int
FROM runs WHERE route IS NOT NULL GROUP BY route ORDER BY 2 DESC;
```

### F2 — tracing

LangSmith, genuinely a zero-code-change env toggle for a LangGraph stack. `.env.example` only:

```
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=
# LANGSMITH_PROJECT=confluence
```

**Do not add these to `app/config.py`.** LangChain reads them from the environment directly; a `settings.langsmith_tracing` nothing consumes is a setting that lies. Document OTel as the swap-in.

### Gate — `scripts/check_observability.py`

1. Two questions on one `thread_id` produce rows with `turn_index` 0 and 1.
2. Turn 1's tokens are **not** re-counted in turn 0's row — compared against a fresh `token_usage()` of the sliced messages.
3. `cost_usd` matches hand arithmetic to within a cent.
4. `route` is recorded on first turns and NULL on continuations.
5. A deliberately failing run records `status='error'` and still returns the error to the caller.
6. **A broken ledger does not break an answer** — monkeypatch `record_run` to raise, confirm `answer()` still returns.

---

## Module G — Streaming

**Steps: G1 (`POST /query/stream`)**

`POST /query/stream` alongside an unchanged `POST /query`. POST rather than GET because the question is a body — so the browser client uses `fetch` + `ReadableStream`, not `EventSource`. That is what a Perplexity-style UI does anyway. `POST /query` stays exactly as it is; its tests and D1's gate depend on it, and a non-streaming client is legitimate.

### Sync, not async — deliberately

Endpoints stay `def`, and the generator inside `StreamingResponse` is a **sync** generator, which Starlette iterates in the anyio threadpool — the same mechanism every existing endpoint uses.

Going async would mean `astream`, async tool execution, and an async psycopg pool, while `embeddings._encode_lock` serialises every encode process-wide and `finnhub._RateLimiter.wait()` sleeps holding its lock. Neither becomes non-blocking by being awaited. There is nothing to win.

### The mechanism: a worker thread and a queue

`graph.stream()` blocks between super-steps, and a long first model turn would leave the connection silent long enough to look dead.

- A worker thread runs `graph.stream(..., stream_mode=["updates", "messages", "custom"])` and pushes onto a `queue.Queue`.
- The response generator pulls with `timeout=15`, emitting an SSE comment heartbeat (`: keepalive\n\n`) on each empty poll.
- The generator's `finally` releases the run slot and signals the worker to stop.

One structure buys heartbeats, clean teardown, and disconnect handling.

### Event vocabulary

| Event | Payload | Source |
|---|---|---|
| `start` | `{run_id, thread_id, route}` | before iteration |
| `tool_call` | `{name, args}` | `updates` on `agent` — the "Searching Apple's 10-K…" line |
| `tool_result` | `{name, ok, summary}` | `updates` on `tools` |
| `citation` | `{index, type, label, source_url}` | as `state["citations"]` grows — the rail fills *before* the prose |
| `token` | `{turn, text}` | `messages` mode text deltas |
| `notice` | `{text}` | `custom` mode — H2's ingest progress |
| `done` | `{answer, citations, route, run_id, thread_id, usage, cost_usd, duration_ms}` | after iteration |
| `error` | `{detail}` | any exception after headers are sent |

**`done` carries the complete final payload deliberately** — it makes the stream a strict superset of `POST /query`, so a client can ignore every delta and still be correct, and the gate can compare the two endpoints directly.

**Stream every turn's text, not just the last.** You cannot know a turn is final until it ends without tool calls, and the intermediate text is worth showing. `turn` lets the UI decide.

### Three things that are easy to get wrong

- **The 503 must happen before the first byte.** Once `200 text/event-stream` is committed there is no status code left. Check `runtime.llm_error` and acquire the semaphore *before* constructing the `StreamingResponse`, then hand slot ownership to the generator, which releases it in `finally`. Do **not** reuse `run_slot()` as a `Depends` — FastAPI's yield-dependency teardown timing relative to a streaming body is a framework detail not worth betting on.
- **A disconnect now actually cancels.** Starlette closes the generator when the client goes away, so `finally` fires, the worker is signalled, and the run stops at the next super-step boundary. LangGraph checkpoints per super-step, so the thread stays resumable. This is strictly better than `POST /query`'s documented behaviour (a disconnected client holds its slot to completion) and is worth recording as a real difference between the two endpoints.
- **Buffering.** Send `Cache-Control: no-cache` and `X-Accel-Buffering: no`, or a reverse proxy holds the whole stream and delivers it at the end.

### CORS — probably not needed

Dev: the Vite dev server proxies `/api` → `localhost:8000`. Prod: FastAPI serves the built UI itself (Module I). Neither is cross-origin. Add `CORSMiddleware` behind an `api_cors_origins: list[str] = []` setting that is **off by default**, for split-origin deployment only.

### Gate — `scripts/check_stream.py`

Reuses `check_api.py`'s `start_server` / `stop_server` / `free_port` (`scripts/check_api.py:141-201`).

1. `start` first, `done` last, exactly one of each.
2. Concatenated `token` events for the final turn equal `done.answer`.
3. `tool_call` precedes its `tool_result`; every streamed `citation` appears in `done.citations` and vice versa.
4. `done.answer` and `done.citations` match a `POST /query` call on a fresh thread in shape.
5. Heartbeats appear during a question with a long first turn.
6. Over the cap is a **503 with `Retry-After` and no SSE body** — not an `error` event.
7. Killing a stream mid-run frees its slot: saturate to `MAX_CONCURRENT_RUNS`, abort one, confirm a new stream is accepted, then confirm `GET /threads/{id}` shows the partial thread.
8. A `simple` question streams `route: "simple"`, no `tool_call` events, and zero citations.

---

## Module H — Coverage

**Steps: H1 (backfill) · H2 (on-demand ingestion) · H3 (price history)**

### H1 — a 25-ticker seed backfill

A version-controlled `data/universe.txt` (one ticker per line, comments allowed), spanning sectors so later gate questions can too — roughly AAPL MSFT GOOGL AMZN NVDA META TSLA AVGO BRK-B JPM V MA BAC UNH JNJ ABBV PFE XOM CVX WMT COST HD PG NFLX AMD.

Three changes to `scripts/ingest.py`:

- `--tickers-file PATH`, alongside the existing repeatable `--ticker`.
- **Per-ticker error isolation.** Today one `UnknownTickerError` aborts the whole batch (`scripts/ingest.py:91-96` has no per-ticker guard). Use `tickers.try_resolve_ticker` (`app/tickers.py:116`) — which exists for exactly this and is currently unused — catch per ticker, and print a failure summary.
- Pin `--years` explicitly in the runner: the CLI defaults it to `None` while `list_filings` defaults it to `2` (`app/edgar.py:145`), and an automated backfill should not depend on which default wins.

Shape: latest 10-K + latest 10-Q per ticker ≈ 50 filings, ~3,100 chunks, ~40 MB of vectors, ~150 MB of cached HTML in `data/raw/` (gitignored, never evicted — note it, don't solve it). ~20–30 minutes, dominated by embedding, which `embeddings._encode_lock` serialises process-wide.

**Scaling watch item, not a task:** the dense CTE in `retriever.hybrid_search` (`app/retriever.py:217-224`) filters by ticker *and* orders by HNSW distance. At 3k chunks Postgres seq-scans and results stay exact. Past roughly 50k chunks, HNSW-with-filter under-returns unless `hnsw.ef_search` is raised or pgvector 0.8's `hnsw.iterative_scan` is enabled. Record it in `docs/architecture/data-pipeline.md` so it is a known seam rather than a surprise at 500 tickers.

### H2 — on-demand ingestion

`docs/architecture/phase-2.md:5-13` already specifies this; its four constraints are honoured as follows.

**Trigger:** the uncovered-symbol branch of the corpus-gap check, `app/tools.py:190-198`. `_covered_tickers()` is documented at `app/tools.py:86` as living in the tool layer precisely to be this hook.

**"Cache the outcome, including failure":**

```sql
CREATE TABLE IF NOT EXISTS ingest_attempts (
    ticker       text PRIMARY KEY,
    status       text        NOT NULL,   -- 'ok' | 'failed'
    reason       text,
    attempted_at timestamptz NOT NULL DEFAULT now()
);
```
A `failed` row suppresses retries for 7 days. Also record `failed` when the parser falls back to whole-document (`SECTION_UNKNOWN`, `app/parser.py:254`) — today `scripts/ingest.py:52` only warns, and an undifferentiated blob is *worse* than a recorded failure: it poisons the ledger, suppresses retries, and produces citations pointing at nothing specific.

**"Cap it at one ingest per turn":** `search_filings` gains a `config: RunnableConfig` parameter. LangChain excludes it from the model-facing schema automatically and LangGraph populates it, so `config["configurable"]["thread_id"]` is readable inside the tool. A lock-guarded per-thread counter in `app/ingest_ondemand.py` enforces the cap; `answer()` resets it per turn. (A `contextvars` approach would look cleaner and fail silently — `ToolNode` runs tools in its own threadpool, where context does not propagate.)

**"Say what's happening":** `langgraph.config.get_stream_writer()` inside the tool, surfaced as G's `notice` events. Better than phase-2.md's suggested `interrupt()`, which requires the client to send a resume — the wrong shape for a progress message nobody needs to answer.

**"Only the latest 10-K":** reuse `edgar.list_filings(company, form_types=("10-K",), limit=1)` and the existing ingest path unchanged, ~20–30s measured. The API pre-warms the embedding model at startup (`app/api.py:163`), so this never pays a cold model load.

### H3 — price history

**New client `app/prices.py`** over `yfinance`, as a *sibling* of `finnhub.py` — a Yahoo scraping breakage must cost charts, not quotes. It normalises to a project `Bar` dataclass rather than passing DataFrames around; that normalisation is what makes the provider swappable, which is the entire reason it is a sibling.

**Two consumers, two paths, and keeping them separate is the design:**

- **The model** gets `get_price_history(symbol, period)` returning a *summary* — start, end, % change, high, low, realised volatility, a handful of anchor points. Never 250 rows of OHLCV; that would spend the context guard's budget on data the model cannot reason over.
- **The chart** comes from a plain `GET /prices/{symbol}?period=1y`, outside the agent entirely. A chart is presentation, not reasoning. This keeps `ToolMessage.artifact` typed as `list[Citation]` and `merge_citations` untouched.

**Two prompt sites must change together**, and missing either leaves the system denying a capability it now has: `SYSTEM_PROMPT`'s *"There is no price history"* clause (`app/agent.py:101`) and `get_quote`'s docstring (`app/tools.py:264-268`).

**This is the sixth tool, and C5 cut `lookup_company` partly to avoid a sixth.** The objection then was that it duplicated something the model already did well; this adds a capability nothing else has, and the honesty argument runs the other way — the system currently refuses a question it could answer. But C5's routing risk is real, so **H3's gate re-runs the whole benchmark set**, not just its own question.

### Gate — `scripts/check_coverage.py`, plus benchmark additions

1. Backfill: `SELECT count(*) FROM filings` matches the universe file; every ticker resolves; a deliberately bad symbol in the file is skipped without aborting.
2. On-demand: a question about an uncovered-but-real ticker (e.g. PLTR) ingests it and answers from the new filing. A second run ingests **nothing** and is measurably faster.
3. Failure caching: force a parse failure, confirm an `ingest_attempts` row with `status='failed'`, confirm a repeat within the window does not re-attempt.
4. Cap: a question naming three uncovered tickers ingests exactly one and reports the gap honestly for the other two.
5. `notice` events reach the SSE stream during an on-demand ingest (uses G's machinery).
6. Price history: the tool returns a summary; `GET /prices/AAPL?period=1y` returns a series; "how has Tesla moved since its last 10-K" now uses it instead of refusing.
7. **Full `check_agent.py` benchmark re-run** — H3 changed both the tool list and the prompt.

---

## Module I — The React UI

**Steps: I1 (app shell + streaming) · I2 (sources, chart, threads)**

### Stack, kept deliberately small

Vite + React + TypeScript in `ui/`. Tailwind v4 (one `@import "tailwindcss"` line, no config file). **No component library, no state manager, no data-fetching library** — there is one endpoint that streams and two that don't.

- SSE parsing: `fetch` + `ReadableStream` + a ~30-line line-buffering parser. `EventSource` cannot POST.
- Chart: `recharts` for the price series. One dependency, declarative, adequate.
- Dev: Vite proxy `/api` → `http://localhost:8000`. No CORS.
- Prod: `npm run build` → `ui/dist`, and FastAPI mounts `StaticFiles` at `/` **when that directory exists**. One process, one origin, no CORS ever; the conditional mount keeps a backend-only checkout working unchanged.
- `.gitignore` gains `node_modules/` and `ui/dist/`.

### The screen

One page, Perplexity-Finance-shaped:

- **Ask bar** at top; the thread scrolls beneath. `thread_id` held in React state and the URL, sent back on every follow-up — multi-turn is already built and free.
- **Activity strip** — `tool_call` events render as live chips ("Searching Apple's 10-K…", "Fetching TSLA quote…"), collapsing to one summary line once the answer starts. This is the payoff for streaming: a 20–60s wait becomes legible instead of blank.
- **Answer** streams token-by-token, markdown-rendered.
- **Sources rail** fills from `citation` events *as they arrive*, before the prose. Grouped and badged by `type` — `filing` visually distinct from `news` and `web`, which is why `Citation.type` exists (`app/tools.py:71`).
- **Route-aware framing.** `simple` and `advisory` render an explicit "answered from general knowledge — not sourced" band. `clarify` renders as a question with the input pre-focused rather than as an answer. This is E's safety net made visible, and the unsourced band is the single most important element on the screen.
- **Price chart** when the question named a ticker, from `GET /prices/{symbol}`.
- **Cost/latency footer** from `done` — small, always present. A research tool that shows what each question cost is more honest than one that hides it.

**Do not resolve `[n]` markers in the UI.** `check_agent.py:317` flags a known mismatch: each tool's `[n]` markers index *its own* artifact list from 1, while `merge_citations` produces one flat de-duplicated list. C4c was cancelled because the pinned model emits no `[n]` markers in prose at all. Render sources as a **list**, never as inline clickable `[n]` resolvers, and the mismatch stays harmless. Leave a comment saying so, or someone will "fix" it into a real bug.

### Gate

The real gate is a human looking at it; what is machine-checkable is the contract, and `scripts/check_stream.py` owns that.

- `npm run build` and `tsc --noEmit` both clean, run from `scripts/check_ui.py` so it fits the existing gate shape and exit codes.
- A written checklist in `docs/architecture/ui.md`: streaming visibly incremental; tool chips appear before the answer; sources fill early; the unsourced band appears for "who is Tesla's CEO"; a `clarify` response reads as a question; a follow-up resolves an antecedent; a mid-stream reload does not wedge the server; chart renders; cost footer non-zero.

---

## Architecture simplifications

Independent of the modules; each lands alongside the module that makes it hurt.

1. **Extract `scripts/_gate.py`.** All 11 gates redefine `FAILURES`, `rule()`, `verdict()`, `indent()`, the summary block, and the `0|1|2` exit convention (`scripts/check_agent.py:84-200`, `scripts/check_api.py:85-126`). Phase 2 adds five more. Move **only those primitives** — nothing question-specific — so each gate stays readable top to bottom. Do this before writing E's gate, not after writing five more copies.

2. **Split `CLAUDE.md`.** It is 47 KB, gitignored (`.gitignore:10`), re-read every session, and Phase 2 will push it past 70 KB — and the project's deepest rationale is currently **not in version control**. Move per-gate findings and per-decision records into the matching `docs/architecture/*.md` (which *are* tracked); let `CLAUDE.md` keep the operational rules, the build-status table, and pointers. Add `docs/architecture/router.md`, `observability.md`, and `ui.md` as the modules land.

3. **`app/ratelimit.py`.** `_RateLimiter` is byte-identical in `app/sec_http.py:70` and `app/finnhub.py:95`. The *separate instances* are correct and must stay — a shared limiter would throttle EDGAR because Finnhub was busy. One class, two instantiations; H3 will want a third for yfinance.

4. **Fix the stale doc.** `docs/architecture/agent.md`'s tool table lists `get_company_news(symbol, days=None)`; the real signature is `(symbol, from_date, to_date, limit)` (`app/tools.py:355`).

5. **Centralise usage arithmetic.** After F1, token→dollar conversion lives only in `app/cost.py`; `scripts/check_agent.py` and `check_router.py` read it rather than carrying copies.

**Explicitly rejected:** merging the five tools into fewer (docstrings *are* the routing logic); adding a synthesis node (the tool-call-free turn is the synthesis); collapsing `app/finnhub.py` and `app/websearch.py` into one live-data module (they are named on opposite principles for a documented reason).

---

## Verification

Each module stops for human verification before the next begins. Setup once:

```bash
docker compose up -d --wait
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest                       # offline, must stay green throughout
```

| Module | Command | Costs money |
|---|---|---|
| E | `.venv/bin/python scripts/check_router.py`, then the full `check_agent.py` | yes (~$0.20) |
| F | `.venv/bin/python scripts/check_observability.py` | yes (~$0.05) |
| G | `.venv/bin/python scripts/check_stream.py` | yes |
| H1 | `.venv/bin/python scripts/ingest.py --tickers-file data/universe.txt --forms 10-K,10-Q --limit 2 --years 2` | no (~25 min) |
| H2/H3 | `.venv/bin/python scripts/check_coverage.py` | yes |
| H (regression) | `.venv/bin/python scripts/check_agent.py` — mandatory after H3 | yes |
| I | `.venv/bin/python scripts/check_ui.py`, then the manual checklist | no |

End-to-end, after all five:

```bash
.venv/bin/python -m uvicorn app.api:app        # serves the built UI at /
```

Ask, in order, and read the `runs` table afterwards to confirm the footer told the truth:

1. *"Who is Tesla's CEO?"* → `simple`, unsourced band, one model call, near-zero cost.
2. *"What's the best stock to buy?"* → `advisory`, declines, names no company.
3. *"Tell me about their risks"* (fresh thread) → `clarify`, asks which company.
4. *"What are Palantir's biggest risks?"* → `research`, on-demand ingest with visible `notice` progress, filing citations.
5. *"Which of those involve suppliers outside the US?"* → router skipped, resolves against the thread, no new retrieval.
