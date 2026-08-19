# API

`app/api.py` — the HTTP layer over the agent. One file, deliberately thin: no retrieval, no prompt, no graph logic. What it owns is what only matters once requests arrive concurrently.

## Endpoints

```
POST /query   {"question": str, "thread_id": str | null}
           -> {"answer": str, "citations": [{"type", "label", "source_url"}], "thread_id": str}

GET  /health -> {"status", "database": bool, "keys": {name: bool}}   200 | 503

GET  /threads/{thread_id}
           -> {"thread_id", "messages": [{"role", "content"}], "citations": [...]}   200 | 404
```

- **`thread_id` is client-held, always returned, never inferred.** Deriving it from an IP or header would let two users behind one NAT share a conversation.
- **Non-streaming.** SSE is real complexity for a service with no UI — stays Phase 2.
- **Citations are the whole thread's, not the turn's** — same reasoning as `agent.answer()` (see [agent.md](agent.md)). Slicing would drop exactly the sources a later turn relies on again.
- **An unknown `thread_id` on `GET /threads` is 404, not 200 with an empty list** — a thread only exists once a turn has run, so "exists but empty" isn't a reachable state, and 200 would make a typo indistinguishable from a real conversation.

## Startup

Everything is built once in the FastAPI `lifespan`, not per request:

1. `db.init_schema()` — idempotent, and the cheapest proof Postgres is reachable.
2. `agent.get_graph()` — builds the connection pool, migrates the checkpointer, compiles the graph.
3. `embeddings.embed_query("warm start")` (if `API_WARM_EMBEDDINGS`, default on) — moves the model load off the first request, and makes a specific cold-start crash structurally unreachable (see below).
4. `agent.get_llm()`, with failure **recorded**, not raised.

**Startup failure is asymmetric.** Postgres unreachable aborts startup — no checkpointer means no graph, so a process that boots without one is lying about being ready. A missing `ANTHROPIC_API_KEY` does *not* abort — `/health`'s whole job is reporting which capabilities are present, and it can't do that if the process refuses to boot over one of them.

## Concurrency

The first thing in this system to serve genuinely parallel requests against process-wide singletons: the embedding model, the psycopg connection pool, the compiled graph.

- **`MAX_CONCURRENT_RUNS = 4`**, a module constant sized against the actual bottlenecks: the connection pool (`max_size=8`, and one request fans out to ~2 borrows per parallel `search_filings` call), the embedding model's process-wide encode lock, and Finnhub's rate limiter (which sleeps *while holding its lock*).
- **Over the cap is an immediate `503` with `Retry-After`, never a queue.** An agent run is 20-60 seconds — queuing behind one is indistinguishable from a hang, and a queue makes "at capacity" and "slow" the same symptom.
- **Single-worker is the shipped configuration.** `uvicorn --workers N` multiplies the cap, and gives each worker its own embedding model (hundreds of MB) and its own pool of 8.

The embedding warm-up at startup matters here specifically: a known bug (four parallel `search_filings` calls in a cold process building four embedding models simultaneously) once segfaulted the interpreter. The locks that fixed it are in `app/embeddings.py`; the warm-up additionally makes that state unreachable through the API by construction.

## Errors

No catch-all handler — unexpected exceptions propagate, so a bug looks like a bug.

| Condition | Status |
|---|---|
| Missing model key | 503, before the run slot is taken |
| Postgres unreachable / pool exhausted | 503 |
| Recursion limit hit (`GraphRecursionError`) | 500 — not 429/503, since a retry re-bills the identical loop |
| Unknown `thread_id` | 404 |
| Blank/oversized question, empty-string `thread_id` | 422 |

## Verifying it

```bash
.venv/bin/python -m pytest tests/test_api.py -q     # offline — no DB, keys, or model
.venv/bin/python scripts/check_api.py                # live: health, single/mixed-source queries,
                                                       # multi-turn pair, and a concurrent burst
                                                       # against a deliberately cold process
.venv/bin/python scripts/check_api.py --skip-burst    # cheaper re-run, skips the concurrency case
```

`check_api.py` starts and stops its own `uvicorn` processes — one normal, one with no model key (health negative control), one cold (the concurrency burst). Costs real money on the live cases (~$0.15-0.25 total).
