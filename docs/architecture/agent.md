# Agent

The tool layer and the LangGraph loop that turns a question into a sourced answer.

```
app/tools.py   — five @tool-decorated functions, each returns (text, citations)
app/agent.py   — the graph: agent ⇄ tools, until the model stops calling tools
```

## The loop is two nodes, not a router

```
START ──▶ agent ──▶ [tool calls?] ──yes──▶ tools ──▶ agent (loop)
                          │no
                          ▼
                         END
```

`agent_node` calls the model; `tool_node` wraps LangGraph's prebuilt `ToolNode` (which already handles parallel calls, argument validation, and error conversion). There's deliberately **no router node and no synthesis node** — a tool-calling model already routes, and the final tool-call-free turn *is* the synthesis.

The load-bearing consequence: **adding a capability means appending one `@tool` function, never touching the graph.** Retrieval is itself a tool (`search_filings`), which is what makes multi-hop questions work — the agent just calls it twice with different filters.

`RECURSION_LIMIT = 25` (roughly a dozen tool calls) — generous for research, low enough that a stuck retry loop fails visibly instead of spending money quietly.

## The five tools (`app/tools.py`)

| Tool | Source | Returns |
|---|---|---|
| `search_filings(query, ticker=None, section=None, form_type=None, fiscal_year=None, k=6)` | Postgres, hybrid search | filing passages |
| `get_quote(symbol)` | Finnhub | price/change — no citation (no article behind a quote) |
| `get_basic_financials(symbol)` | Finnhub | key metrics — no citation |
| `get_company_news(symbol, days=None)` | Finnhub | recent articles |
| `web_search(query, days=None)` | Tavily | web results, domain-restricted |

With five tools and three that plausibly answer "what's going on with Apple?", **tool docstrings are the routing logic** — each states what it's for *and* when to prefer a sibling instead. If the agent picks the wrong source, the fix is the docstring, not the graph.

`search_filings` reports a **corpus gap** (ticker never ingested) as distinct text from an **empty result** (real query, nothing matched) — see [data-pipeline.md](data-pipeline.md). Tool failures (`UnknownSymbolError`, missing keys, bad section names) are caught and returned as a sentence the model can act on, never raised — a raised error tells the model only "something broke"; a sentence tells it what to check.

## Citations, assembled in code

Models mangle URLs and invent accession numbers. The retriever and the vendor clients already know ground truth, so `Citation` objects (`type: "filing"|"news"|"web"`, `label`, `source_url`) are built programmatically as each tool runs — never by the LLM.

Each `@tool(response_format="content_and_artifact")` returns `(text_for_the_model, list[Citation])`. The citations ride on the `ToolMessage.artifact` **without entering the model's context** — `tool_node` reads `msg.artifact` into `state["citations"]` after the prebuilt `ToolNode` runs. State merges through a reducer (`merge_citations`), de-duplicating on `(type, label, source_url)` — never on URL alone, since two sections of one filing share a URL.

## State

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    citations: Annotated[list[Citation], merge_citations]
```

Two fields. The system prompt is prepended per turn (not stored in state), which keeps the cacheable prefix identical every call.

## Prompt caching and retry (C3)

One cache breakpoint on the system block — Anthropic's cacheable prefix is `tools → system → messages`, so one `cache_control` marker covers the ~3.4k-token fixed prefix (five tool schemas + system prompt) that a multi-hop question re-sends five or six times. **Currently inert on Haiku 4.5** — the model's real floor is ~4096 tokens and the prefix sits just under it — but free to leave in, and `check_agent.py` reports the case by name (`[-- ] inert on this model`) rather than failing.

`RETRY_POLICY` retries only `agent_node`, only on `RateLimitError` / `InternalServerError` / `APIConnectionError` — not LangGraph's default "retry almost everything," which would burn three attempts on a bad API key. The tools node is never retried: it converts provider failures to text on purpose, so retrying it would re-bill a live Finnhub/Tavily call to paper over something that isn't an exception.

## Context guard (C4a)

`search_filings` allows `k=12` passages at ~512 tokens each; a comparison question can push 70k+ tokens of filing text into the conversation, re-sent on every later turn. `trim_tool_results()` runs at the top of `agent_node`, before every model call.

- **Truncates tool-result bodies; never drops a message.** Dropping can orphan a `ToolMessage` from the `AIMessage` that requested it, which Anthropic rejects outright. Truncation preserves every pairing by construction.
- **The most recent rounds stay verbatim** (`KEEP_RECENT_TOOL_ROUNDS = 2`) — the model hasn't reasoned over them yet. A "round" is one `AIMessage`'s tool calls plus their results, so parallel calls are kept or elided together.
- **The exemption has its own ceiling** (`PROTECTED_ROUND_BUDGET_TOKENS`, 3x the normal budget). Found live: a model that parallelizes many searches into one round can make that single round dwarf the budget on its own — past the ceiling, the protected round is trimmed too, oldest call first, rather than exempted whole.
- **Applies to the invoke payload, not to state.** Recomputed from full history every turn, so it's idempotent and `state["messages"]` (and C4b's checkpoint of it) always holds the real conversation. Citations are unaffected — they live outside message content entirely.

## Memory (C4b)

`PostgresSaver` over the same connection pool `app/db.py` already owns (not a second pool). `answer(question, thread_id=None)` defaults to a fresh UUID, so single-shot callers are unchanged; passing the same `thread_id` resumes the conversation from Postgres, across process restarts.

- **`answer()` returns the whole thread's messages and citations, not just the turn's** — a follow-up's answer genuinely rests on passages an earlier turn retrieved.
- **A resumed turn often makes no new tool call at all** — if the needed passages are still in context (inside the guard's recent-rounds window), the model answers from them directly. Cheaper than a first turn: one model call, no retrieval.
- **`Citation` is registered with the checkpoint serializer** so it survives round-trips as a typed object, not a discarded/`None` field.

## Scope and refusal (C5)

Prompt text only — no `lookup_company` tool. The model recognizes entities from its own knowledge ("Palantir" → PLTR); the tool that receives the ticker (`get_quote`, `search_filings`, ...) is the verifier, since an invented ticker dies there (`UnknownSymbolError` or a corpus gap). The system prompt separates **recognizing** an entity (allowed) from **asserting a fact** about it (never) — and refuses screening/ranking questions ("good tech stocks to buy") without naming example companies, since a refusal that lists candidates has performed the recommendation it just declined.

## Verifying it

```bash
.venv/bin/python scripts/check_tools.py       # each tool matches its underlying function
.venv/bin/python scripts/check_agent.py       # 8-question benchmark, live, costs money
.venv/bin/python scripts/check_agent.py --guard    # context guard, exercised with a heavy question
.venv/bin/python scripts/check_agent.py --memory   # checkpointer + multi-turn, with negative controls
.venv/bin/python -m pytest tests/test_tools.py tests/test_agent.py -q   # offline
```

Needs `ANTHROPIC_API_KEY`, `FINNHUB_API_KEY`, `TAVILY_API_KEY` — the first model call anywhere in the system. `check_agent.py --ask "..."` runs one ad-hoc question if you just want to poke at something.
