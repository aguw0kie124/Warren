"""C2 · The agent — a two-node ReAct loop over the C1 tools.

    agent ⇄ tools, until the model stops asking for tools.

There is deliberately **no router node and no synthesis node**. A tool-calling
model already routes — a classifier node would be a second LLM call doing the
same job worse — and the final tool-call-free turn *is* the synthesis. What
falls out of that shape is the property worth protecting: **tools are list
entries, not graph nodes.** Adding a capability means appending to `TOOLS` in
app/tools.py and changing nothing here. Multi-hop questions need no planner
either: "compare the last two 10-Ks" is `search_filings` twice with different
`fiscal_year` values, which the loop already permits.

The one thing this file adds beyond the loop is **citation collection**.
Tool results carry `Citation` objects as ToolMessage artifacts (C1), so
`tool_node` reads them off and merges them into state. They are never asked of
the model: an LLM handed an accession number will reformat it, and a citation
that doesn't resolve is worse than none because it looks audited.

C3 adds two things to *how* the model is called, neither of which changes what
the graph does: a prompt-cache breakpoint on the system + tools prefix, and a
retry policy on the `agent` node alone. See `_system_message()` and
`RETRY_POLICY`.

C4a adds one more, on the same principle: a context-window guard that elides
the bodies of old tool results on the way into the model, so a long multi-hop
run stops re-sending every passage it has ever retrieved. See
`trim_tool_results()`.

C4b makes the loop multi-turn by persisting state to Postgres under a
`thread_id`, so a follow-up question can resolve *"which of those…"* against
the conversation that produced it. See `get_checkpointer()` and `answer()`.
"""

import logging
import threading
from typing import Annotated, TypedDict
from uuid import uuid4

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from app.config import settings
from app.db import autocommit_conn, get_pool
from app.tools import TOOLS, Citation

logger = logging.getLogger(__name__)

# Each loop iteration is two steps (agent + tools), so this allows roughly a
# dozen tool calls before LangGraph stops it. Generous for a research question,
# and low enough that a model stuck in a retry loop fails visibly instead of
# spending money quietly.
RECURSION_LIMIT = 25

SYSTEM_PROMPT = """You are a financial research assistant. You answer questions about public companies using SEC filings, live market data, and the financial web — never from memory.

Sourcing discipline, in order of priority:

1. **Prefer filings for what a company itself stated.** Risks, business description, management's discussion, stated outlook, accounting policy — these come from `search_filings`. It is the only audited source you have.
2. **Always date filing claims.** A filing is current only as of its filing date. Say "in its FY2025 10-K, filed October 2025, Apple said…" rather than "Apple says…". If a filing claim might have been overtaken by events, say so, and check `get_company_news` or `web_search` when it matters to the answer.
3. **Attribute every substantive claim to the source you got it from** — filing, news article, web page, or market data — in the prose itself. The reader needs to see which claims are audited and which are a journalist's.
4. **If `search_filings` reports a CORPUS GAP, say so plainly.** It means no filings for that company have been ingested, not that the company disclosed nothing. Do not quietly answer from news or web results instead and let the reader assume you consulted the filings.
5. **Never invent a number, a URL, a date, or a filing reference.** If the tools did not return it, you do not have it. Citations are assembled from tool results automatically, so do not write out URL lists yourself.

Choosing tools:

- One specific company's recent headlines → `get_company_news`.
- Anything not scoped to a single ticker — analyst views, competitors, industry or macro context → `web_search`.
- What the company officially disclosed → `search_filings`.
- Current price → `get_quote`. Valuation ratios and margins → `get_basic_financials`.

Call tools in parallel when the question has independent parts, and call the same tool more than once when a comparison needs it (for example one `search_filings` per fiscal year). If a tool reports a failure, tell the reader what was unavailable rather than working around it silently.

There is **no price history** available: nothing can answer "how has the stock moved since the 10-K". Say that plainly if asked.

Write for someone who reads financial documents. Be specific and quantitative where the sources are, brief where they are not, and never pad an answer to look thorough."""

# C3 · One cache breakpoint, on the system block.
#
# Anthropic's cacheable prefix is ordered **tools → system → messages**, so a
# single breakpoint here also covers the five tool schemas sitting ahead of it:
# together a fixed ~3.0k-token prefix (2.4k of tool schema, 0.6k of prompt)
# that is re-sent on every iteration of a loop a multi-hop question goes around
# five or six times.
#
# This only works because `agent_node` prepends the prompt per call instead of
# storing it in state — the prefix is byte-identical every turn, which is the
# whole precondition for a prefix cache.
#
# **On the pinned Haiku model it is currently inert**, and knowingly so: that
# model ignores a breakpoint under 4096 tokens (measured — the docs say 2048),
# which 3.0k does not clear. Nothing is raised; the cache columns just read
# zero. The breakpoint stays because it is correct, costs nothing, and starts
# paying the moment the prefix grows or the model changes. `scripts/
# check_agent.py` reports the inert case by name rather than failing on it.
#
# **No message-level breakpoint.** A second one on the last message would cache
# the conversation itself, but C4a's `trim_tool_results` rewrites that history
# — invalidating it exactly when the conversation has grown long enough for the
# caching to have mattered. This one sits ahead of the messages, so the guard
# cannot touch it.
def _system_message() -> SystemMessage:
    """The system prompt as a single cache-marked content block.

    Built fresh per call rather than held as a module constant. The *text* is
    identical every time, which is all the cache requires; nothing is shared
    between turns, because the message formatter is free to mutate the dicts
    it is handed and a mutated prefix would stop matching with nothing raised.
    """
    return SystemMessage(
        [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )


# C3 · Retry the model call, and only the model call.
#
# `retry_on` is listed explicitly because LangGraph's default retries almost
# everything it is not told to skip — under that default a bad API key would
# burn three attempts and ~7s of backoff to arrive at the same 401. These three
# are the failures where a second attempt is genuinely a different roll:
# 429s, Anthropic 5xx (including 529 overloaded), and transport errors
# (`APITimeoutError` subclasses `APIConnectionError`).
#
# Not redundant with `ChatAnthropic`'s own `max_retries` — this layers above it
# and survives a client that has exhausted its attempts. Mid-loop is where it
# earns its place: a 429 on the fourth turn otherwise discards three turns of
# completed tool work, and the money already spent on them.
RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    retry_on=(
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    ),
)


# C4a · Context-window guard.
#
# `search_filings` allows k=12 passages of ~512 tokens each, and
# RECURSION_LIMIT permits roughly a dozen tool calls. A comparison question can
# therefore push 70k+ tokens of filing text into the conversation — and every
# later turn re-sends all of it, so cost grows quadratically inside a loop
# designed to be cheap.
#
# Three decisions, each load-bearing:
#
# **Bodies are truncated; messages are never dropped.** Dropping is what
# `trim_messages` does, and it risks orphaning a ToolMessage from the
# AIMessage that requested it — which Anthropic rejects outright. Truncating
# in place preserves every `tool_call_id` pairing by construction, so the
# guard cannot produce an invalid conversation no matter what it elides.
#
# **Only ToolMessages are touched.** The model's own messages *are* its
# reasoning over the passages being elided; keeping them is what makes eliding
# the passages affordable. The human turn is never touched either.
#
# **The trim is applied to the invoke payload, not to state.** `agent_node`
# recomputes it from the full history every turn, so it is idempotent, never
# compounds, and leaves `state["messages"]` — and C4b's checkpoint of it —
# holding the real conversation. Citations are unaffected for the same reason:
# they live in `state["citations"]`, harvested from artifacts before this runs.
# That separation, built in C1 for an unrelated reason, is what makes trimming
# safe at all.

# Measured in characters at 4 per token, deliberately. An exact count means
# either an API round-trip per turn (`count_tokens` is free but is still a
# network call inside the loop) or a second tokenizer that has to agree with
# Anthropic's. This decision needs an order of magnitude, not a number: being
# 20% off moves where the guard engages, not whether it is correct.
TOOL_RESULT_BUDGET_TOKENS = 16_000
CHARS_PER_TOKEN = 4

# The most recent rounds stay verbatim — the model has not finished reasoning
# over them. A "round" is one AIMessage carrying tool calls plus every
# ToolMessage answering it, so parallel calls in one turn are kept or elided
# together. This makes the budget a target rather than a guarantee: two recent
# k=12 searches can exceed it on their own, and should.
KEEP_RECENT_TOOL_ROUNDS = 2

# Below this, eliding costs more in confusion than it saves in tokens — a
# quote, an error sentence, or a corpus-gap notice is nearly all signal.
MIN_ELIDABLE_CHARS = 400


def content_chars(content) -> int:
    """Length of a message body, whether it is a string or content blocks."""
    if isinstance(content, str):
        return len(content)
    return sum(
        len(block.get("text", "")) if isinstance(block, dict) else len(str(block))
        for block in content or []
    )


def tool_result_chars(messages: list[AnyMessage]) -> int:
    """Total characters of tool-result text in a conversation.

    The quantity the guard bounds, and the one worth watching: everything else
    in the prompt is bounded by the question and the model's own brevity.
    """
    return sum(
        content_chars(message.content)
        for message in messages
        if isinstance(message, ToolMessage)
    )


def _protected_tool_call_ids(messages: list[AnyMessage], keep_rounds: int) -> set[str]:
    """The `tool_call_id`s belonging to the last `keep_rounds` tool rounds."""
    protected: set[str] = set()
    rounds = 0
    for message in reversed(messages):
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            rounds += 1
            if rounds > keep_rounds:
                break
            protected.update(
                call["id"] for call in message.tool_calls if call.get("id")
            )
    return protected


def _elision_stub(message: ToolMessage, size: int) -> str:
    """What replaces an elided body — said plainly, in the model's own frame."""
    return (
        f"[Earlier {message.name or 'tool'} result omitted to conserve context "
        f"({size:,} characters) — you have already read it. Its sources are "
        f"recorded and still appear in the answer's citation list.]"
    )


def trim_tool_results(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Elide old tool-result bodies until the conversation fits the budget.

    Returns a new list; the input and its messages are left untouched. Oldest
    first, skipping the recent rounds and anything too small to be worth
    eliding, stopping the moment the total is back under budget — so a run that
    barely crosses the line loses one passage set, not all of them.

    Under budget, the original list is returned unchanged, which is the case
    for every question that does not fan out.
    """
    budget = TOOL_RESULT_BUDGET_TOKENS * CHARS_PER_TOKEN
    total = tool_result_chars(messages)
    if total <= budget:
        return list(messages)

    protected = _protected_tool_call_ids(messages, KEEP_RECENT_TOOL_ROUNDS)
    trimmed = list(messages)
    elided = 0
    freed = 0

    for index, message in enumerate(trimmed):
        if total <= budget:
            break
        if not isinstance(message, ToolMessage) or message.tool_call_id in protected:
            continue
        size = content_chars(message.content)
        if size < MIN_ELIDABLE_CHARS:
            continue
        stub = _elision_stub(message, size)
        trimmed[index] = message.model_copy(update={"content": stub})
        total -= size - len(stub)
        elided += 1
        freed += size - len(stub)

    if elided:
        logger.info(
            "context guard: elided %d old tool result(s), ~%d tokens, "
            "leaving ~%d tokens of tool text",
            elided,
            freed // CHARS_PER_TOKEN,
            total // CHARS_PER_TOKEN,
        )
    return trimmed


def merge_citations(
    existing: list[Citation] | None, new: list[Citation] | None
) -> list[Citation]:
    """State reducer: append new citations, in order, without repeats.

    De-duplication happens here rather than inside a tool because a tool's
    `[n]` markers index its own artifact list position for position. Across
    calls, though, the same filing genuinely recurs — a second search of one
    10-K's risk factors should not cite it twice.

    Keyed on the whole citation, not the URL: two sections of one 10-K share a
    URL and are different sources, and collapsing them would drop one.
    """
    merged = list(existing or [])
    seen = {(c.type, c.label, c.source_url) for c in merged}
    for citation in new or []:
        key = (citation.type, citation.label, citation.source_url)
        if key not in seen:
            seen.add(key)
            merged.append(citation)
    return merged


class AgentState(TypedDict):
    """Two fields. Anything more is state the graph doesn't need."""

    messages: Annotated[list[AnyMessage], add_messages]
    citations: Annotated[list[Citation], merge_citations]


_llm = None


def get_llm():
    """The tool-bound model, built once per process.

    Lazily, so that importing this module — which app/api.py will do at
    startup, and the tests do offline — neither requires an API key nor opens
    a connection.
    """
    global _llm
    if _llm is None:
        # Bound once, here. The list lives in app/tools.py so that adding a
        # capability never touches this file.
        _llm = _build_model().bind_tools(TOOLS)
    return _llm


def _require_key(value: str, name: str) -> str:
    key = value.strip()
    if not key or key.lower().startswith("your"):
        raise RuntimeError(
            f"{name} is not set. Add it to .env before running the agent "
            "(Modules A, B and C1 do not need it — this is the first step "
            "with a model in the loop)."
        )
    return key


def _build_model():
    """The chat model itself, before tools are bound.

    One provider, deliberately. A free-tier second provider lived here long
    enough to prove the wiring — loop termination, artifacts reaching
    `state["citations"]`, citation types matching the tools that ran — then
    was removed: its token ceiling could not seat a multi-tool question, and
    C3's `cache_control` prefix is Anthropic-only, so the path it exercised
    was about to stop resembling the one this system ships.
    """
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=_require_key(settings.anthropic_api_key, "ANTHROPIC_API_KEY"),
        max_tokens=settings.llm_max_tokens,
        timeout=120.0,
    )


def agent_node(state: AgentState) -> dict:
    """One model turn: either tool calls, or the final answer.

    The system prompt is prepended per call rather than stored in state, which
    keeps it out of the conversation the model sees itself as having written
    and keeps the cacheable prefix identical on every turn.

    The context guard runs here rather than in a `pre_model_hook`: this graph
    is hand-built rather than `create_react_agent`, so there is no hook slot,
    and a plain call at the top of the node is both simpler and more honest
    about when it runs. It rewrites only what goes to the model — what comes
    back into state is one message, so the stored history stays whole.
    """
    messages = trim_tool_results(state["messages"])
    response = get_llm().invoke([_system_message(), *messages])
    logger.debug(
        "agent turn: %d tool call(s), %s",
        len(getattr(response, "tool_calls", []) or []),
        _usage_line(response) or "no usage reported",
    )
    return {"messages": [response]}


_tools = ToolNode(TOOLS)


def tool_node(state: AgentState) -> dict:
    """Run the requested tools, and harvest their citations.

    A thin wrapper around the prebuilt node rather than a reimplementation:
    ToolNode already handles parallel calls, argument validation and error
    conversion. All this adds is reading `artifact` off each ToolMessage —
    which is where C1 puts the typed citations, precisely so they can be
    collected without the model ever seeing, or being able to mangle, them.
    """
    result = _tools.invoke({"messages": state["messages"]})
    messages = result["messages"]

    citations = [
        citation
        for message in messages
        for citation in (getattr(message, "artifact", None) or [])
        if isinstance(citation, Citation)
    ]
    if citations:
        logger.debug("collected %d citation(s) from tool results", len(citations))

    return {"messages": messages, "citations": citations}


# C4b · The checkpointer.
#
# Over the pool `app/db.py` already owns, not a second one against the same
# database: two pools would double the connection budget and make "how many
# connections does this process hold" a question with two answers. PostgresSaver
# takes a ConnectionPool directly and sets `row_factory` per cursor, so nothing
# about the pool's existing configuration (pgvector registration included) has
# to change for it.
#
# `setup()` creates and migrates its own tables idempotently. It is called here
# rather than hand-copied into sql/schema.sql for the same reason the schema
# file is not hand-written per table: only one of those two rots when the
# library changes its layout, and it is not this one.
#
# It does not run on the pool, though. Its migrations include `CREATE INDEX
# CONCURRENTLY`, which Postgres rejects inside a transaction block — so the DDL
# gets one short-lived autocommit connection, and everything after it uses the
# pool. Making the *pool* autocommit instead, as LangGraph's own examples do,
# would silently remove transactions from ingestion, which is the one place
# here that needs them.
_checkpointer: PostgresSaver | None = None

# `Citation` is the one project type that rides inside checkpointed state — on
# `ToolMessage.artifact`, where C1 put it. LangGraph's serializer allows
# unregistered types with a logged warning today and will refuse them in a
# future release; refused, a citation comes back as `None` rather than as an
# error, which is the worst available outcome for the one structure whose whole
# value is that every entry resolves. Naming it here makes that a pinned fact
# instead of a deprecation notice nobody read.
CHECKPOINT_SERDE = JsonPlusSerializer(allowed_msgpack_modules=[Citation])

# One lock for both singletons below. Neither construction is thread-safe on
# its own, and D1 will build the graph from an ASGI startup hook while requests
# are already arriving; a race here would run `setup()` twice (harmless) or
# compile two graphs (wasteful). Cheaper to hold a lock for the microsecond
# than to reason about it later.
#
# **Reentrant on purpose**: `get_graph()` holds it and calls `get_checkpointer()`,
# which takes it again. A plain Lock deadlocks there — silently, with no error
# and no traceback, which is exactly how it presented the first time.
_build_lock = threading.RLock()


def get_checkpointer() -> PostgresSaver:
    """The Postgres checkpointer, created and migrated once per process."""
    global _checkpointer
    if _checkpointer is None:
        with _build_lock:
            if _checkpointer is None:
                with autocommit_conn() as conn:
                    PostgresSaver(conn).setup()
                _checkpointer = PostgresSaver(get_pool(), serde=CHECKPOINT_SERDE)
                logger.debug("checkpointer ready")
    return _checkpointer


def build_graph(checkpointer=None):
    """Compile the loop. Two nodes and one conditional edge — that is all of it.

    The checkpointer is a parameter rather than a lookup so that the offline
    tests can compile the real graph without a database, and so a caller that
    wants a one-shot run can pass nothing.
    """
    builder = StateGraph(AgentState)
    # Retry `agent` only. The tools node deliberately converts provider
    # failures into text the model can act on (C1), so retrying it would
    # re-bill live Finnhub and Tavily calls to paper over something that is
    # not an exception in the first place.
    builder.add_node("agent", agent_node, retry_policy=RETRY_POLICY)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    # tools_condition routes to "tools" when the last message carries tool
    # calls, and to END when it doesn't — that tool-call-free turn is the
    # synthesis, which is why there is no separate synthesis node.
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


_graph = None


def get_graph():
    """The compiled graph, built once. app/api.py builds it at startup, not per request."""
    global _graph
    if _graph is None:
        with _build_lock:
            if _graph is None:
                _graph = build_graph(checkpointer=get_checkpointer())
    return _graph


def thread_config(thread_id: str, recursion_limit: int = RECURSION_LIMIT) -> dict:
    """The config every run needs once the graph has a checkpointer.

    In one place because a run without `configurable.thread_id` is not a
    single-shot run — it is an error from LangGraph, and one that only appears
    at the call site that forgot it.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }


def answer(
    question: str,
    thread_id: str | None = None,
    recursion_limit: int = RECURSION_LIMIT,
) -> dict:
    """Run one question to completion, in a conversation.

    Returns `{"answer", "citations", "messages", "thread_id"}`. The messages
    come back too because *which tools were called* is the first thing worth
    inspecting when an answer looks wrong — often before the answer text itself.

    `thread_id` defaults to a fresh uuid, so a caller that has no notion of
    sessions keeps getting exactly one question's worth of context and needs no
    changes. Passing the same id again resumes that conversation from Postgres —
    including across process restarts, which is the whole difference between a
    checkpointer and a list held in memory.

    **`messages` and `citations` are the whole thread's, not this turn's.** The
    reducers accumulate, and that is the honest answer for a follow-up: turn 2's
    answer genuinely rests on the passages turn 1 retrieved, so listing only the
    sources turn 2 happened to re-fetch would under-credit it. A caller wanting
    just this turn's messages can slice from the length it saw last time.
    """
    thread_id = thread_id or str(uuid4())
    state = get_graph().invoke(
        {"messages": [HumanMessage(question)], "citations": []},
        config=thread_config(thread_id, recursion_limit),
    )
    return {
        "answer": final_text(state["messages"]),
        "citations": state.get("citations", []),
        "messages": state["messages"],
        "thread_id": thread_id,
    }


def thread_state(thread_id: str) -> dict:
    """What Postgres holds for one conversation, without running the model.

    The read side of the checkpointer: `{"messages", "citations"}`, empty for a
    thread that has never run. Used by the gate to prove a fresh process can
    still see a thread, and by D1 to render a session's history.
    """
    snapshot = get_graph().get_state(thread_config(thread_id))
    values = snapshot.values or {}
    return {
        "messages": values.get("messages", []),
        "citations": values.get("citations", []),
    }


def final_text(messages: list[AnyMessage]) -> str:
    """The last message's text, flattened across content blocks.

    Anthropic returns content as a list of blocks once thinking or tool use is
    involved, so `.content` is not reliably a string.
    """
    content = messages[-1].content if messages else ""
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def token_usage(messages: list[AnyMessage]) -> list[dict[str, int]]:
    """Per-model-turn token counts, in turn order — the only view of the cache.

    Whether C3's breakpoint is working is invisible in the answers: a prefix
    that silently stops matching costs money and changes nothing else. Turn 1
    should show `cache_creation`, every later turn `cache_read` covering the
    system + tools prefix with near-zero creation.

    `cache_creation` sums the generic count and the per-TTL ones, because
    LangChain zeroes the generic key when the provider reports the ephemeral
    breakdown — reading only one of them under-reports depending on the reply.
    """
    rows: list[dict[str, int]] = []
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        details = usage.get("input_token_details") or {}
        rows.append(
            {
                "input": usage.get("input_tokens", 0) or 0,
                "output": usage.get("output_tokens", 0) or 0,
                "cache_read": details.get("cache_read") or 0,
                "cache_creation": sum(
                    details.get(key) or 0
                    for key in (
                        "cache_creation",
                        "ephemeral_5m_input_tokens",
                        "ephemeral_1h_input_tokens",
                    )
                ),
            }
        )
    return rows


def _usage_line(message: AnyMessage) -> str:
    """One turn's token counts, for the debug log."""
    rows = token_usage([message])
    if not rows:
        return ""
    row = rows[0]
    return (
        f"in={row['input']} out={row['output']} "
        f"cache_read={row['cache_read']} cache_creation={row['cache_creation']}"
    )


def tool_calls_made(messages: list[AnyMessage]) -> list[tuple[str, dict]]:
    """Every (tool name, arguments) pair in a finished run, in call order."""
    return [
        (call["name"], call.get("args", {}))
        for message in messages
        for call in (getattr(message, "tool_calls", None) or [])
    ]
