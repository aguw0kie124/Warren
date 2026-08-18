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
"""

import logging
from typing import Annotated, TypedDict

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from app.config import settings
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
# the conversation itself, but C4's context guard rewrites that history —
# invalidating it exactly when the conversation has grown long enough for the
# caching to have mattered. This one is always valid.
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
    """
    response = get_llm().invoke([_system_message(), *state["messages"]])
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


def build_graph():
    """Compile the loop. Two nodes and one conditional edge — that is all of it."""
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

    return builder.compile()


_graph = None


def get_graph():
    """The compiled graph, built once. app/api.py builds it at startup, not per request."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def answer(question: str, recursion_limit: int = RECURSION_LIMIT) -> dict:
    """Run one question to completion.

    Returns `{"answer": str, "citations": list[Citation], "messages": [...]}`.
    The messages come back too because *which tools were called* is the first
    thing worth inspecting when an answer looks wrong — often before the answer
    text itself.
    """
    state = get_graph().invoke(
        {"messages": [HumanMessage(question)], "citations": []},
        config={"recursion_limit": recursion_limit},
    )
    return {
        "answer": final_text(state["messages"]),
        "citations": state.get("citations", []),
        "messages": state["messages"],
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
