"""Offline tests for the ReAct graph.

No API key, no model, no network. Answer quality and tool *choice* are
judgements for scripts/check_agent.py — what is testable here is the wiring:
that the loop terminates when the model stops asking for tools, that tool
results feed back into the model, and that citations arrive in state with
their types intact and without repeats.

The citation path is the part that fails invisibly. If artifacts stop being
collected, nothing errors — the answers still read fine, they just quietly
lose their sources. C3's two additions fail the same way: a cache breakpoint
that stops being sent costs money and changes no output, and a retry policy is
only observable when the provider is failing.
"""

import anthropic
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app import agent
from app.agent import AgentState, Citation, merge_citations, tool_calls_made


class ScriptedModel:
    """Stands in for the tool-bound ChatAnthropic.

    Returns pre-written AIMessages in order, and records what it was asked, so
    a test can assert the tool results actually came back to the model. An
    exception in the script is raised instead of returned, which is how the
    retry tests make the provider fail.
    """

    def __init__(self, *responses: AIMessage | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages, **kwargs):
        self.calls.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def api_error(error: type[anthropic.APIStatusError], status: int):
    """One Anthropic HTTP failure, shaped the way the SDK raises it."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return error("boom", response=httpx.Response(status, request=request), body=None)


def tool_call(name: str, args: dict, id: str = "call-1") -> dict:
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


@pytest.fixture
def scripted(monkeypatch):
    """Install a scripted model in place of the real one."""

    def install(*responses: AIMessage) -> ScriptedModel:
        model = ScriptedModel(*responses)
        monkeypatch.setattr(agent, "get_llm", lambda: model)
        return model

    return install


@pytest.fixture
def graph():
    return agent.build_graph()


@pytest.fixture
def fast_retry_graph(monkeypatch):
    """The real graph, with the real retry policy's waits collapsed.

    Only the timing is changed — `max_attempts` and, crucially, `retry_on` are
    the shipped values, because which exceptions retry is the whole point.
    """
    monkeypatch.setattr(
        agent,
        "RETRY_POLICY",
        agent.RETRY_POLICY._replace(initial_interval=0.01, jitter=False),
    )
    return agent.build_graph()


@pytest.fixture
def stub_search(monkeypatch):
    """Make search_filings return one known passage without touching Postgres."""
    monkeypatch.setattr(
        "app.tools._covered_tickers", lambda: {"AAPL"}
    )

    from app import retriever
    from datetime import date

    result = retriever.Result(
        accession_number="0000320193-25-000079",
        chunk_index=0,
        section="Item 1A Risk Factors",
        content="The Company's business is subject to supply chain risk.",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type="10-K",
        fiscal_year=2025,
        filing_date=date(2025, 10, 31),
        source_url="https://www.sec.gov/Archives/edgar/data/320193/aapl.htm",
        score=0.03,
    )
    monkeypatch.setattr(retriever, "hybrid_search", lambda query, **kw: [result])
    return result


# --- the loop ---------------------------------------------------------------


def test_no_tool_calls_ends_immediately(scripted, graph):
    model = scripted(AIMessage("There is no price history available."))

    state = graph.invoke({"messages": [HumanMessage("hi")], "citations": []})

    assert len(model.calls) == 1
    assert agent.final_text(state["messages"]) == "There is no price history available."
    assert state["citations"] == []


def test_tool_result_goes_back_to_the_model(scripted, graph, stub_search):
    model = scripted(
        AIMessage(
            "",
            tool_calls=[tool_call("search_filings", {"query": "supply chain", "ticker": "AAPL"})],
        ),
        AIMessage("Apple flagged supply chain risk in its FY2025 10-K."),
    )

    state = graph.invoke(
        {"messages": [HumanMessage("What are Apple's risks?")], "citations": []}
    )

    # Two model turns, and the second one saw the retrieved passage.
    assert len(model.calls) == 2
    second_turn = "\n".join(str(m.content) for m in model.calls[1])
    assert stub_search.content in second_turn
    assert "supply chain risk" in agent.final_text(state["messages"])
    assert tool_calls_made(state["messages"]) == [
        ("search_filings", {"query": "supply chain", "ticker": "AAPL"})
    ]


def test_parallel_tool_calls_in_one_turn(scripted, graph, stub_search, monkeypatch):
    from app import finnhub

    monkeypatch.setattr(
        finnhub, "get_quote", lambda s: (_ for _ in ()).throw(
            finnhub.UnknownSymbolError(s, "/quote")
        )
    )
    scripted(
        AIMessage(
            "",
            tool_calls=[
                tool_call("search_filings", {"query": "risks", "ticker": "AAPL"}, "a"),
                tool_call("get_quote", {"symbol": "AAPL"}, "b"),
            ],
        ),
        AIMessage("Filings answered; the quote was unavailable."),
    )

    state = graph.invoke({"messages": [HumanMessage("q")], "citations": []})

    # A failing tool must not take the run down with it — the loop continues
    # and the model gets to report what was unavailable.
    assert [name for name, _ in tool_calls_made(state["messages"])] == [
        "search_filings",
        "get_quote",
    ]
    assert "unavailable" in agent.final_text(state["messages"])


# --- C3 · prompt caching ----------------------------------------------------


def test_system_prompt_carries_a_cache_breakpoint(scripted, graph):
    model = scripted(AIMessage("done"))

    graph.invoke({"messages": [HumanMessage("hi")], "citations": []})

    system = model.calls[0][0]
    assert system.type == "system"
    # A content-block list, not a bare string — that is the only place
    # cache_control can be attached.
    assert system.content == [
        {
            "type": "text",
            "text": agent.SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_the_cached_prefix_is_identical_across_turns(scripted, graph, stub_search):
    """Byte-identical, or the second turn pays to create the cache again."""
    model = scripted(
        AIMessage("", tool_calls=[tool_call("search_filings", {"query": "q", "ticker": "AAPL"})]),
        AIMessage("done"),
    )

    graph.invoke({"messages": [HumanMessage("q")], "citations": []})

    first, second = model.calls
    assert first[0].content == second[0].content


def test_the_system_block_is_not_shared_between_turns(scripted, graph, stub_search):
    """Each turn gets its own dicts, so a mutating formatter cannot poison the prefix."""
    model = scripted(
        AIMessage("", tool_calls=[tool_call("search_filings", {"query": "q", "ticker": "AAPL"})]),
        AIMessage("done"),
    )

    graph.invoke({"messages": [HumanMessage("q")], "citations": []})

    first, second = model.calls
    assert first[0].content[0] is not second[0].content[0]


def test_token_usage_reads_both_cache_creation_shapes():
    """The provider reports creation generically or per-TTL; both must count.

    LangChain zeroes the generic key when the per-TTL breakdown is present, so
    reading only one of them under-reports depending on which shape came back.
    """
    generic = AIMessage(
        "a",
        usage_metadata={
            "input_tokens": 2400, "output_tokens": 10, "total_tokens": 2410,
            "input_token_details": {"cache_read": 0, "cache_creation": 2000},
        },
    )
    per_ttl = AIMessage(
        "b",
        usage_metadata={
            "input_tokens": 2400, "output_tokens": 10, "total_tokens": 2410,
            "input_token_details": {
                "cache_read": 0, "cache_creation": 0, "ephemeral_5m_input_tokens": 2000,
            },
        },
    )

    assert [row["cache_creation"] for row in agent.token_usage([generic, per_ttl])] == [
        2000,
        2000,
    ]


def test_token_usage_skips_messages_without_usage():
    assert agent.token_usage([HumanMessage("q"), AIMessage("a")]) == []


# --- C3 · node-level retry --------------------------------------------------


def test_a_transient_provider_failure_is_retried(scripted, fast_retry_graph):
    model = scripted(
        api_error(anthropic.InternalServerError, 529),
        api_error(anthropic.InternalServerError, 529),
        AIMessage("Recovered on the third attempt."),
    )

    state = fast_retry_graph.invoke({"messages": [HumanMessage("q")], "citations": []})

    assert len(model.calls) == 3
    assert agent.final_text(state["messages"]) == "Recovered on the third attempt."


def test_a_rate_limit_is_retried(scripted, fast_retry_graph):
    model = scripted(api_error(anthropic.RateLimitError, 429), AIMessage("done"))

    fast_retry_graph.invoke({"messages": [HumanMessage("q")], "citations": []})

    assert len(model.calls) == 2


def test_retry_survives_completed_tool_work(scripted, fast_retry_graph, stub_search):
    """The point of retrying mid-loop: a 429 on a later turn must not discard
    the tool calls already made, or the run pays for that work twice."""
    scripted(
        AIMessage("", tool_calls=[tool_call("search_filings", {"query": "q", "ticker": "AAPL"})]),
        api_error(anthropic.RateLimitError, 429),
        AIMessage("Answered after the rate limit cleared."),
    )

    state = fast_retry_graph.invoke({"messages": [HumanMessage("q")], "citations": []})

    assert len(tool_calls_made(state["messages"])) == 1
    assert len(state["citations"]) == 1


def test_a_bad_api_key_fails_immediately(scripted, fast_retry_graph):
    """A 401 is the same 401 on the third attempt — retrying only spends time.

    LangGraph's default `retry_on` would retry this, which is why the policy
    names its exceptions explicitly.
    """
    model = scripted(api_error(anthropic.AuthenticationError, 401))

    with pytest.raises(anthropic.AuthenticationError):
        fast_retry_graph.invoke({"messages": [HumanMessage("q")], "citations": []})

    assert len(model.calls) == 1


def test_a_bad_request_fails_immediately(scripted, fast_retry_graph):
    model = scripted(api_error(anthropic.BadRequestError, 400))

    with pytest.raises(anthropic.BadRequestError):
        fast_retry_graph.invoke({"messages": [HumanMessage("q")], "citations": []})

    assert len(model.calls) == 1


def test_only_the_agent_node_is_retried(graph):
    """Retrying `tools` would re-bill live Finnhub and Tavily calls to paper
    over failures C1 deliberately returns as text rather than raising."""
    assert agent.RETRY_POLICY in graph.nodes["agent"].retry_policy
    assert graph.nodes["tools"].retry_policy is None


# --- citations --------------------------------------------------------------


def test_citations_are_collected_from_tool_artifacts(scripted, graph, stub_search):
    scripted(
        AIMessage("", tool_calls=[tool_call("search_filings", {"query": "risks", "ticker": "AAPL"})]),
        AIMessage("Apple disclosed supply chain risk."),
    )

    state = graph.invoke({"messages": [HumanMessage("q")], "citations": []})

    assert len(state["citations"]) == 1
    citation = state["citations"][0]
    assert isinstance(citation, Citation)
    assert citation.type == "filing"
    assert citation.source_url == stub_search.source_url
    assert citation.label == "Apple Inc. 10-K, Item 1A Risk Factors, filed 2025-10-31"


def test_a_corpus_gap_yields_no_citations(scripted, graph, monkeypatch):
    monkeypatch.setattr("app.tools._covered_tickers", lambda: {"AAPL"})
    scripted(
        AIMessage("", tool_calls=[tool_call("search_filings", {"query": "risks", "ticker": "NVDA"})]),
        AIMessage("NVDA's filings have not been ingested, so I cannot answer from them."),
    )

    state = graph.invoke({"messages": [HumanMessage("q")], "citations": []})

    assert state["citations"] == []
    # The gap text must have actually reached the model, or the instruction to
    # report it has nothing to act on.
    tool_message = [m for m in state["messages"] if m.type == "tool"][0]
    assert "CORPUS GAP" in tool_message.content


def test_repeated_searches_do_not_duplicate_a_citation(scripted, graph, stub_search):
    scripted(
        AIMessage("", tool_calls=[tool_call("search_filings", {"query": "risks", "ticker": "AAPL"}, "a")]),
        AIMessage("", tool_calls=[tool_call("search_filings", {"query": "supply", "ticker": "AAPL"}, "b")]),
        AIMessage("Answered from two searches of the same filing."),
    )

    state = graph.invoke({"messages": [HumanMessage("q")], "citations": []})

    assert len(tool_calls_made(state["messages"])) == 2
    assert len(state["citations"]) == 1  # same filing, same section, cited once


# --- the reducer ------------------------------------------------------------


def citation(**overrides) -> Citation:
    return Citation(
        **{
            "type": "filing",
            "label": "Apple Inc. 10-K, Item 1A Risk Factors, filed 2025-10-31",
            "source_url": "https://sec.gov/a.htm",
            **overrides,
        }
    )


def test_merge_citations_preserves_order_and_drops_repeats():
    first, second = citation(), citation(label="Apple Inc. 10-K, Item 7 MD&A, filed 2025-10-31")
    merged = merge_citations([first], [first, second])
    assert merged == [first, second]


def test_two_sections_of_one_filing_are_distinct_citations():
    """They share a source_url — keying on URL alone would drop one."""
    risks = citation()
    mda = citation(label="Apple Inc. 10-K, Item 7 MD&A, filed 2025-10-31")
    assert risks.source_url == mda.source_url
    assert len(merge_citations([], [risks, mda])) == 2


def test_merge_citations_handles_empty_state():
    assert merge_citations(None, None) == []
    assert merge_citations([], [citation()]) == [citation()]


# --- the graph shape --------------------------------------------------------


def test_the_graph_is_two_nodes(graph):
    nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"agent", "tools"}


def test_state_has_exactly_messages_and_citations():
    assert set(AgentState.__annotations__) == {"messages", "citations"}
