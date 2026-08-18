"""Offline tests for the ReAct graph.

No API key, no model, no network. Answer quality and tool *choice* are
judgements for scripts/check_agent.py — what is testable here is the wiring:
that the loop terminates when the model stops asking for tools, that tool
results feed back into the model, and that citations arrive in state with
their types intact and without repeats.

The citation path is the part that fails invisibly. If artifacts stop being
collected, nothing errors — the answers still read fine, they just quietly
lose their sources.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app import agent
from app.agent import AgentState, Citation, merge_citations, tool_calls_made


class ScriptedModel:
    """Stands in for the tool-bound ChatAnthropic.

    Returns pre-written AIMessages in order, and records what it was asked, so
    a test can assert the tool results actually came back to the model.
    """

    def __init__(self, *responses: AIMessage) -> None:
        self.responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.responses.pop(0)


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
