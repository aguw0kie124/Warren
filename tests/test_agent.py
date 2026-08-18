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

C4a's guard is the opposite risk — it is the one piece here that *removes*
text from the model's view, so its tests are mostly about restraint: that it
leaves a short conversation alone, that it never drops a message, and that
what it does elide is the oldest tool output and nothing else.
"""

import anthropic
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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


# --- C4a · context guard ----------------------------------------------------


BUDGET_CHARS = agent.TOOL_RESULT_BUDGET_TOKENS * agent.CHARS_PER_TOKEN


def conversation(rounds: int, chars: int = 20_000) -> list:
    """A synthetic multi-hop run: `rounds` search calls, each answered at length."""
    messages: list = [HumanMessage("Compare the last two 10-Ks.")]
    for i in range(rounds):
        messages.append(
            AIMessage("", tool_calls=[tool_call("search_filings", {"query": f"q{i}"}, f"c{i}")])
        )
        messages.append(
            ToolMessage(
                content=f"passage {i} " + "x" * chars,
                tool_call_id=f"c{i}",
                name="search_filings",
            )
        )
    return messages


def elided(messages: list) -> list[int]:
    """Indices of the tool results the guard replaced with a stub."""
    return [
        i
        for i, m in enumerate(messages)
        if isinstance(m, ToolMessage) and "omitted to conserve context" in m.content
    ]


def test_a_conversation_under_budget_is_untouched():
    """The common case. Nothing is elided until there is a reason."""
    messages = conversation(2)
    assert agent.tool_result_chars(messages) < BUDGET_CHARS
    assert [m.content for m in agent.trim_tool_results(messages)] == [
        m.content for m in messages
    ]


def test_the_oldest_tool_results_are_elided_over_budget():
    messages = conversation(5)  # ~100k chars against a ~64k budget
    assert agent.tool_result_chars(messages) > BUDGET_CHARS

    trimmed = agent.trim_tool_results(messages)

    assert elided(trimmed) == [2, 4]  # the first two tool results, in order
    assert agent.tool_result_chars(trimmed) <= BUDGET_CHARS


def test_it_stops_as_soon_as_it_is_back_under_budget():
    """A run that barely crosses the line loses one passage set, not all of them."""
    trimmed = agent.trim_tool_results(conversation(4))
    assert elided(trimmed) == [2]


def test_the_recent_rounds_are_never_elided():
    """The model has not finished reasoning over them — eliding those is the
    one way this guard could change an answer rather than only its price."""
    trimmed = agent.trim_tool_results(conversation(8))
    stubbed = {trimmed[i].tool_call_id for i in elided(trimmed)}
    assert agent.KEEP_RECENT_TOOL_ROUNDS == 2
    assert not {"c6", "c7"} & stubbed


def test_no_message_is_dropped_and_every_pairing_survives():
    """Dropping messages — what trim_messages does — can orphan a ToolMessage
    from the AIMessage that requested it, which Anthropic rejects outright."""
    messages = conversation(6)
    trimmed = agent.trim_tool_results(messages)

    assert len(trimmed) == len(messages)
    assert [type(m) for m in trimmed] == [type(m) for m in messages]
    assert [m.tool_call_id for m in trimmed if isinstance(m, ToolMessage)] == [
        m.tool_call_id for m in messages if isinstance(m, ToolMessage)
    ]
    # Every remaining tool result still answers a call that is still present.
    requested = {
        call["id"] for m in trimmed if isinstance(m, AIMessage) for call in m.tool_calls
    }
    assert all(
        m.tool_call_id in requested for m in trimmed if isinstance(m, ToolMessage)
    )


def test_the_input_messages_are_not_mutated():
    """The trimmed list goes to the model; state keeps the real conversation."""
    messages = conversation(6)
    before = [m.content for m in messages]

    agent.trim_tool_results(messages)

    assert [m.content for m in messages] == before


def test_only_tool_messages_are_elided():
    """The model's own turns are its reasoning over the passages being elided —
    keeping them is what makes eliding the passages affordable."""
    messages = conversation(6)
    messages.insert(1, AIMessage("A long reasoning turn. " * 2_000))

    trimmed = agent.trim_tool_results(messages)

    changed = [new for new, old in zip(trimmed, messages) if new.content != old.content]
    assert changed and all(isinstance(m, ToolMessage) for m in changed)
    assert trimmed[1].content == messages[1].content  # the long AI turn, intact


def test_short_tool_results_are_left_alone():
    """A failure sentence or a corpus-gap notice is nearly all signal — eliding
    it costs more in confusion than it saves in tokens."""
    messages = conversation(5)
    messages[2] = ToolMessage(
        content="CORPUS GAP: NVDA is not in the filings corpus.",
        tool_call_id="c0",
        name="search_filings",
    )

    trimmed = agent.trim_tool_results(messages)

    # Skipped even though it is the oldest candidate: had it been elided, the
    # stub would be *longer* than what it replaced and the loop would have gone
    # on to elide index 6 as well.
    assert trimmed[2].content == messages[2].content
    assert elided(trimmed) == [4]


def test_the_stub_names_what_was_elided():
    messages = conversation(5)
    trimmed = agent.trim_tool_results(messages)

    stub = trimmed[2].content
    assert "search_filings" in stub
    assert f"{len(messages[2].content):,}" in stub  # the loss is visible, not silent
    assert trimmed[2].tool_call_id == "c0"


def test_eliding_preserves_the_artifact_that_carried_the_citations():
    """The elided copy keeps everything but the body — C1's artifact included,
    so a trimmed run is still a fully sourced one."""
    messages = conversation(5)
    sources = [citation()]
    messages[2] = messages[2].model_copy(update={"artifact": sources})

    trimmed = agent.trim_tool_results(messages)

    assert 2 in elided(trimmed)
    assert trimmed[2].artifact == sources


def test_the_node_sends_the_trimmed_history_and_returns_only_the_reply(scripted):
    """The guard rewrites the invoke payload, never state: `agent_node` returns
    one message, so the stored history — and C4b's checkpoint of it — is whole."""
    messages = conversation(5)
    model = scripted(AIMessage("done"))

    result = agent.agent_node({"messages": messages, "citations": []})

    sent = model.calls[0][1:]  # past the system message
    assert len(elided(sent)) == 2
    assert elided(messages) == []  # the caller's list is untouched
    assert result["messages"][0].content == "done"
    assert "citations" not in result


# --- C4b · checkpointer and multi-turn --------------------------------------
#
# The saver here is InMemorySaver, not PostgresSaver: what these tests are
# about is the graph's *use* of a checkpointer — thread isolation, the state a
# second turn resumes, the config `answer()` builds — and none of that is
# Postgres-specific. That the Postgres one actually persists across a process
# boundary is the one claim an offline test cannot make, which is why
# `check_agent.py --memory` proves it by re-running itself as a subprocess.


@pytest.fixture
def saved_graph():
    """The real graph, checkpointed in memory."""
    from langgraph.checkpoint.memory import InMemorySaver

    return agent.build_graph(checkpointer=InMemorySaver())


def turn(graph, question: str, thread: str) -> dict:
    return graph.invoke(
        {"messages": [HumanMessage(question)], "citations": []},
        config=agent.thread_config(thread),
    )


def test_a_second_turn_on_one_thread_sees_the_first(scripted, saved_graph, stub_search):
    model = scripted(
        AIMessage("", tool_calls=[tool_call("search_filings", {"query": "risks", "ticker": "AAPL"})]),
        AIMessage("Apple flagged supply chain risk."),
        AIMessage("The supplier concentration one."),
    )

    first = turn(saved_graph, "What are Apple's key risk factors?", "t1")
    second = turn(saved_graph, "Which of those involve suppliers?", "t1")

    # The follow-up's prompt carries turn 1 — question, tool call, passage and
    # answer — which is the whole point: "those" has no other referent.
    third_turn = "\n".join(str(m.content) for m in model.calls[2])
    assert "What are Apple's key risk factors?" in third_turn
    assert stub_search.content in third_turn
    assert len(second["messages"]) == len(first["messages"]) + 2


def test_a_different_thread_starts_empty(scripted, saved_graph):
    scripted(AIMessage("first"), AIMessage("second"))

    turn(saved_graph, "What are Apple's key risk factors?", "t1")
    other = turn(saved_graph, "Which of those involve suppliers?", "t2")

    assert len(other["messages"]) == 2
    assert "Apple" not in str(other["messages"][0].content)


def test_citations_accumulate_over_a_thread(scripted, saved_graph, stub_search):
    """A follow-up's answer rests on the passages the first turn retrieved, so
    the thread's sources are the honest list — and the reducer still de-dupes."""
    scripted(
        AIMessage("", tool_calls=[tool_call("search_filings", {"query": "risks", "ticker": "AAPL"}, "a")]),
        AIMessage("Apple flagged supply chain risk."),
        AIMessage("", tool_calls=[tool_call("search_filings", {"query": "suppliers", "ticker": "AAPL"}, "b")]),
        AIMessage("The supplier concentration one."),
    )

    turn(saved_graph, "What are Apple's key risk factors?", "t1")
    second = turn(saved_graph, "Which of those involve suppliers?", "t1")

    assert len(second["citations"]) == 1
    assert isinstance(second["citations"][0], Citation)


def test_the_context_guard_never_reaches_the_checkpoint(scripted, saved_graph, stub_search):
    """Trimming is per-turn. A stub that reached state would be checkpointed,
    turning a saving into permanent data loss."""
    scripted(AIMessage("done"))

    state = saved_graph.invoke(
        {"messages": conversation(5), "citations": []},
        config=agent.thread_config("t1"),
    )

    assert elided(state["messages"]) == []


def test_a_graph_compiles_and_runs_without_a_checkpointer(graph, scripted):
    """The single-shot path stays intact — and needs no database."""
    scripted(AIMessage("done"))

    assert graph.checkpointer is None
    assert agent.final_text(
        graph.invoke({"messages": [HumanMessage("q")], "citations": []})["messages"]
    ) == "done"


def test_thread_config_carries_both_the_thread_and_the_limit():
    config = agent.thread_config("abc")

    assert config["configurable"]["thread_id"] == "abc"
    assert config["recursion_limit"] == agent.RECURSION_LIMIT
    assert agent.thread_config("abc", recursion_limit=4)["recursion_limit"] == 4


class RecordingGraph:
    """Stands in for the compiled graph, to see what `answer()` sends it."""

    def __init__(self) -> None:
        self.configs: list[dict] = []

    def invoke(self, state, config):
        self.configs.append(config)
        return {"messages": [*state["messages"], AIMessage("ok")], "citations": []}


def test_answer_defaults_to_a_fresh_thread_each_call(monkeypatch):
    """So every caller that predates sessions keeps getting exactly one
    question's worth of context, with no argument and no leakage between runs."""
    recorder = RecordingGraph()
    monkeypatch.setattr(agent, "get_graph", lambda: recorder)

    first = agent.answer("q")
    second = agent.answer("q")

    threads = [c["configurable"]["thread_id"] for c in recorder.configs]
    assert threads[0] != threads[1]
    assert first["thread_id"] == threads[0]
    assert second["thread_id"] == threads[1]


def test_answer_uses_the_thread_id_it_was_given(monkeypatch):
    recorder = RecordingGraph()
    monkeypatch.setattr(agent, "get_graph", lambda: recorder)

    result = agent.answer("q", thread_id="session-7", recursion_limit=9)

    assert recorder.configs[0] == {
        "configurable": {"thread_id": "session-7"},
        "recursion_limit": 9,
    }
    assert result["thread_id"] == "session-7"
    assert result["answer"] == "ok"


def test_a_checkpointed_conversation_round_trips_through_the_serializer():
    """The serializer is where C4b can lose data quietly.

    LangGraph allows unregistered types with a warning now and will refuse them
    later — refused, a `Citation` revives as `None` rather than raising, so the
    citation list shrinks and nothing says why. `CHECKPOINT_SERDE` names the
    type; this asserts that naming it does not cost the built-in message types,
    which is the other way to get this wrong.
    """
    source = citation()
    state = {
        "messages": [
            HumanMessage("What are Apple's key risk factors?"),
            AIMessage("", tool_calls=[tool_call("search_filings", {"query": "risks"})]),
            ToolMessage(content="[1] a passage", tool_call_id="call-1",
                        name="search_filings", artifact=[source]),
        ],
        "citations": [source],
    }

    revived = agent.CHECKPOINT_SERDE.loads_typed(agent.CHECKPOINT_SERDE.dumps_typed(state))

    assert revived["citations"] == [source]
    assert isinstance(revived["citations"][0], Citation)
    assert [type(m) for m in revived["messages"]] == [HumanMessage, AIMessage, ToolMessage]

    # And the part that does *not* survive as an object: a restored
    # ToolMessage's `artifact` comes back as plain dicts, because LangChain
    # revives the message and treats that field as data. Harmless, and asserted
    # so it stays known: artifacts are read once, by `tool_node`, off messages
    # it has just produced in-process. Nothing reads them back out of a
    # checkpoint — `state["citations"]` above is what a resumed thread uses.
    assert revived["messages"][2].artifact == [source.model_dump()]


def test_nothing_touches_postgres_until_a_checkpointed_graph_is_asked_for(monkeypatch):
    """Importing this module, and compiling a graph, must stay offline — that is
    what lets these tests and app/api.py's import-time wiring work without a
    database or a key."""
    def explode():
        raise AssertionError("the pool was opened")

    monkeypatch.setattr(agent, "get_pool", explode)

    agent.build_graph()  # no checkpointer asked for, so no connection


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
