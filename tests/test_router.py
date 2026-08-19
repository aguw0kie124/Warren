"""Offline tests for E1's router — the wiring, not the classifications.

Whether "who is Tesla's CEO" actually lands on `simple` is a judgement about a
model, and it belongs to scripts/check_router.py where a real one is asked.
What is testable here is everything that would fail *silently* if it broke:

- **The fallback direction.** A label the schema could not seat has to become
  `research`. Every other outcome — `simple` especially — answers a question
  from memory that may have needed evidence, and produces a fluent, confident,
  zero-citation answer while raising nothing.
- **The skip on continuations.** A follow-up must never be classified. Nothing
  errors if it is; C4b's multi-turn capability just quietly starts asking
  "which company did you mean?" about a company named one turn earlier.
- **That `route` is written on every path.** The checkpoint holds the previous
  turn's value, so a node that returns nothing lets a stale route stand.
- **That the respond path binds no tools.** Binding them re-adds the ~2.4k
  prefix this path exists to skip, and nothing about the answers would change.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app import agent, router


class ScriptedClassifier:
    """Stands in for `get_router_llm()`.

    Returns plain AIMessages, because that is what the classifier returns now —
    `with_structured_output` was removed once it measured at ~677 tokens of
    schema overhead around a one-word answer. The reply shape being ordinary
    text is precisely why `parse_label` has to be forgiving, and why these
    tests feed it punctuation and prose as well as bare labels.
    """

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []

    def invoke(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def reply(text: str, input_tokens: int = 300, output_tokens: int = 4) -> AIMessage:
    return AIMessage(text, usage_metadata={
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    })


@pytest.fixture
def classifier(monkeypatch):
    def install(*replies) -> ScriptedClassifier:
        model = ScriptedClassifier(*replies)
        monkeypatch.setattr(router, "get_router_llm", lambda: model)
        return model
    return install


class ScriptedResponder:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.calls: list[list] = []
        self.bind_tools_calls = 0

    def invoke(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.response

    def bind_tools(self, tools):
        self.bind_tools_calls += 1
        return self


@pytest.fixture
def responder(monkeypatch):
    def install(text: str = "answered") -> ScriptedResponder:
        model = ScriptedResponder(AIMessage(text))
        monkeypatch.setattr(agent, "get_respond_llm", lambda: model)
        return model
    return install


# --- classify ---------------------------------------------------------------


@pytest.mark.parametrize("route", ["simple", "research", "advisory", "clarify"])
def test_every_valid_label_passes_through(classifier, route):
    classifier(reply(route))

    assert router.classify("anything").route == route


@pytest.mark.parametrize("text", ["", "cheap", "I am not sure what you mean",
                                  "research or clarify"])
def test_an_unusable_reply_falls_to_research(classifier, text):
    """The direction is the whole safety argument.

    Falling to `simple` here would answer from memory whenever the classifier
    hiccuped — an unsourced answer produced by an infrastructure event rather
    than by a decision anyone made. `"research or clarify"` is in this set on
    purpose: a hedge is not an answer, and it should cost the fallback.
    """
    classifier(reply(text))

    assert router.classify("what are Apple's risks?").route == "research"


@pytest.mark.parametrize("text,expected", [
    ("simple.", "simple"),
    ("**advisory**", "advisory"),
    ("Label: clarify", "clarify"),
    ("  RESEARCH\n", "research"),
])
def test_a_punctuated_or_emphasised_label_still_reads(classifier, text, expected):
    """Tolerance here is cheap and the alternative is expensive: every reply
    the parser rejects is a full research run for a question that did not need
    one."""
    classifier(reply(text))

    assert router.classify("q").route == expected


def test_the_fallback_is_research(classifier):
    """Named rather than inline, so it cannot be quietly made cheaper."""
    assert router.FALLBACK_ROUTE == "research"
    assert router.FALLBACK_ROUTE in router.ROUTES


def test_provider_failures_propagate(monkeypatch):
    """Not swallowed into a default. A 429 is the retry policy's problem, and
    an outage turned into a silently more expensive service is worse than one
    that shows up."""
    class Boom(RuntimeError):
        pass

    class Exploding:
        def invoke(self, messages, **kwargs):
            raise Boom("provider down")

    monkeypatch.setattr(router, "get_router_llm", Exploding)

    with pytest.raises(Boom):
        router.classify("q")


def test_classification_reports_its_own_token_usage(classifier):
    """`router_node` adds no message to state, so `agent.token_usage()` cannot
    see these tokens. If they stop riding along, the router's cost becomes
    unmeasurable and nothing breaks — which is why this is pinned."""
    classifier(reply("simple", input_tokens=207, output_tokens=5))

    routing = router.classify("who runs Tesla?")

    assert routing.usage == {"input": 207, "output": 5}


def test_the_classifier_is_sent_the_router_prompt_and_nothing_else(classifier):
    model = classifier(reply("research"))

    router.classify("what are Apple's risks?")

    system, human = model.calls[0]
    assert system == ("system", router.ROUTER_PROMPT)
    assert human == ("human", "what are Apple's risks?")


def test_the_router_prompt_stays_small():
    """A constraint, not a preference: the savings are the absent prefix, and
    a prompt that grows to rival the research one deletes the whole module's
    reason to exist. ~4 chars/token, the same approximation C4a's guard uses.
    """
    assert len(router.ROUTER_PROMPT) // 4 < 400


def test_the_classifier_binds_no_tools_and_no_schema():
    """The measured reason `with_structured_output` was removed: it cost ~677
    tokens of schema per call, more than the prompt it wrapped. A future
    "let's make this type-safe" would silently restore that, so it is pinned
    here rather than left to the gate's cost check to catch after the fact.
    """
    llm = router.get_router_llm()

    assert not getattr(llm, "kwargs", {}).get("tools")
    assert llm.max_tokens <= 32   # one word is the whole contract


def test_the_router_prompt_states_the_asymmetry():
    """Enforced twice — here and in `classify()`. This is the half a model
    reads."""
    assert "When in doubt, answer research" in router.ROUTER_PROMPT


# --- router_node ------------------------------------------------------------


def test_a_first_turn_is_classified(classifier):
    model = classifier(reply("advisory"))

    result = agent.router_node({"messages": [HumanMessage("what should I buy?")]})

    assert result == {"route": "advisory"}
    assert len(model.calls) == 1


def test_a_continuation_is_never_classified(classifier):
    """The trap this rule exists for: read standalone, "which of those…" is
    textbook `clarify`, and routing it there would send every follow-up in the
    system to a clarification prompt."""
    model = classifier()  # no replies scripted — calling it would IndexError

    result = agent.router_node({
        "messages": [
            HumanMessage("what are Apple's risk factors?"),
            AIMessage("Apple lists supply concentration…"),
            HumanMessage("which of those involve suppliers outside the US?"),
        ]
    })

    assert result == {"route": "research"}
    assert model.calls == []


def test_every_path_writes_route(classifier):
    """Never left to a `.get()` default: the checkpoint holds the previous
    turn's value, and a node returning nothing would let it stand."""
    classifier(reply("simple"))

    first = agent.router_node({"messages": [HumanMessage("who runs Tesla?")]})
    later = agent.router_node({"messages": [HumanMessage("a"), AIMessage("b"),
                                            HumanMessage("c")]})

    assert "route" in first and "route" in later


# --- route_condition --------------------------------------------------------


def test_only_research_reaches_the_loop():
    assert agent.route_condition({"route": "research"}) == "agent"
    for route in ("simple", "advisory", "clarify"):
        assert agent.route_condition({"route": route}) == "respond"


def test_a_missing_route_does_not_reach_the_loop():
    """Absent state is not a research question. It terminates, where the
    clarify prompt asks rather than answers."""
    assert agent.route_condition({}) == "respond"


# --- respond_node -----------------------------------------------------------


@pytest.mark.parametrize("route", ["simple", "advisory", "clarify"])
def test_each_terminal_route_gets_its_own_prompt(responder, route):
    model = responder()

    agent.respond_node({"messages": [HumanMessage("q")], "route": route})

    system = model.calls[0][0]
    assert isinstance(system, SystemMessage)
    assert system.content == router.RESPOND_PROMPTS[route]


def test_an_unknown_route_asks_rather_than_answers(responder):
    model = responder()

    agent.respond_node({"messages": [HumanMessage("q")], "route": "nonsense"})

    assert model.calls[0][0].content == router.CLARIFY_PROMPT


def test_the_respond_path_binds_no_tools(responder):
    """Binding them would re-add the prefix this path exists to skip, and
    would hand a model that is supposed to decline a row of tools to reach
    for. Neither shows up in an answer."""
    model = responder()

    agent.respond_node({"messages": [HumanMessage("q")], "route": "simple"})

    assert model.bind_tools_calls == 0


def test_respond_returns_one_message_and_no_citations(responder):
    responder("From general knowledge rather than a filing: …")

    result = agent.respond_node({"messages": [HumanMessage("q")], "route": "simple"})

    assert list(result) == ["messages"]
    assert len(result["messages"]) == 1


def test_the_respond_prompt_carries_no_cache_breakpoint():
    """All four prompts sit far below any published minimum cacheable prefix,
    so a breakpoint would be decoration — and a decorative one invites the
    padding that C3 warns degrades routing."""
    for prompt in router.RESPOND_PROMPTS.values():
        assert isinstance(prompt, str)   # a plain string, not content blocks


# --- the prompts C5's rules moved into ---------------------------------------


def test_the_advisory_prompt_keeps_c5s_rules():
    """Moved out of SYSTEM_PROMPT, not dropped. tests/test_agent.py asserts
    they left; this asserts they arrived, so a duplicate cannot pass both."""
    prompt = router.ADVISORY_PROMPT
    assert "cannot screen, rank, or scan a universe" in prompt
    assert "does not recommend what to buy, sell, or hold" in prompt
    assert "Do not name example companies" in prompt


def test_the_simple_prompt_forbids_facts_beyond_identity():
    prompt = router.SIMPLE_PROMPT
    assert "You may NOT state a number" in prompt
    assert "hedge" in prompt
    assert "cutoff" in prompt


def test_no_prompt_names_an_example_company():
    """A name offered in passing reads as sourced in a system where every
    other answer carries citations — and these three answer with none at all."""
    for prompt in router.RESPOND_PROMPTS.values():
        for ticker in ("AAPL", "MSFT", "TSLA", "NVDA", "Nvidia", "Palantir"):
            assert ticker not in prompt


# --- the compiled graph -------------------------------------------------------


def test_the_routed_graph_is_the_loop_plus_two():
    nodes = set(agent.build_graph().get_graph().nodes) - {"__start__", "__end__"}

    assert nodes == {"router", "respond", "agent", "tools"}


def test_the_pre_router_shape_still_compiles():
    """`router=False` is what lets E1's gate price the two shapes against each
    other, and what keeps the loop testable without scripting a route."""
    nodes = set(agent.build_graph(router=False).get_graph().nodes) - {"__start__", "__end__"}

    assert nodes == {"agent", "tools"}


def test_a_terminal_route_never_reaches_the_tools_node(classifier, responder):
    """End to end through the real compiled graph: a question classified
    `advisory` produces an answer, no tool call, and no citations."""
    classifier(reply("advisory"))
    responder("I can't rank companies as investments.")

    state = agent.build_graph().invoke(
        {"messages": [HumanMessage("what should I buy?")], "citations": [], "route": ""}
    )

    assert state["route"] == "advisory"
    assert state["citations"] == []
    assert agent.tool_calls_made(state["messages"]) == []
    assert agent.final_text(state["messages"]) == "I can't rank companies as investments."
