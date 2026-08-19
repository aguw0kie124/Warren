"""Offline tests for the HTTP layer.

No database, no keys, no model. Whether the service returns a *good* answer is
scripts/check_api.py's question against a real graph; what is testable here is
the wiring between HTTP and `agent.answer()`, and that wiring's failures are
the quiet kind.

Three of them are worth naming, because none would show up as an error:

- **`messages` leaking into the response body.** `answer()` returns four keys
  and the contract promises three. The fourth is a list of LangChain objects,
  frequently enormous and not JSON-serialisable as it stands; only
  `response_model` keeps it out. Drop that and the first thing that happens is
  a 500 on every query — but reorder it into a `dict` return without the model
  and the body silently grows by two orders of magnitude.
- **A leaked run slot.** The semaphore is bounded, so a slot released twice
  raises, but a slot never released just lowers the cap by one, permanently and
  invisibly, until the service refuses everything.
- **Surface creep.** Three endpoints is the whole spec. A fourth added in
  passing is exactly the kind of thing nothing else notices.

The mechanic that keeps all of this offline: `TestClient(app)` is constructed
but never entered as a context manager, so Starlette never runs the lifespan
and nothing touches Postgres — and `get_runtime` is overridden with a fake, so
no model is ever built. That mirrors `ScriptedModel` in tests/test_agent.py.
"""

import importlib
import sys
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from psycopg_pool import PoolTimeout

from app import api
from app.api import AgentRuntime, app, get_runtime
from app.tools import Citation


def citation(type_: str = "filing", label: str = "AAPL FY2025 10-K · Item 1A") -> Citation:
    return Citation(type=type_, label=label, source_url="https://sec.gov/x.htm")


class FakeRuntime:
    """Stands in for what the lifespan builds.

    Records what `answer` was called with — the `thread_id` the endpoint passed
    down is the whole of the session contract, and it is not visible in the
    response when the client supplied it. An exception in `result` is raised
    instead of returned, which is how the error-mapping tests fail the graph.
    """

    def __init__(self, result=None, state=None, llm_error=None) -> None:
        self._result = result if result is not None else {
            "answer": "Apple lists supply concentration among its risks.",
            "citations": [citation()],
            "messages": [HumanMessage("q"), AIMessage("a")],
            "thread_id": "thread-from-runtime",
        }
        self._state = state if state is not None else {"messages": [], "citations": []}
        self.llm_error = llm_error
        self.graph = object()
        self.started_at = 0.0
        self.calls: list[tuple] = []

    def answer(self, question, thread_id=None, **kwargs):
        self.calls.append((question, thread_id))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def thread_state(self, thread_id):
        return self._state


@pytest.fixture
def runtime():
    """A fake runtime installed over the dependency, torn down after."""
    fake = FakeRuntime()
    app.dependency_overrides[get_runtime] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    # Deliberately not `with TestClient(app) as c`: entering the context manager
    # runs the lifespan, which applies the schema and compiles the graph. These
    # tests must never reach a database.
    return TestClient(app)


@pytest.fixture
def healthy(monkeypatch):
    """A reachable database and every required key present.

    The database is faked at `get_conn` rather than at `_probe_database`, so the
    probe itself — including its exception handling — is the code under test in
    every /health case below, not something the fixture stubbed away.
    """
    @contextmanager
    def reachable():
        yield SimpleNamespace(execute=lambda sql: None)

    monkeypatch.setattr(api, "get_conn", reachable)
    for _, attr in api.REQUIRED_KEYS:
        monkeypatch.setattr(api.settings, attr, "sk-set")


# --- the contract -----------------------------------------------------------


def test_query_returns_exactly_the_three_promised_fields(client, runtime):
    """And in particular *not* `messages`, which `answer()` also returns."""
    response = client.post("/query", json={"question": "What are Apple's risks?"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "citations", "thread_id"}
    assert body["citations"] == [
        {"type": "filing", "label": "AAPL FY2025 10-K · Item 1A",
         "source_url": "https://sec.gov/x.htm"}
    ]


def test_a_supplied_thread_id_reaches_the_agent_and_comes_back(client, runtime):
    runtime._result = {**runtime._result, "thread_id": "session-7"}

    response = client.post("/query", json={"question": "q", "thread_id": "session-7"})

    assert runtime.calls == [("q", "session-7")]
    assert response.json()["thread_id"] == "session-7"


def test_an_omitted_thread_id_is_passed_down_as_none_and_minted_below(client, runtime):
    """The id is the agent's to mint, never the server's to infer — inferring it
    from an IP or a header would make two users behind one NAT share a thread."""
    response = client.post("/query", json={"question": "q"})

    assert runtime.calls == [("q", None)]
    assert response.json()["thread_id"] == "thread-from-runtime"


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "q", "thread_id": ""},
        {"question": "x" * (api.MAX_QUESTION_CHARS + 1)},
    ],
    ids=["missing", "empty", "blank", "empty-thread-id", "oversized"],
)
def test_bad_requests_are_rejected_before_anything_is_spent(client, runtime, payload):
    """The empty `thread_id` is the sharp one: `answer()` does
    `thread_id or uuid4()`, so an empty string silently starts a *new*
    conversation while the client believes it continued one."""
    assert client.post("/query", json=payload).status_code == 422
    assert runtime.calls == []


# --- failures ---------------------------------------------------------------


def test_a_missing_model_key_is_503_and_never_reaches_the_agent(client):
    fake = FakeRuntime(llm_error="ANTHROPIC_API_KEY is not set.")
    app.dependency_overrides[get_runtime] = lambda: fake

    response = client.post("/query", json={"question": "q"})

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
    assert fake.calls == []
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "error, status",
    [
        (PoolTimeout("pool is full"), 503),
        (psycopg.OperationalError("connection refused"), 503),
        (GraphRecursionError("limit"), 500),
    ],
    ids=["pool-timeout", "postgres-down", "recursion"],
)
def test_known_failures_map_to_their_status(client, runtime, error, status):
    runtime._result = error

    assert client.post("/query", json={"question": "q"}).status_code == status


def test_recursion_exhaustion_says_it_is_not_worth_retrying(client, runtime):
    """500 rather than 429/503: the run failed server-side and a retry re-bills
    the identical loop, so a status implying 'try again' costs money to obey."""
    runtime._result = GraphRecursionError("limit")

    response = client.post("/query", json={"question": "q"})

    assert response.status_code == 500
    assert str(api.agent.RECURSION_LIMIT) in response.json()["detail"]


def test_an_unexpected_exception_is_left_to_look_like_a_bug(runtime):
    """No catch-all handler, the same rule app/tools.py applies."""
    runtime._result = ZeroDivisionError("boom")
    client = TestClient(app, raise_server_exceptions=False)

    assert client.post("/query", json={"question": "q"}).status_code == 500


def test_a_request_arriving_before_startup_finished_is_503(client):
    """Rather than the AttributeError that would otherwise surface as a 500."""
    app.dependency_overrides.clear()
    app.state.__dict__.pop("runtime", None)

    assert client.post("/query", json={"question": "q"}).status_code == 503


# --- the concurrency cap ----------------------------------------------------


def test_over_the_cap_is_an_immediate_503_and_the_slot_comes_back(monkeypatch):
    """The cap's two failure modes at once.

    A queue instead of a 503 would make 'at capacity' indistinguishable from
    'slow' — to a client and to the gate. A slot that is taken but never
    returned lowers the cap by one, permanently, with nothing said; the third
    assertion is what catches that.
    """
    monkeypatch.setattr(api, "_run_slots", threading.BoundedSemaphore(1))
    release = threading.Event()
    entered = threading.Event()

    class BlockingRuntime(FakeRuntime):
        def answer(self, question, thread_id=None, **kwargs):
            entered.set()
            release.wait(timeout=5)
            return self._result

    app.dependency_overrides[get_runtime] = lambda: BlockingRuntime()
    client = TestClient(app)
    first: list = []

    holder = threading.Thread(
        target=lambda: first.append(client.post("/query", json={"question": "q"}))
    )
    holder.start()
    assert entered.wait(timeout=5)

    blocked = client.post("/query", json={"question": "q"})
    assert blocked.status_code == 503
    assert blocked.headers["Retry-After"] == str(api.RETRY_AFTER_SECONDS)

    release.set()
    holder.join(timeout=5)
    assert first[0].status_code == 200

    # The slot came back, so the next request is served rather than refused.
    assert client.post("/query", json={"question": "q"}).status_code == 200
    app.dependency_overrides.clear()


# --- /threads ---------------------------------------------------------------


def test_a_stored_thread_renders_roles_and_carries_its_citations(client):
    fake = FakeRuntime(state={
        "messages": [
            HumanMessage("What are Apple's risks?"),
            AIMessage("", tool_calls=[{"name": "search_filings", "args": {},
                                       "id": "c1", "type": "tool_call"}]),
            ToolMessage(content="[1] a passage", tool_call_id="c1",
                        name="search_filings"),
            AIMessage([{"type": "text", "text": "Supply concentration."}]),
        ],
        "citations": [citation()],
    })
    app.dependency_overrides[get_runtime] = lambda: fake

    body = client.get("/threads/session-7").json()

    assert body["thread_id"] == "session-7"
    assert [m["role"] for m in body["messages"]] == ["human", "ai", "tool", "ai"]
    # Content blocks flattened, the same rule agent.final_text applies.
    assert body["messages"][3]["content"] == "Supply concentration."
    assert body["citations"][0]["type"] == "filing"
    app.dependency_overrides.clear()


def test_an_unknown_thread_is_404_not_an_empty_conversation(client, runtime):
    """A 200 with `[]` would make a typo'd id indistinguishable from a real
    thread — silence read as a fact, which is the failure the corpus-gap check
    exists to prevent, arriving through a different door."""
    assert client.get("/threads/nope").status_code == 404


def test_reading_a_thread_never_calls_the_model(client, runtime):
    runtime._state = {"messages": [HumanMessage("q")], "citations": []}

    client.get("/threads/session-7")

    assert runtime.calls == []


# --- /health ----------------------------------------------------------------


def test_health_is_ok_when_postgres_and_every_key_are_there(client, healthy):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] is True
    assert set(body["keys"]) == {name for name, _ in api.REQUIRED_KEYS}
    assert all(body["keys"].values())


@pytest.mark.parametrize("name, attr", api.REQUIRED_KEYS)
def test_a_missing_key_is_503_and_the_body_names_it(client, healthy, monkeypatch,
                                                    name, attr):
    monkeypatch.setattr(api.settings, attr, "")

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["keys"][name] is False


def test_a_placeholder_key_does_not_count_as_present(client, healthy, monkeypatch):
    """The same emptiness rule `agent._require_key` applies, so /health cannot
    call a key present that the model builder is about to reject."""
    monkeypatch.setattr(api.settings, "anthropic_api_key", "your-key-here")

    assert client.get("/health").status_code == 503


def test_health_is_503_when_postgres_is_unreachable(client, healthy, monkeypatch):
    def refuse():
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(api, "get_conn", refuse)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["database"] is False


def test_edgar_user_agent_is_not_a_health_requirement():
    """Nothing in the request path calls SEC — citation source_urls are strings
    the retriever already holds — so it is an ingestion requirement. Checking it
    would report red on a perfectly serviceable read-only deployment."""
    assert "EDGAR_USER_AGENT" not in {name for name, _ in api.REQUIRED_KEYS}


# --- structural -------------------------------------------------------------


def test_the_service_exposes_exactly_three_endpoints():
    """Three is the whole spec. A fourth added in passing is the thing this
    step most easily does by accident, and nothing else would notice."""
    routes = {
        (route.path, method)
        for route in app.routes
        if getattr(route, "methods", None)
        for method in route.methods
        if method not in {"HEAD", "OPTIONS"}
        and not route.path.startswith(("/docs", "/redoc", "/openapi"))
    }

    assert routes == {
        ("/query", "POST"),
        ("/health", "GET"),
        ("/threads/{thread_id}", "GET"),
    }


def test_importing_the_api_opens_no_connection_and_needs_no_key(monkeypatch):
    """The invariant that lets these tests run offline and lets uvicorn import
    the module before its lifespan decides what to build. Mirrors
    test_nothing_touches_postgres_until_a_checkpointed_graph_is_asked_for.
    """
    def explode(*args, **kwargs):
        raise AssertionError("something reached out at import time")

    monkeypatch.setattr(api.agent, "get_pool", explode)
    monkeypatch.setattr(api.agent, "get_llm", explode)

    # Deleted through monkeypatch so the real module is restored afterwards —
    # a reload in place would leave every later test holding a stale one.
    monkeypatch.delitem(sys.modules, "app.api")
    fresh = importlib.import_module("app.api")

    TestClient(fresh.app)
