"""D1 · The HTTP service. One real endpoint, a health check, and a thread reader.

    uvicorn app.api:app --reload

Deliberately thin: no retrieval, no prompt, no graph logic — those already
worked, and this is just the missing way to call them over HTTP. What it does
own is what only matters once requests arrive concurrently: when the singletons
get built (the lifespan, not per-request), how many runs may overlap
(`MAX_CONCURRENT_RUNS`), and what failure looks like as a status code.

`thread_id` is client-held and never inferred — deriving it from an IP or
header would let two users behind one NAT share a conversation. See
`scripts/check_api.py` for the concurrency gate.

`POST /query` and `POST /query/stream` are the same run, reported two ways.
The blocking form stays because every gate and every offline test calls it and
because a client that just wants an answer should not have to parse a protocol
to get one. The streaming form exists because the run takes 20-60 seconds, and
a progress bar that cannot say *what* is being read is only marginally better
than silence.
"""

import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.errors import GraphRecursionError
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, Field, field_validator

from app import agent, embeddings, finnhub, sec_http
from app.config import settings
from app.db import close_pool, get_conn, init_schema
from app.tools import Citation

logger = logging.getLogger(__name__)


# How many agent runs may be in flight at once — arithmetic, not taste, and a
# module constant rather than a setting so nobody tunes it without redoing the
# arithmetic. The binding limit is the psycopg pool (max_size=8): one request
# fans out to ~2 brief borrows per parallel search_filings call, so 4 runs peak
# near 8-9 while anyio's default of 40 threads would peak at 80-160. The
# checkpointer isn't a pressure point (PostgresSaver serialises onto one
# connection via its own lock); `embeddings._encode_lock` and
# `finnhub._RateLimiter` are the other two serialisation points worth knowing
# about. Full arithmetic in CLAUDE.md. Per process — `--workers N` multiplies
# both this and the embedding model's memory footprint.
MAX_CONCURRENT_RUNS = 4

# Over the cap is an immediate 503, never a queue — queuing behind a 20-60s run
# is indistinguishable from a hang, and makes "at capacity" indistinguishable
# from "slow" to both a client and the gate.
RETRY_AFTER_SECONDS = 30

# The longest question in either gate is ~200 characters.
MAX_QUESTION_CHARS = 2000

_run_slots = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)

# EDGAR_USER_AGENT is deliberately absent — nothing in the request path calls
# SEC, so it's an ingestion requirement, not a serving one.
REQUIRED_KEYS = (
    ("ANTHROPIC_API_KEY", "anthropic_api_key"),
    ("FINNHUB_API_KEY", "finnhub_api_key"),
    ("TAVILY_API_KEY", "tavily_api_key"),
)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    # `min_length=1` matters more than it looks. `agent.answer()` does
    # `thread_id or str(uuid4())`, so an empty string silently starts a *new*
    # conversation while the client believes it continued one. Rejecting it at
    # the boundary is the only place that distinction is still visible.
    thread_id: str | None = Field(default=None, min_length=1)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class QueryResponse(BaseModel):
    """`Citation` is imported from app/tools.py, not redeclared.

    It is already a pydantic model with exactly the three fields this response
    promises, and it is the most important contract in the project — a second
    definition of it here would be free to drift from the one the tools build.

    `route` is E1's dispatch decision, and it is reported because the client
    cannot otherwise tell the two empty-citation cases apart: `simple` and
    `advisory` answered without evidence on purpose, while a `research` answer
    with no citations found nothing. A UI renders those differently, and the
    difference is exactly the one this system exists to keep visible.
    """

    answer: str
    citations: list[Citation]
    thread_id: str
    route: str


class ThreadMessage(BaseModel):
    role: str
    content: str


class ThreadResponse(BaseModel):
    thread_id: str
    messages: list[ThreadMessage]
    citations: list[Citation]


class HealthResponse(BaseModel):
    status: str
    database: bool
    keys: dict[str, bool]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentRuntime:
    """What the lifespan built, handed to endpoints by `get_runtime`.

    `answer` and `thread_state` are carried as callables rather than reached
    through the module so the offline tests can override this whole object and
    exercise the HTTP layer with no database, no key, and no model.
    """

    graph: object
    answer: object
    stream_answer: object
    thread_state: object
    llm_error: str | None
    started_at: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build everything once, before the first request can arrive.

    Startup failure is deliberately asymmetric. **Postgres unreachable aborts
    startup** — without it there is no checkpointer, so no graph, and a process
    that boots into that state is lying about being ready. **A missing model key
    does not**: it is a per-request capability, and the endpoint whose entire job
    is reporting which capabilities are present cannot report it if the process
    refuses to boot. It would also leave `scripts/check_api.py` unable to tell
    "server down" from "key missing", which is the distinction its negative
    control rests on.
    """
    init_schema()
    graph = agent.get_graph()  # builds the pool, migrates the checkpointer, compiles

    if settings.api_warm_embeddings:
        # embed_query(), not get_model(): only a real encode warms the Metal
        # kernels, and that's what makes the state that once segfaulted the
        # interpreter (parallel search_filings, cold process) unreachable here.
        # scripts/check_api.py sets this false to reach it on purpose.
        embeddings.embed_query("warm start")

    llm_error: str | None = None
    try:
        agent.get_llm()
    except RuntimeError as exc:
        # Recorded, not raised. /health exists to report exactly this.
        llm_error = str(exc)
        logger.warning("model unavailable: %s", llm_error)

    app.state.runtime = AgentRuntime(
        graph=graph,
        answer=agent.answer,
        stream_answer=agent.stream_answer,
        thread_state=agent.thread_state,
        llm_error=llm_error,
        started_at=time.time(),
    )
    logger.info("api ready (model %s)", settings.anthropic_model)

    yield

    # No checkpointer close — it rides the pool.
    finnhub.close_client()
    sec_http.close_client()
    close_pool()


app = FastAPI(
    title="Confluence",
    description="Financial research over SEC filings, market data, and the web.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_runtime(request: Request) -> AgentRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        # Reachable only if a request arrives before the lifespan finished, or
        # in a test that built the app without one. A 503 rather than the
        # AttributeError that would otherwise surface as a 500.
        raise HTTPException(status_code=503, detail="service is still starting")
    return runtime


def run_slot() -> Iterator[None]:
    """One in-flight agent run, or a 503.

    A generator dependency rather than a `try/finally` in the handler, so
    FastAPI's own teardown releases the slot even when the endpoint raises —
    and so it keeps being released after someone adds an early `return`.
    """
    if not _run_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail=f"at capacity ({MAX_CONCURRENT_RUNS} runs in flight); retry shortly",
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )
    try:
        yield
    finally:
        _run_slots.release()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _present(value: str) -> bool:
    """The same emptiness rule `agent._require_key` applies, so /health cannot
    call a key present that the model builder will reject."""
    key = value.strip()
    return bool(key) and not key.lower().startswith("your")


def _probe_database() -> bool:
    """`SELECT 1`. No model call and no embedding, deliberately — a health check
    that costs money is a health check nothing calls often."""
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        return True
    except (PoolTimeout, psycopg.OperationalError) as exc:
        logger.warning("health probe failed: %s", exc)
        return False


@app.get("/health", response_model=HealthResponse)
def health() -> JSONResponse:
    """Readiness: Postgres reachable and the required keys present.

    **Any one missing is a 503, not a 200 with a sad body.** The consumers here
    are machines — a compose healthcheck, a load balancer, check_api.py's first
    call — and failure reported only inside a 200 body is failure reported in a
    channel nobody reads. The body is identical either way, so this costs no
    diagnosability.
    """
    keys = {name: _present(getattr(settings, attr)) for name, attr in REQUIRED_KEYS}
    database = _probe_database()
    ok = database and all(keys.values())
    body = HealthResponse(
        status="ok" if ok else "degraded", database=database, keys=keys
    )
    return JSONResponse(status_code=200 if ok else 503, content=body.model_dump())


@app.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    runtime: AgentRuntime = Depends(get_runtime),
    _slot: None = Depends(run_slot),
) -> dict:
    """Answer one question, in a conversation.

    **Citations are the whole thread's, not this turn's** — `agent.answer()`
    returns them that way, and slicing would drop exactly the sources a later
    turn relies on again (`merge_citations` de-dupes, so a repeat keeps its
    original index rather than moving to the end).

    `recursion_limit` is deliberately not exposed — a client-tunable spend
    limit is a different feature. And a disconnecting client does not cancel
    the run (`graph.invoke()` is uninterruptible), so its slot holds to the end.
    """
    if runtime.llm_error:
        raise HTTPException(status_code=503, detail=runtime.llm_error)

    # `response_model=QueryResponse` is what drops `messages` (LangChain
    # objects, often enormous) from the four keys `answer()` returns.
    return runtime.answer(payload.question, payload.thread_id)


# ---------------------------------------------------------------------------
# The same run, streamed
# ---------------------------------------------------------------------------
#
# Server-sent events rather than a WebSocket: the traffic is one-directional and
# one-shot, `fetch` can read it without a second protocol, and the run is not
# cancellable anyway (`graph.stream()` is as uninterruptible as `graph.invoke()`
# was), so the return channel a socket would buy has nothing to carry.
#
# **Every frame is `data:` with a `type` inside**, rather than SSE named events.
# One parser on the client, one place to add an event, and `json.dumps` escapes
# newlines — so a frame is always exactly one line and the framing cannot be
# broken by an answer that contains a blank line.
#
# **The run slot is taken by hand here, not through `Depends(run_slot)`.** A
# generator dependency is torn down when the *response* completes, and for a
# StreamingResponse that is a different moment from when the handler returns —
# the handler returns immediately, before the graph has run at all. Depending on
# it would release the slot at the start of the run instead of the end, which is
# a cap that silently stops capping. Acquired before the response is built so
# "at capacity" is still a real 503 with a status line, released in the
# generator's `finally` so it survives a client that disconnects mid-run.


def _frame(event: dict) -> str:
    """One SSE frame. `json.dumps` guarantees it stays a single line."""
    return f"data: {json.dumps(event)}\n\n"


@app.post("/query/stream")
def query_stream(
    payload: QueryRequest, runtime: AgentRuntime = Depends(get_runtime)
) -> StreamingResponse:
    """Answer one question, reporting progress as it goes.

    Same request body, same conversation, same answer as `POST /query`. The
    response is a `text/event-stream` of the events `agent.stream_answer()`
    documents — `start`, `route`, `step`, `sources`, `token`, `reset`, `done`,
    `error`.

    **A failure after the first frame arrives as an `error` event, not a status
    code.** The status line is long gone by then, so the alternative is a
    truncated body, which a client cannot tell from a short answer. The two
    failures that can still be status codes — no model key, and over the
    concurrency cap — are raised before the response is built, and deliberately
    so: those are the two a client can do something about.
    """
    if runtime.llm_error:
        raise HTTPException(status_code=503, detail=runtime.llm_error)

    if not _run_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail=f"at capacity ({MAX_CONCURRENT_RUNS} runs in flight); retry shortly",
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )

    def events():
        try:
            for event in runtime.stream_answer(payload.question, payload.thread_id):
                yield _frame(event)
        finally:
            _run_slots.release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Nginx and friends buffer a proxied response by default, which
            # would hold every frame until the run finished and deliver the
            # whole stream at once — the exact silence this endpoint removes,
            # reintroduced by infrastructure and invisible in development.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/threads/{thread_id}", response_model=ThreadResponse)
def read_thread(
    thread_id: str, runtime: AgentRuntime = Depends(get_runtime)
) -> ThreadResponse:
    """What Postgres holds for one conversation. No model call, so no run slot.

    **An unknown thread is a 404, not a 200 with an empty list.** A thread comes
    into existence only when a turn has run, so "exists but empty" is not a
    reachable state — and a 200 there would make a typo'd id indistinguishable
    from a real conversation. That is the corpus-gap failure shape, silence read
    as a fact, arriving through a different door.
    """
    state = runtime.thread_state(thread_id)
    messages = state.get("messages", [])
    if not messages:
        raise HTTPException(status_code=404, detail=f"no thread {thread_id!r}")

    # Returned whole, including tool results, which on a heavy thread are large.
    # Truncating here would be a second context guard — invisible, in a
    # different file, with different rules from C4a's. A UI filters on `role`.
    return ThreadResponse(
        thread_id=thread_id,
        messages=[
            ThreadMessage(role=message.type, content=_message_text(message))
            for message in messages
        ],
        citations=state.get("citations", []),
    )


def _message_text(message) -> str:
    """Flatten a message body to text.

    Anthropic returns content as a list of blocks once tool use is involved, so
    `.content` is not reliably a string — the same rule `agent.final_text`
    applies to the last message, applied here to all of them.
    """
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------
#
# Three handlers, and no catch-all. Genuinely unexpected exceptions are left to
# surface as 500s with a traceback in the log — the same rule app/tools.py
# applies to tool errors, and for the same reason: a bug should look like a bug.


@app.exception_handler(PoolTimeout)
@app.exception_handler(psycopg.OperationalError)
async def _database_unavailable(request: Request, exc: Exception) -> JSONResponse:
    logger.error("database unavailable: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "database unavailable"},
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
    )


@app.exception_handler(GraphRecursionError)
async def _recursion_exhausted(request: Request, exc: Exception) -> JSONResponse:
    """500, not 429 or 503.

    The run failed server-side and a retry re-bills the identical loop, so any
    status implying "try again" would be a lie that costs money.
    """
    logger.error("recursion limit exhausted: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"the agent exceeded its {agent.RECURSION_LIMIT}-step limit "
            "without finishing; the question may be too broad"
        },
    )
