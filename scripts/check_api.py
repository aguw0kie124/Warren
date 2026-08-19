"""D1 verification: the HTTP service, judged on the contract and on concurrency.

    python scripts/check_api.py                # everything
    python scripts/check_api.py --skip-burst   # cases 1-4 only, cheaper
    python scripts/check_api.py --burst 4      # concurrent runs in case 5
    python scripts/check_api.py --server URL   # against a server you started
    python scripts/check_api.py --quiet        # verdicts only, no answer text

**This script starts and stops its own uvicorn processes.** Every other gate
here is one command, and more importantly the burst's decisive claim is about a
*cold* process: a gate that required a hand-started server could not know
whether the earlier cases had already warmed it. Owning the process is what
makes "cold" a property this script controls rather than assumes. `--server`
opts out for debugging, and skips the two cases that need their own process.

Three servers, and only one of them costs anything:

  A  normal settings                — cases 1-4
  B  ANTHROPIC_API_KEY=""           — the health negative control; never calls
                                      a model, so it is free
  C  API_WARM_EMBEDDINGS=false      — untouched until the burst, so its first
                                      four requests race the embedding
                                      singleton from cold, in parallel

Needs Postgres up, all three API keys, and EDGAR_USER_AGENT — the last one for
*this script*, not for the server: case 2 fetches filing citation URLs through
`app.sec_http`, and nothing in the request path calls SEC.

Costs money: 4 questions in cases 2-4 plus 4 in the burst, roughly $0.15-0.25
on claude-haiku-4-5. Run the Module C gates first — this asserts on answers
from the same graph, so a routing or checkpointer failure would present here as
a FastAPI bug that it is not.

## What case 5 is for, and what it is not

`docs/PLAN.md` records the failure this exists to catch: after C2's gate passed,
an ad-hoc question segfaulted the interpreter, because four parallel
`search_filings` calls built four embedding models and four simultaneous Metal
initialisations killed CPython. The locks that fixed it are in place, and the
service pre-warms the model so the state cannot arise at all — which is exactly
why the burst runs against a server with warming *off*. What it establishes is
that the locks hold under concurrent cold parallel encoding; what the pre-warm
establishes is that a shipped server never enters that state. Neither claim
covers the other.

Four assertions, because three of them can pass for the wrong reason alone:

  survived      — the recorded failure was `signal 11`, not an exception, so no
                  status code can express it. The evidence has to be the
                  interpreter's own liveness afterwards.
  overlapped    — a server that silently serialises everything also survives.
                  Survival alone establishes nothing about concurrency.
  no cross-talk — process-wide-state bugs are quiet: they show up as one
                  conversation's passages in another's citations. Four
                  *identical* questions would hide that, which is why the four
                  requests name four different company pairs.
  cap engaged   — a fifth request must be refused rather than queued. If it is
                  served, MAX_CONCURRENT_RUNS is doing nothing.

**What the burst does not establish**, per the standing rule that a gate
establishes what it exercised and nothing more: nothing about sustained load,
nothing about Anthropic's rate ceiling at higher concurrency, and nothing about
`uvicorn --workers N` — each worker gets its own embedding model and its own
pool of 8, and the cap is per process.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import exit_code, indent, note, rule, summary, verdict  # noqa: E402

from app import api, finnhub  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import close_pool  # noqa: E402
from app.sec_http import close_client, sec_get  # noqa: E402

# pid -> uvicorn log path, so a dead server's last words survive it.
_LOGS: dict[int, str] = {}

# Long enough for a multi-hop first turn. httpx's 5-second default would fail
# this gate spuriously and look like a server bug.
QUERY_TIMEOUT = 240.0

FILINGS_QUESTION = "What are Apple's key risk factors in its most recent 10-K?"
MIXED_QUESTION = (
    "What did Apple say about supply chain risk, and has anything happened recently?"
)
# Meaningless on its own — which is the point. It can only resolve against a
# stored thread, so the pair of runs (with and without a thread_id) is the
# evidence, not either run alone.
FOLLOW_UP = "Which of those are new this year?"

# Four different pairs, deliberately. Identical questions would survive a
# shared-state bug without showing it; these make cross-talk visible as a
# company appearing in a thread that never asked about it.
BURST_PAIRS = [("Apple", "Microsoft"), ("Meta", "Tesla"),
               ("Apple", "Tesla"), ("Microsoft", "Meta")]
BURST_QUESTION = (
    "Compare how {a} and {b} describe competition risk in their most recent 10-K."
)
BURST_TICKERS = {"Apple": "AAPL", "Microsoft": "MSFT", "Meta": "META", "Tesla": "TSLA"}


# ---------------------------------------------------------------------------
# Servers
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(label: str, **env_overrides: str) -> tuple[subprocess.Popen, str]:
    """Boot one uvicorn process and wait for it to answer.

    Waits on /health answering *at all*, not on it answering 200 — server B
    exists precisely to return 503 there, and a readiness check that required
    200 could never start it.
    """
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, **env_overrides}
    # A file rather than a PIPE: a segfaulting server is exactly the case whose
    # output has to survive, and a pipe nobody drains fills and blocks the
    # process that is being tested. `_LOGS` keeps the handle so the tail can be
    # read after the process is gone.
    log = tempfile.NamedTemporaryFile("w+", prefix=f"uvicorn-{label}-", suffix=".log",
                                      delete=False)
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _LOGS[process.pid] = log.name

    deadline = time.time() + 180
    while time.time() < deadline:
        if process.poll() is not None:
            print(f"\n  server {label} died during startup:\n{indent(server_log(process))}")
            raise SystemExit(2)
        try:
            httpx.get(f"{url}/health", timeout=5.0)
            print(f"  server {label} up on {url}"
                  + (f"  ({', '.join(f'{k}={v!r}' for k, v in env_overrides.items())})"
                     if env_overrides else ""))
            return process, url
        except httpx.RequestError:
            time.sleep(0.5)

    process.terminate()
    print(f"\n  server {label} never became ready:\n{indent(server_log(process))}")
    raise SystemExit(2)


def server_log(process: subprocess.Popen, tail: int = 3000) -> str:
    path = _LOGS.get(process.pid)
    if not path or not os.path.exists(path):
        return "(no log)"
    with open(path) as handle:
        return handle.read()[-tail:]


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def post_query(url: str, question: str, thread_id: str | None = None) -> httpx.Response:
    payload: dict = {"question": question}
    if thread_id:
        payload["thread_id"] = thread_id
    return httpx.post(f"{url}/query", json=payload, timeout=QUERY_TIMEOUT)


def show(response: httpx.Response, quiet: bool) -> dict:
    body = response.json()
    types = [c["type"] for c in body.get("citations", [])]
    print(f"\n    citations: {', '.join(sorted(set(types))) or 'none'} "
          f"({len(types)} total)")
    if not quiet:
        print(f"\n{indent(body.get('answer', ''))}")
    return body


# ---------------------------------------------------------------------------
# 1 · health
# ---------------------------------------------------------------------------


def check_health(url: str) -> None:
    rule("1. GET /health")

    response = httpx.get(f"{url}/health", timeout=10.0)
    body = response.json()
    print(f"    {json.dumps(body)}")

    verdict(response.status_code == 200, "health is 200")
    verdict(body.get("database") is True, "Postgres reachable")
    for name, present in body.get("keys", {}).items():
        verdict(present is True, f"{name} present")


def check_health_negative_control() -> None:
    """A server missing the model key.

    Without this, case 1 is a check that can only pass — it asserts on a
    correctly configured server and would report the same green if /health
    returned a constant. The second half also proves /query refuses *before*
    reaching the model, which is the difference between a cheap 503 and a
    request that spends money to fail.
    """
    rule("1b. a server with no ANTHROPIC_API_KEY (negative control, free)")

    process, url = start_server("B", ANTHROPIC_API_KEY="")
    try:
        response = httpx.get(f"{url}/health", timeout=10.0)
        body = response.json()
        verdict(response.status_code == 503, "health is 503 without the model key")
        verdict(body.get("keys", {}).get("ANTHROPIC_API_KEY") is False,
                "the body names ANTHROPIC_API_KEY as missing")
        verdict(body.get("database") is True,
                "and still reports Postgres reachable, so the two are separable")

        started = time.time()
        query = post_query(url, FILINGS_QUESTION)
        elapsed = time.time() - started
        verdict(query.status_code == 503, "/query is 503 rather than a model error")
        verdict(elapsed < 5.0, f"and refuses in {elapsed:.2f}s, so it never called out")
    finally:
        stop_server(process)


# ---------------------------------------------------------------------------
# 2 · one filings question
# ---------------------------------------------------------------------------


def check_single_query(url: str, quiet: bool) -> dict:
    rule(f"2. POST /query · {FILINGS_QUESTION}")

    response = post_query(url, FILINGS_QUESTION)
    if not verdict(response.status_code == 200, f"200 (got {response.status_code})"):
        print(f"    {response.text[:500]}")
        return {}
    body = show(response, quiet)

    # The contract is answer/citations/thread_id/route. `answer()` returns a
    # fifth field too, the full message list — enormous, and not something a
    # client should ever have to ignore.
    verdict(set(body) == {"answer", "citations", "thread_id", "route"},
            f"body is exactly answer/citations/thread_id/route (got {sorted(body)})")
    verdict(len(body.get("answer", "")) > 200, "the answer is substantive, not a stub")
    verdict(body.get("thread_id"), "a thread_id came back, unasked")
    verdict(any(c["type"] == "filing" for c in body.get("citations", [])),
            "at least one filing citation")
    return body


def check_citation_urls(citations: list[dict]) -> None:
    """Only the EDGAR ones — they go through this project's SEC client, which
    carries the required User-Agent and rate limit. News and web URLs are the
    providers' to keep working, and hammering them proves nothing about this
    code. A citation nobody can open is not a citation."""
    rule("2b. filing citation URLs resolve")

    filings = {c["source_url"]: c for c in citations if c["type"] == "filing"}
    if not filings:
        print("  (no filing citations in that run)")
        return
    for citation in filings.values():
        try:
            ok = sec_get(citation["source_url"], accept="text/html").status_code == 200
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            ok = False
            print(f"        {type(exc).__name__}: {exc}")
        verdict(ok, citation["label"][:80])


# ---------------------------------------------------------------------------
# 3 · mixed sources
# ---------------------------------------------------------------------------


def check_mixed_sources(url: str, quiet: bool) -> None:
    """One question that only a multi-source system can answer well.

    The filings half is dated by definition — a 10-K states the company's
    position as of its filing date — so an answer carrying *only* filing
    citations has quietly dropped half the question.
    """
    rule(f"3. POST /query · {MIXED_QUESTION}")

    response = post_query(url, MIXED_QUESTION)
    if not verdict(response.status_code == 200, f"200 (got {response.status_code})"):
        return
    body = show(response, quiet)

    types = [c["type"] for c in body.get("citations", [])]
    histogram = {t: types.count(t) for t in sorted(set(types))}
    print(f"    by type: {histogram}")
    verdict(len(set(types)) >= 2, f"mixed citation types ({sorted(set(types))})")
    verdict("filing" in types, "including a filing, so the SEC half was really consulted")


# ---------------------------------------------------------------------------
# 4 · multi-turn, and the thread reader
# ---------------------------------------------------------------------------


def check_multi_turn(url: str, thread_id: str, quiet: bool) -> dict:
    """The same follow-up twice, differing only in whether a thread was named.

    That pair is the evidence, not either run alone: a model guessing the
    most-discussed company would pass the first half by itself. It reuses case
    2's thread rather than opening a new one, so this costs one turn, not two.

    Note what is deliberately *not* asserted — that the resumed turn re-runs
    `search_filings`. C4b's own gate demanded that and failed a run for
    answering correctly without it: the passages were still in context and
    inside the guard's recent rounds, so searching again would have bought
    nothing. A gate that specifies *how* will eventually fail a model for doing
    better.
    """
    rule(f"4. multi-turn · {FOLLOW_UP}")

    print("    ── with the thread_id from case 2 ──")
    resumed = post_query(url, FOLLOW_UP, thread_id=thread_id)
    if not verdict(resumed.status_code == 200, f"200 (got {resumed.status_code})"):
        return {}
    resumed_body = show(resumed, quiet)
    verdict(resumed_body["thread_id"] == thread_id, "the same thread_id came back")

    print("\n    ── the same question, no thread_id (control) ──")
    fresh = post_query(url, FOLLOW_UP)
    fresh_body = show(fresh, quiet)
    verdict(fresh_body["thread_id"] != thread_id, "the control got a different thread")

    resumed_text = resumed_body["answer"].lower()
    fresh_text = fresh_body["answer"].lower()
    verdict("apple" in resumed_text,
            "the resumed turn resolved 'those' against the stored conversation")
    fresh_filings = [c for c in fresh_body["citations"] if c["type"] == "filing"]
    verdict("apple" not in fresh_text and not fresh_filings,
            "and the control resolved nothing — no Apple, no filing citations")

    if not quiet:
        print("\n    ↳ the human reading: does the control ask what is meant, rather "
              "than\n      inventing a company to be talking about?")
    return resumed_body


def check_thread_endpoint(url: str, thread_id: str, posted: list[dict]) -> None:
    rule("4b. GET /threads/{thread_id} (free — reads Postgres, calls no model)")

    response = httpx.get(f"{url}/threads/{thread_id}", timeout=30.0)
    if not verdict(response.status_code == 200, f"200 (got {response.status_code})"):
        return
    body = response.json()
    roles = [m["role"] for m in body["messages"]]
    print(f"    {len(roles)} messages: {', '.join(roles[:10])}"
          + (" …" if len(roles) > 10 else ""))

    verdict(roles.count("human") >= 2, "both turns are in the stored thread")
    # The write side and the read side must agree, which is the one thing
    # neither endpoint can establish alone — `posted` is what the last POST
    # returned, read back here from Postgres by a different code path.
    verdict(body["citations"] == posted,
            f"its citation list matches what POST /query returned "
            f"({len(body['citations'])} vs {len(posted)})")

    missing = httpx.get(f"{url}/threads/{uuid4()}", timeout=30.0)
    verdict(missing.status_code == 404, "an unknown thread is 404, not an empty 200")


# ---------------------------------------------------------------------------
# 5 · the cold concurrent burst
# ---------------------------------------------------------------------------


def check_cold_burst(concurrency: int, quiet: bool) -> None:
    rule(f"5. {concurrency} concurrent questions against a cold process, plus one over the cap")

    process, url = start_server("C", API_WARM_EMBEDDINGS="false")
    try:
        pairs = (BURST_PAIRS * 4)[:concurrency]
        questions = [BURST_QUESTION.format(a=a, b=b) for a, b in pairs]

        def run(question: str, pair: tuple[str, str]) -> dict:
            started = time.perf_counter()
            try:
                response = post_query(url, question)
                body = response.json() if response.status_code == 200 else {}
                status = response.status_code
            except Exception as exc:  # noqa: BLE001 - reporting, not handling
                body, status = {}, f"{type(exc).__name__}: {exc}"
            # The pair travels with its own result, so a request that fails
            # cannot shift every later answer onto the wrong companies and turn
            # the cross-talk check into noise.
            return {"question": question, "pair": pair, "status": status,
                    "body": body, "start": started, "end": time.perf_counter()}

        with ThreadPoolExecutor(max_workers=concurrency + 1) as pool:
            futures = [pool.submit(run, q, p) for q, p in zip(questions, pairs, strict=True)]
            # Late enough that the cap is genuinely full, early enough that the
            # first runs are nowhere near finishing.
            time.sleep(3)
            overflow_future = pool.submit(
                lambda: httpx.post(f"{url}/query",
                                   json={"question": "What is Apple's stock price?"},
                                   timeout=30.0))
            results = [f.result() for f in futures]
            overflow_response = overflow_future.result()

        # --- survived ---
        alive = process.poll() is None
        if not alive:
            print(f"    process exited with returncode {process.returncode}")
            print(indent((process.stdout.read() or "")[-2000:]))
        verdict(alive, "the interpreter survived concurrent cold parallel embedding")
        if alive:
            probe = httpx.get(f"{url}/health", timeout=10.0)
            verdict(probe.status_code == 200, "and still serves /health afterwards")

        served = [r for r in results if r["status"] == 200]
        verdict(len(served) == concurrency,
                f"all {concurrency} requests were served "
                f"({[r['status'] for r in results]})")

        # --- overlapped ---
        base = min(r["start"] for r in results)
        print("\n    request intervals (seconds from the first start):")
        for r in results:
            print(f"      {r['start'] - base:6.1f} → {r['end'] - base:6.1f}  "
                  f"{r['question'][:56]}")
        peak = max(
            sum(1 for r in results if r["start"] <= t < r["end"])
            for t in (r["start"] for r in results)
        )
        wall = max(r["end"] for r in results) - base
        serial = sum(r["end"] - r["start"] for r in results)
        print(f"    peak overlap {peak}, wall {wall:.1f}s vs {serial:.1f}s if serialised")
        verdict(peak >= 2, f"requests genuinely overlapped (peak {peak})")
        verdict(wall < 0.8 * serial,
                f"and the server did not serialise them ({wall:.1f}s < "
                f"{0.8 * serial:.1f}s)")

        # --- no cross-talk ---
        threads = {r["body"].get("thread_id") for r in served}
        verdict(len(threads) == len(served), "every request got its own thread")
        leaked = []
        for r in served:
            a, b = r["pair"]
            allowed = {BURST_TICKERS[a], BURST_TICKERS[b]}
            for c in r["body"].get("citations", []):
                if c["type"] != "filing":
                    continue
                found = {t for t in BURST_TICKERS.values() if t in c["label"].upper()}
                if found - allowed:
                    leaked.append(f"{c['label']} in the {a}/{b} thread")
        verdict(not leaked,
                "no thread's filing citations mention a company it never asked about"
                + (f" ({leaked[:3]})" if leaked else ""))

        # --- the cap engaged ---
        # Only meaningful when the burst actually fills it. A smaller --burst
        # says so rather than failing a cap that was never reached.
        if concurrency >= api.MAX_CONCURRENT_RUNS:
            verdict(overflow_response.status_code == 503,
                    f"the request over the cap is 503, not queued "
                    f"(got {overflow_response.status_code})")
            verdict("Retry-After" in overflow_response.headers,
                    "and carries Retry-After, so the client knows what to do")
        else:
            note(f"cap not exercised: {concurrency} < "
                 f"MAX_CONCURRENT_RUNS={api.MAX_CONCURRENT_RUNS}")

        if not quiet:
            print("\n    ↳ the human reading: does each answer discuss only its own two "
                  "companies?")
            for r in served:
                a, b = r["pair"]
                print(f"\n    ── {a} / {b} ──\n{indent(r['body'].get('answer', '')[:700])}")
    finally:
        stop_server(process)


# ---------------------------------------------------------------------------


def preflight() -> None:
    """Fail on a setup mistake before anything is spent or started."""
    rule(f"model: {settings.anthropic_model}")
    missing = [
        name for name, attr in (("ANTHROPIC_API_KEY", "anthropic_api_key"),
                                ("FINNHUB_API_KEY", "finnhub_api_key"),
                                ("TAVILY_API_KEY", "tavily_api_key"),
                                ("EDGAR_USER_AGENT", "edgar_user_agent"))
        if not getattr(settings, attr).strip()
    ]
    if missing:
        print(f"\n  missing from .env: {', '.join(missing)}")
        print("  (EDGAR_USER_AGENT is needed by this script, not by the server — "
              "case 2b fetches\n   filing URLs through app/sec_http.py.)")
        raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", metavar="URL",
                    help="use a server you started; skips the two cases that need "
                         "their own process")
    ap.add_argument("--burst", type=int, default=4,
                    help="concurrent requests in case 5 (default 4)")
    ap.add_argument("--skip-burst", action="store_true",
                    help="cases 1-4 only — cheaper to re-run")
    ap.add_argument("--quiet", action="store_true", help="verdicts only, no answer text")
    args = ap.parse_args()

    preflight()

    process = None
    try:
        if args.server:
            url = args.server
            print(f"  using {url}; cases 1b and 5 need their own process and are skipped")
        else:
            rule("starting servers")
            process, url = start_server("A")

        check_health(url)
        if not args.server:
            check_health_negative_control()

        body = check_single_query(url, args.quiet)
        if body:
            check_citation_urls(body["citations"])
            check_mixed_sources(url, args.quiet)
            resumed = check_multi_turn(url, body["thread_id"], args.quiet)
            # After the follow-up, so the stored thread holds both turns, and
            # compared against what that POST returned — the read side and the
            # write side reaching the same list through different code.
            if resumed:
                check_thread_endpoint(url, body["thread_id"], resumed["citations"])

        if not args.skip_burst and not args.server:
            check_cold_burst(args.burst, args.quiet)

        summary(
            on_failure="Answer-quality and tool-choice failures belong to Module "
                       "C, not to this layer.\n  Run scripts/check_agent.py "
                       "before suspecting app/api.py.",
            on_success="now read the answers themselves, and confirm the burst "
                       "answers stayed\n  inside their own two companies.",
        )
    finally:
        if process is not None:
            stop_server(process)
        finnhub.close_client()
        close_client()
        close_pool()

    raise SystemExit(exit_code())


if __name__ == "__main__":
    main()
