"""C2 verification: the ReAct graph, judged on tool choice before answer quality.

    python scripts/check_agent.py                    # the benchmark set
    python scripts/check_agent.py --only 3 4         # just the news/web pair
    python scripts/check_agent.py --ask "..."        # one ad-hoc question
    python scripts/check_agent.py --quiet            # verdicts only, no answers

**Which tools were called is the primary reading.** With five tools and three
that plausibly answer "what's going on with Apple", tool choice is the thing
most likely to be wrong, and it is wrong in a way a good-sounding answer hides:
a fluent response assembled from the wrong source still reads as correct.

Questions 3 and 4 are the pair that matters. They are deliberately similar —
both about Apple, both about opinion and events — and separated only by whether
the question is scoped to one ticker. If they route wrongly, **fix the
docstrings in app/tools.py, not this graph.** That is where tool selection
lives.

Needs Postgres up, FINNHUB_API_KEY, TAVILY_API_KEY, and ANTHROPIC_API_KEY —
this is the first step with a model in the loop, and it costs money to run.
(Roughly $0.10 for the whole set on claude-haiku-4-5: a ~1.8k-token fixed
prefix per turn, plus ~3.2k per k=6 search result.)

There is one provider on purpose. A free-tier smoke provider lived here
briefly so the wiring could be exercised before there was a paid key; it was
removed once it had done that, because its token ceiling could not seat a
multi-tool question and C3's `cache_control` prefix would have diverged the
path it tested from the one this system ships. Every verdict below is
therefore a real verdict — nothing is advisory, nothing is uncounted.
"""

import argparse
import logging
import re

from app import agent, finnhub
from app.config import settings
from app.db import close_pool
from app.sec_http import close_client, sec_get

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

FAILURES: list[str] = []

# (question, tools that must be called, tools that must NOT be).
#
# Taken from the plan verbatim, because they encode decisions rather than
# coverage: 1-2 are baseline routing, 3-4 are the news/web boundary the B gates
# showed is genuinely ambiguous, 5 is multi-source, and 6 is the corpus gap —
# C1's headline behaviour, seen for the first time through a model that has to
# choose to report it.
BENCHMARK = [
    (
        "What are Apple's key risk factors?",
        {"search_filings"},
        {"get_company_news", "web_search"},
    ),
    (
        "What's AAPL trading at right now?",
        {"get_quote"},
        {"search_filings"},
    ),
    (
        "What's the latest news on Apple?",
        {"get_company_news"},
        {"web_search"},
    ),
    (
        "What do analysts think about Apple's AI strategy compared to Google's?",
        {"web_search"},
        {"get_company_news"},
    ),
    (
        "How does Apple's stated revenue outlook compare to its current valuation?",
        {"search_filings", "get_basic_financials"},
        set(),
    ),
    (
        "What are Nvidia's key risk factors according to its SEC filings?",
        {"search_filings"},
        set(),
    ),
]

# Question 6 only: the answer has to admit the gap rather than paper over it.
GAP_MARKERS = ("not in the", "not been ingested", "not available", "no filings",
               "unavailable", "not in our", "don't have", "do not have", "cannot")


def rule(title: str) -> None:
    print(f"\n{'━' * 78}\n{title}\n{'━' * 78}")


def verdict(ok: bool, label: str) -> bool:
    """Record one check."""
    print(f"  [{'ok ' if ok else 'BAD'}] {label}")
    if not ok:
        FAILURES.append(label)
    return ok


def run_streaming(question: str) -> dict:
    """Run one question, printing each tool call as the graph makes it.

    Streaming rather than a plain invoke because the interesting event is the
    tool call, and it happens well before the answer exists.
    """
    state: dict = {"messages": [], "citations": []}
    stream = agent.get_graph().stream(
        {"messages": [("user", question)], "citations": []},
        config={"recursion_limit": agent.RECURSION_LIMIT},
        stream_mode="values",
    )
    for state in stream:
        last = state["messages"][-1] if state["messages"] else None
        for call in getattr(last, "tool_calls", None) or []:
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items() if v is not None)
            print(f"    → {call['name']}({args[:110]})")
    return state


def check_question(index: int, question: str, required: set, forbidden: set,
                   quiet: bool) -> dict:
    rule(f"{index}. {question}")

    state = run_streaming(question)
    called = {name for name, _ in agent.tool_calls_made(state["messages"])}
    citations = state.get("citations", [])

    print(f"\n    tools called: {', '.join(sorted(called)) or 'none'}")

    # Both routing checks are judgments the model made, not properties of the
    # graph. They are the primary reading of this script.
    missing = required - called
    verdict(not missing, f"called {', '.join(sorted(required))}"
                         + (f" (missing {', '.join(sorted(missing))})" if missing else ""))
    if forbidden:
        wrong = forbidden & called
        verdict(
            not wrong,
            f"avoided {', '.join(sorted(forbidden))}"
            + (f" (called {', '.join(sorted(wrong))} — fix the docstring)" if wrong else ""),
        )

    answer = agent.final_text(state["messages"])
    verdict(bool(answer.strip()), "produced a non-empty answer")

    by_type: dict[str, int] = {}
    for citation in citations:
        by_type[citation.type] = by_type.get(citation.type, 0) + 1
    print(f"    citations: {by_type or 'none'}")

    # A citation whose type doesn't match the tools that ran means the
    # collection wiring is crossing sources, which no answer would reveal.
    if "filing" in by_type:
        verdict("search_filings" in called, "filing citations only where filings were searched")
    if "news" in by_type:
        verdict("get_company_news" in called, "news citations only where news was fetched")
    if "web" in by_type:
        verdict("web_search" in called, "web citations only where the web was searched")

    if index == 6:
        lowered = answer.lower()
        verdict(
            any(marker in lowered for marker in GAP_MARKERS),
            "states plainly that Nvidia's filings are unavailable (corpus gap)",
        )

    # C4c's input: whether the model writes bracketed markers at all. Each
    # tool's [n] restart at 1 and index that call's artifact list, while
    # merge_citations produces one flat list — so a reproduced [2] can point
    # at the wrong source. Never a failure, always worth seeing.
    markers = sorted(set(re.findall(r"\[(\d+)\]", answer)))
    if markers:
        print(f"    [n] markers in answer: {', '.join(markers)}"
              f"  (citation list has {len(citations)}) ← C4c input")

    if not quiet:
        print(f"\n{indent(answer)}")
        if citations:
            print("\n    sources:")
            for i, citation in enumerate(citations, start=1):
                print(f"      {i:>2}. [{citation.type}] {citation.label[:88]}")
                print(f"          {citation.source_url}")

    return state


def indent(text: str, prefix: str = "    │ ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def check_citation_urls(state: dict) -> None:
    """Fetch the filing citations from one run — a citation nobody can open is not one.

    Only the EDGAR ones: those go through the project's own SEC client, which
    carries the required User-Agent and rate limit. News and web URLs are the
    providers' to keep working, and hammering them here would prove nothing
    about this code.
    """
    rule("filing citation URLs resolve")
    filings = [c for c in state.get("citations", []) if c.type == "filing"]
    if not filings:
        print("  (no filing citations in that run)")
        return
    for citation in {c.source_url: c for c in filings}.values():
        try:
            ok = sec_get(citation.source_url, accept="text/html").status_code == 200
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            ok = False
            print(f"        {type(exc).__name__}: {exc}")
        verdict(ok, citation.label[:80])


def configure() -> None:
    """Build the model before the first question runs.

    Eagerly, rather than on the first question: a missing key is a setup
    mistake, and it should surface as one line before anything runs instead of
    a graph traceback three questions in.
    """
    rule(f"model: {settings.anthropic_model}")
    try:
        agent.get_llm()
    except RuntimeError as exc:
        print(f"\n  {exc}")
        raise SystemExit(2) from None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", help="benchmark numbers to run (1-6)")
    ap.add_argument("--ask", help="run one ad-hoc question instead of the set")
    ap.add_argument("--quiet", action="store_true", help="verdicts only, no answer text")
    args = ap.parse_args()

    configure()

    last_state: dict = {}
    try:
        if args.ask:
            rule(args.ask)
            state = run_streaming(args.ask)
            called = {name for name, _ in agent.tool_calls_made(state["messages"])}
            print(f"\n    tools called: {', '.join(sorted(called)) or 'none'}")
            print(f"\n{indent(agent.final_text(state['messages']))}")
            for citation in state.get("citations", []):
                print(f"      [{citation.type}] {citation.label}\n          {citation.source_url}")
            return

        for i, (question, required, forbidden) in enumerate(BENCHMARK, start=1):
            if args.only and i not in args.only:
                continue
            state = check_question(i, question, required, forbidden, args.quiet)
            # Keep the first run that produced filing citations, so the URL
            # check below costs nothing extra.
            if not last_state and any(c.type == "filing" for c in state.get("citations", [])):
                last_state = state

        check_citation_urls(last_state)

        rule("summary")
        if FAILURES:
            print(f"  {len(FAILURES)} check(s) FAILED:")
            for failure in FAILURES:
                print(f"    - {failure}")
            print("\n  Tool-choice failures are docstring bugs. Fix app/tools.py, "
                  "not app/agent.py.")
        else:
            print("  all checks passed — now read the answers themselves: are the "
                  "filing claims\n  actually supported by the retrieved passages, and "
                  "is filing data dated?")
    finally:
        finnhub.close_client()
        close_client()
        close_pool()

    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
