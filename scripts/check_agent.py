"""C2/C3 verification: the ReAct graph, judged on tool choice before answer quality.

    python scripts/check_agent.py                    # the benchmark set
    python scripts/check_agent.py --only 3 4         # just the news/web pair
    python scripts/check_agent.py --guard            # C4a's context guard alone
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

C3 adds the token counts. Prompt caching changes no answer and raises no
error when it stops working — it only changes the bill — so the per-turn
`in / out · cache read / created` line under each question, and the summary
check that reads it, are the only place the breakpoint is observable at all.
The retry policy is covered offline in tests/test_agent.py, since making a
real provider return 529 on demand is not something a gate can arrange.

C4a is `--guard`, a separate mode rather than a seventh benchmark question.
Its question is deliberately expensive — six k=12 searches, ~150k characters
of filing text — and the benchmark set should stay cheap enough to re-run
freely. It is also the one check here whose evidence is arithmetic rather than
judgement: history only ever grows, so a prompt that gets *smaller* between
turns cannot happen without the guard.
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

# C3 · The prefix length below which Anthropic ignores a cache breakpoint.
#
# **Measured on 2026-08-18, not read from the docs**, which give 2048 for the
# Haiku family. `claude-haiku-4-5-20251001` did not cache a 4,007-token prefix
# and did cache a 4,569-token one — a 4096 floor, double the published figure.
# Nothing is raised either way: an ignored breakpoint returns an ordinary
# response with zeroes in the cache columns, which is exactly what the numbers
# below exist to make visible.
CACHE_MINIMUM_TOKENS = 4096 if "haiku" in settings.anthropic_model else 1024

# C4a · The question the context guard exists for, and it is not subtle: three
# companies × two fiscal years at the maximum k, which is ~150k characters of
# filing text against a ~64k budget. The instruction to retrieve 12 passages is
# in the question on purpose — the guard is what makes a heavy question
# affordable, so the gate has to ask a heavy one rather than hope for it.
GUARD_QUESTION = (
    "Compare the risk factors Apple, Microsoft and Meta each disclosed in "
    "their two most recent 10-K filings, and say what changed year over year "
    "for each company. Retrieve 12 passages per search so the comparison is "
    "thorough."
)

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

    print_usage(state)

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


def print_usage(state: dict) -> None:
    """Per-turn tokens. Read the cache columns, not the totals."""
    rows = agent.token_usage(state.get("messages", []))
    if not rows:
        return
    print("\n    tokens per turn (in / out · cache read / created):")
    for i, row in enumerate(rows, start=1):
        print(f"      {i}. {row['input']:>6} / {row['output']:<6}"
              f" · {row['cache_read']:>6} / {row['cache_creation']:>6}")


def check_prompt_cache(state: dict) -> None:
    """C3's only observable.

    A prefix that silently stops matching produces identical answers at a
    higher price, so nothing above this line would notice. Turn 1 writes the
    system + tool-schema prefix to cache; every later turn should read it back
    and create nothing, because `agent_node` rebuilds a byte-identical prefix
    each call. A later turn that re-creates the cache means something upstream
    of the messages is varying between turns — that is the bug.
    """
    rule("prompt cache (C3)")
    rows = agent.token_usage(state.get("messages", []))
    if len(rows) < 2:
        print("  (needs a run with at least two model turns — none of the "
              "questions run made a tool call)")
        return

    first, rest = rows[0], rows[1:]

    # A prefix under the model's floor is a property of the model, not a bug in
    # this code, so it is reported rather than failed. Said plainly because the
    # tempting reading of three zeroes is "caching is broken" — it is not; the
    # breakpoint is correct and starts paying the moment the prefix clears the
    # floor or the model changes to one with a lower one.
    turn_one_prompt = first["input"] + first["cache_read"] + first["cache_creation"]
    touched_cache = any(row["cache_read"] or row["cache_creation"] for row in rows)
    if not touched_cache and turn_one_prompt < CACHE_MINIMUM_TOKENS:
        print(f"  [-- ] inert on this model: the ~{turn_one_prompt}-token prompt is under "
              f"{settings.anthropic_model}'s\n         {CACHE_MINIMUM_TOKENS}-token minimum "
              "cacheable prefix, so Anthropic accepts the breakpoint\n         and ignores "
              "it. Not counted as a failure — see CLAUDE.md.")
        return

    # Written *or* read, not written: the ephemeral cache outlives the process
    # by five minutes, so a re-run inside that window legitimately opens on a
    # hit. Requiring a write here would fail the gate for running it twice.
    wrote, read = first["cache_creation"], first["cache_read"]
    if wrote:
        how = f"wrote {wrote} tokens"
    elif read:
        how = f"read {read} tokens — still warm from an earlier run"
    else:
        how = "wrote and read nothing"
    if not verdict(bool(wrote or read), f"turn 1 reached the cache ({how})"):
        print(f"        ↳ and the ~{turn_one_prompt}-token prompt clears "
              f"{CACHE_MINIMUM_TOKENS}, so length is not the reason this time.\n"
              "          Check that the system block still reaches the provider as a "
              "content-block\n          list carrying cache_control — a bare string "
              "caches nothing and raises nothing.")

    turns = "turn 2" if len(rest) == 1 else f"turns 2-{len(rows)}"
    verdict(
        all(row["cache_read"] > 0 for row in rest),
        f"{turns} read the prefix from cache "
        f"({', '.join(str(row['cache_read']) for row in rest)} tokens)",
    )

    recreated = [i for i, row in enumerate(rest, start=2) if row["cache_creation"] > 0]
    verdict(
        not recreated,
        "no later turn re-created the cache"
        + (f" (turn(s) {recreated} did — something upstream of the messages "
           "is varying between turns)" if recreated else ""),
    )


def prompt_tokens(row: dict) -> int:
    """What a turn actually sent. `input` alone omits the part read from cache."""
    return row["input"] + row["cache_read"] + row["cache_creation"]


def check_context_guard(state: dict) -> None:
    """C4a's only observable, and the arithmetic is what makes it one.

    A conversation only ever grows, so without the guard every turn's prompt is
    at least as large as the last. **A prompt that shrinks between turns is
    therefore proof the guard fired** — no estimate, no token-counting
    approximation, nothing to argue with. The rest is confirming it took only
    what it was supposed to: tool bodies, out of the payload, never out of
    state, and never a citation.
    """
    rule("context-window guard (C4a)")
    messages = state.get("messages", [])
    rows = agent.token_usage(messages)

    stored = agent.tool_result_chars(messages)
    # What the *next* turn would send — i.e. exactly what the last turn sent,
    # since the guard recomputes from the full history every time.
    trimmed = agent.trim_tool_results(messages)
    elided = sum(
        1 for m in trimmed
        if getattr(m, "type", "") == "tool" and "omitted to conserve context" in str(m.content)
    )

    print(f"  tool-result text in state: {stored:,} chars "
          f"(~{stored // agent.CHARS_PER_TOKEN:,} tokens) against a "
          f"{agent.TOOL_RESULT_BUDGET_TOKENS:,}-token budget")
    print(f"  prompt per turn (tokens):  "
          f"{' → '.join(f'{prompt_tokens(row):,}' for row in rows)}")

    if not elided:
        print("  [-- ] not exercised: the run never crossed the budget, so the guard "
              "had nothing\n         to do. That is the correct behaviour for a light "
              "question — but it means\n         this run proves nothing about the "
              "guard. Ask something heavier.")
        return

    print(f"  elided {elided} old tool result(s) from the last turn's payload")

    peak = max(prompt_tokens(row) for row in rows)
    shrank = [
        i for i in range(1, len(rows))
        if prompt_tokens(rows[i]) < prompt_tokens(rows[i - 1])
    ]
    verdict(
        bool(shrank),
        f"the prompt shrank between turns (at turn {shrank[0] + 1 if shrank else '—'}), "
        f"which only trimming can do — peak {peak:,} tokens",
    )

    # The guard rewrites the invoke payload, not the conversation. If a stub
    # ever reaches state, C4b's checkpointer persists it and the elision
    # becomes permanent rather than per-turn.
    leaked = [
        m for m in messages
        if getattr(m, "type", "") == "tool" and "omitted to conserve context" in str(m.content)
    ]
    label = "state still holds the untrimmed conversation"
    verdict(not leaked, label + (f" ({len(leaked)} stub(s) leaked into it)" if leaked else ""))

    # Citations live outside the messages, which is the whole reason trimming
    # is safe: eliding the passage must not cost the source it came from.
    filings = [c for c in state.get("citations", []) if c.type == "filing"]
    years = sorted({year for c in filings for year in re.findall(r"\b20\d{2}\b", c.label)})
    verdict(bool(filings), f"filing citations survived the trim ({len(filings)} of them)")
    print(f"    filing years cited: {', '.join(years) or 'none'} — the answer "
          "should still compare both")


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
    ap.add_argument("--guard", action="store_true",
                    help="run C4a's heavy question and check the context guard")
    ap.add_argument("--ask", help="run one ad-hoc question instead of the set")
    ap.add_argument("--quiet", action="store_true", help="verdicts only, no answer text")
    args = ap.parse_args()

    configure()

    last_state: dict = {}
    cache_state: dict = {}
    try:
        if args.guard:
            # One question, and the summary below still runs — the guard's
            # verdicts are counted like any other, and the answer is printed
            # because "did it still compare both years" is a human reading.
            rule(f"C4a · {GUARD_QUESTION}")
            state = run_streaming(GUARD_QUESTION)
            answer = agent.final_text(state["messages"])
            print_usage(state)
            check_context_guard(state)
            check_citation_urls(state)
            if not args.quiet:
                print(f"\n{indent(answer)}")

        elif args.ask:
            rule(args.ask)
            state = run_streaming(args.ask)
            called = {name for name, _ in agent.tool_calls_made(state["messages"])}
            print(f"\n    tools called: {', '.join(sorted(called)) or 'none'}")
            print_usage(state)
            print(f"\n{indent(agent.final_text(state['messages']))}")
            for citation in state.get("citations", []):
                print(f"      [{citation.type}] {citation.label}\n          {citation.source_url}")
            return

        else:
            for i, (question, required, forbidden) in enumerate(BENCHMARK, start=1):
                if args.only and i not in args.only:
                    continue
                state = check_question(i, question, required, forbidden, args.quiet)
                # Keep the first run that produced filing citations, so the URL
                # check below costs nothing extra.
                if not last_state and any(c.type == "filing" for c in state.get("citations", [])):
                    last_state = state
                # And the first multi-turn run, which is the only kind that can
                # show a cache read: turn 1 can only ever create.
                if not cache_state and len(agent.token_usage(state["messages"])) > 1:
                    cache_state = state

            check_citation_urls(last_state)
            check_prompt_cache(cache_state)

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
