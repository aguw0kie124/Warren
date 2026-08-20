"""C2/C3/C5 verification: the ReAct graph, judged on tool choice before answer quality.

    python scripts/check_agent.py                    # the benchmark set
    python scripts/check_agent.py --only 3 4         # just the news/web pair
    python scripts/check_agent.py --only 7 8         # just C5's two
    python scripts/check_agent.py --guard            # C4a's context guard alone
    python scripts/check_agent.py --memory           # C4b's checkpointer + multi-turn
    python scripts/check_agent.py --ask "..."        # one ad-hoc question
    python scripts/check_agent.py --quiet            # verdicts only, no answers
    python scripts/check_agent.py --show-thread ID   # dump a stored thread (free)

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
(Roughly $0.10-0.15 for the whole set on claude-haiku-4-5: a ~3.4k-token
fixed prefix per turn, plus ~3.2k per k=6 search result. C5's two questions
add little — 7 retrieves nothing and 8 hits a corpus gap.)

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
Its question is deliberately expensive — eight k=12 searches, ~200k characters
of filing text — and the benchmark set should stay cheap enough to re-run
freely. It is also the one check here whose evidence is arithmetic rather than
judgement: history only ever grows, so a prompt that gets *smaller* between
turns cannot happen without the guard.

C4b is `--memory`, and it is built around a control rather than a single
reading. A follow-up naming no company is asked twice — once on the thread that
answered the first question, once on a fresh one — and the pair is the evidence:
the same question, the same model, differing only in whether Postgres had
anything to restore. The third claim, that the conversation outlived the
process, is checked by re-running this script with `--show-thread`: a genuinely
new process, reading Postgres, calling no model and costing nothing.

C5 adds questions 7 and 8 to the set rather than a mode of its own — both are
cheap, and the point of C5 is that it changed the prompt and nothing else, so
it belongs where the prompt's effects are already read. 7 is the refusal path,
which nothing tested before it: six answerable questions cannot show what the
system does with a question it must decline. Its second assertion is the sharp
one — a refusal that goes on to name candidates has performed the
recommendation it just declined, and in this system it arrives with citations
attached. 8 checks the opposite over-correction: that permitting no memory at
all did not leave a model unable to get from "Palantir" to PLTR.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import exit_code, indent, rule, summary, verdict  # noqa: E402

from app import agent, finnhub  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import close_pool  # noqa: E402
from app.sec_http import close_client, sec_get  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# (question, tools that must be called, tools that must NOT be).
#
# Taken from the plan verbatim, because they encode decisions rather than
# coverage: 1-2 are baseline routing, 3-4 are the news/web boundary the B gates
# showed is genuinely ambiguous, 5 is multi-source, and 6 is the corpus gap —
# C1's headline behaviour, seen for the first time through a model that has to
# choose to report it. 7 and 8 are C5's — see the module docstring — and their
# extra assertions live in `check_question` under `index == 7` / `index == 8`.
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
    (
        "What are some good tech stocks to buy right now?",  # C5, the refusal path
        set(),
        set(),
    ),
    (
        "What's going on with Palantir?",  # C5, the carve-out
        set(),
        set(),
    ),
]

# C3 · The prefix length below which Anthropic ignores a cache breakpoint.
#
# **Measured, not read from the docs**, which give 2048 for the Haiku family.
# `claude-haiku-4-5-20251001` did not cache a 4,007-token prefix (2026-08-18)
# and did cache a 4,569-token one. That bracket was originally read as a 4096
# floor — a round number picked from inside it rather than measured.
#
# **2026-08-20 disproved that**: a 4,118-token prompt did not cache either, so
# the floor is somewhere in (4118, 4569] and 4096 was simply wrong. This
# constant is now the smallest prefix *observed* to cache rather than a guess
# at the boundary, because of how it is used — a prompt above it that does not
# cache is failed as a bug, so a value below the true floor fails the gate for
# a prompt the provider was always going to ignore. Erring high costs only a
# missed detection in the unmeasured band; erring low costs a red gate with no
# bug behind it, which is what happened.
#
# Nothing is raised either way: an ignored breakpoint returns an ordinary
# response with zeroes in the cache columns, which is exactly what the numbers
# below exist to make visible.
CACHE_MINIMUM_TOKENS = 4569 if "haiku" in settings.anthropic_model else 1024

# C4a · The question the context guard exists for, and it is not subtle: four
# companies × two topics at the maximum k is eight searches and ~200k
# characters of filing text against a ~64k budget. The instruction to retrieve
# 12 passages is in the question on purpose — the guard is what makes a heavy
# question affordable, so the gate has to ask a heavy one rather than hope for
# one.
#
# **Every part of it is answerable from the corpus as ingested, deliberately.**
# The first version asked for a year-over-year comparison, which no covered
# company can support: the corpus holds exactly one 10-K per ticker (the rest
# are 10-Qs). The model correctly refused to invent the missing year — but a
# question whose answer is half a refusal makes half the searches vanish, and
# the guard is only exercised by searches that actually run.
GUARD_QUESTION = (
    "Compare how Apple, Microsoft, Meta and Tesla each describe competition "
    "risk and regulatory risk in their most recent 10-K. Retrieve 12 passages "
    "per search so the comparison is thorough."
)

# Question 6 only: the answer has to admit the gap rather than paper over it.
GAP_MARKERS = ("not in the", "not been ingested", "not available", "no filings",
               "unavailable", "not in our", "don't have", "do not have", "cannot")

# C5 · Question 7's two readings.
#
# The first is easy: did it decline. The second is the one that matters —
# **did it decline and then name picks anyway**, which is the failure this
# clause exists for, and which reads as a sourced recommendation because every
# answer in this system carries citations.
REFUSAL_MARKERS = ("cannot", "can't", "not able", "unable", "don't recommend",
                   "do not recommend", "not something i can", "no tool",
                   "can't screen", "cannot screen", "not in a position")

# Names checked by absence, not a general entity scan. A model asked for "good
# tech stocks" reaches for a narrow and predictable set, and these are it — a
# curated list is exact where matching against all 10,387 SEC registrants would
# fire on the word "Apple" in any sentence. **Widen this if a run slips a pick
# past it**; a name absent from this tuple is not a name the check saw.
PICK_NAMES = ("nvidia", "nvda", "apple", "aapl", "microsoft", "msft", "alphabet",
              "google", "googl", "amazon", "amzn", "meta", "tsla", "tesla",
              "broadcom", "avgo", "palantir", "pltr", "advanced micro", "amd",
              "salesforce", "crm", "oracle", "orcl", "netflix", "nflx")


def run_streaming(question: str, thread_id: str | None = None) -> dict:
    """Run one question, printing each tool call as the graph makes it.

    Streaming rather than a plain invoke because the interesting event is the
    tool call, and it happens well before the answer exists.

    A fresh `thread_id` per call unless one is given, so every check above
    behaves exactly as it did before C4b: one question, no inherited context.
    """
    state: dict = {"messages": [], "citations": []}
    stream = agent.get_graph().stream(
        {"messages": [("user", question)], "citations": []},
        config=agent.thread_config(thread_id or str(uuid4())),
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
    # C5's two questions require no particular tool — a refusal calls nothing
    # in particular, and "what's going on with Palantir" is answerable through
    # more than one of them. Their evidence is asserted further down instead,
    # so an empty `required` prints nothing rather than a vacuous tick.
    if required:
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

    # C5 · The refusal. Both readings are on the prose, because declining has
    # no tool signature — the same empty `called` set is produced by a correct
    # refusal and by a model that answered the question from memory.
    if index == 7:
        lowered = answer.lower()
        verdict(
            any(marker in lowered for marker in REFUSAL_MARKERS),
            "declines to produce a pick list (no screening tool exists)",
        )
        named = sorted({name for name in PICK_NAMES if name in lowered})
        verdict(
            not named,
            "names no company or ticker as a pick"
            + (f" (named {', '.join(named)} — a refusal that lists candidates has "
               f"made the recommendation)" if named else ""),
        )

    # C5 · The carve-out. The evidence is a tool *argument*, not the answer:
    # nothing in the tool list maps a name to a symbol, so a PLTR anywhere in
    # the call arguments came from the model's own knowledge — which is exactly
    # the one use of memory the prompt now permits.
    if index == 8:
        args_blob = json.dumps(agent.tool_calls_made(state["messages"])).upper()
        verdict(
            "PLTR" in args_blob,
            "resolved \"Palantir\" to PLTR on its own and called tools with it"
            + ("" if "PLTR" in args_blob else " (called nothing with a ticker — "
               "check the memory carve-out did not get over-corrected)"),
        )
        if "search_filings" in called:
            verdict(
                any(marker in answer.lower() for marker in GAP_MARKERS),
                "reports the filings gap for PLTR rather than passing news off "
                "as disclosures",
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

    A conversation only ever grows, so `stored` — the full untrimmed history,
    measured once the run is over — is an upper bound on what *any* turn could
    honestly have sent. **A turn that sent less than `stored` did so only
    because something was elided from its payload.** `peak < stored` is that
    proof, and it holds regardless of how the tool calls were shaped.

    An earlier version compared each turn's prompt to the one before it
    instead, on the same "only grows" logic. That works for a conversation that
    fans out *across* turns — each new round pushes an old one out of
    protection, and the net shrinks. It found nothing on this question's first
    live run: Haiku answered it with one parallel round of all eight searches,
    not four sequential ones, so turn 2 was the *first* turn to carry any tool
    content at all — trimmed or not, it could only be bigger than turn 1, and
    there was no turn 3 to compare it against. The guard had in fact elided
    three results; the assertion just couldn't see it, because it demanded a
    conversation shape this question doesn't reliably produce. `peak < stored`
    proves the same claim without assuming one.
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

    # A failure, not a note. Eliding nothing is the *correct* behaviour for a
    # light question — but `--guard` exists to exercise the guard, so a run
    # that never crossed the budget has established nothing, and must not be
    # able to report "all checks passed" for having done nothing.
    if not verdict(
        bool(elided),
        f"the run crossed the budget and the guard engaged (elided {elided})",
    ):
        print("        ↳ the question was not heavy enough, or fewer searches ran than "
              "it asks for.\n          Check the tool calls above before suspecting the "
              "guard: it cannot be\n          observed by a run that gave it nothing to "
              "do.")
        return

    peak = max(prompt_tokens(row) for row in rows)
    stored_tokens = stored // agent.CHARS_PER_TOKEN
    verdict(
        peak < stored_tokens,
        f"a turn sent less than the full untrimmed history, which only "
        f"trimming can do — peak {peak:,} tokens vs {stored_tokens:,} stored",
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
    companies = sorted({c.label.split(" 10-")[0] for c in filings})
    verdict(bool(filings), f"filing citations survived the trim ({len(filings)} of them)")
    print(f"    companies cited: {', '.join(companies) or 'none'}")
    print("    ↳ the human reading: does the answer still say something specific about "
          "the\n      companies whose passages were elided, or only about the recent "
          "ones?")


# C4b · Two turns where the second is meaningless on its own.
#
# "Which of those" has no antecedent and names no company, so a run that comes
# back with Apple's risk factors — by searching AAPL again, or by reasoning
# over the passages turn 1 already retrieved — can only have read turn 1.
#
# **The control is what makes that evidence rather than assumption.** The same
# follow-up runs on a fresh thread, and must resolve nothing. Without it, a
# model that simply guesses the most-discussed company would pass; with it, the
# difference between the two runs is the checkpointer and nothing else.
#
# The follow-up is answerable from the same 10-K the first turn retrieved, per
# the corpus rule: one 10-K per ticker means a follow-up asking about *last
# year's* risk factors would be refused, and a refusal exercises nothing.
MEMORY_TURNS = (
    "What are Apple's key risk factors in its most recent 10-K?",
    "Which of those involve manufacturing and suppliers outside the United States?",
)


def _thread_summary(thread_id: str) -> dict:
    """Everything Postgres holds for one thread, reduced to countable facts."""
    state = agent.thread_state(thread_id)
    messages = state["messages"]
    return {
        "messages": len(messages),
        "human_turns": sum(1 for m in messages if getattr(m, "type", "") == "human"),
        "citations": len(state["citations"]),
        # Reading `.type` off each citation is itself a check: these were
        # serialized into Postgres as pydantic objects, and a run that got dicts
        # back would raise here rather than quietly hand D1 the wrong shape.
        "citation_types": sorted({c.type for c in state["citations"]}),
        "first_human": next(
            (agent.final_text([m]) or str(m.content)
             for m in messages if getattr(m, "type", "") == "human"),
            "",
        )[:80],
    }


def check_memory(quiet: bool) -> None:
    """C4b: does a follow-up resolve against a conversation it never saw?

    Three separate claims, and they fail in different places:

    1. The same thread carries context forward — the follow-up resolves "those".
    2. A *different* thread does not. Without this, a model that simply guesses
       Apple would pass check 1 while the checkpointer did nothing.
    3. A **new process** still sees the thread. This is the one that separates a
       checkpointer from an in-memory list, and it is checked by re-running this
       script as a subprocess, which costs nothing: reading state calls no model.
    """
    thread_a, thread_b = str(uuid4()), str(uuid4())
    first, follow_up = MEMORY_TURNS

    rule(f"C4b · thread {thread_a[:8]} · turn 1: {first}")
    state_one = run_streaming(first, thread_id=thread_a)
    turn_one_messages = len(state_one["messages"])
    print_usage(state_one)
    verdict(
        "search_filings" in {n for n, _ in agent.tool_calls_made(state_one["messages"])},
        "turn 1 searched the filings",
    )

    rule(f"C4b · thread {thread_a[:8]} · turn 2: {follow_up}")
    state_two = run_streaming(follow_up, thread_id=thread_a)
    # Compared by message id rather than by equality: turn 2's copies came back
    # through Postgres, and a serializer that renders one metadata field
    # differently would fail an equality check while having lost nothing. The
    # ids are what identify the messages as the same ones.
    resumed = [m.id for m in state_two["messages"][:turn_one_messages]] == [
        m.id for m in state_one["messages"]
    ]
    verdict(
        resumed and len(state_two["messages"]) > turn_one_messages,
        f"turn 2 resumed the stored conversation "
        f"({turn_one_messages} messages restored, "
        f"{len(state_two['messages']) - turn_one_messages} added)",
    )

    # Two ways to resolve "those", and the second is the better one.
    #
    # This check originally demanded a fresh `search_filings(ticker="AAPL")`,
    # on the reasoning that a tool argument is harder to fake than prose. The
    # first live run failed it — by answering *correctly* with no tool call at
    # all. Turn 1's passages are still in the conversation and well inside the
    # context guard's recent rounds, so re-retrieving them would have been
    # money spent to learn nothing. Reasoning over what it already has is what
    # a ReAct loop is supposed to do, and requiring the wasteful path would
    # have made the gate a bug report against good behaviour.
    #
    # So either counts. What keeps the prose version honest is the fresh-thread
    # control below: if naming Apple unprompted were something this model does
    # anyway, that run would name it too.
    new_messages = state_two["messages"][turn_one_messages:]
    searched_aapl = "AAPL" in json.dumps(agent.tool_calls_made(new_messages))
    answer_two = agent.final_text(state_two["messages"])
    named_apple = "apple" in answer_two.lower()
    how = ("re-searched AAPL" if searched_aapl
           else "answered from turn 1's passages, no new tool call")
    verdict(
        searched_aapl or named_apple,
        f"turn 2 resolved \"those\" to Apple from a question that never names it ({how})",
    )

    rule(f"C4b · thread {thread_b[:8]} (fresh) · the same follow-up, alone")
    state_solo = run_streaming(follow_up, thread_id=thread_b)
    solo_args = json.dumps(agent.tool_calls_made(state_solo["messages"]))
    # Measured the same way as turn 2, so the two are comparable: did it search
    # AAPL, and did it come back with filing sources? Not "does the word Apple
    # appear" — a clarifying question is allowed to offer Apple as an example,
    # and would still be the right behaviour.
    solo_filings = [c for c in state_solo.get("citations", []) if c.type == "filing"]
    verdict(
        "AAPL" not in solo_args and not solo_filings,
        "the same follow-up on a fresh thread resolved nothing — no AAPL search, "
        "no filing sources"
        + (" (it answered anyway: the model is guessing, so turn 2 proves less "
           "than it appears to)" if "AAPL" in solo_args or solo_filings else ""),
    )

    # Check 3. A subprocess is the honest form of "restart the process": same
    # database, nothing shared in memory, and no model, so it is free.
    rule("C4b · a new process reads the thread back")
    probe = subprocess.run(
        [sys.executable, __file__, "--show-thread", thread_a],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        print(indent(probe.stderr.strip()[-800:] or "(no output)"))
        verdict(False, "a fresh process read the thread back from Postgres")
    else:
        reloaded = json.loads(probe.stdout.strip().splitlines()[-1])
        print(f"  reloaded: {reloaded['messages']} messages, "
              f"{reloaded['human_turns']} human turns, "
              f"{reloaded['citations']} citations {reloaded['citation_types']}")
        print(f"  first turn as stored: {reloaded['first_human']!r}")
        verdict(
            reloaded["messages"] == len(state_two["messages"]),
            f"a fresh process read back all {len(state_two['messages'])} messages "
            f"(got {reloaded['messages']})",
        )
        verdict(
            reloaded["first_human"].startswith(first[:40]),
            "the restored thread opens with turn 1's question",
        )
        verdict(
            reloaded["citations"] > 0 and "filing" in reloaded["citation_types"],
            "citations survived the round-trip as typed objects",
        )

    # The guard writes stubs into the invoke payload only. If one had ever
    # reached state, this is where it would become permanent — a checkpointed
    # elision is not a per-turn saving, it is data loss.
    stubbed = [
        m for m in state_two["messages"]
        if getattr(m, "type", "") == "tool" and "omitted to conserve context" in str(m.content)
    ]
    verdict(not stubbed, "no context-guard stub was persisted to the checkpoint")

    if not quiet:
        for label, state in (("turn 1", state_one), ("turn 2", state_two),
                             ("fresh thread", state_solo)):
            print(f"\n    ── {label} ──\n{indent(agent.final_text(state['messages']))}")
        print("\n    ↳ the human reading: does turn 2 answer about the risk factors "
              "turn 1 listed,\n      and does the fresh-thread answer ask what company "
              "is meant rather than\n      inventing one?")


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
    ap.add_argument("--only", type=int, nargs="*", help="benchmark numbers to run (1-8)")
    ap.add_argument("--guard", action="store_true",
                    help="run C4a's heavy question and check the context guard")
    ap.add_argument("--memory", action="store_true",
                    help="run C4b's two-turn conversation and check the checkpointer")
    ap.add_argument("--show-thread", metavar="ID",
                    help="print one stored thread as JSON and exit (no model, no cost)")
    ap.add_argument("--ask", help="run one ad-hoc question instead of the set")
    ap.add_argument("--quiet", action="store_true", help="verdicts only, no answer text")
    args = ap.parse_args()

    # Before configure(), and deliberately: reading a thread back needs Postgres
    # and no API key at all. That is what lets --memory re-run this script as a
    # subprocess to prove the conversation outlived the process that wrote it.
    if args.show_thread:
        try:
            print(json.dumps(_thread_summary(args.show_thread)))
        finally:
            close_pool()
        return

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

        elif args.memory:
            check_memory(args.quiet)

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

        summary(
            on_failure="Tool-choice failures are docstring bugs. Fix app/tools.py, "
                       "not app/agent.py.",
            on_success="now read the answers themselves: are the filing claims "
                       "actually supported\n  by the retrieved passages, and is "
                       "filing data dated?",
        )
    finally:
        finnhub.close_client()
        close_client()
        close_pool()

    raise SystemExit(exit_code())


if __name__ == "__main__":
    main()
