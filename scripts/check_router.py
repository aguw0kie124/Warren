"""E1 verification: the query router, judged on where it sends things.

    python scripts/check_router.py                 # the labelled set + answers
    python scripts/check_router.py --classify-only # labels only, no answers (cheap)
    python scripts/check_router.py --cost          # price the two graph shapes
    python scripts/check_router.py --quiet         # verdicts only, no answer text

**Classification is the primary reading, and it is not symmetric.** Three of
the four routes answer with no tools and therefore no citations, so a
misroute here does not look like a failure — it looks like a confident answer.
The gradient matters:

- `research → advisory` or `research → clarify` is a bad answer. Visible.
- `research → simple` is an **unsourced** answer to a question that needed
  evidence — fluent, dated-looking, and carrying nothing a reader can check.
  That is the failure this whole system is built to prevent, arriving through
  a door C1's corpus-gap check does not cover, so it is failed by name and
  separately from everything else.

The other direction is deliberately tolerated. `simple → research` costs money
and returns a correct, sourced answer; that is the direction to be wrong in,
and `ROUTER_PROMPT` says so.

**The cost case is the evidence this module was built before the observability
that would have sized it.** E was built ahead of F by decision, which means
nothing had measured how much a cheap question overpays. `--cost` runs the
same questions through `build_graph(router=True)` and `build_graph(router=False)`
and prints both totals, so the delta arrives at the gate rather than never.

Needs Postgres up and ANTHROPIC_API_KEY. Costs money — roughly $0.05 for the
labelled set, and about the same again for `--cost`.
"""

import argparse
import logging
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import FAILURES, exit_code, indent, note, rule, summary, verdict  # noqa: E402

from app import agent, finnhub, router  # noqa: E402
from app.config import settings  # noqa: E402
from app.cost import estimate_usd  # noqa: E402
from app.db import close_pool  # noqa: E402
from app.sec_http import close_client  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# (question, expected route). Sixteen, four per class, and the traps are
# deliberate rather than decorative:
#
#   - "Who is Tesla's CEO now, and did that change recently?" must be RESEARCH,
#     not simple. Recency defeats the whitelist — the model's knowledge is
#     frozen at its snapshot and leadership is the least stable thing on it.
#   - "What does Nvidia actually do?" must be SIMPLE even though Nvidia is not
#     in the corpus; the question is about identity, not disclosure.
#   - "Which is the better investment, Apple or Microsoft?" must be ADVISORY
#     even though it names two real, covered companies — a comparison framed as
#     a judgement is still a judgement.
#   - "How did the quarter go?" must be CLARIFY: no company, and "the quarter"
#     is not a fiscal period until someone says whose.
LABELLED = [
    ("Who is Tesla's CEO?",                                          "simple"),
    ("What does Nvidia actually do?",                                "simple"),
    ("What industry is JPMorgan in?",                                "simple"),
    ("Where is Microsoft headquartered?",                            "simple"),

    ("What are Apple's key risk factors?",                           "research"),
    ("What's AAPL trading at right now?",                            "research"),
    ("What's the latest news on Meta?",                              "research"),
    ("Who is Tesla's CEO now, and did that change recently?",        "research"),

    ("What are some good tech stocks to buy right now?",             "advisory"),
    ("Should I buy Tesla?",                                          "advisory"),
    ("Find me undervalued semiconductor names.",                     "advisory"),
    ("Which is the better investment, Apple or Microsoft?",          "advisory"),

    ("Tell me about their risks.",                                   "clarify"),
    ("How did the quarter go?",                                      "clarify"),
    ("Is it expensive?",                                             "clarify"),
    ("What happened last week?",                                     "clarify"),
]

# The subset `--cost` prices through both graph shapes. Four rather than eight:
# each question runs twice, and the point is a ratio, not a census.
COST_QUESTIONS = [
    ("Who is Tesla's CEO?", "simple"),
    ("Where is Microsoft headquartered?", "simple"),
    ("What are some good tech stocks to buy right now?", "advisory"),
    ("Should I buy Tesla?", "advisory"),
]

# Reused from check_agent.py's question 7, which is the same claim read on the
# same kind of answer. Widen it here and there together if a run slips a pick
# past it — a name absent from this tuple is not a name the check saw.
PICK_NAMES = ("nvidia", "nvda", "apple", "aapl", "microsoft", "msft", "alphabet",
              "google", "googl", "amazon", "amzn", "meta", "tsla", "tesla",
              "broadcom", "avgo", "palantir", "pltr", "advanced micro", "amd",
              "salesforce", "crm", "oracle", "orcl", "netflix", "nflx")

def introduced_names(answer: str, question: str) -> set[str]:
    """Companies the *answer* put on the table that the *question* did not.

    The rule being enforced is "a refusal that lists candidates has made the
    recommendation", and that is about names the assistant supplies. A name the
    user already chose is not a candidate — it is the subject — and forbidding
    it costs the one thing that makes a decline useful: "I can't rate <their
    company>, but I can analyse its margins — want that?" Echoing their choice
    back is not a pick; introducing a second company is.
    """
    asked = question.lower()
    said = answer.lower()
    return {name for name in PICK_NAMES if name in said and name not in asked}


REFUSAL_MARKERS = ("cannot", "can't", "not able", "unable", "don't recommend",
                   "do not recommend", "not something i can", "no tool",
                   "can't screen", "cannot screen", "not in a position")

# The `simple` path's whole safety story is that it says where the answer came
# from. An unhedged one is indistinguishable from a sourced answer whose
# citations went missing.
HEDGE_MARKERS = ("general knowledge", "not from a filing", "rather than a filing",
                 "my own knowledge", "not sourced", "from memory",
                 "without consulting", "may be out of date", "as of my")


def classify_all() -> list[tuple[str, str, str, dict]]:
    """Label every question. Returns (question, expected, actual, usage)."""
    rows = []
    for question, expected in LABELLED:
        routing = router.classify(question)
        rows.append((question, expected, routing.route, routing.usage))
    return rows


def check_classification(rows: list[tuple[str, str, str, dict]]) -> None:
    rule("1. classification")

    by_class: dict[str, list[tuple[str, str]]] = {}
    for question, expected, actual, _ in rows:
        by_class.setdefault(expected, []).append((question, actual))

    for expected in ("simple", "research", "advisory", "clarify"):
        for question, actual in by_class.get(expected, []):
            verdict(
                actual == expected,
                f"{expected:<8} ← {question[:58]}"
                + (f"   (got {actual})" if actual != expected else ""),
            )

    rule("1b. the asymmetry")
    # The one that is not just "a wrong label". A research question answered
    # from the simple prompt produces prose with no citations and no admission
    # that none were consulted.
    leaked = [q for q, exp, act, _ in rows if exp == "research" and act == "simple"]
    verdict(
        not leaked,
        "no research question was routed to `simple`"
        + (f" ({len(leaked)} was: {leaked[0][:50]!r} — this answers from memory "
           f"a question that needed evidence)" if leaked else ""),
    )

    stranded = [q for q, exp, act, _ in rows if exp == "research" and act == "clarify"]
    verdict(
        not stranded,
        "no research question was routed to `clarify`"
        + (f" ({stranded[0][:50]!r} was)" if stranded else ""),
    )

    # Cheap questions falling into the loop is the tolerated direction, so it
    # is reported and not failed — it costs money and returns a good answer.
    overpaid = [q for q, exp, act, _ in rows if exp in ("simple",) and act == "research"]
    if overpaid:
        note(f"{len(overpaid)} `simple` question(s) fell to research — the safe "
             f"direction, costing money rather than correctness")

    rule("1c. confusion matrix")
    routes = ("simple", "research", "advisory", "clarify")
    print(f"    {'expected → got':<16}" + "".join(f"{r:>10}" for r in routes))
    for expected in routes:
        counts = {r: 0 for r in routes}
        for _, exp, act, _ in rows:
            if exp == expected:
                counts[act] = counts.get(act, 0) + 1
        print(f"    {expected:<16}" + "".join(f"{counts[r]:>10}" for r in routes))

    tokens = sum(u.get("input", 0) + u.get("output", 0) for *_, u in rows)
    per_call = tokens / max(len(rows), 1)
    print(f"\n    router cost: {tokens} tokens over {len(rows)} calls "
          f"(~{per_call:.0f}/call, ${estimate_usd(settings.anthropic_model, input_tokens=tokens) or 0.0:.4f} total)")
    verdict(
        per_call < 500,
        f"the classifier prompt stays small (~{per_call:.0f} tokens/call)"
        + ("" if per_call < 500 else " — the savings are the absent prefix; a "
           "classifier this big deletes them"),
    )


def run(question: str, thread_id: str | None = None, use_router: bool = True) -> dict:
    """One question through a freshly built graph of the requested shape."""
    graph = agent.get_graph() if use_router else agent.build_graph(router=False)
    return graph.invoke(
        {"messages": [("user", question)], "citations": [], "route": ""},
        config=agent.thread_config(thread_id or str(uuid4())),
    )


def check_answers(quiet: bool) -> None:
    """One answer per terminal route — the prose claims, which have no tool signature."""
    rule("2. what the terminal routes actually say")

    cases = [
        ("Who is Tesla's CEO?", "simple"),
        ("What are some good tech stocks to buy right now?", "advisory"),
        # The named-company advisory question, which is the shape people
        # actually type. It must decline the rating without declining the
        # company: the offer is worthless if it cannot say which company it is
        # offering to research.
        ("Apple stock analysis, need buy rating for a tech heavy investor", "advisory"),
        ("Tell me about their risks.", "clarify"),
    ]
    for i, (question, expected) in enumerate(cases):
        rule(f"2{'abcdefg'[i]}. {expected} · {question}")
        state = run(question)
        answer = agent.final_text(state["messages"])
        lowered = answer.lower()
        called = {name for name, _ in agent.tool_calls_made(state["messages"])}

        verdict(state.get("route") == expected,
                f"routed to `{expected}` (got `{state.get('route')}`)")
        verdict(not called, "called no tools"
                + (f" (called {', '.join(sorted(called))})" if called else ""))
        verdict(not state.get("citations"), "produced no citations")
        verdict(bool(answer.strip()), "produced a non-empty answer")

        if expected == "simple":
            verdict(
                any(m in lowered for m in HEDGE_MARKERS),
                "says the answer came from general knowledge, not a source"
                + ("" if any(m in lowered for m in HEDGE_MARKERS)
                   else " — an unhedged memory answer is indistinguishable from "
                        "a sourced one whose citations went missing"),
            )
        if expected == "advisory":
            verdict(any(m in lowered for m in REFUSAL_MARKERS),
                    "declines to produce a pick list")
            named = sorted(introduced_names(answer, question))
            verdict(
                not named,
                "introduces no company or ticker of its own"
                + (f" (named {', '.join(named)} — a refusal that lists candidates "
                   f"has made the recommendation)" if named else ""),
            )
            verdict(answer.rstrip().endswith("?"),
                    "ends by offering to research something instead"
                    + ("" if answer.rstrip().endswith("?")
                       else " — a decline with no offer is a dead end, and most "
                            "questions arrive underspecified rather than settled"))
        if expected == "clarify":
            verdict("?" in answer, "asks a question back")
            named = sorted(introduced_names(answer, question))
            verdict(not named,
                    "offers no menu of companies to pick from"
                    + (f" (named {', '.join(named)})" if named else ""))

        if not quiet:
            print(f"\n{indent(answer)}")


def check_follow_up_skips_the_router(quiet: bool) -> None:
    """C4b's multi-turn, with E1 in front of it.

    The claim is negative — that the second turn was *not* classified — so the
    evidence is the model-turn count. A first turn on a research question runs
    router + agent + agent; a follow-up that skipped the router runs one agent
    turn and nothing else. If the router ran, the answer would be a question
    back rather than an answer.
    """
    rule("3. a follow-up is not re-classified")

    thread = str(uuid4())
    first = run("What are Apple's key risk factors?", thread_id=thread)
    verdict(first.get("route") == "research", "turn 1 routed to `research`")

    second = run("Which of those involve suppliers outside the US?", thread_id=thread)
    answer = agent.final_text(second["messages"])

    verdict(second.get("route") == "research",
            f"turn 2 routed to `research` without a classifier call "
            f"(got `{second.get('route')}`)")
    verdict(
        not answer.strip().endswith("?"),
        "turn 2 answered rather than asking which company was meant"
        + ("" if not answer.strip().endswith("?")
           else " — the skip-on-history rule is the thing to check"),
    )

    if not quiet:
        print(f"\n{indent(answer)}")


def check_cost() -> None:
    """Price the routed shape against the pre-router one, on the same questions.

    This is the measurement F would have supplied had it been built first. It
    is a real comparison rather than an estimate: both graphs are compiled from
    the same code, given the same questions, and their token counts summed the
    same way.
    """
    rule("4. what the router saves")

    totals = {True: {"tokens": 0, "usd": 0.0}, False: {"tokens": 0, "usd": 0.0}}
    for question, expected in COST_QUESTIONS:
        line = {}
        for use_router in (True, False):
            state = run(question, use_router=use_router)
            rows = agent.token_usage(state["messages"])
            tokens = sum(r["input"] + r["output"] + r["cache_read"] for r in rows)
            usd = sum(
                estimate_usd(
                    settings.anthropic_model,
                    input_tokens=r["input"],
                    output_tokens=r["output"],
                    cache_read_tokens=r["cache_read"],
                    cache_write_tokens=r["cache_creation"],
                ) or 0.0
                for r in rows
            )
            if use_router:
                # The classifier's own tokens are invisible to token_usage,
                # because router_node adds no message to state. Counting the
                # routed shape without them would flatter it.
                routing = router.classify(question)
                tokens += routing.usage.get("input", 0) + routing.usage.get("output", 0)
                usd += estimate_usd(
                    settings.anthropic_model,
                    input_tokens=routing.usage.get("input", 0),
                    output_tokens=routing.usage.get("output", 0),
                ) or 0.0
            totals[use_router]["tokens"] += tokens
            totals[use_router]["usd"] += usd
            line[use_router] = (tokens, usd, len(rows))

        print(f"\n    {question[:60]}  ({expected})")
        print(f"      routed : {line[True][0]:>7} tokens  ${line[True][1]:.5f}  "
              f"{line[True][2]} model turn(s)")
        print(f"      loop   : {line[False][0]:>7} tokens  ${line[False][1]:.5f}  "
              f"{line[False][2]} model turn(s)")

    routed, loop = totals[True], totals[False]
    saved = loop["usd"] - routed["usd"]
    ratio = (loop["usd"] / routed["usd"]) if routed["usd"] else 0.0

    print(f"\n    over {len(COST_QUESTIONS)} cheap questions:")
    print(f"      routed : {routed['tokens']:>7} tokens  ${routed['usd']:.5f}")
    print(f"      loop   : {loop['tokens']:>7} tokens  ${loop['usd']:.5f}")
    print(f"      saved  : ${saved:.5f}  ({ratio:.1f}x)")

    verdict(
        routed["usd"] < loop["usd"],
        f"the routed shape is cheaper on questions that need no evidence "
        f"({ratio:.1f}x)"
        + ("" if routed["usd"] < loop["usd"]
           else " — the classifier is costing more than the prefix it skips; "
                "check ROUTER_PROMPT's size and that respond binds no tools"),
    )
    print("\n    ↳ the human reading: this is the cheap-question case only. "
          "Research questions\n      pay the classifier and save nothing — "
          "F's `runs` table is what shows the mix.")


def configure() -> None:
    """Build both models before anything runs, so a missing key is one line."""
    rule(f"model: {settings.anthropic_model}")
    try:
        agent.get_llm()
        router.get_router_llm()
    except RuntimeError as exc:
        print(f"\n  {exc}")
        raise SystemExit(2) from None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify-only", action="store_true",
                    help="labels only — no answers generated (much cheaper)")
    ap.add_argument("--cost", action="store_true",
                    help="price the routed graph against the pre-router one")
    ap.add_argument("--quiet", action="store_true", help="verdicts only, no answer text")
    args = ap.parse_args()

    configure()

    try:
        if args.cost:
            check_cost()
        else:
            check_classification(classify_all())
            if not args.classify_only:
                check_answers(args.quiet)
                check_follow_up_skips_the_router(args.quiet)

        summary(
            on_failure="A wrong label is a ROUTER_PROMPT bug — fix app/router.py, "
                       "not the graph.\n  A right label with a wrong answer is a "
                       "RESPOND_PROMPTS bug, in the same file.",
            on_success="now read the answers: does the simple one hedge without "
                       "hedging into\n  uselessness, and does the clarify one ask "
                       "the narrowest question it could?",
        )
    finally:
        finnhub.close_client()
        close_client()
        close_pool()

    raise SystemExit(exit_code())


if __name__ == "__main__":
    main()
