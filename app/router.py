"""E1 · The query router — pre-dispatch, ahead of the ReAct loop.

`app/agent.py` records that there is deliberately no router node, because a
tool-calling model already routes. That argument is about routing *within* a
research question, and it still holds — `tools_condition` is untouched.

This does a different job. It decides **which prompt and which tool set run at
all**, and it adds a terminal path the two-node loop cannot express: a question
that cannot be acted on as written gets asked about rather than guessed at.
Tools are still list entries, not graph nodes; adding a capability still means
appending one `@tool`.

Four classes, three of which terminate immediately:

    research   → the ReAct loop, unchanged. THE DEFAULT.
    simple     → answered from the model's own knowledge, hedged. No tools.
    advisory   → declined, with C5's rules intact. No tools.
    clarify    → one question back to the user. No tools.

**The asymmetry is the whole safety argument, and it is enforced twice** — in
`ROUTER_PROMPT` and again in `classify()`, which maps anything unrecognised to
`research`. A `research → advisory` slip is an annoyance. A `research → simple`
slip produces a fluent, zero-citation, memory-sourced answer to a question that
needed evidence, which is the exact failure the corpus-gap check and the whole
citation design exist to prevent. Wrong-and-expensive is recoverable;
wrong-and-unsourced is not.

**The classifier is deliberately not tool-bound.** Binding the five research
tool schemas to it would spend exactly the savings this module exists to
produce — the prefix, not the answer, is what a cheap question overpays for.
`ROUTER_PROMPT` is kept under ~300 tokens for the same reason; that is a
constraint on edits to this file, not a preference.

**No `cache_control` on any prompt here.** All four sit far below any published
minimum cacheable prefix, so a breakpoint would be decoration.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal, get_args

from langchain_anthropic import ChatAnthropic

from app.config import settings

logger = logging.getLogger(__name__)

Route = Literal["simple", "research", "advisory", "clarify"]

ROUTES: tuple[str, ...] = get_args(Route)

# Where anything unrecognised, ambiguous, or malformed lands. Named rather than
# written inline at each site, because every one of those sites is a place
# someone could later "simplify" into a cheaper default.
FALLBACK_ROUTE = "research"

# The routes that terminate at `respond_node` rather than entering the loop.
TERMINAL_ROUTES = tuple(r for r in ROUTES if r != "research")


@dataclass(frozen=True)
class Routing:
    """A classification plus what it cost.

    The usage rides along because the router's tokens are otherwise invisible:
    `router_node` adds no message to state, so `agent.token_usage()` — which
    reads `usage_metadata` off messages — cannot see them. A gate comparing the
    routed graph against the two-node one has to be able to count them, and a
    number nobody can obtain is a number nobody will check.
    """

    route: str
    usage: dict = field(default_factory=dict)


ROUTER_PROMPT = """You classify one question for a financial research system. Reply with exactly one label and nothing else.

- research — needs SEC filings, market data, news, or the web. Anything touching numbers, prices, risks, performance, events, disclosures, strategy, comparisons, or "what's going on with X". THE DEFAULT.
- simple — answerable from general knowledge of a company's identity alone: who leads it, what it does in a sentence, what industry it is in, where it is based, which exchange it trades on. Nothing numeric, dated, or financial. If the question implies currency or change — "who is the CEO now", "did they just replace their CFO", anything naming a date — it is research, not simple.
- advisory — asks for a pick, a ranking, a screen, or a buy/sell/hold judgment. "What should I buy", "best stocks right now", "is X a good investment", "find me undervalued names".
- clarify — cannot be acted on as written: no company is identifiable in it, or it has more than one reading that would lead to genuinely different research.

When in doubt, answer research. Being wrong toward research costs money and time. Being wrong toward simple produces an unsourced answer to a question that needed evidence, which is worse."""


SIMPLE_PROMPT = """You are a financial research assistant. This question was routed here because it asks only about a company's identity — who leads it, what it does, what industry it is in, where it is based, or where it trades.

Answer from your own knowledge, and mark that plainly. Open the substantive sentence with a hedge — "From general knowledge rather than a filing or live source: ..." — and close by offering the sourced version: that the company's SEC filings, current market data, or recent news can be pulled if they want something verifiable.

Hard limits. You may state only: who leads the company, what it does in a sentence, its industry, its headquarters, its listing venue. You may NOT state a number, a date, a price, a metric, a ratio, a growth rate, a risk, an event, a filing reference, or a quotation. Those come from tools, and no tool ran here. If answering properly needs any of them, say so and offer to research it instead of reaching for memory.

Your knowledge has a cutoff and company leadership changes. If you are not confident the answer is still current, say so explicitly rather than stating it flatly.

Two or three sentences. Do not pad."""


ADVISORY_PROMPT = """You are a financial research assistant. This question asks for a pick, a ranking, a screen, or a buy/sell/hold judgment, and this system does not do that.

**Decline the rating in one line, then offer the work you can actually do, then ask whether to go ahead.** A dead-end refusal is the failure here: most people do not arrive with a precisely scoped research question, and "I can't do that" sends them away from an answer that was one sentence of redirection from existing.

Write three or four sentences, in this shape:

1. **One opening line.** You cannot give a buy, sell or hold rating or an investment recommendation. One sentence — no apology, no paragraph on risk tolerance or consulting a professional.
2. **What you can do instead**, as two to four concrete angles taken from what they actually asked: revenue and margin trend, earnings quality, balance-sheet strength, capital allocation, valuation multiples, what the filings flag as risks, recent news. Say these come from SEC filings, reported financials, live market data and the financial web.
3. **A closing question** that offers to start — "Want me to start with ...?" or "Which of those would be most useful?" End on that question.

Two rules that hold regardless:

- It does not recommend what to buy, sell, or hold, and does not rank companies as investments.
- It cannot screen, rank, or scan a universe — no tool here takes a criterion ("cheap", "strong margins", "undervalued") and returns a list of companies matching it. Every tool takes a ticker. If no company was named, say a name is needed and ask which one.

**Do not name example companies.** If the user named a company, use that one and only that one — echoing back their own choice is not a recommendation, and the offer is useless without it. Introducing any *other* company is a recommendation, because every answer in this system carries citations, so a name offered in passing reads as a sourced pick. Which companies are on the table is the user's move, not yours.

Be brief and direct. No headings, no bullet list in the reply itself, no closing disclaimer."""


CLARIFY_PROMPT = """You are a financial research assistant. This question cannot be acted on as written — either no company is identifiable in it, or it has more than one reading that would lead to genuinely different research.

Ask for exactly what you need in order to proceed, and nothing more. Name the ambiguity concretely rather than asking the user to start over.

Do not answer the question. Do not guess which company was meant. Do not offer a menu of companies to choose from — naming candidates is the user's move, not yours, for the same reason a refusal that lists picks has made the recommendation.

One or two sentences, ending with the question itself."""


RESPOND_PROMPTS: dict[str, str] = {
    "simple": SIMPLE_PROMPT,
    "advisory": ADVISORY_PROMPT,
    "clarify": CLARIFY_PROMPT,
}


# The answer is one of four words, so it comes back as text and is parsed
# here. **`with_structured_output` was tried first and removed**: measured on
# 2026-08-19, it cost 974 input tokens per call against 297 for the prompt and
# question alone — ~677 tokens of tool schema and forced `tool_choice` wrapped
# around a single enum field. That is more overhead than the entire prompt it
# was protecting, and this module's whole economic premise is that classifying
# is cheaper than the ~2.4k-token tool prefix it lets a question skip. A
# schema that triples the cost of the thing it validates is a bad trade when
# the validation it buys — "the label is one of these four" — is three lines
# of Python that `classify()` needs anyway for the fallback case.
_LABEL_STRIP = " \t\n.,:;'\"`*-_[]()"

_router_llm = None


def get_router_llm():
    """The classifier, built once per process.

    Same pinned model as the agent — it is already the cheapest available, and
    a second model id would be a second thing to keep in step with the gates.
    What makes this cheap is the absent tool schemas and the short prompt, not
    a smaller model.

    `max_tokens` is tiny on purpose. One word is the whole contract, and a
    reply truncated mid-preamble fails to parse and falls to `research` — the
    safe direction — rather than costing a paragraph of tokens to say
    "simple".

    Lazily, like `agent.get_llm()`, so importing this module needs no API key.
    """
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            max_tokens=16,
            timeout=30.0,
        )
    return _router_llm


def _usage_of(message) -> dict:
    """The token counts off a classifier reply, or an empty dict."""
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input": usage.get("input_tokens", 0) or 0,
        "output": usage.get("output_tokens", 0) or 0,
    }


def _text_of(content) -> str:
    """Flatten a reply body, which is a block list once anything else is on."""
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text", "") for block in content or [] if isinstance(block, dict)
    )


def parse_label(content) -> str | None:
    """The one route named in a reply, or None if that is not exactly one.

    Tolerant of a model that punctuates or emphasises — "simple.",
    "**advisory**", "Label: clarify" all read correctly. Deliberately *not*
    tolerant of a reply naming two routes ("research or clarify"), which is a
    model hedging rather than answering; None sends that to the fallback,
    which is what hedging should cost.
    """
    words = {word.strip(_LABEL_STRIP).lower() for word in _text_of(content).split()}
    found = [route for route in ROUTES if route in words]
    return found[0] if len(found) == 1 else None


def classify(question: str) -> Routing:
    """Label one question. Never raises on a bad label — falls to research.

    Genuine failures — a 429, a 5xx, a bad key — are left to propagate. The
    node's retry policy covers the transient ones, and swallowing the rest
    would turn an outage into a silently more expensive service.
    """
    reply = get_router_llm().invoke(
        [("system", ROUTER_PROMPT), ("human", question)]
    )
    usage = _usage_of(reply)
    route = parse_label(getattr(reply, "content", reply))

    if route not in ROUTES:
        # Not a bug and not an error: the model said something that was not one
        # of the four words. Research is the complete path, so this costs money
        # rather than correctness — which is the direction to be wrong in.
        logger.warning("router returned %r; falling back to %s", route, FALLBACK_ROUTE)
        route = FALLBACK_ROUTE

    logger.debug("routed %r → %s (%s)", question[:60], route, usage)
    return Routing(route=route, usage=usage)
