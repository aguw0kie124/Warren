"""C1 verification: the tools, called directly with no LLM in the loop.

    python scripts/check_tools.py
    python scripts/check_tools.py --ticker MSFT
    python scripts/check_tools.py --show search_filings   # full model-visible text

C1 is plumbing, so most of this is a parity check: each tool is run beside the
plain function it wraps, and the two are compared. Confirming the tools are a
pure pass-through is what makes any later weirdness attributable to the model
rather than the wiring.

The parts that are not plumbing get their own sections. The corpus-gap check
is the one piece of real behaviour C1 adds, and it has to be visibly different
from an ordinary empty result — those two are the same `[]` to a retriever and
opposite facts to a reader. And every tool's failure path has to come back as
readable text, since a traceback tells a model nothing it can act on.

Needs Postgres up, plus FINNHUB_API_KEY and TAVILY_API_KEY. No ANTHROPIC_API_KEY:
there is deliberately no model here.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import note, rule, summary, verdict, exit_code  # noqa: E402

from app import finnhub, retriever, websearch  # noqa: E402
from app.db import close_pool  # noqa: E402
from app.tools import TOOLS, Citation, VALID_SECTIONS, search_filings  # noqa: E402
from app.tools import get_basic_financials, get_company_news, get_quote, web_search  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def run(tool, **args):
    """Invoke a tool the way ToolNode will, so the artifact comes back too."""
    return tool.invoke(
        {"name": tool.name, "args": args, "id": "check-1", "type": "tool_call"}
    )


def excerpt(text: str, limit: int = 900) -> str:
    body = text if len(text) <= limit else text[:limit] + f"\n… (+{len(text)-limit} chars)"
    return "\n".join("    │ " + line for line in body.splitlines())


def check_citations(message, expected_type: str, tool_name: str) -> None:
    art = message.artifact
    verdict(
        all(isinstance(c, Citation) for c in art),
        f"{tool_name}: artifact is a list of Citation objects ({len(art)})",
    )
    verdict(
        bool(art) and all(c.type == expected_type for c in art),
        f"{tool_name}: every citation is typed {expected_type!r}",
    )
    verdict(
        all(c.source_url.startswith("http") and c.label for c in art),
        f"{tool_name}: every citation has a label and an http URL",
    )
    # The [n] markers the model reads must index the citation list position for
    # position, or a model that writes "[2]" cites the wrong source.
    aligned = all(
        f"[{i}] {c.label}" in message.content for i, c in enumerate(art, start=1)
    )
    verdict(aligned, f"{tool_name}: [n] markers align with the citation list")


# ---------------------------------------------------------------------------
# 1. pass-through parity
# ---------------------------------------------------------------------------


def check_search_filings(ticker: str, show: bool) -> None:
    rule(f"search_filings vs retriever.hybrid_search  ({ticker})")
    query = "What are the main business risks?"

    direct = retriever.hybrid_search(query, ticker=ticker, k=3)
    message = run(search_filings, query=query, ticker=ticker, k=3)

    print(f"  hybrid_search returned {len(direct)} result(s):")
    for r in direct:
        note(f"{r.score:.5f}  {r.section}  {r.accession_number}#{r.chunk_index}")

    verdict(
        len(message.artifact) == len(direct),
        f"same result count through the tool ({len(message.artifact)})",
    )
    verdict(
        [c.label for c in message.artifact] == [r.citation.strip("[]") for r in direct],
        "same citations, in the same order",
    )
    verdict(
        all(r.content in message.content for r in direct),
        "every retrieved passage appears verbatim in the tool's text",
    )
    verdict("as of their filing date" in message.content, "staleness note present")
    check_citations(message, "filing", "search_filings")
    if show:
        print(excerpt(message.content, 1800))


def check_quote(ticker: str) -> None:
    rule(f"get_quote vs finnhub.get_quote  ({ticker})")
    direct = finnhub.get_quote(ticker)
    message = run(get_quote, symbol=ticker)

    note(f"finnhub.get_quote: {direct.current_price:.2f} at {direct.as_of}")
    verdict(f"{direct.current_price:.2f}" in message.content, "same price in the text")
    verdict(direct.as_of in message.content, "timestamp carried through")
    verdict(message.artifact == [], "no citations (a quote has no article behind it)")
    print(excerpt(message.content))


def check_financials(ticker: str) -> None:
    rule(f"get_basic_financials vs finnhub.get_basic_financials  ({ticker})")
    direct = finnhub.get_basic_financials(ticker)
    present = direct.present()
    message = run(get_basic_financials, symbol=ticker)

    note(f"{len(present)} metric(s) available: {', '.join(sorted(present))}")
    verdict(
        all(name in message.content for name in present),
        "every available metric appears in the text",
    )
    verdict(
        "None" not in message.content,
        "absent metrics are omitted, not rendered as None",
    )
    verdict(message.artifact == [], "no citations")
    print(excerpt(message.content))


def check_news(ticker: str, show: bool) -> None:
    rule(f"get_company_news vs finnhub.get_company_news  ({ticker})")
    direct = finnhub.get_company_news(ticker, limit=5)
    message = run(get_company_news, symbol=ticker, limit=5)

    note(f"finnhub returned {len(direct)} article(s) for the trailing week")
    for item in direct[:3]:
        note(f"{item.published_date}  {item.source}: {item.headline[:60]}")

    if not direct:
        verdict("quiet period" in message.content, "empty feed reads as a quiet period")
        return

    # A live feed can gain an article between two calls, so the counts are
    # compared as a set of URLs rather than asserted equal.
    tool_urls = {c.source_url for c in message.artifact}
    overlap = tool_urls & {a.url for a in direct}
    verdict(
        len(overlap) >= min(len(direct), len(tool_urls)) - 1,
        f"same articles through the tool ({len(overlap)} of {len(direct)} shared)",
    )
    check_citations(message, "news", "get_company_news")
    if show:
        print(excerpt(message.content, 1400))


def check_web(show: bool) -> None:
    rule("web_search vs websearch.web_search")
    query = "Apple services revenue growth analyst commentary"
    direct = websearch.web_search(query)
    message = run(web_search, query=query)

    note(f"websearch returned {len(direct)} result(s): "
         f"{', '.join(r.domain for r in direct)}")

    verdict(len(message.artifact) <= websearch.DEFAULT_MAX_RESULTS,
            f"result count respects the module default ({len(message.artifact)})")
    verdict(
        all(websearch._allowed(c.source_url, websearch.FINANCE_DOMAINS)
            for c in message.artifact),
        "every citation URL is on the finance allowlist",
    )
    verdict(
        len({c.source_url for c in message.artifact}) == len(message.artifact),
        "no duplicate URLs survive into the citations",
    )
    check_citations(message, "web", "web_search")
    if show:
        print(excerpt(message.content, 1400))


# ---------------------------------------------------------------------------
# 2. corpus gap vs empty result vs hits
# ---------------------------------------------------------------------------


def check_corpus_gap(ticker: str) -> None:
    rule("corpus gap vs ordinary empty result")
    covered = sorted({r[0] for r in _covered()})
    note(f"ingested: {', '.join(covered)}")

    missing = next(
        (t for t in ("NVDA", "AMZN", "GOOGL", "ZZZZ") if t not in covered), "ZZZZ"
    )
    gap = run(search_filings, query="risk factors", ticker=missing)
    print(f"\n  1. uningested ticker ({missing}):")
    print(excerpt(gap.content, 500))
    verdict("CORPUS GAP" in gap.content, f"{missing} reports a corpus gap")
    verdict(all(t in gap.content for t in covered),
            "the gap names the companies that ARE covered")

    # With dense retrieval in the fusion, a covered ticker practically always
    # returns *something*, so the genuine empty path is reached through
    # filters, not through a bad query.
    empty = run(search_filings, query="risk factors", ticker=ticker, fiscal_year=1999)
    print(f"\n  2. covered ticker, impossible filter ({ticker}, fiscal_year=1999):")
    print(excerpt(empty.content, 500))
    verdict("CORPUS GAP" not in empty.content and "No passages matched" in empty.content,
            "an empty result is NOT reported as a corpus gap")

    nonsense = run(search_filings, query="lunar regolith mining permits", ticker=ticker, k=2)
    print(f"\n  3. covered ticker, nonsense query ({ticker}):")
    print(excerpt(nonsense.content, 400))
    verdict(
        "CORPUS GAP" not in nonsense.content,
        "a nonsense query is neither a gap nor (with dense retrieval) empty",
    )


def _covered():
    from app.db import get_conn

    with get_conn() as conn:
        return conn.execute("SELECT DISTINCT ticker FROM filings").fetchall()


# ---------------------------------------------------------------------------
# 3. bad input comes back as text
# ---------------------------------------------------------------------------


def check_bad_input(ticker: str, live: bool = True) -> None:
    rule("bad input is reported as text, not raised")

    for tool in (get_quote, get_basic_financials, get_company_news) if live else ():
        message = run(tool, symbol="APPL")
        ok = message.status == "success" and "APPL" in message.content
        verdict(ok, f"{tool.name}('APPL') explains the unknown symbol")
        note(message.content.splitlines()[0][:100])

    bad_section = run(search_filings, query="risks", ticker=ticker, section="Risk Factors")
    verdict(
        all(s in bad_section.content for s in VALID_SECTIONS),
        "an invalid section is answered with the valid vocabulary",
    )
    note(bad_section.content[:150])

    if not live:
        return

    empty_query = run(web_search, query="   ")
    verdict(empty_query.status == "success" and empty_query.artifact == [],
            "an empty web_search query fails cleanly")
    note(empty_query.content[:100])


# ---------------------------------------------------------------------------


def check_surface() -> None:
    rule("the tool surface C2 will bind")
    for tool in TOOLS:
        first = (tool.description or "").splitlines()[0]
        print(f"  {tool.name:<22} {len(tool.description):>4} chars  {first[:44]}")
        verdict(tool.response_format == "content_and_artifact",
                f"{tool.name}: returns content + artifact")
    verdict(len(TOOLS) == 6, "six tools registered")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL", help="an ingested ticker to exercise")
    ap.add_argument("--show", action="store_true",
                    help="print the full model-visible text for the source tools")
    ap.add_argument("--skip-live", action="store_true",
                    help="filings and corpus checks only; no Finnhub or Tavily calls")
    args = ap.parse_args()

    try:
        check_surface()
        check_search_filings(args.ticker, args.show)
        check_corpus_gap(args.ticker)
        if not args.skip_live:
            check_quote(args.ticker)
            check_financials(args.ticker)
            check_news(args.ticker, args.show)
            check_web(args.show)
        check_bad_input(args.ticker, live=not args.skip_live)

        summary(
            on_success="read the rendered tool text above before moving to C2; "
                       "the docstrings\n  and this text are what the model "
                       "actually routes on.",
        )
    finally:
        finnhub.close_client()
        close_pool()

    raise SystemExit(exit_code())


if __name__ == "__main__":
    main()
