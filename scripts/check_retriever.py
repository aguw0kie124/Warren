"""A7 verification: dense retrieval quality, filters, and citation resolution.

    python scripts/check_retriever.py                 # run the benchmark set
    python scripts/check_retriever.py --save          # record results for A8
    python scripts/check_retriever.py --query "going concern" --ticker MSFT

The question set is fixed on purpose. A8 replaces dense search with hybrid
BM25+RRF and has to be judged against the *same* questions, so --save writes a
snapshot to data/retrieval_a7.json for that comparison to be concrete rather
than remembered.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import exit_code, rule, summary, verdict  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import close_pool  # noqa: E402
from app.retriever import hybrid_search, search  # noqa: E402
from app.sec_http import close_client, sec_get  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

SNAPSHOT = settings.cache_dir.parent / "retrieval_a7.json"

# Two kinds deliberately mixed. Semantic questions are what dense search is good
# at; the exact-phrase ones are where it is expected to struggle, and are the
# reason A8 exists — they are the comparison that will justify hybrid search.
QUESTIONS = [
    ("semantic", "What are Apple's main business risks?", {"ticker": "AAPL"}),
    ("semantic", "How is the company growing revenue?", {"ticker": "MSFT"}),
    ("semantic", "What could disrupt manufacturing and supply?", {"ticker": "TSLA"}),
    ("semantic", "Risks from artificial intelligence investment", {"ticker": "META"}),
    ("exact", "effective tax rate", {"ticker": "AAPL"}),
    ("exact", "goodwill impairment", {}),
    ("exact", "remaining performance obligation", {"ticker": "MSFT"}),
    ("exact", "Reality Labs segment results", {"ticker": "META"}),
]

# (query, the literal string a correct hit must contain, filters).
#
# Every phrase here was confirmed present in the corpus first. An earlier
# version of this set used "material weakness" and "going concern", which
# appear in ZERO chunks — these are four large, healthy filers who report
# neither. Those queries could not distinguish a good retriever from a bad
# one, because there was no right answer to find.
EXACT_TERMS = [
    ("effective tax rate", "effective tax rate", {"ticker": "AAPL"}),
    ("goodwill impairment", "goodwill impairment", {}),
    ("remaining performance obligation", "remaining performance obligation", {"ticker": "MSFT"}),
    ("What is Reality Labs?", "Reality Labs", {"ticker": "META"}),
    ("Who is the Technoking?", "Technoking", {"ticker": "TSLA"}),
    ("Full Self-Driving capability", "Full Self-Driving", {"ticker": "TSLA"}),
    ("Is the company dependent on Elon Musk?", "Elon Musk", {"ticker": "TSLA"}),
]


def run(label: str, query: str, k: int, **filters) -> list[dict]:
    results = search(query, k=k, **filters)
    flt = ", ".join(f"{k}={v}" for k, v in filters.items()) or "no filter"
    rule(f"[{label}] {query!r}  ({flt})")

    if not verdict(bool(results), "results returned"):
        return []

    for r in results:
        print(f"  {r.score:.3f}  {r.ticker} {r.form_type} FY{r.fiscal_year}  {r.section}")
        print(f"         {r.content[:180].replace(chr(10), ' ')}")

    return [
        {
            "query": query, "kind": label, "filters": filters,
            "rank": i, "score": round(r.score, 4),
            "ticker": r.ticker, "form_type": r.form_type, "section": r.section,
            "accession_number": r.accession_number, "chunk_index": r.chunk_index,
            "content": r.content,
        }
        for i, r in enumerate(results)
    ]


def first_hit(results, phrase: str) -> int | None:
    """1-based rank of the first result literally containing the phrase."""
    for i, r in enumerate(results, start=1):
        if phrase.lower() in r.content.lower():
            return i
    return None


def compare(k: int) -> None:
    """Dense vs hybrid on the queries hybrid is supposed to win.

    Measured, not eyeballed: the score is the rank at which the first chunk
    literally containing the phrase appears. Lower is better; a dash means it
    never appeared in the top k at all.
    """
    print(f"\n=== dense vs hybrid: rank of first literal match (k={k}) ===")
    print(f"  {'query':<38} {'phrase':<26} {'dense':>6} {'hybrid':>7}  verdict")

    wins = losses = 0
    for query, phrase, filters in EXACT_TERMS:
        d = first_hit(search(query, k=k, **filters), phrase)
        h = first_hit(hybrid_search(query, k=k, **filters), phrase)

        if (h or 99) < (d or 99):
            verdict, _ = "hybrid better", (wins := wins + 1)
        elif (h or 99) > (d or 99):
            verdict, _ = "DENSE better", (losses := losses + 1)
        else:
            verdict = "tie"

        print(f"  {query[:37]:<38} {phrase[:25]:<26} "
              f"{(d or '-'):>6} {(h or '-'):>7}  {verdict}")

    print(f"\n  hybrid wins {wins}, loses {losses}, of {len(EXACT_TERMS)}")


def show_fusion(query: str, k: int, **filters) -> None:
    """Where each ranker placed the fused results — the diagnostic view."""
    print(f"\n=== fusion detail: {query!r} ===")
    for i, r in enumerate(hybrid_search(query, k=k, **filters), start=1):
        found = "both" if r.dense_rank and r.sparse_rank else (
            "dense only" if r.dense_rank else "sparse only")
        print(f"  {i}. score={r.score:.5f}  dense={r.dense_rank or '-':>3} "
              f"sparse={r.sparse_rank or '-':>3}  ({found})")
        print(f"     {r.ticker} {r.section}: {r.content[:120].replace(chr(10), ' ')}")


def check_url(url: str) -> bool:
    """Fetch the citation URL through the project's own SEC client.

    Not urllib: sec_get carries the required User-Agent, the rate limit, and a
    working TLS trust store. Checking citations through a different HTTP path
    than the one that fetched them tests the wrong thing anyway.
    """
    try:
        return sec_get(url, accept="text/html").status_code == 200
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        print(f"         {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="run one ad-hoc query instead of the set")
    ap.add_argument("--ticker")
    ap.add_argument("--section")
    ap.add_argument("--form-type")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--save", action="store_true", help="write the A8 comparison snapshot")
    ap.add_argument("--compare", action="store_true", help="A8: dense vs hybrid")
    ap.add_argument("--hybrid", action="store_true", help="use hybrid for --query")
    args = ap.parse_args()

    snapshot: list[dict] = []
    try:
        if args.compare:
            compare(args.k)
            show_fusion("Is the company dependent on Elon Musk?", 3, ticker="TSLA")
            show_fusion("goodwill impairment", 3)
            return

        if args.query:
            filters = {
                k: v for k, v in
                {"ticker": args.ticker, "section": args.section, "form_type": args.form_type}.items()
                if v
            }
            engine = hybrid_search if args.hybrid else search
            for r in engine(args.query, k=args.k, **filters):
                print(f"  {r.score:.5f}  {r.ticker} {r.form_type} {r.section}")
                print(f"         {r.content[:180].replace(chr(10), ' ')}")
            return

        for label, query, filters in QUESTIONS:
            snapshot += run(label, query, args.k, **filters)

        # Filters must actually narrow, not merely be accepted.
        rule("filter behaviour")
        broad = search("revenue growth", k=20)
        narrow = search("revenue growth", ticker="AAPL", section="Item 7 MD&A", k=20)
        print(f"  unfiltered:            {len(broad):>3} results, "
              f"{len({r.ticker for r in broad})} ticker(s)")
        print(f"  ticker+section filter: {len(narrow):>3} results, "
              f"{len({r.ticker for r in narrow})} ticker(s)")
        verdict(
            bool(narrow) and {r.ticker for r in narrow} == {"AAPL"}
            and {r.section for r in narrow} == {"Item 7 MD&A"},
            "filtered results honour every filter",
        )

        # A citation nobody can open is not a citation.
        rule("citation URLs resolve")
        urls = sorted({(r.source_url, r.citation) for r in broad[:3]})
        for url, citation in urls:
            verdict(check_url(url), f"{citation}\n         {url}")

        if args.save:
            SNAPSHOT.write_text(json.dumps(snapshot, indent=2))
            print(f"\nsaved {len(snapshot)} result(s) to {SNAPSHOT} for the A8 comparison")

        summary()
    finally:
        close_client()
        close_pool()

    raise SystemExit(exit_code())


if __name__ == "__main__":
    main()
