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

from app.config import settings
from app.db import close_pool
from app.retriever import search
from app.sec_http import close_client, sec_get

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

SNAPSHOT = settings.cache_dir.parent / "retrieval_a7.json"

# Two kinds deliberately mixed. Semantic questions are what dense search is good
# at; the exact-phrase ones are where it is expected to struggle, and are the
# reason A8 exists — they are the comparison that will justify hybrid search.
QUESTIONS = [
    ("semantic", "What are Apple's main business risks?", {"ticker": "AAPL"}),
    ("semantic", "How is the company growing revenue?", {"ticker": "MSFT"}),
    ("semantic", "Is the companies value driven by elon?", {"ticker": "TSLA"}),
    ("semantic", "Risks from artificial intelligence investment", {"ticker": "META"}),
    ("exact", "material weakness in internal control", {}),
    ("exact", "going concern", {}),
    ("exact", "Item 7A quantitative and qualitative disclosures", {}),
    ("exact", "effective tax rate", {"ticker": "AAPL"}),
]


def run(label: str, query: str, k: int, **filters) -> list[dict]:
    results = search(query, k=k, **filters)
    flt = ", ".join(f"{k}={v}" for k, v in filters.items()) or "no filter"
    print(f"\n=== [{label}] {query!r}  ({flt}) ===")

    if not results:
        print("  [BAD] no results")
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
    args = ap.parse_args()

    snapshot: list[dict] = []
    try:
        if args.query:
            filters = {
                k: v for k, v in
                {"ticker": args.ticker, "section": args.section, "form_type": args.form_type}.items()
                if v
            }
            run("adhoc", args.query, args.k, **filters)
            return

        for label, query, filters in QUESTIONS:
            snapshot += run(label, query, args.k, **filters)

        # Filters must actually narrow, not merely be accepted.
        print("\n=== filter behaviour ===")
        broad = search("revenue growth", k=20)
        narrow = search("revenue growth", ticker="AAPL", section="Item 7 MD&A", k=20)
        print(f"  unfiltered:            {len(broad):>3} results, "
              f"{len({r.ticker for r in broad})} ticker(s)")
        print(f"  ticker+section filter: {len(narrow):>3} results, "
              f"{len({r.ticker for r in narrow})} ticker(s)")
        ok = narrow and {r.ticker for r in narrow} == {"AAPL"} \
            and {r.section for r in narrow} == {"Item 7 MD&A"}
        print(f"  [{'ok ' if ok else 'BAD'}] filtered results honour every filter")

        # A citation nobody can open is not a citation.
        print("\n=== citation URLs resolve ===")
        urls = sorted({(r.source_url, r.citation) for r in broad[:3]})
        for url, citation in urls:
            print(f"  [{'ok ' if check_url(url) else 'BAD'}] {citation}\n         {url}")

        if args.save:
            SNAPSHOT.write_text(json.dumps(snapshot, indent=2))
            print(f"\nsaved {len(snapshot)} result(s) to {SNAPSHOT} for the A8 comparison")
    finally:
        close_client()
        close_pool()


if __name__ == "__main__":
    main()
