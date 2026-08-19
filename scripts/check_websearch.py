"""B2 verification: Tavily web search, and proof the domain allowlist is real.

    python scripts/check_websearch.py
    python scripts/check_websearch.py --query "Nvidia data center demand outlook"
    python scripts/check_websearch.py --open-web    # same query, filter off

The allowlist is the part worth verifying. It is one line of configuration that
silently stops mattering if the parameter name changes or the provider ignores
it — and the failure mode is invisible: the agent just starts citing worse
sources. So this runs the same query twice, filtered and unfiltered, and shows
you the domains side by side. If both lists look the same, the filter is not
being applied.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import rule  # noqa: E402

from app.websearch import FINANCE_DOMAINS, WebSearchError, _allowed, web_search  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

DEFAULT_QUERY = "Apple services revenue growth analyst commentary"

# Deliberately the kind of query that attracts promotional stock content — the
# reason the allowlist exists at all.
SPAM_MAGNET_QUERY = "best stocks to buy now for huge gains"


def show(results: list, check_allowlist: bool) -> None:
    if not results:
        print("  (no results)")
        return
    for r in results:
        flag = ""
        if check_allowlist:
            flag = "✔" if _allowed(r.url, FINANCE_DOMAINS) else "✘ OFF-ALLOWLIST"
        date = f" · {r.published_date}" if r.published_date else ""
        print(f"  {flag} [{r.domain}] {r.title[:64]}{date}")
        print(f"      {r.url}")
        print(f"      {r.content[:160].strip()}…\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument(
        "--open-web",
        action="store_true",
        help="run the comparison query with the allowlist disabled",
    )
    args = parser.parse_args()

    try:
        rule(f"filtered · {args.query!r}")
        filtered = web_search(args.query, max_results=args.max_results)
        show(filtered, check_allowlist=True)
        print(f"  {len(filtered)} result(s), all from the {len(FINANCE_DOMAINS)}-domain "
              "allowlist.")
        print("  ✔ every URL above should be a source you would cite in research.")

        rule(f"the filter's job · {SPAM_MAGNET_QUERY!r}")
        # An unfiltered search here returns listicles and promotional content.
        # Filtered, it should return few results or none — and "none" is the
        # correct answer to this question from a research assistant.
        show(web_search(SPAM_MAGNET_QUERY, max_results=args.max_results), True)

        if args.open_web:
            rule(f"allowlist OFF · {SPAM_MAGNET_QUERY!r}")
            # include_domains=[] is the explicit opt-out. Compare the domains
            # against the block above: if they match, the filter is not working.
            show(
                web_search(
                    SPAM_MAGNET_QUERY,
                    max_results=args.max_results,
                    include_domains=[],
                ),
                check_allowlist=True,
            )
            print("  ✘-marked rows are what the allowlist keeps out of the agent.")
        else:
            print("\n  re-run with --open-web to see what the filter excludes.")

    except WebSearchError as exc:
        print(f"\n✘ {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

    print("\nB2 checks complete.\n")


if __name__ == "__main__":
    main()
