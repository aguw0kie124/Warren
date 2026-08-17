"""A4 verification: split cached filings into sections and eyeball the result.

    python scripts/check_parser.py                 # every file in data/raw/
    python scripts/check_parser.py --show "Item 1A Risk Factors"

Reads only from data/raw/ — no network. Run scripts/check_edgar.py first to
populate it.
"""

import argparse
import logging
import re
from pathlib import Path

from app.config import settings
from app.parser import SECTION_UNKNOWN, parse_sections

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# data/raw filenames look like: AAPL_10-K_2025_0000320193-25-000079.html
NAME_RE = re.compile(r"^(?P<ticker>[^_]+)_(?P<form>10-[KQ])_(?P<fy>\d{4})_")

# A parsed section must read as prose, not as a table-of-contents fragment.
EXPECT_MIN_CHARS = {"Item 1A Risk Factors": 10_000, "Item 7 MD&A": 5_000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="print the opening of this section")
    ap.add_argument("--chars", type=int, default=400)
    args = ap.parse_args()

    files = sorted(settings.raw_dir.glob("*.html"))
    if not files:
        print(f"No filings in {settings.raw_dir}. Run scripts/check_edgar.py first.")
        return

    problems = 0
    for path in files:
        meta = NAME_RE.match(path.name)
        if not meta:
            print(f"\n{path.name}: unrecognized filename, skipping")
            continue
        form = meta["form"]

        print(f"\n=== {path.name} ===")
        sections = parse_sections(path.read_text(errors="replace"), form)

        if SECTION_UNKNOWN in sections:
            problems += 1
            print("  [BAD] fell back to whole-document (no sections identified)")
            continue

        for key, body in sections.items():
            floor = EXPECT_MIN_CHARS.get(key, 0)
            # 10-Q risk factors are legitimately short ("no material changes").
            if form == "10-Q" and key == "Item 1A Risk Factors":
                floor = 0
            ok = len(body) >= floor
            problems += not ok
            print(f"  [{'ok ' if ok else 'BAD'}] {key:<24} {len(body):>8,} chars")

            first = body.split("\n", 1)[0]
            print(f"        heading: {first[:70]!r}")

        if args.show and args.show in sections:
            print(f"\n  --- {args.show}, first {args.chars} chars ---")
            print("  " + sections[args.show][: args.chars].replace("\n", "\n  "))

    print("\nA4:", "PASS" if problems == 0 else f"FAIL ({problems} problem(s))")


if __name__ == "__main__":
    main()
