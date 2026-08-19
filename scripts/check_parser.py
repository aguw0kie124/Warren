"""A4 verification: split cached filings into sections and eyeball the result.

    python scripts/check_parser.py                 # every file in data/raw/
    python scripts/check_parser.py --show "Item 1A Risk Factors"

Reads only from data/raw/ — no network. Run scripts/check_edgar.py first to
populate it.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import exit_code, rule, summary, verdict  # noqa: E402

from app.config import settings  # noqa: E402
from app.parser import SECTION_UNKNOWN, parse_sections  # noqa: E402

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

    for path in files:
        meta = NAME_RE.match(path.name)
        if not meta:
            print(f"\n{path.name}: unrecognized filename, skipping")
            continue
        form = meta["form"]

        rule(path.name)
        sections = parse_sections(path.read_text(errors="replace"), form)

        if not verdict(SECTION_UNKNOWN not in sections,
                        "sections identified (no whole-document fallback)"):
            continue

        for key, body in sections.items():
            floor = EXPECT_MIN_CHARS.get(key, 0)
            # 10-Q risk factors are legitimately short ("no material changes").
            if form == "10-Q" and key == "Item 1A Risk Factors":
                floor = 0
            verdict(len(body) >= floor, f"{key:<24} {len(body):>8,} chars")

            first = body.split("\n", 1)[0]
            print(f"        heading: {first[:70]!r}")

        if args.show and args.show in sections:
            print(f"\n  --- {args.show}, first {args.chars} chars ---")
            print("  " + sections[args.show][: args.chars].replace("\n", "\n  "))

    summary()
    raise SystemExit(exit_code())


if __name__ == "__main__":
    main()
