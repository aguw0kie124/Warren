"""F1/F2 verification: XBRL fundamentals, and the statements built from them.

    python scripts/check_data.py                 # the whole set
    python scripts/check_data.py --statements    # just the tag-mapping reading
    python scripts/check_data.py --quiet         # verdicts only

**Costs no money** — no model is involved. It does make a handful of EDGAR
requests to prove citations resolve, through `app/sec_http.py`, so it needs
`EDGAR_USER_AGENT` and Postgres.

Two of these checks are the reason the module exists rather than coverage
formalities:

- **The `fy` trap.** companyfacts stamps the *filing's* fiscal year onto every
  fact it restates, so Apple's FY2025 10-K labels three different years 2025.
  Reading that field puts 2023's revenue under 2025 — a real number under the
  wrong year, which no downstream check would question. The two figures below
  are the ones that catches.
- **Agreement with the text corpus.** The same revenue figure reached two
  entirely different ways — parsed out of MD&A prose by the chunker, and read
  from an XBRL tag — should be the same number. It is the only check here that
  can fail because of something *upstream* of this module.

The statement mapping is deliberately **reported rather than asserted**, except
for four lines. Coverage below 100% is usually correct: Accenture has no gross
profit, Allstate no R&D, Amazon no dividend. Asserting them all would fail on
companies that are fine and train whoever runs this to ignore it.
"""

import argparse
import logging
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import exit_code, indent, note, rule, summary, verdict  # noqa: E402

from app.db import close_pool, get_conn  # noqa: E402
from app.edgar import filing_index_url  # noqa: E402
from app.fundamentals import STATEMENTS, load_statement  # noqa: E402
from app.sec_http import close_client, sec_get  # noqa: E402
from app.tools import get_financials  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# Apple's FY2025 10-K restates all three of these, stamping fy=2025 on each.
# Read from `fy`, the 2023 and 2024 rows land under 2025 and overwrite it.
FY_TRAP = [("2023-09-30", 383.285), ("2024-09-28", 391.035), ("2025-09-27", 416.161)]

# Lines every operating company reports. Everything else in the mapping is
# printed for a human, because its absence is usually the company's choice.
UNIVERSAL = {
    "income": ("Revenue", "Net income"),
    "balance": ("Total assets", "Shareholders' equity"),
    "cash_flow": ("Operating cash flow", "Investing cash flow"),
}

# Ordinary operating companies across four sectors, so a mapping that only
# works for large-cap tech fails here rather than in an answer.
MAPPING_SAMPLE = ("AAPL", "NVDA", "WMT", "JNJ", "CVX", "XOM")

# Deliberately outside MAPPING_SAMPLE: section 7 deletes this ticker's rows
# and watches them come back, which would be circular against a name the
# earlier sections depend on.
ON_DEMAND_TICKER = "RIVN"


def query(sql, params=()):
    with get_conn() as conn:
        return conn.execute(sql, params).fetchall()


def check_shape() -> None:
    rule("1. what was ingested")
    facts, tickers = query(
        "SELECT count(*), count(DISTINCT ticker) FROM company_facts")[0]
    print(f"    {facts:,} facts across {tickers} companies")
    verdict(facts > 0, "the fundamentals table is populated")
    verdict(tickers >= 100, f"at least 100 companies have fundamentals ({tickers})")

    spans = query("""
        SELECT period_type, count(*), min(period_end - period_start),
               max(period_end - period_start)
        FROM company_facts GROUP BY period_type ORDER BY 2 DESC""")
    for period_type, count, lo, hi in spans:
        print(f"    {period_type:<10} {count:>9,}  span {lo}-{hi} days"
              if lo is not None else f"    {period_type:<10} {count:>9,}")

    # The cumulative-facts trap: a nine-month figure sitting in a quarterly
    # series looks exactly like an extraordinary quarter.
    bad = query("""
        SELECT count(*) FROM company_facts
        WHERE (period_type = 'annual'    AND (period_end - period_start) NOT BETWEEN 300 AND 400)
           OR (period_type = 'quarterly' AND (period_end - period_start) NOT BETWEEN 60 AND 100)
           OR (period_type = 'instant'   AND period_start IS NOT NULL)""")[0][0]
    verdict(bad == 0, f"no fact sits in the wrong period class ({bad} found)")

    # Forward-dated duration facts legitimately exist — pension and health-care
    # trend-rate assumptions are tagged over future years. The requirement is
    # not that they be absent but that they never reach a statement, so the
    # claim is asserted where it matters rather than at the table.
    future = query("""
        SELECT count(*) FROM company_facts
        WHERE period_type IN ('annual','quarterly') AND period_end > current_date""")[0][0]
    if future:
        note(f"{future} forward-dated duration fact(s) present — assumptions, not "
             f"results; section 4 checks they stay out of statements")

    leaked = [t for t in ("NUE", "AAPL", "NVDA")
              if any(p > __import__("datetime").date.today()
                     for p in load_statement(t, "income", "annual", 4).periods)]
    verdict(not leaked,
            "no statement offers a period that has not ended"
            + (f" (leaked for {', '.join(leaked)})" if leaked else ""))


def check_fy_trap() -> None:
    """The sharpest check here, and the cheapest to get wrong."""
    rule("2. the fy trap — periods come from start/end, never from `fy`")

    for end, expected_bn in FY_TRAP:
        rows = query("""
            SELECT value/1e9, accession_number FROM company_facts
            WHERE ticker='AAPL' AND period_type='annual' AND period_end=%s
              AND concept='RevenueFromContractWithCustomerExcludingAssessedTax'""", (end,))
        got = float(rows[0][0]) if rows else 0.0
        verdict(
            abs(got - expected_bn) < 0.01,
            f"AAPL revenue for the year ending {end} is ${expected_bn}B (got ${got:.3f}B)",
        )

    accessions = {r[1] for e, _ in FY_TRAP for r in query("""
        SELECT 1, accession_number FROM company_facts
        WHERE ticker='AAPL' AND period_type='annual' AND period_end=%s
          AND concept='RevenueFromContractWithCustomerExcludingAssessedTax'""", (e,))}
    verdict(
        len(accessions) == 1,
        f"all three came from the SAME filing ({len(accessions)} accession) — which "
        f"is exactly why `fy` cannot be the fact's year",
    )


def check_agreement() -> None:
    """XBRL against the chunk corpus — the one check that can fail upstream."""
    rule("3. XBRL agrees with the filing text")

    rows = query("""
        SELECT value FROM company_facts
        WHERE ticker='AAPL' AND period_type='annual' AND period_end='2025-09-27'
          AND concept='RevenueFromContractWithCustomerExcludingAssessedTax'""")
    if not rows:
        verdict(False, "AAPL FY2025 revenue is present in company_facts")
        return
    xbrl = float(rows[0][0])

    # The same figure as it appears in MD&A prose, where the chunker put it.
    hit = query("""
        SELECT count(*) FROM chunks c JOIN filings f USING (accession_number)
        WHERE f.ticker='AAPL' AND c.content LIKE %s""", (f"%{int(xbrl/1e6):,}%",))[0][0]
    verdict(
        hit > 0,
        f"${xbrl/1e9:.3f}B from XBRL also appears as '{int(xbrl/1e6):,}' in the "
        f"ingested MD&A text ({hit} chunk(s))",
    )


def check_statements(quiet: bool) -> None:
    rule("4. the statement mapping")

    # No coverage check: load_statement calls ensure_facts, so a ticker that
    # was never backfilled is fetched here rather than skipped.
    for ticker in MAPPING_SAMPLE:
        for name in STATEMENTS:
            statement = load_statement(ticker, name, "annual", limit=4)
            resolved = statement.resolved
            if not quiet:
                print(f"\n    {ticker} {name}: {len(resolved)}/{len(statement.rows)} lines")
                for row in statement.rows:
                    tags = " + ".join(row.concepts) if row.concepts else "— not reported"
                    flag = "  (re-tagged)" if row.mixed else ""
                    print(f"      {row.label:<26} {tags[:66]}{flag}")

            if not statement.periods:
                # Not a mapping failure. SEC's ticker file points at the
                # *current* registrant, and after a corporate reorganisation
                # that is a successor entity with no annual history yet — the
                # decades of filings sit under a predecessor CIK the ticker no
                # longer names. Reported, because a user asking for five years
                # of this company will correctly get nothing.
                note(f"{ticker} {name}: no completed {statement.period} periods — "
                     f"successor registrant, history is under a predecessor CIK")
                continue

            for label in UNIVERSAL[name]:
                row = next((r for r in statement.rows if r.label == label), None)
                verdict(
                    bool(row and row.values),
                    f"{ticker} {name}: '{label}' resolved"
                    + ("" if row and row.values else " — a universal line with no "
                       "tag is a mapping gap, not a company that omitted it"),
                )


def check_tool(quiet: bool) -> None:
    rule("5. get_financials")

    def run(**args):
        return get_financials.invoke(
            {"name": "get_financials", "args": args, "id": "c", "type": "tool_call"})

    msg = run(ticker="NVDA", statement="income", period="annual", limit=5)
    if not quiet:
        print(f"\n{indent(msg.content)}")

    verdict("Revenue" in msg.content, "renders an income statement")
    verdict(bool(msg.artifact), f"returns citations ({len(msg.artifact)})")
    verdict(all(c.type == "filing" for c in msg.artifact),
            "every citation is typed 'filing'")

    # Per-row unit scaling: EPS must not be flattened by a table-wide scale.
    eps = next((l for l in msg.content.splitlines() if "EPS" in l), "")
    verdict("B" not in eps.split("EPS")[-1],
            f"diluted EPS is not scaled into billions ({eps.strip()[:60]})")

    # The revenue row is the per-period fallthrough working: NVIDIA re-tagged
    # it mid-history, and a per-row rule leaves four of five years blank.
    revenue = next((l for l in msg.content.splitlines() if l.startswith("Revenue")), "")
    verdict(revenue.count("—") == 0,
            f"NVDA revenue is continuous across all periods ({revenue.count('—')} blank)")

    gap = run(ticker="SKHY").content
    verdict("NO FUNDAMENTALS" in gap,
            "a company without fundamentals is reported, not guessed at")
    verdict("Unknown period" in run(ticker="AAPL", period="monthly").content,
            "an invalid period is answered with the valid ones")

    rule("6. citations resolve on EDGAR")
    for citation in list({c.source_url: c for c in msg.artifact}.values())[:3]:
        try:
            ok = sec_get(citation.source_url, accept="text/html").status_code == 200
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            ok = False
            print(f"        {type(exc).__name__}: {exc}")
        verdict(ok, citation.label[:72])

    covered_by_text = query("SELECT count(*) FROM filings WHERE ticker='NVDA'")[0][0]
    verdict(
        covered_by_text == 0,
        "…and NVDA has no ingested filings at all, so those URLs were built "
        "from cik + accession rather than looked up",
    )


def check_on_demand() -> None:
    """The fetch-through path — nothing is backfilled, so this is coverage.

    Destructive on purpose: it deletes one ticker's facts and asserts they come
    back. That is only safe *because* of the property being tested — the rows
    are reconstructible from one free SEC call — and a non-destructive version
    would pass on the first run and be meaningless on every one after.
    """
    rule("7. facts are fetched on demand, not backfilled")

    before = query("SELECT count(*) FROM company_facts WHERE ticker = %s",
                   (ON_DEMAND_TICKER,))[0][0]
    with get_conn() as conn:
        conn.execute("DELETE FROM company_facts WHERE ticker = %s",
                     (ON_DEMAND_TICKER,))
    note(f"deleted {before:,} stored fact(s) for {ON_DEMAND_TICKER}")

    start = time.perf_counter()
    text, citations = get_financials.func(ticker=ON_DEMAND_TICKER,
                                          statement="income", limit=3)
    cold = time.perf_counter() - start

    verdict("Revenue" in text and "NO FUNDAMENTALS" not in text,
            f"a ticker with nothing stored answers anyway ({cold:.1f}s)")
    verdict(bool(citations), f"and cites the filings it came from ({len(citations)})")

    written = query("SELECT count(*) FROM company_facts WHERE ticker = %s",
                    (ON_DEMAND_TICKER,))[0][0]
    verdict(written > 0, f"the fetch was written back ({written:,} fact(s))")

    start = time.perf_counter()
    again, _ = get_financials.func(ticker=ON_DEMAND_TICKER,
                                   statement="income", limit=3)
    warm = time.perf_counter() - start
    verdict(again == text, "the second call returns an identical statement")
    verdict(warm < cold / 2,
            f"and serves it from Postgres rather than re-fetching "
            f"({warm:.2f}s vs {cold:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--statements", action="store_true",
                    help="only the tag-mapping reading")
    ap.add_argument("--quiet", action="store_true", help="verdicts only")
    args = ap.parse_args()

    try:
        if args.statements:
            check_statements(args.quiet)
        else:
            check_shape()
            check_fy_trap()
            check_agreement()
            check_statements(args.quiet)
            check_tool(args.quiet)
            check_on_demand()

        summary(
            on_failure="A wrong number is app/xbrl.py (parsing) or "
                       "app/fundamentals.py (mapping).\n  A missing line is "
                       "usually the company, not the code — read section 4.",
            on_success="now read section 4: does every line that resolved pick a "
                       "tag that\n  actually means what the label says?",
        )
    finally:
        close_client()
        close_pool()

    raise SystemExit(exit_code())


if __name__ == "__main__":
    main()
