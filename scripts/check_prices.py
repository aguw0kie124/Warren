"""Gate for B1 · app/prices.py — market data on yfinance.

Four things, all assertable, so this is a real gate rather than an eyeball one.
Costs nothing: yfinance needs no API key and no database.

    python scripts/check_prices.py
    python scripts/check_prices.py --symbol MSFT --loss-maker LCID
    python scripts/check_prices.py --census      # dump raw info keys

`--census` is the one-time reconnaissance that resolves a units question rather
than asserting anything: it prints Yahoo's raw keys beside the values this
module derived from them. That is how `dividendYield` was established to be a
percent while `trailingAnnualDividendYield` is a fraction — by reading what the
field returns for a known payer, not by trusting what it is called.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _gate import exit_code, indent, note, rule, summary, verdict  # noqa: E402

from app import prices  # noqa: E402
from app.prices import (  # noqa: E402
    MAX_QUOTE_STALENESS_DAYS,
    UnknownSymbolError,
    get_key_stats,
    get_price_history,
    get_quote,
)

BOGUS = "ZZQQNOTREAL"


def check_quote(symbol: str) -> None:
    rule(f"1 · {symbol} quotes fresh and non-zero")
    quote = get_quote(symbol)
    print(indent(f"{quote.price:,.2f} on {quote.as_of}"))

    verdict(quote.price > 0, f"{symbol} quotes a non-zero price ({quote.price:,.2f})")
    verdict(quote.low <= quote.price <= quote.high, "the price sits inside its own session range")

    age = (date.today() - quote.session_date).days
    verdict(age <= MAX_QUOTE_STALENESS_DAYS,
            f"the session date is within {MAX_QUOTE_STALENESS_DAYS} days ({quote.session_date}, {age}d ago)")
    # Deliberately redundant beside the staleness check above. This is the
    # invariant by name, so it survives someone widening that window.
    verdict(quote.session_date.year >= date.today().year - 1,
            "the quote is not dated 1970 — the fabrication guard holds")
    note("compare the price against a browser before believing this gate")


def check_unknown_symbol() -> None:
    rule(f"2 · an unknown symbol raises rather than returning $0.00 / 1970")
    for label, call in (
        ("get_quote", lambda: get_quote(BOGUS)),
        ("get_key_stats", lambda: get_key_stats(BOGUS)),
        ("get_price_history", lambda: get_price_history(BOGUS)),
    ):
        try:
            result = call()
        except UnknownSymbolError:
            verdict(True, f"{label}({BOGUS}) raises UnknownSymbolError")
        except Exception as exc:
            verdict(False, f"{label}({BOGUS}) raises UnknownSymbolError, not {type(exc).__name__}: {exc}")
        else:
            # Printing what came back is the point: this is what a regression to
            # the $0.00/1970 quote actually looks like on the terminal.
            verdict(False, f"{label}({BOGUS}) raises rather than returning a value")
            print(indent(repr(result)[:200]))
    note("yfinance logs an HTTP 404 line for each of these — that is expected,")
    note("and it is NOT the signal. UnknownSymbolError is.")


def check_history(symbol: str) -> None:
    rule(f"3 · {symbol} 1y summary is internally consistent")
    summary_ = get_price_history(symbol, period="1y")
    print(indent(f"{summary_.start_date} -> {summary_.end_date}, "
                 f"{summary_.percent_change:+.2f}%, {summary_.sessions} sessions"))
    for day, close in summary_.anchors:
        print(indent(f"  {day}  {close:,.2f}"))

    # Recomputed from a SECOND, independent fetch rather than from the summary's
    # own fields — so this also proves the summary is not serving a stale frame.
    closes = prices._fetch("history", symbol, period="1y", interval="1d",
                           auto_adjust=True, actions=False)["Close"].dropna()
    by_hand = (float(closes.iloc[-1]) - float(closes.iloc[0])) / float(closes.iloc[0]) * 100

    verdict(abs(summary_.percent_change - by_hand) < 1e-6,
            f"pct_change matches first/last close computed by hand "
            f"({summary_.percent_change:+.4f}% vs {by_hand:+.4f}%)")
    verdict(summary_.anchors[0][1] == summary_.start_close
            and summary_.anchors[-1][1] == summary_.end_close,
            "the first and last anchors ARE the period's first and last close")
    verdict(all(summary_.low <= c <= summary_.high for _, c in summary_.anchors),
            "every anchor sits inside the reported high/low")
    verdict([d for d, _ in summary_.anchors] == sorted(d for d, _ in summary_.anchors),
            f"the {len(summary_.anchors)} anchors are in date order")
    verdict(summary_.realised_vol_pct is not None and 0 < summary_.realised_vol_pct < 200,
            f"annualised volatility is plausible ({summary_.realised_vol_pct:.1f}%)")


def check_key_stats(symbol: str, loss_maker: str) -> None:
    rule(f"4 · {symbol} has a plausible P/E; {loss_maker} has none")
    stats = get_key_stats(symbol)
    for name, value in stats.present().items():
        print(indent(f"{name:<24} {value:,.3f}"))

    verdict(stats.pe_ratio is not None and 3 < stats.pe_ratio < 200,
            f"{symbol} P/E is plausible ({stats.pe_ratio})")
    verdict(None not in stats.present().values(), "present() carries no Nones")

    loss = get_key_stats(loss_maker)
    print(indent(f"{loss_maker} eps_ttm={loss.eps_ttm} pe_ratio={loss.pe_ratio!r} "
                 f"forward_pe={loss.forward_pe!r}"))
    if loss.eps_ttm is not None and loss.eps_ttm > 0:
        # A gate that fails because a company started making money is measuring
        # the wrong thing.
        note(f"{loss_maker} now reports positive trailing EPS ({loss.eps_ttm}) — "
             f"pick another --loss-maker. Not judged.")
        return
    verdict(loss.pe_ratio is None, f"{loss_maker} has no earnings and no P/E (got {loss.pe_ratio!r})")
    verdict("pe_ratio" not in loss.present(), "the absent P/E is omitted, not rendered")


def census(symbol: str) -> None:
    rule(f"census · raw Yahoo info keys for {symbol}")
    info = prices._fetch("info", symbol)
    note(f"{len(info)} keys returned")
    for key in ("dividendYield", "trailingAnnualDividendYield", "trailingPE",
                "forwardPE", "returnOnEquity", "profitMargins", "revenueGrowth"):
        print(indent(f"{key:<32} {info.get(key, '<ABSENT>')}"))
    note("dividendYield is a PERCENT; trailingAnnualDividendYield is a FRACTION.")
    note("They must never share a fallback tuple — see _INFO_KEYS.")
    print(indent(f"all keys: {sorted(info)}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--loss-maker", default="RIVN",
                        help="a company with negative trailing EPS, so it has no P/E")
    parser.add_argument("--census", action="store_true", help="dump raw info keys and exit")
    args = parser.parse_args()

    try:
        if args.census:
            census(args.symbol)
            return 0
        check_quote(args.symbol)
        check_unknown_symbol()
        check_history(args.symbol)
        check_key_stats(args.symbol, args.loss_maker)
    except prices.PriceDataError as exc:
        print(f"\nsetup problem: {exc}")
        return 2

    summary(on_failure="market data is not trustworthy yet — fix before wiring the tools",
            on_success="quotes, key stats and history are consistent and the "
                       "unknown-symbol guard holds")
    return exit_code()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        prices.close_client()
