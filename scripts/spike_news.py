"""DISPOSABLE · Does a dedicated news tool still earn its place?

Not a gate. It prints measurements and makes no pass/fail claim — there are no
`verdict()` calls below on purpose. Precedent: the E1 router ablation harness was
disposable and was not kept. **Delete this file once the finding is written into
docs/phase-2-plan.md.**

## The question

Phase 2 capability 3 moves quotes, ratios and price history onto yfinance and
strips app/finnhub.py down. What is left of Finnhub is `get_company_news`, and
it is not obvious that a dedicated per-ticker news tool is worth a provider, an
API key and a rate limiter — `web_search(days=7)` already switches Tavily to its
news topic, and yfinance carries `Ticker.news` for free alongside the prices.

## The decision rule, fixed BEFORE the run

    Default: the dedicated news tool is REMOVED. It survives only if it CLEARLY
    beats web_search(days=7) on BOTH per-ticker recall (does it surface
    company-specific items the web search misses?) AND source quality
    (aggregator share, citation laundering). If it survives, the provider is
    whichever of Finnhub or yfinance scores better on source quality — with
    yfinance additionally required to return nothing, or raise, for a bogus
    ticker, never market-wide filler.

Fixed in advance and not to be adjusted after seeing output: the ticker set
below, AGGREGATORS, and the promotional-headline judgement — which is made by
reading the shuffled headline dump at the end, where the provider labels are
withheld until the key beneath it.

## Why these criteria

Straight out of app/websearch.py:61-71. That allowlist was not built from
reputation — each domain "was checked against the promotional query below and
returned nothing for it — that test, not reputation, is what earns a place
here." And nasdaq.com was *removed* for syndicating Motley Fool / InvestorPlace
listicles under its own hostname, because "a domain that launders spam through a
reputable hostname is worse than one that is obviously spam, because the
allowlist is what the agent trusts instead of judging sources."

`finance.yahoo.com` is structurally the same object. So criterion 1 is not
quality but **laundering**: does the URL a citation would carry name the
publisher, or the aggregator? A yfinance item whose citation reads
finance.yahoo.com for a Motley Fool listicle is the nasdaq.com failure wearing a
new hostname.

Finnhub is already known-bad on this axis — app/finnhub.py:48-56 records that
ten articles bought "two or three usable headlines". That is the bar. It is low.
"""

import argparse
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _gate import indent, note, rule  # noqa: E402

from app import finnhub, websearch  # noqa: E402

# Saturated / retail-attention magnet / mid-cap / thin micro-cap. PLTR is the
# direct analogue of the "best stocks to buy now for huge gains" probe that
# built the allowlist: it is where promotional content concentrates, and it is
# the ticker that will separate the three providers if anything does.
TICKERS = [("AAPL", "Apple"), ("PLTR", "Palantir"), ("CROX", "Crocs"), ("CULP", "Culp")]
BOGUS = "ZZQQNOTREAL"

# Fixed before the run. Publishers whose business model is volume commentary
# rather than reporting — the ones that took 4 of 8 slots on the promotional
# query in app/websearch.py:62-64.
AGGREGATORS = {
    "fool.com", "zacks.com", "investorplace.com", "benzinga.com",
    "simplywall.st", "gurufocus.com", "247wallst.com", "tipranks.com",
    "insidermonkey.com", "barchart.com", "stocktwits.com", "invezz.com",
}

PER_TICKER = 5
LOOKBACK_DAYS = 7


@dataclass
class Item:
    provider: str
    ticker: str
    title: str
    publisher: str
    url: str
    published: date | None
    # yfinance only: Yahoo's own flag for "this is our hosted copy".
    hosted: bool | None = None
    resolved: int | None = field(default=None)

    @property
    def url_host(self) -> str:
        host = (urlparse(self.url).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host

    @property
    def launders(self) -> bool:
        """True when the citation would name an aggregator, not the publisher.

        Compared loosely — "Barrons.com" vs barrons.com, "The Motley Fool" vs
        fool.com. The test is whether a reader clicking the citation lands on
        the named publisher's own site.
        """
        pub = self.publisher.lower().replace(" ", "").replace(".com", "")
        host = self.url_host.replace(".com", "").replace(".", "")
        return bool(pub) and pub not in host and host not in pub

    @property
    def is_aggregator(self) -> bool:
        pub = self.publisher.lower().replace(" ", "")
        return self.url_host in AGGREGATORS or any(
            a.replace(".com", "").replace(".", "") in pub for a in AGGREGATORS
        )


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


def from_finnhub(symbol: str, name: str) -> list[Item]:
    articles = finnhub.get_company_news(symbol, limit=PER_TICKER)
    return [
        Item("finnhub", symbol, a.headline, a.source, a.url, a.published_date)
        for a in articles
    ]


def from_yfinance(symbol: str, name: str) -> list[Item]:
    import yfinance as yf

    items: list[Item] = []
    for raw in (yf.Ticker(symbol).news or [])[:PER_TICKER]:
        c = raw.get("content") or {}
        # canonicalUrl is the publisher's own URL; clickThroughUrl is Yahoo's
        # hosted copy and is often null. Prefer the former — the whole point of
        # criterion 1 is which of the two a citation would carry.
        url = ((c.get("canonicalUrl") or {}).get("url")
               or (c.get("clickThroughUrl") or {}).get("url") or "")
        if not url or not c.get("title"):
            continue
        items.append(Item(
            "yfinance", symbol, c["title"],
            (c.get("provider") or {}).get("displayName", ""),
            url, _parse_date(c.get("pubDate")), hosted=bool(c.get("isHosted")),
        ))
    return items


def from_websearch(symbol: str, name: str) -> list[Item]:
    results = websearch.web_search(
        f"{name} ({symbol}) stock news", max_results=PER_TICKER, days=LOOKBACK_DAYS
    )
    return [
        Item("web_search", symbol, r.title, r.domain, r.url, _parse_date(r.published_date))
        for r in results
    ]


PROVIDERS = {"finnhub": from_finnhub, "yfinance": from_yfinance, "web_search": from_websearch}


def _parse_date(value) -> date | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip().replace("Z", "+00:00")
    for parse in (lambda t: datetime.fromisoformat(t).date(),
                  lambda t: datetime.strptime(t[:25], "%a, %d %b %Y %H:%M:%S").date()):
        try:
            return parse(text)
        except (ValueError, TypeError):
            continue
    return None


def resolve(items: list[Item]) -> None:
    """Does the URL a citation would carry actually open?

    Every citation in this system exists so that it resolves — get_quote returns
    no citation at all precisely because "a fabricated URL in a list whose entire
    value is that every entry resolves is worse than no entry."
    """
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    with httpx.Client(follow_redirects=True, timeout=15.0, headers=headers) as client:
        for item in items:
            try:
                item.resolved = client.get(item.url).status_code
            except Exception:
                item.resolved = 0


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def pct(numerator: int, denominator: int) -> str:
    return "  n/a" if not denominator else f"{100 * numerator / denominator:5.1f}%"


def report(items: list[Item]) -> None:
    by_provider: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_provider[item.provider].append(item)

    rule("1 · citation laundering — does the URL name the publisher or the aggregator?")
    note("the nasdaq.com test (app/websearch.py:66-71). lower is better.")
    print(f"\n  {'provider':<12} {'items':>6} {'laundered':>10} {'yahoo-hosted':>13}")
    for name, group in by_provider.items():
        hosted = sum(1 for i in group if i.hosted)
        hosted_s = pct(hosted, len(group)) if any(i.hosted is not None for i in group) else "   —"
        print(f"  {name:<12} {len(group):>6} {pct(sum(i.launders for i in group), len(group)):>10} {hosted_s:>13}")

    rule("2 · resolvability — does every citation open?")
    print(f"\n  {'provider':<12} {'200':>6} {'other':>6} {'failed':>7}")
    for name, group in by_provider.items():
        ok = sum(1 for i in group if i.resolved == 200)
        dead = sum(1 for i in group if not i.resolved)
        print(f"  {name:<12} {ok:>6} {len(group) - ok - dead:>6} {dead:>7}   ({pct(ok, len(group))} clean)")
        for bad in (i for i in group if i.resolved != 200):
            print(indent(f"{bad.resolved or 'ERR'} {bad.url[:96]}"))

    rule("3 · per-ticker recall — what does each provider surface alone?")
    note("the criterion the KEEP branch turns on: a dedicated tool must find")
    note("company-specific items web_search misses, not just find something.")
    for symbol, _ in TICKERS:
        print(f"\n  {symbol}")
        seen: dict[str, set[str]] = {
            n: {i.url_host + urlparse(i.url).path[:40] for i in g if i.ticker == symbol}
            for n, g in by_provider.items()
        }
        for name, urls in seen.items():
            others = set().union(*(v for k, v in seen.items() if k != name)) or set()
            unique = urls - others
            print(f"    {name:<12} {len(urls):>2} items, {len(unique):>2} unique to it")
            for item in (i for i in by_provider[name] if i.ticker == symbol):
                mark = "*" if item.url_host + urlparse(item.url).path[:40] in unique else " "
                print(indent(f"{mark} [{item.publisher[:22]:<22}] {item.title[:70]}"))

    rule("4 · aggregator / promotional share")
    note(f"fixed list, {len(AGGREGATORS)} publishers, chosen before the run.")
    print(f"\n  {'provider':<12} {'aggregator share':>18}")
    for name, group in by_provider.items():
        print(f"  {name:<12} {pct(sum(i.is_aggregator for i in group), len(group)):>18}")

    rule("5 · datedness — can the agent date what it cites?")
    print(f"\n  {'provider':<12} {'dated':>8} {'median age':>12}")
    for name, group in by_provider.items():
        ages = [(date.today() - i.published).days for i in group if i.published]
        median = f"{statistics.median(ages):.0f}d" if ages else "—"
        print(f"  {name:<12} {pct(len(ages), len(group)):>8} {median:>12}")


def check_bogus() -> None:
    rule(f"6 · the bogus ticker ({BOGUS})")
    note("[] or a raise is fine. MARKET-WIDE FILLER IS A HARD FAIL — that is the")
    note("$0.00 quote in news form: plausible content confidently attached to a")
    note("company that does not exist.")
    print()
    for name, fetch in PROVIDERS.items():
        try:
            got = fetch(BOGUS, "Nonexistent Holdings")
        except Exception as exc:
            print(f"  {name:<12} raised {type(exc).__name__} — safe")
            print(indent(str(exc)[:110]))
            continue
        if not got:
            print(f"  {name:<12} returned [] — safe")
        else:
            print(f"  {name:<12} *** returned {len(got)} articles for a company that does not exist ***")
            for item in got:
                print(indent(f"[{item.publisher[:22]:<22}] {item.title[:70]}"))


def blind_headlines(items: list[Item]) -> None:
    rule("7 · blind headline read — judge these BEFORE looking at the key")
    note("promotional shape: '3 Reasons to Buy...', 'Should You Buy X Now?',")
    note("'...Is a Screaming Buy'. Mark them, THEN read the key below.")
    shuffled = items[:]
    random.shuffle(shuffled)
    print()
    for n, item in enumerate(shuffled, start=1):
        print(f"  {n:>3}. [{item.ticker}] {item.title[:96]}")
    print("\n  ── key ──")
    for n, item in enumerate(shuffled, start=1):
        print(f"  {n:>3}. {item.provider:<11} {item.publisher[:28]:<28} {item.url_host}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260820, help="shuffle seed, for a reproducible blind read")
    args = parser.parse_args()
    random.seed(args.seed)

    rule("news provider spike — DISPOSABLE, delete after the finding is recorded")
    note(f"tickers: {', '.join(s for s, _ in TICKERS)} + {BOGUS}")
    note(f"{PER_TICKER} items each, {LOOKBACK_DAYS}-day lookback")
    note("DEFAULT IS REMOVAL. A dedicated news tool must beat web_search on BOTH")
    note("recall AND source quality to survive. A tie is removal.")

    items: list[Item] = []
    for symbol, name in TICKERS:
        for provider, fetch in PROVIDERS.items():
            try:
                items.extend(fetch(symbol, name))
            except Exception as exc:
                print(f"  !! {provider} failed for {symbol}: {type(exc).__name__}: {exc}")

    if not items:
        print("\nno items from any provider — nothing to measure.")
        return 2

    resolve(items)
    report(items)
    check_bogus()
    blind_headlines(items)

    rule("done")
    note("record the finding in docs/phase-2-plan.md, then DELETE this file.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        finnhub.close_client()
