"""Offline tests for statement assembly.

No database. `load_statement` reads Postgres and its correctness against real
filers is `scripts/check_data.py`'s question; what is testable here is the two
pieces of pure logic between the rows and the table, both of which fail by
producing a *plausible* statement rather than an error:

- **`_resolve_line`** turns a line's candidate tags into one series. The
  obvious rule — first candidate with any data — renders NVIDIA with four
  blank revenue years under a full gross-profit row, because NVIDIA re-tagged
  revenue partway through its history. That is a wrong statement, not a
  crash.
- **`_format_value`** scales by each row's own unit. A single table-wide scale
  prints $4.90 diluted EPS as `0.00B` beside revenue in billions, which reads
  as a company that earned nothing.
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.fundamentals import StatementRow, _resolve_line
from app.tools import _format_value

P2025, P2024, P2023 = date(2025, 9, 27), date(2024, 9, 28), date(2023, 9, 30)
PERIODS = [P2025, P2024, P2023]


def facts(**by_concept):
    """{concept: {period: value}} -> the nested shape _resolve_line consumes."""
    return {
        concept: {"USD": {p: Decimal(str(v)) for p, v in periods.items()}}
        for concept, periods in by_concept.items()
    }


# --- resolving a line -------------------------------------------------------


def test_a_single_tag_covering_every_period():
    row = _resolve_line("Revenue", ("Revenues",),
                        facts(Revenues={P2025: 3, P2024: 2, P2023: 1}), PERIODS)

    assert row.concepts == ("Revenues",)
    assert row.values == {P2025: Decimal("3"), P2024: Decimal("2"), P2023: Decimal("1")}
    assert not row.mixed


def test_a_retagged_line_reassembles_across_candidates():
    """The NVIDIA case, and the reason this function exists. The preferred tag
    covers only the oldest period; the fallback covers the rest."""
    row = _resolve_line(
        "Revenue",
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"),
        facts(
            RevenueFromContractWithCustomerExcludingAssessedTax={P2023: 1},
            Revenues={P2025: 3, P2024: 2, P2023: 99},
        ),
        PERIODS,
    )

    assert set(row.values) == set(PERIODS)
    assert row.mixed
    # Preference order still wins the period both tags cover.
    assert row.values[P2023] == Decimal("1")


def test_first_with_any_data_would_have_been_wrong():
    """Pinned as the regression it is: under the old rule this row had one
    value and three blanks, beside a complete row below it."""
    data = facts(Preferred={P2023: 1}, Fallback={P2025: 3, P2024: 2})

    row = _resolve_line("Revenue", ("Preferred", "Fallback"), data, PERIODS)

    assert len(row.values) == 3


def test_preference_order_decides_an_overlap():
    row = _resolve_line("Revenue", ("First", "Second"),
                        facts(First={P2025: 10}, Second={P2025: 20}), PERIODS)

    assert row.values[P2025] == Decimal("10")
    assert row.concepts == ("First",)


def test_a_candidate_contributing_nothing_is_not_recorded():
    """`concepts` is what actually filled the row, so the gate's "which tag
    matched" reading means something."""
    row = _resolve_line("Revenue", ("Empty", "Real"),
                        facts(Real={P2025: 1}), PERIODS)

    assert row.concepts == ("Real",)


def test_no_candidate_with_data_leaves_the_line_blank():
    row = _resolve_line("Gross profit", ("GrossProfit",), {}, PERIODS)

    assert row.concepts == ()
    assert row.concept is None
    assert row.values == {}
    assert row.unit is None


def test_units_are_never_mixed_within_a_row():
    """A dollars-per-share figure filling a gap in a dollars row would be
    invisible in the table and wrong by nine orders of magnitude."""
    data = {
        "Dollars": {"USD": {P2025: Decimal("1000")}},
        "PerShare": {"USD/shares": {P2024: Decimal("2.5")}},
    }

    row = _resolve_line("Something", ("Dollars", "PerShare"), data, PERIODS)

    assert row.unit == "USD"
    assert P2024 not in row.values
    assert row.concepts == ("Dollars",)


def test_resolution_stops_once_every_period_is_filled():
    data = facts(First={P2025: 1, P2024: 2, P2023: 3}, Second={P2025: 9})

    row = _resolve_line("Revenue", ("First", "Second"), data, PERIODS)

    assert row.concepts == ("First",)


def test_a_concept_reported_in_several_units_picks_by_priority():
    data = {"X": {"shares": {P2025: Decimal("5")}, "USD": {P2025: Decimal("7")}}}

    row = _resolve_line("X", ("X",), data, PERIODS)

    assert row.unit == "USD"
    assert row.values[P2025] == Decimal("7")


# --- formatting -------------------------------------------------------------


@pytest.mark.parametrize("value,unit,expected", [
    (Decimal("416161000000"), "USD", "416.16B"),
    (Decimal("-187000000"),   "USD", "-187.00M"),
    (Decimal("54321"),        "USD", "54.32K"),
    (Decimal("412"),          "USD", "412.00"),
    (Decimal("6.08"),         "USD/shares", "6.08"),
    (Decimal("15000000000"),  "shares", "15.00B"),
    (Decimal("1.75"),         "pure", "1.75"),
])
def test_each_unit_scales_on_its_own(value, unit, expected):
    assert _format_value(value, unit) == expected


def test_earnings_per_share_is_not_scaled_into_nothing():
    """The bug this rule exists for: one table-wide scale renders $4.90 as
    0.00B beside revenue in billions."""
    assert _format_value(Decimal("4.90"), "USD/shares") == "4.90"
    assert _format_value(Decimal("4.90"), "USD") != "4.90B"


# --- the row's own reporting ------------------------------------------------


def test_mixed_is_true_only_when_more_than_one_tag_contributed():
    assert not StatementRow("x", ("A",), "USD", {}).mixed
    assert StatementRow("x", ("A", "B"), "USD", {}).mixed
    assert not StatementRow("x", (), None, {}).mixed


# --- fetching on demand -----------------------------------------------------
#
# `ensure_facts` is what replaced the backfill: nothing is pre-loaded, so a
# ticker is fetched the first time it is asked for and written back. Every
# failure mode here is silent by nature — a fetch that should have happened and
# didn't looks exactly like a company with no data, and a fetch that happens on
# every call just makes the tool slow. Both need pinning.
#
# Offline like the rest of the file: `get_conn` is faked, and the three
# functions `ensure_facts` imports lazily are patched at their source modules,
# which is where a function-local import resolves them.


class _FakeConn:
    def __init__(self, newest, calls):
        self._newest, self._calls = newest, calls

    def execute(self, sql, params=None):
        self._calls.append(sql)
        return self

    def fetchone(self):
        return (self._newest,)


def _patch_db(monkeypatch, newest):
    """Pretend company_facts holds one ticker whose newest fact was filed then."""
    import contextlib

    calls: list[str] = []

    @contextlib.contextmanager
    def fake_get_conn():
        yield _FakeConn(newest, calls)

    monkeypatch.setattr("app.fundamentals.get_conn", fake_get_conn)
    return calls


def _patch_sec(monkeypatch, fetched, *, raises=None, resolves=True):
    """Count SEC fetches and capture what got written."""
    import types

    state = {"fetches": 0, "written": []}

    def fake_fetch(cik):
        state["fetches"] += 1
        if raises is not None:
            raise raises
        return {"cik": cik}

    monkeypatch.setattr("app.xbrl.fetch_company_facts", fake_fetch)
    monkeypatch.setattr("app.xbrl.parse_facts", lambda payload, ticker: list(fetched))
    monkeypatch.setattr(
        "app.tickers.try_resolve_ticker",
        lambda t: types.SimpleNamespace(cik="0000320193") if resolves else None,
    )
    monkeypatch.setattr(
        "app.store.upsert_facts",
        lambda conn, facts: state["written"].extend(facts) or len(facts),
    )
    return state


def test_a_ticker_with_no_stored_facts_is_fetched_and_written(monkeypatch):
    from app.fundamentals import ensure_facts

    _patch_db(monkeypatch, newest=None)
    state = _patch_sec(monkeypatch, fetched=["f1", "f2"])

    assert ensure_facts("NVDA") is True
    assert state["fetches"] == 1
    assert state["written"] == ["f1", "f2"]


def test_recently_filed_facts_are_served_without_touching_sec(monkeypatch):
    """The warm-start half. Without this the tool re-downloads ~3.8 MB per call."""
    from app.fundamentals import ensure_facts

    _patch_db(monkeypatch, newest=date.today() - timedelta(days=10))
    state = _patch_sec(monkeypatch, fetched=["f1"])

    assert ensure_facts("AAPL") is True
    assert state["fetches"] == 0


def test_stale_facts_are_refetched(monkeypatch):
    """A company filed a 10-Q since the rows were written. Serving what we hold
    would answer a question about *last year* as though it were current."""
    from app.fundamentals import REFETCH_AFTER_DAYS, ensure_facts

    _patch_db(monkeypatch, newest=date.today() - timedelta(days=REFETCH_AFTER_DAYS + 1))
    state = _patch_sec(monkeypatch, fetched=["f1"])

    assert ensure_facts("AAPL") is True
    assert state["fetches"] == 1


def test_a_filer_sec_has_nothing_for_reports_absence(monkeypatch):
    from app.fundamentals import ensure_facts

    _patch_db(monkeypatch, newest=None)
    _patch_sec(monkeypatch, fetched=[])

    assert ensure_facts("ZZZZ") is False


def test_an_unresolvable_ticker_reports_absence(monkeypatch):
    from app.fundamentals import ensure_facts

    _patch_db(monkeypatch, newest=None)
    state = _patch_sec(monkeypatch, fetched=["f1"], resolves=False)

    assert ensure_facts("ZZZZ") is False
    assert state["fetches"] == 0


def test_a_failed_fetch_returns_false_rather_than_raising(monkeypatch):
    """Tool failures are returned as text, so this must not propagate — an
    exception here would surface a traceback where an absence belongs."""
    from app.fundamentals import ensure_facts

    _patch_db(monkeypatch, newest=None)
    _patch_sec(monkeypatch, fetched=["f1"], raises=RuntimeError("SEC 503"))

    assert ensure_facts("AAPL") is False


def test_a_failed_refetch_still_serves_what_is_already_stored(monkeypatch):
    """Stale beats nothing. SEC being down must not turn a company we have
    years of history for into a corpus gap."""
    from app.fundamentals import ensure_facts

    _patch_db(monkeypatch, newest=date(2020, 1, 1))
    _patch_sec(monkeypatch, fetched=["f1"], raises=RuntimeError("SEC 503"))

    assert ensure_facts("AAPL") is True
