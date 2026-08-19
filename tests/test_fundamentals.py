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

from datetime import date
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
