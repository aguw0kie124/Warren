"""Offline tests for retrieval plumbing.

Ranking quality needs a live corpus and a human reading the output — that's
scripts/check_retriever.py. What's testable offline is the part that silently
corrupts results without failing: filter construction, and row-to-Result
mapping. A8 adds the RRF fusion maths here.
"""

from datetime import date

from app.retriever import Result, _filters, _to_results, _where, rrf_score

ROW = (
    "0000320193-25-000079", 3, "Item 1A Risk Factors", "Our business is subject to risk.",
    "AAPL", "Apple Inc.", "10-K", 2025, date(2025, 10, 31),
    "https://www.sec.gov/Archives/edgar/data/320193/x.htm", 0.61,
)


def test_no_filters_produces_no_where_clause():
    clauses, params = _filters(None, None, None, None)
    assert clauses == []
    assert params == []
    assert _where(clauses) == ""


def test_ticker_is_uppercased():
    # Tickers are stored uppercase; a lowercase filter would silently match
    # nothing and look like "no relevant results" rather than a bug.
    _, params = _filters("aapl", None, None, None)
    assert params == ["AAPL"]


def test_form_type_is_uppercased():
    _, params = _filters(None, None, "10-k", None)
    assert params == ["10-K"]


def test_each_filter_contributes_one_clause_and_one_param():
    clauses, params = _filters("AAPL", "Item 7 MD&A", "10-K", 2025)
    assert len(clauses) == len(params) == 4
    assert params == ["AAPL", "Item 7 MD&A", "10-K", 2025]


def test_clauses_and_params_stay_in_step():
    # Order matters: params are positional, so a clause emitted without its
    # parameter (or vice versa) shifts every later value by one.
    clauses, params = _filters("AAPL", None, "10-Q", None)
    assert clauses == ["f.ticker = %s", "f.form_type = %s"]
    assert params == ["AAPL", "10-Q"]


def test_where_joins_with_and():
    assert _where(["a = %s", "b = %s"]) == "WHERE a = %s AND b = %s"


def test_row_maps_onto_result_fields_in_order():
    # The SELECT column order and this mapping have to move together; a
    # mismatch yields plausible-looking results with swapped metadata.
    result = _to_results([ROW])[0]
    assert result.ticker == "AAPL"
    assert result.section == "Item 1A Risk Factors"
    assert result.chunk_index == 3
    assert result.fiscal_year == 2025
    assert result.score == 0.61
    assert result.source_url.endswith("x.htm")


def test_citation_is_assembled_from_stored_metadata():
    assert _to_results([ROW])[0].citation == (
        "[Apple Inc. 10-K, Item 1A Risk Factors, filed 2025-10-31]"
    )


def test_empty_rows_map_to_empty_results():
    assert _to_results([]) == []


def test_result_is_immutable():
    # Results flow into citation assembly downstream; letting a caller mutate
    # one would let a citation drift from the row it came from.
    result = _to_results([ROW])[0]
    try:
        result.score = 1.0
    except Exception as exc:
        assert type(exc).__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("Result should be frozen")


def test_result_dataclass_exposes_every_citation_field():
    fields = set(Result.__dataclass_fields__)
    assert {"company_name", "form_type", "section", "filing_date", "source_url"} <= fields


# --- A8: RRF fusion maths ---


def test_rrf_uses_rank_not_score():
    # The entire point: cosine similarity and ts_rank_cd are on incomparable
    # scales, so only position may influence the fused score.
    assert rrf_score(1, None) == 1 / 61
    assert rrf_score(None, 1) == 1 / 61


def test_rrf_is_symmetric_across_the_two_rankers():
    # Neither ranker is privileged, so swapping the ranks must not change the
    # score — which is also why exact ties are common and expected.
    assert rrf_score(1, 2) == rrf_score(2, 1)


def test_agreement_beats_a_single_strong_opinion():
    # A chunk both rankers place 2nd outranks one that only dense puts 1st.
    # That is the behaviour hybrid search is bought for.
    assert rrf_score(2, 2) > rrf_score(1, None)


def test_a_chunk_found_by_only_one_ranker_still_scores():
    # FULL OUTER JOIN semantics: exact-phrase hits dense misses entirely must
    # still be able to place.
    assert rrf_score(None, 3) > 0


def test_missing_from_both_rankers_scores_zero():
    assert rrf_score(None, None) == 0.0


def test_score_decreases_monotonically_with_rank():
    scores = [rrf_score(r, None) for r in range(1, 10)]
    assert scores == sorted(scores, reverse=True)


def test_damping_constant_compresses_top_ranks():
    # k=60 is large relative to the ranks that matter, so rank 1 and rank 2
    # differ by little — no single ranker's top hit can dominate the fusion.
    gap = rrf_score(1, None) - rrf_score(2, None)
    assert gap < 0.1 * rrf_score(1, None)


def test_rrf_constant_is_configurable():
    assert rrf_score(1, None, k=0) == 1.0
