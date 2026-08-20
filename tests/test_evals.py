"""Offline tests for retrieval eval scoring.

No database, no corpus, no LangSmith account. What is testable here is the
arithmetic — whether recall counts the right things and whether MRR notices
ordering — because that is the part a wrong answer would hide. A metric that is
subtly wrong does not error; it prints a plausible number that gets compared
against later plausible numbers, and the whole point of the eval is that those
comparisons mean something.

The golden set itself is also asserted here, because its *shape* is a contract:
a row that silently loses its labels stops being scored and the average quietly
improves.
"""

import json

import pytest

from app import evals
from app.evals import GoldenQuery, load_golden, mean, recall_at_k, reciprocal_rank

A = ("0000320193-25-000079", 14)
B = ("0000320193-25-000079", 24)
C = ("0001628280-26-003952", 21)


# --- recall@k ---------------------------------------------------------------


def test_recall_counts_only_the_top_k():
    """A labelled chunk at position 7 does not count towards recall@6.

    The cap is the whole meaning of the metric: the agent reads k passages, so a
    chunk ranked below k was not retrieved as far as the answer is concerned.
    """
    retrieved = [("x", i) for i in range(6)] + [A]
    assert recall_at_k(retrieved, (A,), k=6) == 0.0
    assert recall_at_k(retrieved, (A,), k=7) == 1.0


def test_recall_is_a_share_of_the_labels_not_of_the_results():
    assert recall_at_k([A, ("x", 1)], (A, B), k=6) == 0.5
    assert recall_at_k([A, B], (A, B), k=6) == 1.0


def test_recall_ignores_duplicate_retrievals():
    """A retriever returning one chunk twice must not score it twice."""
    assert recall_at_k([A, A, A], (A, B), k=6) == 0.5


def test_recall_with_no_labels_raises_rather_than_returning_zero():
    """Zero would read as 'retrieved nothing relevant' — a measurement.

    No labels means *nothing was measured*, which is a different fact, and the
    one an unlabelled golden row carries. Silently returning 0.0 would drag an
    average down and look like a retrieval regression.
    """
    with pytest.raises(ValueError):
        recall_at_k([A], (), k=6)


# --- reciprocal rank --------------------------------------------------------


def test_reciprocal_rank_is_one_based():
    assert reciprocal_rank([A], (A,)) == 1.0
    assert reciprocal_rank([("x", 1), A], (A,)) == 0.5
    assert reciprocal_rank([("x", 1), ("y", 2), A], (A,)) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_the_first_label_found():
    """Not the best-ranked label overall — the first one the model would read."""
    assert reciprocal_rank([B, A], (A, B)) == 1.0


def test_reciprocal_rank_is_zero_when_nothing_relevant_came_back():
    assert reciprocal_rank([("x", 1), ("y", 2)], (A,)) == 0.0


def test_reciprocal_rank_with_no_labels_raises():
    with pytest.raises(ValueError):
        reciprocal_rank([A], ())


def test_recall_and_mrr_disagree_on_ordering():
    """The reason both metrics exist.

    Same retrieved set, opposite orders: recall cannot tell them apart, MRR can.
    A reranker that improves ordering and nothing else shows up only in MRR.
    """
    good = [A, ("x", 1), ("y", 2)]
    bad = [("x", 1), ("y", 2), A]
    assert recall_at_k(good, (A,), k=6) == recall_at_k(bad, (A,), k=6)
    assert reciprocal_rank(good, (A,)) > reciprocal_rank(bad, (A,))


# --- mean -------------------------------------------------------------------


def test_mean_of_nothing_is_zero_not_an_error():
    assert mean([]) == 0.0
    assert mean([1.0, 0.0]) == 0.5


# --- the golden set ---------------------------------------------------------


def test_the_golden_set_loads_and_every_query_is_addressable():
    golden = load_golden()
    assert golden
    assert len({q.id for q in golden}) == len(golden), "ids must be unique"
    assert all(q.query.strip() for q in golden)


def test_every_status_is_one_the_scorer_understands():
    """A typo'd status silently drops a query out of scoring."""
    known = {evals.CONFIRMED, evals.SEEDED, evals.UNLABELLED}
    assert {q.status for q in load_golden()} <= known


def test_unlabelled_queries_carry_no_labels_and_are_not_scorable():
    """The rejected seeds. If one ever gained labels without being re-reviewed,
    it would re-enter scoring carrying exactly the output that was rejected."""
    for query in load_golden():
        if query.status == evals.UNLABELLED:
            assert query.relevant == ()
            assert not query.scorable
            assert query.note, "a rejection must say why"


def test_scorable_queries_have_labels():
    scorable = [q for q in load_golden() if q.scorable]
    assert scorable
    assert all(q.relevant for q in scorable)


def test_labels_are_accession_and_chunk_index_pairs():
    """Not section, which many chunks share — that would make recall trivially
    satisfiable by any chunk from the right part of the filing."""
    for query in load_golden():
        for accession, index in query.relevant:
            assert isinstance(accession, str) and accession
            assert isinstance(index, int)


def test_the_seed_records_that_it_is_not_ground_truth():
    """The file's own warnings are load-bearing, not decoration.

    Someone reading only the JSON must find out that it was seeded from a dense
    retrieval dump — otherwise the dense-vs-hybrid comparison looks meaningful.
    """
    document = json.loads(evals.GOLDEN_PATH.read_text())
    assert "NOT GROUND TRUTH" in document["_readme"]
    assert "BIASED TOWARDS DENSE" in document["_bias_warning"]


def test_a_confirmed_query_would_be_scorable():
    """Guards the transition the whole file is waiting on."""
    query = GoldenQuery(id="x", query="q", kind="exact", status=evals.CONFIRMED,
                        note="", relevant=(A,))
    assert query.scorable
