"""Phase 2 verification: retrieval quality, scored on LangSmith.

    python scripts/check_retrieval_quality.py              # hybrid, the default
    python scripts/check_retrieval_quality.py --dense      # dense-only comparison
    python scripts/check_retrieval_quality.py --k 12       # sweep a different k
    python scripts/check_retrieval_quality.py --local      # score without LangSmith
    python scripts/check_retrieval_quality.py --sync       # push the golden set and stop

**Recall@k and MRR, no LLM judge.** Both are arithmetic against a labelled set,
so this costs nothing to run and can go on every change. The maths lives in
`app/evals.py` and is unit-tested offline; this file owns LangSmith.

**The golden set is not ground truth yet, and the gate says so out loud.**
`data/retrieval_golden.json` was seeded from `data/retrieval_a7.json`, which is
a dump of what the retriever *returned* — treating that as what it *should* have
returned bakes today's behaviour in as the target. Three of the eight seeds were
read and rejected outright (see the file's notes): the exact-match queries for
'material weakness in internal control' and 'going concern' retrieved
cybersecurity and tax prose respectively, which is the retriever missing rather
than a label. Those carry no labels and are excluded from scoring rather than
quietly counted as satisfied.

**The seed is also biased towards dense retrieval**, which is worse than it
sounds and is why the gate refuses to draw a conclusion from it: A7 *is* the
dense retriever, so `data/retrieval_a7.json` holds what dense returned, and
`--dense` therefore scores near 1.0 by reproducing its own output. Measured on
the seed at k=6, dense reads 1.000 and hybrid 0.800 — which establishes nothing
about hybrid, only that hybrid ranks differently from the thing that wrote the
labels. The comparison becomes meaningful once rows reach status='confirmed'.

So the number this prints is **provisional until rows reach status='confirmed'**,
and the gate reports the split every run rather than averaging it away.

Needs Postgres up with the corpus ingested. LangSmith needs LANGSMITH_API_KEY;
`--local` skips it and scores in-process, which is the same arithmetic without
the experiment history.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import exit_code, note, rule, summary, verdict  # noqa: E402

from app import evals, retriever  # noqa: E402
from app.db import close_pool  # noqa: E402
from app.evals import GoldenQuery  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

DATASET_NAME = "confluence-retrieval"


def search(query: GoldenQuery, k: int, dense: bool) -> list[tuple[str, int]]:
    """Run one golden query through the retriever, in rank order.

    The filters ride from the golden set rather than being inferred, because a
    query labelled under `ticker=AAPL` scored without that filter is a different
    question with the same words.
    """
    fn = retriever.search if dense else retriever.hybrid_search
    results = fn(query.query, k=k, **query.filters)
    return [(r.accession_number, r.chunk_index) for r in results]


def score_one(query: GoldenQuery, k: int, dense: bool) -> dict:
    retrieved = search(query, k, dense)
    return {
        "id": query.id,
        "recall": evals.recall_at_k(retrieved, query.relevant, k),
        "rr": evals.reciprocal_rank(retrieved, query.relevant),
        "retrieved": retrieved,
    }


# ---------------------------------------------------------------------------
# The golden set
# ---------------------------------------------------------------------------


def check_golden_set(golden: list[GoldenQuery]) -> list[GoldenQuery]:
    rule("1. the golden set")

    scorable = [q for q in golden if q.scorable]
    confirmed = [q for q in golden if q.status == evals.CONFIRMED]
    unlabelled = [q for q in golden if q.status == evals.UNLABELLED]

    verdict(bool(golden), f"{len(golden)} queries loaded from data/retrieval_golden.json")
    verdict(
        bool(scorable),
        f"{len(scorable)} carry labels and can be scored",
    )
    note(f"{len(confirmed)} confirmed by a human, "
         f"{len(scorable) - len(confirmed)} still seeded (scored, provisional)")

    if unlabelled:
        note(f"{len(unlabelled)} excluded — seed read and REJECTED, no labels yet:")
        for query in unlabelled:
            note(f"    {query.id}: {query.note.splitlines()[0]}")

    if not confirmed:
        note("NOTHING IS CONFIRMED YET. Every score below is provisional: it measures "
             "\n         agreement with the seed, not with a human's judgement. Confirm "
             "rows in\n         data/retrieval_golden.json before treating any of this "
             "as a baseline.")
        note("AND THE SEED IS BIASED TOWARDS DENSE. data/retrieval_a7.json is a dump of "
             "\n         A7, the *dense* retriever (its scores are cosine similarities, "
             "not RRF's\n         ~0.03). So `--dense` scores near 1.0 by reproducing its "
             "own output, while\n         hybrid is penalised for ranking differently "
             "rather than for ranking worse.\n         **Do not read the dense/hybrid "
             "delta until rows are confirmed.**")
    return scorable


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def check_local(scorable: list[GoldenQuery], k: int, dense: bool) -> None:
    """Score in-process. The same arithmetic LangSmith runs, without the account."""
    rule(f"2. {'dense' if dense else 'hybrid'} retrieval @ k={k}")

    rows = [score_one(query, k, dense) for query in scorable]
    for query, row in zip(scorable, rows):
        note(f"{query.id:<26} recall@{k}={row['recall']:.2f}  rr={row['rr']:.3f}"
             f"  ({query.kind})")

    recall = evals.mean([row["recall"] for row in rows])
    mrr = evals.mean([row["rr"] for row in rows])
    print(f"\n    recall@{k} {recall:.3f}    MRR {mrr:.3f}    over {len(rows)} queries")

    verdict(
        all(row["rr"] > 0 for row in rows),
        "every scorable query retrieved at least one labelled chunk",
    )
    for query, row in zip(scorable, rows):
        if row["rr"] == 0:
            note(f"    {query.id} retrieved none of its labels — {row['retrieved'][:3]}")


# ---------------------------------------------------------------------------
# LangSmith
# ---------------------------------------------------------------------------


def require_langsmith():
    """The client, or a setup error naming exactly what is missing.

    Exit 2 rather than 1, per the gate convention: "you have not finished
    configuring your machine" is not "the thing under test is broken".
    """
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("\n  LANGSMITH_API_KEY is not set. Add it to .env (see .env.example), or "
              "run\n  with --local to score in-process without LangSmith.")
        raise SystemExit(2)
    from langsmith import Client

    return Client()


def sync_dataset(client, golden: list[GoldenQuery]) -> None:
    """Push the golden set to LangSmith, replacing what is there.

    Only labelled queries are uploaded. An unlabelled row in a LangSmith dataset
    would show up as an example with no reference output, which scores as a
    failure rather than as the exclusion it actually is.
    """
    rule("dataset sync")

    scorable = [q for q in golden if q.scorable]
    if client.has_dataset(dataset_name=DATASET_NAME):
        client.delete_dataset(dataset_name=DATASET_NAME)
        note(f"replaced the existing {DATASET_NAME!r} dataset")

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "SEC filing retrieval golden set. Seeded from a system-output dump; "
            "rows are provisional until confirmed by a human."
        ),
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"query": q.query, "filters": q.filters, "kind": q.kind},
                "outputs": {"relevant": [list(key) for key in q.relevant]},
                "metadata": {"id": q.id, "status": q.status, "kind": q.kind},
            }
            for q in scorable
        ],
    )
    verdict(True, f"pushed {len(scorable)} labelled queries to {DATASET_NAME!r}")


def check_langsmith(client, scorable: list[GoldenQuery], k: int, dense: bool) -> None:
    """Run the experiment through LangSmith so the scores get a history.

    The evaluators are the same two functions `--local` uses. LangSmith adds
    versioning, a diff against previous experiments, and per-example traces —
    which is the whole reason a tuning change (reranking, a different k, RRF
    weights) can be compared to a baseline rather than argued about.
    """
    rule(f"2. {'dense' if dense else 'hybrid'} retrieval @ k={k} · LangSmith")

    from langsmith import evaluate

    by_query = {q.query: q for q in scorable}

    def target(inputs: dict) -> dict:
        query = by_query[inputs["query"]]
        return {"retrieved": [list(key) for key in search(query, k, dense)]}

    def recall(outputs: dict, reference_outputs: dict) -> dict:
        retrieved = [tuple(item) for item in outputs["retrieved"]]
        relevant = tuple(tuple(item) for item in reference_outputs["relevant"])
        return {"key": f"recall@{k}", "score": evals.recall_at_k(retrieved, relevant, k)}

    def mrr(outputs: dict, reference_outputs: dict) -> dict:
        retrieved = [tuple(item) for item in outputs["retrieved"]]
        relevant = tuple(tuple(item) for item in reference_outputs["relevant"])
        return {"key": "reciprocal_rank", "score": evals.reciprocal_rank(retrieved, relevant)}

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[recall, mrr],
        experiment_prefix=f"{'dense' if dense else 'hybrid'}-k{k}",
        metadata={"k": k, "mode": "dense" if dense else "hybrid"},
        max_concurrency=1,  # the embedding model serialises on _encode_lock anyway
    )

    scores: dict[str, list[float]] = {}
    for row in results:
        for result in row["evaluation_results"]["results"]:
            scores.setdefault(result.key, []).append(result.score or 0.0)

    for key, values in sorted(scores.items()):
        print(f"\n    {key}: {evals.mean(values):.3f} over {len(values)} queries")

    verdict(bool(scores), "LangSmith returned scored results")
    verdict(
        all(score > 0 for score in scores.get("reciprocal_rank", [])),
        "every scorable query retrieved at least one labelled chunk",
    )
    note(f"experiment: {getattr(results, 'experiment_name', DATASET_NAME)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true", help="dense-only, for comparison")
    ap.add_argument("--k", type=int, default=retriever.DEFAULT_K, help="results per query")
    ap.add_argument("--local", action="store_true", help="score in-process, no LangSmith")
    ap.add_argument("--sync", action="store_true", help="push the golden set and stop")
    args = ap.parse_args()

    golden = evals.load_golden()

    try:
        scorable = check_golden_set(golden)
        if not scorable:
            summary(on_failure="Nothing is labelled — there is nothing to measure.")
            raise SystemExit(exit_code())

        if args.local:
            check_local(scorable, args.k, args.dense)
        else:
            client = require_langsmith()
            sync_dataset(client, golden)
            if not args.sync:
                check_langsmith(client, scorable, args.k, args.dense)

        summary(
            on_failure="A zero recall is a retrieval bug or a wrong label — read the "
                       "retrieved\n  chunks before assuming which.",
            on_success="the score is provisional until data/retrieval_golden.json rows "
                       "reach\n  status='confirmed'. Confirm them, then tune k, RRF "
                       "weights or a reranker\n  against this baseline.",
        )
    finally:
        close_pool()

    raise SystemExit(exit_code())


if __name__ == "__main__":
    main()
