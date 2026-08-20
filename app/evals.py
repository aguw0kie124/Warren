"""Retrieval eval scoring — recall@k and MRR, as arithmetic.

**No LLM judge.** Both metrics are counting exercises against a labelled set,
so scoring costs nothing, is deterministic, and can run on every change. A
judge would add a second model, a second bill, and a second source of variance
to a measurement whose whole job is to be stable enough to compare against
itself.

Split out from `scripts/check_retrieval_quality.py` so the maths is unit-tested
offline — no database, no LangSmith account, no corpus. The gate owns talking to
LangSmith; this module owns being right.

**A chunk is identified by `(accession_number, chunk_index)`.** Not by section,
which several chunks of one filing share, and not by content, which would make
a re-ingest with different chunk boundaries look like a retrieval regression.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.config import PROJECT_ROOT

GOLDEN_PATH = PROJECT_ROOT / "data" / "retrieval_golden.json"

# A row counts as a label only once a human has signed off. `seeded` rows are
# scored but reported as provisional; `unlabelled` rows carry no labels at all
# and are excluded, because scoring against an empty relevant set is a division
# by zero dressed up as a measurement.
CONFIRMED = "confirmed"
SEEDED = "seeded"
UNLABELLED = "unlabelled"
SCORABLE = (CONFIRMED, SEEDED)

# The identity of a retrieved chunk.
ChunkKey = tuple[str, int]


@dataclass(frozen=True)
class GoldenQuery:
    """One labelled query: what to ask, how to filter, and what should come back."""

    id: str
    query: str
    kind: str
    status: str
    note: str
    filters: dict = field(default_factory=dict)
    relevant: tuple[ChunkKey, ...] = ()

    @property
    def scorable(self) -> bool:
        return self.status in SCORABLE and bool(self.relevant)


def load_golden(path: Path | str = GOLDEN_PATH) -> list[GoldenQuery]:
    """Read the golden set. Every query, including the ones with no labels."""
    document = json.loads(Path(path).read_text())
    return [
        GoldenQuery(
            id=row["id"],
            query=row["query"],
            kind=row["kind"],
            status=row["status"],
            note=row.get("note", ""),
            filters=row.get("filters") or {},
            relevant=tuple(
                (item["accession_number"], item["chunk_index"])
                for item in row.get("relevant") or []
            ),
        )
        for row in document["queries"]
    ]


def recall_at_k(
    retrieved: list[ChunkKey], relevant: tuple[ChunkKey, ...] | set[ChunkKey], k: int
) -> float:
    """Share of the labelled chunks that appear in the top k.

    Capped at k on the way in rather than trusting the caller to have truncated,
    because the retriever is free to return fewer or more than asked for and a
    silently longer list would inflate this.
    """
    if not relevant:
        raise ValueError("recall is undefined with no relevant chunks")
    found = set(retrieved[:k]) & set(relevant)
    return len(found) / len(set(relevant))


def reciprocal_rank(
    retrieved: list[ChunkKey], relevant: tuple[ChunkKey, ...] | set[ChunkKey]
) -> float:
    """1/(rank of the first labelled chunk), or 0 if none was retrieved.

    Ranks are 1-based: a labelled chunk at the top scores 1.0. This is the half
    of the pair that notices *ordering* — recall@k is blind to whether the right
    passage came back first or sixth, and the model reads the first one.
    """
    if not relevant:
        raise ValueError("reciprocal rank is undefined with no relevant chunks")
    wanted = set(relevant)
    for position, key in enumerate(retrieved, start=1):
        if key in wanted:
            return 1.0 / position
    return 0.0


def mean(values: list[float]) -> float:
    """Zero for an empty set rather than a ZeroDivisionError.

    The empty case is real — a golden set where nothing is labelled yet — and it
    means "nothing was measured", which is what a caller printing 0.000 against
    a count of 0 queries will show.
    """
    return sum(values) / len(values) if values else 0.0
