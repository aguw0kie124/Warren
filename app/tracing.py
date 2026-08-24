"""LangSmith spans for the parts of this system LangChain cannot see.

**Almost all tracing here is free and lives in no file.** LangChain and LangGraph
emit spans for every graph node, model call and tool call on their own, driven
entirely by `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` in
the environment. That is the P2·2 decision and it is unchanged: nothing in this
module turns tracing on, and there is still no `settings.langsmith_*` field.

What automatic tracing cannot see is what happens *inside* a tool. `ToolNode`
reports one span per `@tool` call — its arguments and its returned string — so a
`search_filings` that took nine seconds is nine opaque seconds. The embedding
of the query, the fused SQL, and the fetch-through to SEC are all invisible, and
those are the three places this system actually spends wall-clock time.

`@traceable` is what the `langsmith-trace` skill prescribes for exactly this
case, and it inherits the same env-driven contract: **with tracing off it is a
pass-through.** Measured at ~8µs per call on this machine against functions
whose fastest member is a network round trip, which is why the decorators can
sit on the hot path unconditionally rather than behind a flag.

**What is deliberately NOT traced.** `prices` and `websearch` calls are one
HTTP request each behind a tool that already reports its own arguments and
result, so a span would restate the parent. `get_key_stats` is three sequential
requests inside one attribute access, which is a cost the parent span's duration
already shows, not a hidden decision. The parser and chunker run in
`scripts/ingest.py`, which is an offline batch job and not a request path. The
rule applied here: instrument a function when the tool span above it hides a
decision or a cost, not merely because it is on the path.

**Payload shaping is not cosmetic.** `embed_documents` receives a whole batch of
chunk texts and returns a list of 768-float vectors; traced naively, one
ingestion run would ship megabytes of duplicated corpus to LangSmith and make
every trace unreadable. Each decorator below states what it keeps and why.
"""

import logging
from typing import Any

from langsmith import traceable

logger = logging.getLogger(__name__)

__all__ = ["traceable", "retriever_documents", "hide_embedding_io", "hide_vectors"]


def retriever_documents(outputs: Any) -> dict:
    """Shape `list[Result]` into what LangSmith renders as retrieved documents.

    A span marked `run_type="retriever"` gets a purpose-built view — ranked
    documents with their scores and metadata — instead of a JSON dump, but only
    if the payload is `{"documents": [{"page_content": ..., "metadata": ...}]}`.
    Returning the dataclasses raw costs that view for nothing.

    **The metadata carries `accession_number` and `chunk_index` first**, because
    that pair *is* a chunk's identity everywhere else in this project — it is
    what `app/evals.py` scores recall against and what the golden set stores. A
    trace that identified a chunk by section instead would be describing
    something several chunks of one filing share, and could not be lined up
    against an eval row.

    `dense_rank` and `sparse_rank` are included because their asymmetry is the
    whole diagnostic value of hybrid search: a chunk with a sparse rank and no
    dense rank is BM25 catching an exact-match token the embedding blurred, and
    that is visible here and nowhere else.
    """
    return {
        "documents": [
            {
                "page_content": result.content,
                "metadata": {
                    "accession_number": result.accession_number,
                    "chunk_index": result.chunk_index,
                    "ticker": result.ticker,
                    "form_type": result.form_type,
                    "section": result.section,
                    "fiscal_year": result.fiscal_year,
                    "filing_date": result.filing_date.isoformat(),
                    "score": result.score,
                    "dense_rank": result.dense_rank,
                    "sparse_rank": result.sparse_rank,
                },
            }
            for result in outputs
        ]
    }


def hide_embedding_io(inputs: dict) -> dict:
    """Replace a batch of chunk texts with its shape.

    `embed_documents` is called once per batch during ingestion, with up to 16
    chunks of ~512 tokens each. Those texts are already in Postgres and already
    in the trace of whatever retrieved them; shipping them again would put the
    corpus into the trace store twice and bury the number actually worth seeing,
    which is how long the encode took.
    """
    texts = inputs.get("texts") or []
    return {
        "count": len(texts),
        "chars": sum(len(text) for text in texts),
        "batch_size": inputs.get("batch_size"),
    }


def hide_vectors(outputs: Any) -> dict:
    """Replace embedding vectors with their shape.

    768 floats say nothing a human or a diff can read, and a query embedding
    plus its results would otherwise dominate the span. The dimension is kept
    because it is the one thing that can silently go wrong here: it is coupled
    across the model, `sql/schema.sql` and `settings.embedding_dim`, and a
    mismatch surfaces as bad retrieval rather than as an error.
    """
    if outputs and isinstance(outputs[0], (list, tuple)):
        return {"vectors": len(outputs), "dim": len(outputs[0])}
    return {"vectors": 1, "dim": len(outputs) if outputs else 0}
