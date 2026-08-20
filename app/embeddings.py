"""Local embeddings via sentence-transformers.

The single swap point for the embedding model. Nothing outside this file names
the model, applies a prompt template, or knows the vector width — so moving to
a different model (or a hosted API) touches only here, plus the vector(N)
column and settings.embedding_dim it must stay in step with.
"""

import logging
import threading
from collections.abc import Sequence

from app.config import settings
from app.tracing import hide_embedding_io, hide_vectors, traceable

logger = logging.getLogger(__name__)

Vector = list[float]


_model = None
# `lru_cache` memoises but does not serialise: threads that arrive before the
# first call returns all run the body, so N concurrent callers built N models.
# That is not hypothetical here — the agent issues parallel `search_filings`
# calls, ToolNode runs them in threads, and four simultaneous Metal/MPS
# initialisations segfault the interpreter rather than raising. A plain lock
# is the fix; the double check keeps it off the hot path.
_init_lock = threading.Lock()

# Loading once is necessary but not sufficient: concurrent `encode` on one
# loaded model crashes too (SIGTRAP on MPS), so encoding is serialised as well.
# Measured, not assumed — fixing only the load race took the repro from 4 model
# loads to 1 and it still died. The cost is nil in practice: a query embedding
# is a single short string, and the calls this serialises are already waiting
# behind an LLM turn and a Postgres round-trip.
_encode_lock = threading.Lock()


def get_model():
    """Load the encoder once per process (a few seconds, several hundred MB).

    Thread-safe: concurrent first-callers block on the lock, and exactly one
    of them constructs the model.
    """
    global _model
    if _model is None:
        with _init_lock:
            if _model is None:
                _model = _build_model()
    return _model


def _build_model():
    # Imported lazily so that merely importing this module doesn't drag in torch.
    from sentence_transformers import SentenceTransformer

    logger.info("loading embedding model %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)

    # Renamed in sentence-transformers 5.7; the old name still works but warns.
    measure = getattr(model, "get_embedding_dimension", None) or (
        model.get_sentence_embedding_dimension
    )
    dim = measure()
    if dim != settings.embedding_dim:
        # Caught here rather than as an opaque Postgres error thousands of
        # chunks into an ingest run.
        raise RuntimeError(
            f"{settings.embedding_model} emits {dim}-dim vectors but "
            f"settings.embedding_dim is {settings.embedding_dim}. The vector(N) "
            f"column in sql/schema.sql must match too."
        )
    return model


@traceable(
    run_type="embedding",
    name="embed_documents",
    process_inputs=hide_embedding_io,
    process_outputs=hide_vectors,
)
def embed_documents(texts: Sequence[str], batch_size: int = 16) -> list[Vector]:
    """Embed passages for storage."""
    if not texts:
        return []
    model = get_model()
    with _encode_lock:
        vectors = model.encode_document(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return [v.tolist() for v in vectors]


# The query text is kept in the span and the 768-float result is not. Which
# template a query was embedded under is invisible from the outside and
# swapping them degrades retrieval *silently* (see the module docstring), so
# having the exact string that reached `encode_query` recorded next to what came
# back is the only external evidence the right one was used.
@traceable(run_type="embedding", name="embed_query", process_outputs=hide_vectors)
def embed_query(text: str) -> Vector:
    """Embed a search query."""
    model = get_model()
    with _encode_lock:
        vector = model.encode_query(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return vector.tolist()
