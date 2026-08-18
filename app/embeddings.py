"""Local embeddings via sentence-transformers.

The single swap point for the embedding model. Nothing outside this file names
the model, applies a prompt template, or knows the vector width — so moving to
a different model (or a hosted API) touches only here, plus the vector(N)
column and settings.embedding_dim it must stay in step with.
"""

import logging
from collections.abc import Sequence
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

Vector = list[float]


@lru_cache(maxsize=1)
def get_model():
    """Load the encoder once per process (a few seconds, several hundred MB)."""
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


def embed_documents(texts: Sequence[str], batch_size: int = 16) -> list[Vector]:
    """Embed passages for storage."""
    if not texts:
        return []
    vectors = get_model().encode_document(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> Vector:
    """Embed a search query."""
    vector = get_model().encode_query(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()
