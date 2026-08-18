"""Thread-safety of the embedding model singleton.

These tests exist because of a real crash, not a hypothetical one. The agent
issues parallel `search_filings` calls, LangGraph's `ToolNode` runs them in
threads, and each one embeds a query. `get_model()` was memoised with
`@lru_cache`, which caches the *result* but does not serialise the *call* — so
every thread that arrived before the first call returned ran the body too. Four
simultaneous Metal/MPS initialisations segfault CPython rather than raising,
which is why the failure looked like a crash with no traceback.

No torch here: `_build_model` is replaced with a slow fake, so these run
offline in milliseconds and still fail if the locking is removed. The real
model is exercised by `scripts/check_retriever.py`.
"""

import threading
import time

import pytest

from app import embeddings


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test starts cold, and leaves nothing behind for the next one."""
    embeddings._model = None
    yield
    embeddings._model = None


class _FakeModel:
    """Stands in for SentenceTransformer, and records concurrent entry."""

    def __init__(self) -> None:
        self.max_concurrent = 0
        self._active = 0
        self._lock = threading.Lock()

    def _enter(self) -> None:
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)

    def _exit(self) -> None:
        with self._lock:
            self._active -= 1

    def _encode(self):
        self._enter()
        try:
            time.sleep(0.02)  # long enough that unserialised callers overlap
        finally:
            self._exit()

    def encode_query(self, text, **kwargs):
        self._encode()
        return _FakeVector()

    def encode_document(self, texts, **kwargs):
        self._encode()
        return [_FakeVector() for _ in texts]


class _FakeVector:
    def tolist(self):
        return [0.0, 1.0]


def _run_concurrently(target, n=8):
    threads = [threading.Thread(target=target) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_concurrent_get_model_builds_exactly_one(monkeypatch):
    """The regression itself: N threads, one construction.

    Asserting on the *build count* rather than on identity is the point.
    `lru_cache` would still hand every caller the same object afterwards, so an
    identity check alone passes against the broken version — it is the
    duplicate construction that crashed.
    """
    builds = []
    build_lock = threading.Lock()

    def slow_build():
        with build_lock:
            builds.append(1)
        time.sleep(0.05)  # widen the window a racing thread would slip through
        return _FakeModel()

    monkeypatch.setattr(embeddings, "_build_model", slow_build)

    seen = []
    _run_concurrently(lambda: seen.append(embeddings.get_model()), n=8)

    assert len(builds) == 1, f"model was constructed {len(builds)} times, not once"
    assert len({id(m) for m in seen}) == 1, "callers received different models"


def test_concurrent_embed_query_is_serialised(monkeypatch):
    """Loading once is necessary but not sufficient — encoding must not overlap.

    Fixing only the construction race left the crash in place: concurrent
    `encode` on a single loaded model still died. This asserts the second lock.
    """
    fake = _FakeModel()
    monkeypatch.setattr(embeddings, "_build_model", lambda: fake)

    _run_concurrently(lambda: embeddings.embed_query("AI capex risk"), n=8)

    assert fake.max_concurrent == 1, (
        f"{fake.max_concurrent} threads encoded at once; encoding must be serialised"
    )


def test_concurrent_embed_documents_is_serialised(monkeypatch):
    """Same guarantee on the ingestion path, which shares the one model."""
    fake = _FakeModel()
    monkeypatch.setattr(embeddings, "_build_model", lambda: fake)

    _run_concurrently(lambda: embeddings.embed_documents(["a", "b"]), n=8)

    assert fake.max_concurrent == 1


def test_embed_empty_documents_never_touches_the_model(monkeypatch):
    """The early return must stay ahead of get_model(), or an empty batch
    would pay a multi-second model load for nothing."""

    def explode():
        raise AssertionError("get_model() was called for an empty batch")

    monkeypatch.setattr(embeddings, "_build_model", explode)

    assert embeddings.embed_documents([]) == []
