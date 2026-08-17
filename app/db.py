"""Postgres connection pool and schema bootstrap."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from psycopg import Connection
from psycopg_pool import ConnectionPool

from app.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Return the process-wide connection pool, opening it on first use."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=8,
            open=True,
            # Fail fast with a clear error instead of hanging when the container
            # isn't up yet — the most common local failure by far.
            timeout=10.0,
        )
    return _pool


@contextmanager
def get_conn() -> Iterator[Connection]:
    """Borrow a connection from the pool. Commits on clean exit, rolls back on error."""
    with get_pool().connection() as conn:
        yield conn


def init_schema() -> None:
    """Apply sql/schema.sql. Every statement in it is idempotent, so this is
    safe to call on every startup."""
    sql = SCHEMA_PATH.read_text()
    with get_conn() as conn:
        conn.execute(sql)
    logger.info("schema applied from %s", SCHEMA_PATH)


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
