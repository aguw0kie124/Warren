"""A0 verification: apply the schema, then print what actually landed in Postgres.

Stands in for `psql \\dt` / `\\d chunks` so you don't need a host psql client.

    python scripts/check_db.py
"""

import logging

from app.db import close_pool, get_conn, init_schema

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

QUERIES = {
    "extensions": """
        SELECT extname, extversion FROM pg_extension
        WHERE extname IN ('vector') ORDER BY extname
    """,
    "tables": """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' ORDER BY table_name
    """,
    "chunks columns": """
        SELECT column_name, data_type, is_generated
        FROM information_schema.columns
        WHERE table_name = 'chunks' ORDER BY ordinal_position
    """,
    "indexes": """
        SELECT tablename, indexname, indexdef FROM pg_indexes
        WHERE schemaname = 'public' ORDER BY tablename, indexname
    """,
    "row counts": """
        SELECT 'filings' AS table, count(*) FROM filings
        UNION ALL
        SELECT 'chunks', count(*) FROM chunks
    """,
}


def main() -> None:
    init_schema()
    # Re-apply immediately: every statement in schema.sql is meant to be
    # idempotent, and this is the cheapest place to prove it.
    init_schema()

    with get_conn() as conn:
        for title, sql in QUERIES.items():
            print(f"\n=== {title} ===")
            for row in conn.execute(sql).fetchall():
                # indexdef is long; print it on its own line for readability
                if title == "indexes":
                    print(f"  {row[0]}.{row[1]}\n      {row[2]}")
                else:
                    print("  " + " | ".join(str(v) for v in row))

    print("\nA0 checklist:")
    print("  [ ] 'vector' listed under extensions")
    print("  [ ] both 'filings' and 'chunks' listed under tables")
    print("  [ ] chunks.embedding is USER-DEFINED (the vector type)")
    print("  [ ] chunks.content_tsv shows is_generated = ALWAYS")
    print("  [ ] chunks_embedding_hnsw_idx uses hnsw, chunks_content_tsv_gin_idx uses gin")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_pool()
