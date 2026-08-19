"""A0 verification: apply the schema, then print what actually landed in Postgres.

Stands in for `psql \\dt` / `\\d chunks` so you don't need a host psql client.

    python scripts/check_db.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import exit_code, rule, summary, verdict  # noqa: E402

from app.db import close_pool, get_conn, init_schema  # noqa: E402

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

    results = {}
    with get_conn() as conn:
        for title, sql in QUERIES.items():
            rule(title)
            rows = conn.execute(sql).fetchall()
            results[title] = rows
            for row in rows:
                # indexdef is long; print it on its own line for readability
                if title == "indexes":
                    print(f"  {row[0]}.{row[1]}\n      {row[2]}")
                else:
                    print("  " + " | ".join(str(v) for v in row))

    rule("A0 checklist")
    verdict("vector" in {r[0] for r in results["extensions"]},
            "'vector' listed under extensions")
    tables = {r[0] for r in results["tables"]}
    verdict({"filings", "chunks"} <= tables, "both 'filings' and 'chunks' listed under tables")
    columns = {r[0]: (r[1], r[2]) for r in results["chunks columns"]}
    verdict(columns.get("embedding", (None,))[0] == "USER-DEFINED",
            "chunks.embedding is USER-DEFINED (the vector type)")
    verdict(columns.get("content_tsv", (None, None))[1] == "ALWAYS",
            "chunks.content_tsv shows is_generated = ALWAYS")
    index_defs = {r[1]: r[2] for r in results["indexes"]}
    verdict("hnsw" in index_defs.get("chunks_embedding_hnsw_idx", ""),
            "chunks_embedding_hnsw_idx uses hnsw")
    verdict("gin" in index_defs.get("chunks_content_tsv_gin_idx", ""),
            "chunks_content_tsv_gin_idx uses gin")

    summary()


if __name__ == "__main__":
    try:
        main()
    finally:
        close_pool()

    raise SystemExit(exit_code())
