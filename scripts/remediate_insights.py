#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.db import get_connection
from rag.graph_db import get_graph_driver
from rag.insight_extraction import extract_and_store_insights


_PENDING_SQL = """
SELECT DISTINCT c.source_id
FROM chunks c
WHERE c.deleted_at IS NULL
  AND NOT EXISTS (
    SELECT 1 FROM chunk_insights ci WHERE ci.chunk_id = c.id
  )
ORDER BY c.source_id
"""

_CHUNK_ROWS_SQL = """
SELECT id::text, content
FROM chunks
WHERE source_id = %s AND deleted_at IS NULL
ORDER BY chunk_index
"""


_SOURCE_LINK_COUNT_SQL = """
SELECT COUNT(*)
FROM chunk_insights ci
JOIN chunks c ON c.id = ci.chunk_id
WHERE c.source_id = %s
"""


_DELETE_SOURCE_LINKS_SQL = """
DELETE FROM chunk_insights
WHERE chunk_id IN (
  SELECT id FROM chunks WHERE source_id = %s
)
"""


_DELETE_ORPHAN_INSIGHTS_SQL = """
DELETE FROM insights i
WHERE NOT EXISTS (
  SELECT 1 FROM chunk_insights ci WHERE ci.insight_id = i.id
)
"""


def _cleanup_source_insights(conn, driver, source_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(_DELETE_SOURCE_LINKS_SQL, (source_id,))
        cur.execute(_DELETE_ORPHAN_INSIGHTS_SQL)
    with driver.session() as session:
        session.run("MATCH (i:Insight) WHERE NOT (i)<-[:CONTAINS]-() DETACH DELETE i")
    conn.commit()


def _load_chunk_rows(conn, source_id: str) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(_CHUNK_ROWS_SQL, (source_id,))
        return [(row[0], row[1]) for row in cur.fetchall()]


def _process_source(conn, driver, source_id: str) -> bool:
    chunk_rows = _load_chunk_rows(conn, source_id)
    if not chunk_rows:
        print(f"  [SKIP] {source_id} - no chunks")
        return False

    result = extract_and_store_insights(conn, driver, source_id, chunk_rows)
    print(
        f"  [OK] {source_id} - "
        f"{result['chunks_processed']} chunks, "
        f"{result['insights_extracted']} new, "
        f"{result['insights_reused']} reused"
    )
    return True


def _has_existing_links(conn, source_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(_SOURCE_LINK_COUNT_SQL, (source_id,))
        row = cur.fetchone()
    return bool(row and int(row[0]) > 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill insights for already-ingested sources."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of sources to process per batch.",
    )
    parser.add_argument("--source-id", default=None, help="Backfill a single source UUID.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --source-id, delete existing insight links for that source and rebuild.",
    )
    args = parser.parse_args(argv)

    with get_connection() as conn, get_graph_driver() as driver:
        if args.source_id:
            if _has_existing_links(conn, args.source_id):
                if not args.force:
                    print(
                        f"Source {args.source_id} already has insight links. "
                        "Use --force to clean up and rebuild."
                    )
                    return 0
                print(f"Force rebuilding insights for source {args.source_id}.")
                _cleanup_source_insights(conn, driver, args.source_id)

            try:
                processed = _process_source(conn, driver, args.source_id)
            except Exception as exc:
                print(f"  [ERROR] {args.source_id} - {exc}")
                conn.rollback()
                return 1
            print(f"\nDone. Processed: {1 if processed else 0}, Errors: 0")
            return 0

        with conn.cursor() as cur:
            cur.execute(_PENDING_SQL)
            pending = [str(row[0]) for row in cur.fetchall()]

        total = len(pending)
        print(f"Found {total} sources pending insight extraction.")
        if total == 0:
            print("Nothing to do.")
            return 0

        processed = 0
        errors = 0
        for offset in range(0, total, args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            batch_number = offset // args.batch_size + 1
            print(
                f"\nBatch {batch_number}: sources {offset + 1}-"
                f"{min(offset + args.batch_size, total)} of {total}"
            )

            for source_id in batch:
                try:
                    if _process_source(conn, driver, source_id):
                        processed += 1
                except Exception as exc:
                    print(f"  [ERROR] {source_id} - {exc}")
                    conn.rollback()
                    errors += 1

        print(f"\nDone. Processed: {processed}, Errors: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
