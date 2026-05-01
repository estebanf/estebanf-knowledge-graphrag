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
    args = parser.parse_args(argv)

    with get_connection() as conn, get_graph_driver() as driver:
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
                    with conn.cursor() as cur:
                        cur.execute(_CHUNK_ROWS_SQL, (source_id,))
                        chunk_rows = [(row[0], row[1]) for row in cur.fetchall()]

                    if not chunk_rows:
                        print(f"  [SKIP] {source_id} - no chunks")
                        continue

                    result = extract_and_store_insights(conn, driver, source_id, chunk_rows)
                    print(
                        f"  [OK] {source_id} - "
                        f"{result['chunks_processed']} chunks, "
                        f"{result['insights_extracted']} new, "
                        f"{result['insights_reused']} reused"
                    )
                    processed += 1
                except Exception as exc:
                    print(f"  [ERROR] {source_id} - {exc}")
                    conn.rollback()
                    errors += 1

        print(f"\nDone. Processed: {processed}, Errors: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
