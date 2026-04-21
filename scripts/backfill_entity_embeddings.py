#!/usr/bin/env python3
"""Backfill embeddings for entities that have embedding IS NULL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.db import get_connection
from rag.embedding import get_embeddings

_BATCH_SIZE = 32


def main():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, canonical_name FROM entities WHERE embedding IS NULL ORDER BY id"
        ).fetchall()

    total = len(rows)
    if total == 0:
        print("No entities missing embeddings.")
        return

    print(f"Backfilling {total} entities in batches of {_BATCH_SIZE}...")
    updated = 0

    for i in range(0, total, _BATCH_SIZE):
        batch = rows[i : i + _BATCH_SIZE]
        ids = [str(r[0]) for r in batch]
        names = [r[1] for r in batch]

        try:
            vecs = get_embeddings(names)
        except Exception as e:
            print(f"  Batch {i}–{i+len(batch)-1}: embedding failed — {e}")
            continue

        if len(vecs) != len(names):
            print(f"  Batch {i}–{i+len(batch)-1}: length mismatch ({len(vecs)} vs {len(names)}), skipping")
            continue

        with get_connection() as conn:
            for entity_id, vec in zip(ids, vecs):
                embedding_str = f"[{','.join(str(v) for v in vec)}]"
                conn.execute(
                    "UPDATE entities SET embedding = %s::vector WHERE id = %s",
                    (embedding_str, entity_id),
                )
            conn.commit()

        updated += len(batch)
        print(f"  {updated}/{total} done")

    print(f"Backfill complete. {updated}/{total} entities updated.")


if __name__ == "__main__":
    main()
