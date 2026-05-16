#!/usr/bin/env python3
"""Merge duplicate Entity nodes (same canonical_name + entity_type) across Memgraph and PostgreSQL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

from rag.db import get_connection
from rag.graph_db import get_graph_driver


def fetch_duplicate_groups(conn) -> list[dict]:
    """Return groups where more than one entity shares canonical_name + entity_type.

    Each group dict has:
      - name: str
      - entity_type: str
      - members: list of dicts with keys id, has_embedding (bool), aliases, created_at
    """
    rows = conn.execute(
        """
        SELECT e.id, e.canonical_name, e.entity_type, e.aliases,
               e.embedding IS NOT NULL AS has_embedding, e.created_at
        FROM entities e
        WHERE (e.canonical_name, e.entity_type) IN (
            SELECT canonical_name, entity_type
            FROM entities
            GROUP BY canonical_name, entity_type
            HAVING count(*) > 1
        )
        ORDER BY e.canonical_name, e.entity_type, e.created_at
        """
    ).fetchall()

    groups: dict[tuple, dict] = {}
    for row in rows:
        key = (row[1], row[2])  # canonical_name, entity_type
        if key not in groups:
            groups[key] = {"name": row[1], "entity_type": row[2], "members": []}
        groups[key]["members"].append({
            "id": str(row[0]),
            "aliases": row[3] or [],
            "has_embedding": row[4],
            "created_at": row[5],
        })

    return list(groups.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge duplicate entities.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Print what would be merged.")
    group.add_argument("--execute", action="store_true", help="Perform the merge.")
    args = parser.parse_args()

    with get_connection() as conn:
        groups = fetch_duplicate_groups(conn)

    print(f"Found {len(groups)} duplicate groups ({sum(len(g['members']) for g in groups)} total rows).")
    if args.dry_run:
        for g in groups:
            print(f"  [{g['entity_type']}] {g['name']} — {len(g['members'])} copies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
