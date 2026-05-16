#!/usr/bin/env python3
"""Merge duplicate Entity nodes (same canonical_name + entity_type) across Memgraph and PostgreSQL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

from rag.db import get_connection
from rag.graph_db import get_graph_driver


def pick_survivor(members: list[dict]) -> dict:
    """Pick the entity to keep.

    Prefer: has_embedding=True first, then oldest created_at.
    """
    if not members:
        raise ValueError("pick_survivor requires at least one member")
    return sorted(members, key=lambda m: (not m["has_embedding"], m["created_at"]))[0]


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


def merge_postgres(conn, survivor_id: str, dup_ids: list[str], all_members: list[dict]) -> None:
    """Merge aliases from duplicates into survivor, then delete duplicate rows."""
    if not dup_ids:
        return
    all_aliases: set[str] = set()
    for m in all_members:
        all_aliases.update(m["aliases"])

    conn.execute(
        "UPDATE entities SET aliases = %s WHERE id = %s",
        (sorted(all_aliases), survivor_id),
    )
    conn.execute(
        "DELETE FROM entities WHERE id = ANY(%s::uuid[])",
        (dup_ids,),
    )


def merge_memgraph(session, survivor_id: str, dup_ids: list[str]) -> int:
    """Re-point MENTIONS edges from duplicates to survivor, delete duplicate nodes.

    Returns the number of edges re-pointed.
    """
    edges_repointed = 0
    for dup_id in dup_ids:
        result = session.run(
            "MATCH (c:Chunk)-[old:MENTIONS]->(e:Entity {entity_id: $dup_id}) "
            "MATCH (s:Entity {entity_id: $survivor_id}) "
            "MERGE (c)-[:MENTIONS {confidence: old.confidence}]->(s) "
            "WITH e, count(old) AS repointed "
            "DETACH DELETE e "
            "RETURN repointed",
            dup_id=dup_id,
            survivor_id=survivor_id,
        )
        record = result.single()
        if record:
            edges_repointed += record["repointed"]

    return edges_repointed


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
            survivor = pick_survivor(g["members"])
            dup_ids = [m["id"] for m in g["members"] if m["id"] != survivor["id"]]
            print(
                f"  [{g['entity_type']}] {g['name']} — "
                f"keep {survivor['id'][:8]}…, "
                f"delete {len(dup_ids)} duplicate(s)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
