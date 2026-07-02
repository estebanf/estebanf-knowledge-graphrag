#!/usr/bin/env python3
"""Merge duplicate Entity nodes (same canonical_name + entity_type) across Memgraph and PostgreSQL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

from rag.db import get_connection, vacuum_analyze_entities
from rag.graph_db import get_graph_driver


def pick_survivor(members: list[dict]) -> dict:
    """Pick the entity to keep.

    Prefer: has_embedding=True first, then oldest created_at.
    """
    if not members:
        raise ValueError("pick_survivor requires at least one member")
    return sorted(members, key=lambda m: (not m["has_embedding"], m["created_at"]))[0]


def fetch_cross_type_duplicate_groups(conn) -> list[dict]:
    """Return groups where more than one entity shares canonical_name but has different entity_type.

    Each group dict has the same shape as fetch_duplicate_groups but members may differ by entity_type.
    """
    rows = conn.execute(
        """
        SELECT e.id, e.canonical_name, e.entity_type, e.aliases,
               e.embedding IS NOT NULL AS has_embedding, e.created_at
        FROM entities e
        WHERE e.canonical_name IN (
            SELECT canonical_name
            FROM entities
            GROUP BY canonical_name
            HAVING count(DISTINCT entity_type) > 1
        )
        ORDER BY e.canonical_name, e.entity_type, e.created_at
        """
    ).fetchall()

    groups: dict[str, dict] = {}
    for row in rows:
        key = row[1]  # canonical_name only
        if key not in groups:
            groups[key] = {"name": row[1], "entity_type": None, "members": []}
        groups[key]["members"].append({
            "id": str(row[0]),
            "entity_type": row[2],
            "aliases": row[3] or [],
            "has_embedding": row[4],
            "created_at": row[5],
        })

    return list(groups.values())


def pick_survivor_cross_type(members: list[dict]) -> dict:
    """Pick the entity to keep for cross-type groups.

    Prefer the most-frequent entity_type among members, then has_embedding=True, then oldest created_at.
    """
    from collections import Counter
    type_counts = Counter(m["entity_type"] for m in members)
    dominant_type = type_counts.most_common(1)[0][0]
    preferred = [m for m in members if m["entity_type"] == dominant_type]
    return sorted(preferred, key=lambda m: (not m["has_embedding"], m["created_at"]))[0]


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

        # Re-point outgoing RELATED_TO edges
        session.run(
            "MATCH (e:Entity {entity_id: $dup_id})-[r:RELATED_TO]->(other:Entity) "
            "WHERE other.entity_id <> $survivor_id "
            "MATCH (s:Entity {entity_id: $survivor_id}) "
            "MERGE (s)-[:RELATED_TO {type: r.type, confidence: r.confidence, chunk_id: r.chunk_id}]->(other)",
            dup_id=dup_id,
            survivor_id=survivor_id,
        )
        # Re-point incoming RELATED_TO edges
        session.run(
            "MATCH (other:Entity)-[r:RELATED_TO]->(e:Entity {entity_id: $dup_id}) "
            "WHERE other.entity_id <> $survivor_id "
            "MATCH (s:Entity {entity_id: $survivor_id}) "
            "MERGE (other)-[:RELATED_TO {type: r.type, confidence: r.confidence, chunk_id: r.chunk_id}]->(s)",
            dup_id=dup_id,
            survivor_id=survivor_id,
        )
        # Unconditional cleanup: delete node if it had no MENTIONS edges (idempotent if already deleted above)
        session.run(
            "MATCH (e:Entity {entity_id: $dup_id}) DETACH DELETE e",
            dup_id=dup_id,
        )

    return edges_repointed


def merge_group(conn, session, group: dict, dry_run: bool, survivor_picker=None) -> int:
    picker = survivor_picker or pick_survivor
    survivor = picker(group["members"])
    dup_ids = [m["id"] for m in group["members"] if m["id"] != survivor["id"]]

    entity_type_label = survivor.get("entity_type") or group.get("entity_type") or "?"
    if dry_run:
        print(
            f"  [{entity_type_label}] {group['name']} — "
            f"keep {survivor['id'][:8]}…, delete {len(dup_ids)} duplicate(s)"
        )
        return 0

    try:
        edges = merge_memgraph(session, survivor["id"], dup_ids)
        merge_postgres(conn, survivor["id"], dup_ids, group["members"])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    print(
        f"  [{entity_type_label}] {group['name']} — "
        f"merged {len(dup_ids)} duplicate(s), re-pointed {edges} edge(s)"
    )
    return len(dup_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge duplicate entities.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Print what would be merged.")
    group.add_argument("--execute", action="store_true", help="Perform the merge.")
    parser.add_argument(
        "--cross-type",
        action="store_true",
        help="Also merge entities sharing canonical_name but with different entity_type.",
    )
    args = parser.parse_args()
    dry_run = args.dry_run

    total_merged = 0
    with get_connection() as conn, get_graph_driver() as driver:
        groups = fetch_duplicate_groups(conn)
        print(
            f"{'[DRY RUN] ' if dry_run else ''}"
            f"Found {len(groups)} duplicate groups "
            f"({sum(len(g['members']) for g in groups)} total rows)."
        )

        with driver.session() as session:
            for g in groups:
                total_merged += merge_group(conn, session, g, dry_run=dry_run)

        if args.cross_type:
            cross_groups = fetch_cross_type_duplicate_groups(conn)
            print(
                f"{'[DRY RUN] ' if dry_run else ''}"
                f"Found {len(cross_groups)} cross-type duplicate groups "
                f"({sum(len(g['members']) for g in cross_groups)} total rows)."
            )
            with driver.session() as session:
                for g in cross_groups:
                    total_merged += merge_group(
                        conn, session, g, dry_run=dry_run, survivor_picker=pick_survivor_cross_type
                    )

    if total_merged > 0:
        print(f"Reclaiming space from {total_merged} merged row(s)...", flush=True)
        vacuum_analyze_entities()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
