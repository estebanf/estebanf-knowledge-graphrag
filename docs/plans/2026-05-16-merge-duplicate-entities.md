# Merge Duplicate Entities Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write an idempotent script that merges duplicate `(canonical_name, entity_type)` entity groups across both Memgraph and PostgreSQL.

**Architecture:** The script queries Postgres for duplicate groups, picks a survivor per group (non-null embedding first, then oldest `created_at`), merges aliases into the survivor, re-points all `MENTIONS` edges in Memgraph to the survivor, deletes the duplicate Memgraph nodes, and deletes the duplicate Postgres rows. Re-running is safe because already-merged groups no longer appear as duplicates.

**Tech Stack:** Python, psycopg (Postgres), neo4j driver (Memgraph), `rag.db.get_connection`, `rag.graph_db.get_graph_driver`

---

### Task 1: Script skeleton with argument parsing

**Files:**
- Create: `scripts/merge_duplicate_entities.py`

**Step 1: Create the file with imports and argparse**

```python
#!/usr/bin/env python3
"""Merge duplicate Entity nodes (same canonical_name + entity_type) across Memgraph and PostgreSQL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

from rag.db import get_connection
from rag.graph_db import get_graph_driver


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge duplicate entities.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Print what would be merged.")
    group.add_argument("--execute", action="store_true", help="Perform the merge.")
    args = parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 2: Run to verify it parses correctly**

```bash
cd /Users/estebanf/development/knowledge-graphrag
python scripts/merge_duplicate_entities.py --dry-run
```

Expected: exits 0 with no output.

**Step 3: Commit**

```bash
git add scripts/merge_duplicate_entities.py
git commit -m "feat: add merge_duplicate_entities script skeleton"
```

---

### Task 2: Duplicate detection query

**Files:**
- Modify: `scripts/merge_duplicate_entities.py`

**Step 1: Add `fetch_duplicate_groups` function**

Add this function above `main()`:

```python
def fetch_duplicate_groups(conn) -> list[dict]:
    """Return groups where more than one entity shares canonical_name + entity_type.

    Each group dict has:
      - name: str
      - entity_type: str
      - members: list of dicts with keys id, embedding (bool), created_at
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
```

**Step 2: Wire into main and print summary**

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="Merge duplicate entities.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    with get_connection() as conn:
        groups = fetch_duplicate_groups(conn)

    print(f"Found {len(groups)} duplicate groups ({sum(len(g['members']) for g in groups)} total rows).")
    if args.dry_run:
        for g in groups:
            print(f"  [{g['entity_type']}] {g['name']} — {len(g['members'])} copies")
    return 0
```

**Step 3: Run dry-run against real data**

```bash
python scripts/merge_duplicate_entities.py --dry-run
```

Expected: prints the count of duplicate groups and lists each one.

**Step 4: Commit**

```bash
git add scripts/merge_duplicate_entities.py
git commit -m "feat: add duplicate group detection to merge script"
```

---

### Task 3: Survivor selection

**Files:**
- Modify: `scripts/merge_duplicate_entities.py`

**Step 1: Add `pick_survivor` function**

Add above `fetch_duplicate_groups`:

```python
def pick_survivor(members: list[dict]) -> dict:
    """Pick the entity to keep.

    Prefer: has_embedding=True first, then oldest created_at.
    """
    return sorted(members, key=lambda m: (not m["has_embedding"], m["created_at"]))[0]
```

**Step 2: Print survivor in dry-run**

Update the dry-run loop in `main`:

```python
    if args.dry_run:
        for g in groups:
            survivor = pick_survivor(g["members"])
            dup_ids = [m["id"] for m in g["members"] if m["id"] != survivor["id"]]
            print(
                f"  [{g['entity_type']}] {g['name']} — "
                f"keep {survivor['id'][:8]}…, "
                f"delete {len(dup_ids)} duplicate(s)"
            )
```

**Step 3: Run dry-run to verify survivor selection**

```bash
python scripts/merge_duplicate_entities.py --dry-run
```

Expected: each line shows which entity will be kept and how many will be deleted.

**Step 4: Commit**

```bash
git add scripts/merge_duplicate_entities.py
git commit -m "feat: add survivor selection to merge script"
```

---

### Task 4: Postgres alias merge and row deletion

**Files:**
- Modify: `scripts/merge_duplicate_entities.py`

**Step 1: Add `merge_postgres` function**

```python
def merge_postgres(conn, survivor_id: str, dup_ids: list[str], all_members: list[dict]) -> None:
    """Merge aliases from duplicates into survivor, then delete duplicate rows."""
    # Collect all unique aliases across the group
    all_aliases: set[str] = set()
    for m in all_members:
        all_aliases.update(m["aliases"])

    conn.execute(
        "UPDATE entities SET aliases = %s WHERE id = %s",
        (list(all_aliases), survivor_id),
    )
    conn.execute(
        "DELETE FROM entities WHERE id = ANY(%s::uuid[])",
        (dup_ids,),
    )
```

**Step 2: Commit**

```bash
git add scripts/merge_duplicate_entities.py
git commit -m "feat: add postgres merge logic to merge script"
```

---

### Task 5: Memgraph edge re-pointing and node deletion

**Files:**
- Modify: `scripts/merge_duplicate_entities.py`

**Step 1: Add `merge_memgraph` function**

```python
def merge_memgraph(session, survivor_id: str, dup_ids: list[str]) -> int:
    """Re-point MENTIONS edges from duplicates to survivor, delete duplicate nodes.

    Returns the number of edges re-pointed.
    """
    edges_repointed = 0
    for dup_id in dup_ids:
        # Find all chunks that mention this duplicate
        result = session.run(
            "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {entity_id: $dup_id}) "
            "RETURN c.chunk_id AS chunk_id",
            dup_id=dup_id,
        )
        chunk_ids = [r["chunk_id"] for r in result]
        edges_repointed += len(chunk_ids)

        # Re-point each edge to the survivor
        for chunk_id in chunk_ids:
            session.run(
                "MATCH (c:Chunk {chunk_id: $chunk_id}), (s:Entity {entity_id: $survivor_id}) "
                "MERGE (c)-[:MENTIONS {confidence: 1.0}]->(s)",
                chunk_id=chunk_id,
                survivor_id=survivor_id,
            )

        # Delete the duplicate node (and any remaining edges)
        session.run(
            "MATCH (e:Entity {entity_id: $dup_id}) DETACH DELETE e",
            dup_id=dup_id,
        )

    return edges_repointed
```

**Step 2: Commit**

```bash
git add scripts/merge_duplicate_entities.py
git commit -m "feat: add memgraph merge logic to merge script"
```

---

### Task 6: Wire execute mode end-to-end

**Files:**
- Modify: `scripts/merge_duplicate_entities.py`

**Step 1: Add `merge_group` function and update main**

Add above `main`:

```python
def merge_group(conn, session, group: dict, dry_run: bool) -> None:
    survivor = pick_survivor(group["members"])
    dup_ids = [m["id"] for m in group["members"] if m["id"] != survivor["id"]]

    if dry_run:
        print(
            f"  [{group['entity_type']}] {group['name']} — "
            f"keep {survivor['id'][:8]}…, delete {len(dup_ids)} duplicate(s)"
        )
        return

    edges = merge_memgraph(session, survivor["id"], dup_ids)
    merge_postgres(conn, survivor["id"], dup_ids, group["members"])
    conn.commit()
    print(
        f"  [{group['entity_type']}] {group['name']} — "
        f"merged {len(dup_ids)} duplicate(s), re-pointed {edges} edge(s)"
    )
```

Replace `main` with:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="Merge duplicate entities.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    dry_run = args.dry_run

    with get_connection() as conn, get_graph_driver() as driver:
        groups = fetch_duplicate_groups(conn)
        print(
            f"{'[DRY RUN] ' if dry_run else ''}"
            f"Found {len(groups)} duplicate groups "
            f"({sum(len(g['members']) for g in groups)} total rows)."
        )

        with driver.session() as session:
            for g in groups:
                merge_group(conn, session, g, dry_run=dry_run)

    print("Done.")
    return 0
```

**Step 2: Run dry-run one final time to verify full flow**

```bash
python scripts/merge_duplicate_entities.py --dry-run
```

Expected: lists all groups with survivor and duplicate count, no data changed.

**Step 3: Run execute**

```bash
python scripts/merge_duplicate_entities.py --execute
```

Expected: each group prints merged count and edges re-pointed.

**Step 4: Verify no duplicates remain**

```bash
python scripts/merge_duplicate_entities.py --dry-run
```

Expected: `Found 0 duplicate groups`.

**Step 5: Commit**

```bash
git add scripts/merge_duplicate_entities.py
git commit -m "feat: complete merge_duplicate_entities script"
```

---

## Final verification

Run `--dry-run` again after `--execute` completes — it should report 0 groups. That confirms idempotency: re-running a second `--execute` would also be a no-op.
