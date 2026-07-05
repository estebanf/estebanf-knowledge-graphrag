#!/usr/bin/env python3
"""Weekly corpus maintenance sweep (U7, R12/R13).

One command keeps the corpus deduplicated and the vector indexes healthy:

  1. Entity merge       — reuses scripts/merge_semantic_duplicates.py's
                           run_entity_merge() (fuzzy + cosine candidate
                           detection, union-find clustering, Postgres +
                           Memgraph merge). It vacuums `entities` itself.
  2. Insight merge       — the load-bearing phase (doc-review P2): finds
                           near-duplicate insights via the same binary-HNSW
                           prefilter + full-precision rerank pattern U1 uses
                           for intake (src/rag/insight_extraction.py's
                           upsert_insight), scoped by default to insights
                           created in the last `--since` days (probes)
                           against the full corpus (candidates) — never an
                           unscoped corpus-wide self-join. `--full` opts into
                           a whole-corpus sweep (every insight is a probe);
                           documented as slower and occasional, not the
                           weekly default.
  3. Orphan/consistency  — reuses src/rag/ingestion.py's
                           _cleanup_orphan_insights() directly; additionally
                           *reports* (does not auto-fix) Postgres rows with
                           no matching Memgraph node.
  4. Index/stats health  — VACUUM ANALYZE insights/chunks/entities, print
                           table + index sizes, prewarm the binary HNSW
                           vector indexes.

Argparse mode convention (mirrors scripts/merge_semantic_duplicates.py):
`--dry-run` and `--execute` are a required, mutually exclusive group —
`--dry-run` is the safe, no-writes default choice an operator reaches for
first; there is no silent-default mode, so it's always explicit which one
ran.

Concurrency guard (P1, doc-review): before any `--execute` phase mutates
data, this script (a) refuses to run if any `jobs` row is non-terminal
(`status = 'pending'` or `status LIKE 'processing:%'` — intake's own status
vocabulary, see src/rag/worker.py) and (b) acquires a fixed-key Postgres
advisory lock (pg_try_advisory_lock) so two overlapping `--execute` runs of
this script can't race each other either. Rationale: an in-flight intake job
can hold or re-create an insight id this script is merging away between
fetch and write, corrupting either side. `--dry-run` never acquires the lock
or checks for active jobs — it performs no writes, so there is nothing to
race. The lock is held for the entire `--execute` run, including phase 4
(VACUUM ANALYZE is safe to run concurrently with intake on its own, but
holding the same lock through phase 4 keeps the guard's semantics simple:
one lock, one full `--execute` run).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.config import settings
from rag.db import (
    get_connection,
    prewarm_vector_indexes,
    set_hnsw_ef_search,
    vacuum_analyze_chunks,
    vacuum_analyze_entities,
    vacuum_analyze_insights,
)
from rag.graph_db import get_graph_driver
from rag.ingestion import _cleanup_orphan_insights

sys.path.insert(0, str(Path(__file__).parent))
import merge_semantic_duplicates as entity_merge  # noqa: E402

UnionFind = entity_merge.UnionFind

# Fixed application-specific advisory lock key. Any 64-bit int works; this one
# has no other meaning. Do not reuse this constant for an unrelated lock.
MAINTENANCE_LOCK_KEY = 875_321_001

_MERGE_RERANK_LIMIT = 20  # top-N reranked candidates per probe, then threshold-filtered


# ── Phase 2: insight merge ────────────────────────────────────────────────────

def _embedding_literal(embedding) -> str:
    if isinstance(embedding, str):
        return embedding
    return "[" + ",".join(str(v) for v in embedding) + "]"


def fetch_probe_insights(conn, since_days: int | None) -> list[tuple[str, str]]:
    """Return (id, embedding_text) rows to use as merge-candidate probes.

    `since_days=None` means `--full`: every embedded insight is a probe.
    Otherwise scoped to insights created within the last `since_days` days —
    the default, load-bearing scoping requirement (R12/doc-review P2) that
    keeps this phase index-assisted and boundable instead of an unscoped
    corpus-wide cross-product.
    """
    if since_days is None:
        rows = conn.execute(
            "SELECT id::text, embedding::text FROM insights WHERE embedding IS NOT NULL"
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id::text, embedding::text FROM insights
            WHERE embedding IS NOT NULL
              AND created_at >= now() - (%s || ' days')::interval
            """,
            (since_days,),
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def find_insight_candidate_pairs(
    conn,
    probes: list[tuple[str, str]],
    cosine_threshold: float,
    prefilter_candidates: int,
) -> list[tuple[str, str, float]]:
    """For each probe insight, prefilter+rerank against the full corpus
    server-side (no client-side vector literals for a whole-corpus compare)
    using the same binary-HNSW prefilter + full-precision-rerank shape as
    `src/rag/insight_extraction.upsert_insight` (U1). Returns (probe_id,
    candidate_id, cosine_similarity) triples above `cosine_threshold`.
    """
    if not probes:
        return []

    set_hnsw_ef_search(conn, prefilter_candidates)
    pairs: list[tuple[str, str, float]] = []
    with conn.cursor() as cur:
        for probe_id, emb_text in probes:
            cur.execute(
                """
                WITH insight_prefilter AS MATERIALIZED (
                    SELECT id
                    FROM insights
                    WHERE embedding IS NOT NULL
                      AND id != %s
                    ORDER BY binary_quantize(embedding)::bit(4096) <~> binary_quantize(%s::vector)::bit(4096)
                    LIMIT %s
                )
                SELECT i.id, 1 - (i.embedding <=> %s::vector) AS sim
                FROM insight_prefilter p
                JOIN insights i ON i.id = p.id
                ORDER BY i.embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    probe_id,
                    emb_text,
                    prefilter_candidates,
                    emb_text,
                    emb_text,
                    _MERGE_RERANK_LIMIT,
                ),
            )
            for cand_id, sim in cur.fetchall():
                cand_id = str(cand_id)
                sim = float(sim)
                if cand_id != probe_id and sim >= cosine_threshold:
                    pairs.append((probe_id, cand_id, sim))
    return pairs


def fetch_insight_meta(conn, ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    rows = conn.execute(
        "SELECT id::text, created_at FROM insights WHERE id = ANY(%s::uuid[])",
        (ids,),
    ).fetchall()
    return {r[0]: {"id": r[0], "created_at": r[1]} for r in rows}


def pick_insight_survivor(members: list[dict]) -> dict:
    """Survivor = oldest row in the cluster (R13)."""
    return sorted(members, key=lambda m: (m["created_at"] is None, m["created_at"]))[0]


def merge_insights_postgres(conn, survivor_id: str, dup_ids: list[str]) -> None:
    """Re-point `chunk_insights` from losers to the survivor, then delete the
    loser `insights` rows.

    `chunk_insights` has a composite PRIMARY KEY (chunk_id, insight_id)
    (scripts/migrate/007_insights.sql) — if a chunk already links to BOTH the
    survivor and a loser, blindly repointing the loser's row would collide
    with the survivor's existing row. The UPDATE below only repoints rows
    whose chunk has no existing link to the survivor; any chunk_insights row
    left pointing at a loser (i.e. the conflicting case) is then removed for
    free by `ON DELETE CASCADE` when the loser `insights` row is deleted, so
    a single surviving row remains for that chunk (no explicit
    ON CONFLICT/DELETE-then-skip dance needed).
    """
    if not dup_ids:
        return
    conn.execute(
        """
        UPDATE chunk_insights SET insight_id = %s
        WHERE insight_id = ANY(%s::uuid[])
          AND chunk_id NOT IN (
              SELECT chunk_id FROM chunk_insights WHERE insight_id = %s
          )
        """,
        (survivor_id, dup_ids, survivor_id),
    )
    conn.execute("DELETE FROM insights WHERE id = ANY(%s::uuid[])", (dup_ids,))


def merge_insights_memgraph(session, survivor_id: str, dup_ids: list[str]) -> int:
    """Re-point Memgraph `CONTAINS` and `RELATED_TO` edges from each loser to
    the survivor, then delete the loser `Insight` node. Mirrors
    `merge_semantic_duplicates.merge_memgraph`'s entity-merge shape.

    Self-edges: a `RELATED_TO` re-point from a loser to `other` is skipped
    when `other.insight_id == survivor_id` (the `WHERE other.insight_id <>
    $survivor_id` clauses below) so merging never creates a
    survivor-to-itself edge.
    """
    edges_repointed = 0
    for dup_id in dup_ids:
        result = session.run(
            "MATCH (c:Chunk)-[old:CONTAINS]->(i:Insight {insight_id: $dup_id}) "
            "MATCH (s:Insight {insight_id: $survivor_id}) "
            "MERGE (c)-[:CONTAINS {topics: old.topics}]->(s) "
            "WITH i, count(old) AS repointed "
            "RETURN repointed",
            dup_id=dup_id,
            survivor_id=survivor_id,
        )
        record = result.single()
        if record:
            edges_repointed += record["repointed"]

        session.run(
            "MATCH (i:Insight {insight_id: $dup_id})-[r:RELATED_TO]->(other:Insight) "
            "WHERE other.insight_id <> $survivor_id "
            "MATCH (s:Insight {insight_id: $survivor_id}) "
            "MERGE (s)-[:RELATED_TO {similarity: r.similarity}]->(other)",
            dup_id=dup_id,
            survivor_id=survivor_id,
        )
        session.run(
            "MATCH (other:Insight)-[r:RELATED_TO]->(i:Insight {insight_id: $dup_id}) "
            "WHERE other.insight_id <> $survivor_id "
            "MATCH (s:Insight {insight_id: $survivor_id}) "
            "MERGE (other)-[:RELATED_TO {similarity: r.similarity}]->(s)",
            dup_id=dup_id,
            survivor_id=survivor_id,
        )
        session.run(
            "MATCH (i:Insight {insight_id: $dup_id}) DETACH DELETE i",
            dup_id=dup_id,
        )

    return edges_repointed


def run_insight_merge(
    dry_run: bool,
    since_days: int | None,
    cosine_threshold: float,
    prefilter_candidates: int = settings.INSIGHT_PREFILTER_CANDIDATES,
) -> int:
    """Phase 2: near-duplicate insight merge. Returns total merged count."""
    prefix = "[DRY RUN] " if dry_run else ""
    scope_desc = "whole corpus (--full)" if since_days is None else f"last {since_days} day(s)"
    print(f"{prefix}Insight merge: probe scope = {scope_desc}.", flush=True)

    with get_connection() as conn:
        probes = fetch_probe_insights(conn, since_days)
        print(f"{prefix}Probing {len(probes)} insight(s) against the corpus...", flush=True)
        pairs = find_insight_candidate_pairs(conn, probes, cosine_threshold, prefilter_candidates)

    print(f"{prefix}Insight candidate pairs above threshold: {len(pairs)}.", flush=True)

    uf = UnionFind()
    for id_a, id_b, _sim in pairs:
        uf.union(id_a, id_b)
    clusters = uf.groups()

    if not clusters:
        print(f"{prefix}No insight duplicate clusters found.")
        return 0

    with get_connection() as conn:
        all_ids = [eid for members in clusters.values() for eid in members]
        meta = fetch_insight_meta(conn, all_ids)

    total_merged = 0
    total_edges = 0
    if dry_run:
        for cluster_ids in clusters.values():
            members = [meta[i] for i in cluster_ids if i in meta]
            if len(members) < 2:
                continue
            survivor = pick_insight_survivor(members)
            dup_ids = [m["id"] for m in members if m["id"] != survivor["id"]]
            print(f"  survivor={survivor['id']} absorbs: {dup_ids}")
            total_merged += len(dup_ids)
        print(f"{prefix}Would merge {total_merged} duplicate insight(s) into survivors.")
        return total_merged

    with get_connection() as conn, get_graph_driver() as driver:
        with driver.session() as session:
            for cluster_ids in clusters.values():
                members = [meta[i] for i in cluster_ids if i in meta]
                if len(members) < 2:
                    continue
                survivor = pick_insight_survivor(members)
                dup_ids = [m["id"] for m in members if m["id"] != survivor["id"]]
                try:
                    edges = merge_insights_memgraph(session, survivor["id"], dup_ids)
                    merge_insights_postgres(conn, survivor["id"], dup_ids)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                print(
                    f"  survivor={survivor['id']} merged {len(dup_ids)} duplicate(s), "
                    f"re-pointed {edges} edge(s)"
                )
                total_merged += len(dup_ids)
                total_edges += edges

    print(f"Merged {total_merged} duplicate insight(s), re-pointed {total_edges} edge(s).")
    if total_merged > 0:
        vacuum_analyze_insights()
    return total_merged


# ── Phase 3: orphan/consistency sweep ────────────────────────────────────────

def report_missing_memgraph_nodes(conn, driver) -> dict[str, int]:
    """Report (do not fix) Postgres `insights`/`entities` rows with no
    corresponding Memgraph node — the reverse direction of
    `_cleanup_orphan_insights`, which only removes Memgraph-side orphans.
    """
    counts = {"insights_missing_node": 0, "entities_missing_node": 0}
    if not driver:
        return counts

    with driver.session() as session:
        insight_ids = {
            r["id"] for r in session.run("MATCH (i:Insight) RETURN i.insight_id AS id")
        }
        entity_ids = {
            r["id"] for r in session.run("MATCH (e:Entity) RETURN e.entity_id AS id")
        }

    pg_insight_ids = {str(r[0]) for r in conn.execute("SELECT id FROM insights").fetchall()}
    pg_entity_ids = {str(r[0]) for r in conn.execute("SELECT id FROM entities").fetchall()}

    counts["insights_missing_node"] = len(pg_insight_ids - insight_ids)
    counts["entities_missing_node"] = len(pg_entity_ids - entity_ids)
    return counts


def run_consistency_sweep(dry_run: bool) -> dict[str, int]:
    prefix = "[DRY RUN] " if dry_run else ""
    with get_connection() as conn, get_graph_driver() as driver:
        counts = report_missing_memgraph_nodes(conn, driver)
        print(
            f"{prefix}Postgres rows with no Memgraph node (report-only): "
            f"{counts['insights_missing_node']} insight(s), "
            f"{counts['entities_missing_node']} entity/entities."
        )
        if dry_run:
            print(f"{prefix}Would run orphan-insight cleanup (no chunk_insights links).")
            return counts

        _cleanup_orphan_insights(conn, driver)
        conn.commit()
        print("Orphan-insight cleanup complete.")
    return counts


# ── Phase 4: index/stats health ──────────────────────────────────────────────

_HEALTH_TABLES = ("insights", "chunks", "entities")


def run_health_phase(dry_run: bool) -> None:
    prefix = "[DRY RUN] " if dry_run else ""
    if dry_run:
        print(f"{prefix}Would VACUUM ANALYZE {', '.join(_HEALTH_TABLES)} and prewarm vector indexes.")
        return

    vacuum_analyze_insights()
    vacuum_analyze_chunks()
    vacuum_analyze_entities()

    with get_connection() as conn:
        for table in _HEALTH_TABLES:
            size = conn.execute("SELECT pg_relation_size(%s)", (table,)).fetchone()[0]
            idx_size = conn.execute("SELECT pg_indexes_size(%s)", (table,)).fetchone()[0]
            print(f"  {table}: table={size} bytes, indexes={idx_size} bytes")

    blocks = prewarm_vector_indexes()
    print(f"Prewarmed vector indexes: {blocks}")


# ── Concurrency guard ─────────────────────────────────────────────────────────

def count_active_jobs(conn) -> int:
    """Non-terminal `jobs` rows — intake's own status vocabulary
    (`src/rag/worker.py`): 'pending' or 'processing:<stage>'. 'completed' and
    'failed:<stage>' are terminal.
    """
    return conn.execute(
        "SELECT count(*) FROM jobs WHERE status = 'pending' OR status LIKE 'processing:%'"
    ).fetchone()[0]


def try_acquire_maintenance_lock(conn) -> bool:
    return conn.execute(
        "SELECT pg_try_advisory_lock(%s)", (MAINTENANCE_LOCK_KEY,)
    ).fetchone()[0]


def release_maintenance_lock(conn) -> None:
    conn.execute("SELECT pg_advisory_unlock(%s)", (MAINTENANCE_LOCK_KEY,))


# ── Cache invalidation ────────────────────────────────────────────────────────


def _invalidate_entity_edge_cache(merged_entity_ids: set[str]) -> None:
    """Delete entity_semantic_edges rows touching merged entities and orphaned
    entity ids that no longer exist in `entities`.  Skip entirely when
    `merged_entity_ids` is empty and no orphans exist (mirroring the existing
    "only vacuum when total_merged > 0" discipline)."""

    if not merged_entity_ids:
        with get_connection() as conn:
            orphan_count = conn.execute(
                """SELECT count(*) FROM entity_semantic_edges ese
                   WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = ese.entity_a)
                      OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = ese.entity_b)"""
            ).fetchone()[0]
        if orphan_count == 0:
            print("Cache invalidation: nothing to do (no merges, no orphans).")
            return
        print(f"Cache invalidation: deleting {orphan_count} orphaned edge row(s)...")
        with get_connection() as conn:
            conn.execute(
                """DELETE FROM entity_semantic_edges
                   WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = entity_semantic_edges.entity_a)
                      OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = entity_semantic_edges.entity_b)"""
            )
        print(f"Cache invalidation: {orphan_count} orphaned edge row(s) deleted.")
        return

    entity_list = list(merged_entity_ids)
    with get_connection() as conn:
        touched = conn.execute(
            """DELETE FROM entity_semantic_edges
               WHERE entity_a = ANY(%s::uuid[]) OR entity_b = ANY(%s::uuid[])""",
            (entity_list, entity_list),
        ).rowcount or 0

        orphan = conn.execute(
            """DELETE FROM entity_semantic_edges
               WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = entity_semantic_edges.entity_a)
                  OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = entity_semantic_edges.entity_b)"""
        ).rowcount or 0

    total = touched + orphan
    if total > 0:
        print(
            f"Cache invalidation: deleted {touched} edge row(s) touching merged entities, "
            f"{orphan} orphaned row(s) ({total} total)."
        )
    else:
        print("Cache invalidation: nothing to delete (merged entities had no cached edges).")


# ── Community-run concurrency guard ───────────────────────────────────────────


def _reap_stale_community_runs(conn) -> int:
    """Mark stale community_runs rows as failed. Returns number of rows reaped."""
    return conn.execute(
        """UPDATE community_runs
           SET status = 'failed',
               error = 'stale',
               updated_at = now()
           WHERE status = 'running'
             AND updated_at < now() - make_interval(secs => %s)""",
        (settings.COMMUNITY_RUN_STALE_SECONDS,),
    ).rowcount or 0


def _count_active_community_runs(conn) -> int:
    return conn.execute(
        "SELECT count(*) FROM community_runs WHERE status = 'running'"
    ).fetchone()[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Weekly corpus maintenance sweep: entity merge, insight merge, "
            "orphan/consistency sweep, index/stats health. "
            "--execute refuses to run while any `jobs` row is non-terminal "
            "(status 'pending' or 'processing:<stage>') and holds a fixed "
            "Postgres advisory lock for the whole run, so it never races an "
            "in-flight intake job or another concurrent --execute run. "
            "--dry-run performs no writes and skips both the active-job "
            "check and the lock."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Report counts, write nothing (default-safe mode).")
    mode.add_argument("--execute", action="store_true", help="Apply merges/cleanup/vacuum for real.")

    parser.add_argument("--skip-entities", action="store_true", help="Skip phase 1 (entity merge).")
    parser.add_argument("--skip-insights", action="store_true", help="Skip phase 2 (insight merge).")
    parser.add_argument("--skip-consistency", action="store_true", help="Skip phase 3 (orphan/consistency sweep).")
    parser.add_argument("--skip-health", action="store_true", help="Skip phase 4 (index/stats health).")

    parser.add_argument(
        "--since", type=int, default=7,
        help=(
            "Insight-merge phase 2 default scoping: only insights created in the last "
            "N days are used as probes against the full corpus (R12/doc-review P2). "
            "Ignored if --full is given."
        ),
    )
    parser.add_argument(
        "--full", action="store_true",
        help=(
            "Insight-merge phase 2: probe every insight against the whole corpus "
            "instead of scoping to --since. Slower, occasional operation — not the "
            "weekly default; do not schedule this as the routine weekly run at corpus "
            "sizes around 100k+ insights."
        ),
    )
    parser.add_argument(
        "--insight-cosine-threshold", type=float,
        default=settings.INSIGHT_DEDUP_COSINE_THRESHOLD,
        help="Cosine similarity threshold for insight duplicate candidates.",
    )
    return parser


def _run_phases(args: argparse.Namespace) -> int:
    dry_run = not args.execute
    since_days = None if args.full else args.since

    if not args.skip_entities:
        print("== Phase 1: entity merge ==", flush=True)
        merged_count, merged_entity_ids = entity_merge.run_entity_merge(dry_run=dry_run)
        if not dry_run:
            _invalidate_entity_edge_cache(merged_entity_ids)
    else:
        print("== Phase 1: entity merge (skipped) ==")
        merged_count = 0

    if not args.skip_insights:
        print("== Phase 2: insight merge ==", flush=True)
        run_insight_merge(
            dry_run=dry_run,
            since_days=since_days,
            cosine_threshold=args.insight_cosine_threshold,
        )
    else:
        print("== Phase 2: insight merge (skipped) ==")

    if not args.skip_consistency:
        print("== Phase 3: orphan/consistency sweep ==", flush=True)
        run_consistency_sweep(dry_run=dry_run)
    else:
        print("== Phase 3: orphan/consistency sweep (skipped) ==")

    if not args.skip_health:
        print("== Phase 4: index/stats health ==", flush=True)
        run_health_phase(dry_run=dry_run)
    else:
        print("== Phase 4: index/stats health (skipped) ==")

    print("Done.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.execute:
        return _run_phases(args)

    with get_connection() as guard_conn:
        active = count_active_jobs(guard_conn)
        if active > 0:
            print(
                f"Refusing --execute: {active} job(s) in flight (status pending/"
                f"processing:*). Wait for them to finish, or rerun with --dry-run."
            )
            return 1

        reaped = _reap_stale_community_runs(guard_conn)
        if reaped > 0:
            print(
                f"Reaped {reaped} stale community run(s) (status 'running' with "
                f"updated_at older than {settings.COMMUNITY_RUN_STALE_SECONDS}s)."
            )

        active_runs = _count_active_community_runs(guard_conn)
        if active_runs > 0:
            print(
                f"Refusing --execute: {active_runs} community run(s) in flight "
                f"(status 'running'). Wait for them to finish, or rerun with --dry-run."
            )
            return 1

        if not try_acquire_maintenance_lock(guard_conn):
            print(
                "Refusing --execute: another weekly_maintenance --execute run "
                "holds the maintenance advisory lock."
            )
            return 1

        try:
            return _run_phases(args)
        finally:
            release_maintenance_lock(guard_conn)


if __name__ == "__main__":
    raise SystemExit(main())
