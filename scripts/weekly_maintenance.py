#!/usr/bin/env python3
"""Weekly corpus maintenance sweep (U7, R12/R13).

One command keeps the corpus deduplicated and the vector indexes healthy:

  0. Entity audit        — OPTIONAL, Jev-backed (off unless
                           MAINTENANCE_JEV_ENABLED=true or `--jev`; see
                           src/rag/entity_audit.py). Scores every chunk->entity
                           MENTIONS edge with Jev into `entity_mention_audit`
                           (scoring writes ONLY that table, so it also runs
                           while intake jobs are active), then applies the
                           audit: drop mentions that are not real named
                           entities (and entities left with no mention),
                           retype off-schema entities, and write salience into
                           MENTIONS.confidence. Apply mutates the corpus, so it
                           follows the same --execute gate as every other
                           mutating phase. Runs before the entity merge.
  1. Entity merge       — reuses scripts/merge_semantic_duplicates.py's
                           run_entity_merge() (fuzzy + cosine candidate
                           detection, union-find clustering, Postgres +
                           Memgraph merge). It vacuums `entities` itself. The
                           survivor of a cluster is its row with the most
                           MENTIONS edges. With Jev on, every candidate pair is
                           first confirmed by Jev: only confirmed pairs merge,
                           borderline ones go to `entity_merge_review`.
  1b. Entity names       — OPTIONAL, Jev-backed (same gate as phase 0; skip
                           with `--skip-entity-names`): for the hub entities
                           (>= ENTITY_NAME_REVIEW_MIN_ALIASES aliases, at most
                           `--name-limit`), Jev picks the correct spelling among
                           canonical name + aliases; a confident rename either
                           updates the name or merges into the existing row that
                           already has it (src/rag/entity_names.py).
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
                           weekly default. With Jev on, pairs whose cosine is in
                           [INSIGHT_NEAR_DUP_COSINE_LOW, threshold) are also
                           collected (same single prefilter pass) and confirmed
                           by Jev: same claim -> merge, related -> the
                           `insight_pair_review` table, else dropped. At most
                           `--near-dup-limit` pairs per run.
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

Jev scoring is the one exception to "nothing runs before the guard": it only
inserts into `entity_mention_audit`, never touches entities/chunks/Memgraph,
so it runs BEFORE the active-job check and may proceed while jobs are in
flight (a 2-hour `--full` scoring run can overlap intake). Every corpus
mutation (including the audit apply) still happens only after the guard and
under the advisory lock. By default scoring runs under `--execute` and not
under `--dry-run`; `--dry-run --score` spends Jev credits but still writes only
the audit table.

The other Jev phases (merge confirmation, entity names, insight near-duplicates)
judge pairs/entities with Jev in a `--dry-run` too, which spends Jev credits, but
a dry-run writes nothing to the corpus and nothing to the review tables. Only
`--execute` (behind the active-job check and advisory lock) merges, renames and
inserts review rows. A Jev error or a missing OPENROUTER_API_KEY never causes a
merge or rename: the item stays unconfirmed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag import entity_audit, entity_names, jev_dedup
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


def fetch_insight_texts(conn, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    rows = conn.execute(
        "SELECT id::text, content FROM insights WHERE id = ANY(%s::uuid[])",
        (ids,),
    ).fetchall()
    return {r[0]: r[1] or "" for r in rows}


def confirm_near_duplicates(
    band_pairs: list[tuple[str, str, float]],
    dry_run: bool,
    near_dup_limit: int,
) -> list[tuple[str, str, float]]:
    """Jev-confirm insight pairs whose cosine is in the near-duplicate band.

    Pairs are normalized/deduplicated first (with `--since`, both members are
    often probes, which would double the Jev calls and slip the reversed copy
    past the review table's UNIQUE constraint), pairs that already have an
    `insight_pair_review` row (any status) are dropped, the rest are sorted by
    cosine descending and capped at `near_dup_limit`. Routing (`rag.jev_dedup`):
    score >= 1.5 -> returned for merging; 0.5..1.5 -> `insight_pair_review`
    under --execute (only reported in a dry-run); lower -> dropped; Jev error ->
    unconfirmed (not merged). Calls Jev even in a dry-run.
    """
    prefix = "[DRY RUN] " if dry_run else ""
    pairs = jev_dedup.dedupe_pairs(band_pairs)
    if not pairs:
        print(f"{prefix}Insight near-duplicate band: no candidate pairs.", flush=True)
        return []

    with get_connection() as conn:
        reviewed = jev_dedup.fetch_reviewed_insight_pairs(conn)
    fresh = [p for p in pairs if jev_dedup.normalize_pair(p[0], p[1]) not in reviewed]
    fresh.sort(key=lambda p: -p[2])
    capped = fresh[:near_dup_limit]
    print(
        f"{prefix}Insight near-duplicate band: {len(pairs)} pair(s), "
        f"{len(pairs) - len(fresh)} already in insight_pair_review; judging "
        f"{len(capped)} (cap --near-dup-limit {near_dup_limit}, highest cosine first)...",
        flush=True,
    )
    if not capped:
        return []

    with get_connection() as conn:
        texts = fetch_insight_texts(conn, [i for p in capped for i in p[:2]])
    judgements, stats = jev_dedup.judge_insight_pairs(capped, texts)
    counts = jev_dedup.summarize(judgements)
    print(
        f"{prefix}Jev insight pairs: {counts['merge']} same claim (merge), {counts['review']} related "
        f"(review), {counts['drop']} different, {counts['error']} unconfirmed (Jev error); "
        f"{stats.describe()}",
        flush=True,
    )
    if counts["review"]:
        if dry_run:
            print(f"{prefix}Would add {counts['review']} pair(s) to insight_pair_review.")
        else:
            with get_connection() as conn:
                added = jev_dedup.insert_insight_reviews(conn, judgements)
            print(f"Added {added} pair(s) to insight_pair_review.")
    return [(j.id_a, j.id_b, j.similarity) for j in judgements if j.route == jev_dedup.ROUTE_MERGE]


def run_insight_merge(
    dry_run: bool,
    since_days: int | None,
    cosine_threshold: float,
    prefilter_candidates: int = settings.INSIGHT_PREFILTER_CANDIDATES,
    use_jev: bool = False,
    near_dup_low: float = settings.INSIGHT_NEAR_DUP_COSINE_LOW,
    near_dup_limit: int = 2000,
) -> int:
    """Phase 2: near-duplicate insight merge. Returns total merged count.

    With `use_jev`, one prefilter pass runs at `min(near_dup_low, cosine_threshold)`
    and the result is split client-side: pairs at or above `cosine_threshold`
    merge as before, the band [near_dup_low, cosine_threshold) goes to Jev
    (`confirm_near_duplicates`) and only its confirmed pairs join the merge.
    """
    prefix = "[DRY RUN] " if dry_run else ""
    scope_desc = "whole corpus (--full)" if since_days is None else f"last {since_days} day(s)"
    print(f"{prefix}Insight merge: probe scope = {scope_desc}.", flush=True)

    search_threshold = min(near_dup_low, cosine_threshold) if use_jev else cosine_threshold
    with get_connection() as conn:
        probes = fetch_probe_insights(conn, since_days)
        print(f"{prefix}Probing {len(probes)} insight(s) against the corpus...", flush=True)
        pairs = find_insight_candidate_pairs(conn, probes, search_threshold, prefilter_candidates)

    if use_jev:
        band_pairs = [p for p in pairs if p[2] < cosine_threshold]
        pairs = [p for p in pairs if p[2] >= cosine_threshold]
        print(f"{prefix}Insight candidate pairs above threshold: {len(pairs)}.", flush=True)
        pairs = pairs + confirm_near_duplicates(band_pairs, dry_run, near_dup_limit)
    else:
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
            # get_connection() is not autocommit and closes without committing,
            # so without this the DELETE is silently rolled back.
            conn.commit()
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
        conn.commit()  # not autocommit: see the note above

    total = touched + orphan
    if total > 0:
        print(
            f"Cache invalidation: deleted {touched} edge row(s) touching merged entities, "
            f"{orphan} orphaned row(s) ({total} total)."
        )
    else:
        print("Cache invalidation: nothing to delete (merged entities had no cached edges).")


# ── Phase 0: Jev entity audit ─────────────────────────────────────────────────


def jev_enabled(args: argparse.Namespace) -> bool:
    """`--no-jev` forces off, `--jev` forces on, otherwise MAINTENANCE_JEV_ENABLED."""
    if args.no_jev:
        return False
    return bool(args.jev or settings.MAINTENANCE_JEV_ENABLED)


def entity_audit_active(args: argparse.Namespace) -> bool:
    return jev_enabled(args) and not args.skip_entity_audit


def should_score(args: argparse.Namespace) -> bool:
    """Explicit `--score`/`--no-score` wins; otherwise score under --execute only."""
    if args.score is not None:
        return bool(args.score)
    return bool(args.execute)


def run_entity_audit_scoring(args: argparse.Namespace):
    """Jev-score unscored mentions in scope. Writes ONLY `entity_mention_audit`
    (resumable, committed per batch), so it is safe with intake jobs running.
    Returns the `ScoreStats`, or None when scoring cannot start."""
    if not settings.OPENROUTER_API_KEY:
        print("Entity audit scoring needs OPENROUTER_API_KEY (Jev runs on OpenRouter); not set.")
        return None
    since_days = None if args.full else args.since
    with get_connection() as conn, get_graph_driver() as driver:
        with driver.session() as session:
            return entity_audit.score_mentions(
                conn, session, since_days, rescore=args.rescore
            )


def run_entity_audit_apply(dry_run: bool, since_days: int | None, rescore: bool) -> None:
    """Apply (or, in dry-run, report) the audit. Corpus mutations: callers
    must hold the maintenance lock with no active jobs."""
    if dry_run:
        with get_connection() as conn, get_graph_driver() as driver:
            with driver.session() as session:
                entity_audit.dry_run_report(conn, session, since_days, rescore=rescore)
        return

    with get_connection() as conn, get_graph_driver() as driver:
        with driver.session() as session:
            result = entity_audit.apply_audit(conn, session, since_days)
    _invalidate_entity_edge_cache(result.deleted_entity_ids)
    if result.deleted_entity_ids:
        vacuum_analyze_entities()


# ── Phase 1b: Jev entity name correction ──────────────────────────────────────


def entity_names_active(args: argparse.Namespace) -> bool:
    return jev_enabled(args) and not args.skip_entity_names


def _merge_entities_into(session, conn, survivor_id: str, dup_ids: list[str], members: list[dict]) -> int:
    """The existing entity merge (Memgraph MENTIONS repoint first, then
    Postgres aliases + delete), handed to `rag.entity_names` so renames that
    collide with an existing row merge exactly like every other merge. The
    caller commits."""
    edges = entity_merge.merge_memgraph(session, survivor_id, dup_ids)
    entity_merge.merge_postgres(conn, survivor_id, dup_ids, members)
    return edges


def run_entity_name_correction(dry_run: bool, name_limit: int) -> set[str]:
    """Phase 1b. Returns the entity ids touched (empty in a dry-run); the
    semantic-edge cache is invalidated and `entities` vacuumed here when
    anything changed."""
    with get_connection() as conn, get_graph_driver() as driver:
        with driver.session() as session:
            result = entity_names.run_name_correction(
                conn,
                session,
                dry_run=dry_run,
                limit=name_limit,
                merge_fn=_merge_entities_into,
                pick_survivor=entity_merge.pick_survivor,
            )
    if dry_run or not result.touched_ids:
        return set()
    _invalidate_entity_edge_cache(result.touched_ids)
    if result.deleted_ids:
        vacuum_analyze_entities()
    return set(result.touched_ids)


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

    jev_group = parser.add_mutually_exclusive_group()
    jev_group.add_argument(
        "--jev", action="store_true",
        help=(
            "Enable the Jev phases for this run (default: MAINTENANCE_JEV_ENABLED): entity audit, "
            "Jev-confirmed entity merges, entity name correction, Jev-confirmed near-duplicate "
            "insights. They spend Jev credits even with --dry-run."
        ),
    )
    jev_group.add_argument(
        "--no-jev", action="store_true",
        help="Force every Jev maintenance phase off for this run.",
    )
    parser.add_argument(
        "--skip-entity-audit", action="store_true",
        help="Skip phase 0 (Jev entity audit) even when Jev maintenance is enabled.",
    )
    parser.add_argument(
        "--score", action=argparse.BooleanOptionalAction, default=None,
        help=(
            "Entity audit: Jev-score unscored mentions (writes only entity_mention_audit and "
            "spends Jev credits; ~$6 / ~2h for --full over ~160k chunks at concurrency 8). "
            "Default: score under --execute, do not score under --dry-run. "
            "--dry-run --score is allowed and still writes only the audit table."
        ),
    )
    parser.add_argument(
        "--rescore", action="store_true",
        help="Entity audit: re-score mentions that already have an audit row.",
    )
    parser.add_argument(
        "--skip-entity-names", action="store_true",
        help="Skip phase 1b (Jev entity name correction) even when Jev maintenance is enabled.",
    )
    parser.add_argument(
        "--name-limit", type=int, default=500,
        help=(
            "Phase 1b: review at most this many hub entities per run (most aliases first; hubs "
            "have >= ENTITY_NAME_REVIEW_MIN_ALIASES aliases). One Jev call each."
        ),
    )
    parser.add_argument(
        "--near-dup-limit", type=int, default=2000,
        help=(
            "Phase 2 with Jev: judge at most this many near-duplicate insight pairs "
            "(cosine in [INSIGHT_NEAR_DUP_COSINE_LOW, --insight-cosine-threshold), highest "
            "cosine first) per run. One Jev call each."
        ),
    )
    parser.add_argument("--skip-entities", action="store_true", help="Skip phase 1 (entity merge).")
    parser.add_argument("--skip-insights", action="store_true", help="Skip phase 2 (insight merge).")
    parser.add_argument("--skip-consistency", action="store_true", help="Skip phase 3 (orphan/consistency sweep).")
    parser.add_argument("--skip-health", action="store_true", help="Skip phase 4 (index/stats health).")

    parser.add_argument(
        "--since", type=int, default=7,
        help=(
            "Insight-merge phase 2 default scoping: only insights created in the last "
            "N days are used as probes against the full corpus (R12/doc-review P2). "
            "Also scopes the entity audit (chunks created in the last N days). "
            "Ignored if --full is given."
        ),
    )
    parser.add_argument(
        "--full", action="store_true",
        help=(
            "Insight-merge phase 2 (and entity audit phase 0): probe every insight / "
            "audit every chunk instead of scoping to --since. Slower, occasional operation — not the "
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

    if entity_audit_active(args):
        print("== Phase 0: entity audit ==", flush=True)
        run_entity_audit_apply(dry_run=dry_run, since_days=since_days, rescore=args.rescore)

    use_jev = jev_enabled(args)

    if not args.skip_entities:
        print("== Phase 1: entity merge ==", flush=True)
        merged_count, merged_entity_ids = entity_merge.run_entity_merge(
            dry_run=dry_run, use_jev=use_jev
        )
        if not dry_run:
            _invalidate_entity_edge_cache(merged_entity_ids)
    else:
        print("== Phase 1: entity merge (skipped) ==")
        merged_count = 0

    if entity_names_active(args):
        print("== Phase 1b: entity name correction ==", flush=True)
        run_entity_name_correction(dry_run=dry_run, name_limit=args.name_limit)

    if not args.skip_insights:
        print("== Phase 2: insight merge ==", flush=True)
        run_insight_merge(
            dry_run=dry_run,
            since_days=since_days,
            cosine_threshold=args.insight_cosine_threshold,
            use_jev=use_jev,
            near_dup_limit=args.near_dup_limit,
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

    scoring_aborted = False
    if entity_audit_active(args) and should_score(args):
        # Scoring only inserts into entity_mention_audit, so it deliberately
        # runs BEFORE (and outside) the active-job guard and the lock.
        print("== Phase 0: entity audit scoring (writes only entity_mention_audit) ==", flush=True)
        stats = run_entity_audit_scoring(args)
        if stats is None:
            return 1
        scoring_aborted = stats.aborted
    elif not entity_audit_active(args) and (args.score or args.rescore):
        print(
            "Note: --score/--rescore have no effect "
            + ("(entity audit skipped via --skip-entity-audit)."
               if jev_enabled(args)
               else "(Jev maintenance is off; pass --jev or set MAINTENANCE_JEV_ENABLED=true).")
        )

    rc = _run_guarded(args)
    if rc == 0 and scoring_aborted:
        print("Entity audit scoring stopped early (Jev calls kept failing); exiting non-zero.")
        return 1
    return rc


def _run_guarded(args: argparse.Namespace) -> int:
    """Run the phases; under --execute only after the active-job/community-run
    guard and with the advisory lock held."""
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
