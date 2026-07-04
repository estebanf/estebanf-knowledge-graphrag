---
title: "Postgres TOAST Bloat, Memgraph Schema Drift, and Silent hnsw.ef_search Capping"
date: 2026-07-02
last_refreshed: 2026-07-03
category: database-issues
module: rag storage layer (postgres entities/vector retrieval + memgraph schema)
problem_type: database_issue
component: database
symptoms:
  - "entities table grew to ~2.9GB of dead TOAST with pg_stat_user_tables.last_autovacuum NULL, never autovacuumed"
  - "SHOW INDEX INFO / SHOW CONSTRAINT INFO on live Memgraph missing the Insight(insight_id) index and unique constraint declared in memgraph_init.cypher"
  - "every MATCH (i:Insight {insight_id: ...}) lookup in insight_extraction.py and retrieval.py did a full label scan across 112,873 nodes instead of an index lookup"
  - "RETRIEVAL_DENSE_PREFETCH_COUNT=1000 had no effect on dense_retrieve/insight_dense_retrieve candidate pool because pgvector's hnsw.ef_search GUC silently capped it at its default of 40, confirmed via EXPLAIN (ANALYZE, BUFFERS) returning only 40 rows"
  - "rag retrieve latency measured at ~30.1s under the silent 40-candidate cap versus ~40.1s once ef_search was forced to 1000"
root_cause: incomplete_setup
resolution_type: code_fix
severity: high
related_components:
  - src/rag/db.py
  - src/rag/graph_db.py
  - src/rag/retrieval.py
  - src/rag/api/main.py
  - scripts/merge_semantic_duplicates.py
  - scripts/merge_duplicate_entities.py
  - scripts/init/postgres/02_schema.sql
  - scripts/init/memgraph_init.cypher
  - scripts/migrate/010_entities_autovacuum_tuning.sql
tags: [postgres, autovacuum, toast-bloat, memgraph, schema-drift, hnsw, pgvector, hnsw-ef-search]
---

# Postgres TOAST Bloat, Memgraph Schema Drift, and Silent hnsw.ef_search Capping

## Problem

Three storage- and index-layer issues were quietly degrading a self-hosted RAG stack (pgvector on Postgres + Memgraph), each discovered by inspecting the live database directly rather than by assumption:

1. Postgres's `entities` table had never been autovacuumed, and the merge scripts that periodically deduplicate entities were generating dead rows with nothing ever reclaiming them.
2. Memgraph's live instance was missing the `Insight` index/constraint declared in `scripts/init/memgraph_init.cypher` — the init script had drifted out of sync with what was actually applied to the running database.
3. pgvector's `hnsw.ef_search` GUC was silently capping the dense-retrieval candidate pool at its default of 40, regardless of the much larger `LIMIT` the application code requested.

All three share a pattern worth naming: a config value, schema declaration, or `LIMIT` clause *looked* correct in the codebase, but the actual runtime behavior of Postgres/pgvector/Memgraph didn't match it, and nothing was verifying that it did.

## Symptoms

- `entities` measured 4917MB via `pg_total_relation_size` for only 113,835 rows, versus an expected TOAST size of ~1.86GB for `vector(4096)` embeddings — roughly 2.6x bloat. `pg_stat_user_tables.last_autovacuum` and `last_vacuum` were both `NULL` for `entities`.
- `SHOW INDEX INFO;` / `SHOW CONSTRAINT INFO;` against the live Memgraph instance listed `Chunk`, `Entity`, `Source` — no `Insight` at all — despite 112,873 `Insight` nodes existing and being looked up by `insight_id` in both `insight_extraction.py`'s dedup/store path and `retrieval.py`'s `expand_seed_insight`/`_load_related_insights` (called per seed insight, up to `RETRIEVAL_INSIGHT_SEED_COUNT` times per `rag retrieve` call).
- `EXPLAIN (ANALYZE, BUFFERS)` on the dense-retrieval prefilter CTE in `dense_retrieve()`/`insight_dense_retrieve()` showed the HNSW index scan materializing only 40 rows despite an explicit `LIMIT 1000` (`RETRIEVAL_DENSE_PREFETCH_COUNT`) — confirmed by sweeping `SET hnsw.ef_search = 40/100/500/1000` and watching returned row counts track `ef_search` exactly, not the SQL `LIMIT`.

## What Didn't Work

- **Unconditional Memgraph schema reconciliation in the FastAPI lifespan.** The first version called `reconcile_schema(driver)` directly with no error handling. A 9-reviewer code review pass (testing, correctness, reliability, and adversarial reviewers independently) found this hard-crashed the *entire* backend startup — not just graph-dependent routes — on any transient Memgraph unavailability (a container restart race, a healthcheck that only checks TCP-port-open rather than actual Bolt readiness). It also broke `tests/test_mcp_server.py`'s two tests, since `with TestClient(app) as client:` triggers the FastAPI lifespan; those tests only exercise MCP auth middleware and have nothing to do with the graph store, yet silently gained a hard Memgraph dependency. The `correctness` reviewer reproduced this empirically: `MEMGRAPH_URL=bolt://localhost:1 pytest tests/test_mcp_server.py` failed with `neo4j.exceptions.ServiceUnavailable` — it had only passed in the dev sandbox because a live Memgraph container happened to be running there, masking the regression. Fixed by wrapping the call in try/except that logs loudly (`log.exception(...)`) and lets the app continue starting, preserving the "make schema drift visible" intent without letting transient Memgraph unavailability take down unrelated routes. Re-running the same `MEMGRAPH_URL=bolt://localhost:1` reproduction confirmed the fix.
- **A one-off Python vector-fetch sanity check produced garbled output.** Tried `list(row[0])` on a `pgvector` embedding column fetched via psycopg without `register_vector()` registered. Since the codebase never registers the pgvector adapter (embeddings are only ever built as literal `[v1,v2,...]` strings for SQL, never parsed back in Python), the raw string was iterated character-by-character instead of as a vector. Not a real bug — a verification-script mistake — fixed by fetching `embedding::text` and parsing manually for the one-off check.
- **Autovacuum tuning was applied live but never made durable.** After tuning `entities`'s autovacuum thresholds via `ALTER TABLE ... SET (...)` directly on the running database, a code reviewer found the change was only documented as a manual README command — not baked into `scripts/init/postgres/02_schema.sql` for fresh installs, nor into any `scripts/migrate/*.sql` file for other existing deployments. The bloat-recurrence risk was closed only for the one database this session touched directly. Fixed by adding the storage parameters to the init schema's `CREATE TABLE entities` and adding a dedicated migration file.

## Solution

**1. Postgres `entities` TOAST bloat.**

One-time reclaim and tightened autovacuum thresholds on the live table:

```sql
VACUUM FULL VERBOSE entities;  -- ~113s; removed 15,544 dead row versions; 4917MB -> 2049MB
ANALYZE entities;              -- reltuples was stale (102,822 vs actual 113,835)

ALTER TABLE entities SET (
  autovacuum_vacuum_scale_factor = 0.02,
  autovacuum_vacuum_threshold = 500,
  autovacuum_analyze_scale_factor = 0.02,
  autovacuum_analyze_threshold = 500
);
```

Added a vacuum helper to `src/rag/db.py` and wired it into both merge scripts:

```python
def vacuum_analyze_entities() -> None:
    """Reclaim dead rows left by entity-merge scripts.

    VACUUM cannot run inside a transaction block, so this opens its own
    autocommit connection rather than reusing a caller's connection.
    """
    conn = psycopg.connect(settings.POSTGRES_URL, autocommit=True)
    try:
        conn.execute("VACUUM (ANALYZE) entities")
    finally:
        conn.close()
```

`scripts/merge_semantic_duplicates.py` and `scripts/merge_duplicate_entities.py` now track `total_merged` across all clusters/groups and call `vacuum_analyze_entities()` once at the end, but only when `total_merged > 0` — never on dry-run, never when zero duplicates were found. The tightened thresholds were also baked into `scripts/init/postgres/02_schema.sql`'s `CREATE TABLE entities (...) WITH (autovacuum_vacuum_scale_factor = 0.02, ...)` for fresh installs, plus a new `scripts/migrate/010_entities_autovacuum_tuning.sql` for existing databases.

**2. Missing Memgraph `Insight` index/constraint.**

Applied live via `mgconsole`, then made durable in `src/rag/graph_db.py`:

```python
SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT ON (s:Source) ASSERT s.source_id IS UNIQUE;",
    "CREATE CONSTRAINT ON (c:Chunk) ASSERT c.chunk_id IS UNIQUE;",
    "CREATE CONSTRAINT ON (e:Entity) ASSERT e.entity_id IS UNIQUE;",
    "CREATE CONSTRAINT ON (i:Insight) ASSERT i.insight_id IS UNIQUE;",
    "CREATE INDEX ON :Entity(canonical_name);",
    "CREATE INDEX ON :Entity(entity_id);",
    "CREATE INDEX ON :Entity(entity_type);",
    "CREATE INDEX ON :Chunk(chunk_id);",
    "CREATE INDEX ON :Chunk(source_id);",
    "CREATE INDEX ON :Insight(insight_id);",
]

def reconcile_schema(driver) -> None:
    """Idempotently (re-)apply the Memgraph index/constraint statements."""
    with driver.session() as session:
        for statement in SCHEMA_STATEMENTS:
            session.run(statement)
```

Wired into `src/rag/api/main.py`'s FastAPI `lifespan` so it runs once on every backend startup, wrapped so failures don't take down unrelated routes (see "What Didn't Work" above for why the try/except is load-bearing, not decorative):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with get_graph_driver() as driver:
            reconcile_schema(driver)
    except Exception:
        log.exception(
            "Memgraph schema reconciliation failed at startup; graph-dependent "
            "routes may fail until this is resolved, but the API will still serve "
            "other traffic."
        )
    if mcp_app is not None and hasattr(mcp_app, "lifespan_context"):
        ...
```

**3. pgvector `hnsw.ef_search` silently capping dense-retrieval prefetch.**

```python
def set_hnsw_ef_search(conn, prefetch_count: int) -> None:
    """Widen the HNSW candidate search so a binary prefilter query actually
    returns ``prefetch_count`` rows.

    pgvector's ``hnsw.ef_search`` GUC defaults to 40 and silently caps the
    candidate pool below any larger ``LIMIT`` requested against an HNSW
    index -- the prefilter query returns at most ``ef_search`` rows
    regardless of the SQL ``LIMIT``. Session-scoped ``SET`` (not
    ``SET LOCAL``) is fine here: each call site holds its own short-lived
    connection for a single query.
    """
    conn.execute(f"SET hnsw.ef_search = {int(prefetch_count)}")
```

Originally a private helper local to `retrieval.py`; later extracted into `src/rag/db.py` as a shared, public `set_hnsw_ef_search()` when a second call site needed the identical fix (see Related Issues) — both `retrieval.py` and `insight_extraction.py` now import it from `db.py` rather than each keeping their own copy. Called at the top of the `if prefetch_count > top_n:` branch in both `dense_retrieve` and `insight_dense_retrieve`, right before the prefilter CTE query, sized to `RETRIEVAL_DENSE_PREFETCH_COUNT` rather than a hardcoded constant. Measured latency tradeoff on `rag retrieve "insurance triage"` (`rag search` unaffected — it doesn't use the multi-variant fanout path):

| Config | rag retrieve wall time | Dense-retrieve candidate pool |
|---|---|---|
| Before fix (silent ~40 cap) | ~30.1s | ~40 (regardless of configured prefetch) |
| ef_search=1000 (matching original config) | ~40.1s | ~1000 |
| ef_search=250 (final chosen default) | ~32.2s | ~250 |

Given the ~10s latency cost at full 1000, `RETRIEVAL_DENSE_PREFETCH_COUNT`'s default was lowered from 1000 to 250 — still ~6x wider than the silent 40-candidate cap, while keeping latency close to the pre-fix baseline. This remains a user-tunable config default, not a hardcoded limit.

## Why This Works

All three fixes close the gap between "what the code/config claims" and "what the database engine actually does at runtime":

- The `entities` bloat wasn't a one-time cleanup problem — it was a recurring generator (merge scripts) with no corresponding reclaim step and autovacuum thresholds too loose (20% of table size) to ever trigger on a table whose only churn is periodic merge bursts. Tightening the scale factor to 2% and threshold to 500 rows means autovacuum fires promptly after a merge run, and the app-triggered `vacuum_analyze_entities()` closes the loop immediately rather than waiting on autovacuum's polling interval — but only fires when there was actually something to reclaim, so it doesn't add needless I/O to dry-runs or no-op merge invocations.
- The Memgraph fix works because `reconcile_schema()` is idempotent (verified empirically — rerunning `CREATE INDEX`/`CREATE CONSTRAINT` a second time is a safe no-op in Memgraph) and now runs on every startup instead of relying on someone remembering to re-run `memgraph_init.cypher` by hand whenever the script and the live instance diverge. The try/except boundary scopes the blast radius of a Memgraph outage to "graph-dependent routes may be degraded" instead of "the whole API is down," which matches the actual dependency graph of the application.
- The `ef_search` fix works because it makes the HNSW candidate pool match the SQL `LIMIT` the application already believed it was getting — pgvector's `hnsw.ef_search` is a session GUC, not something the planner infers from `LIMIT`, so it has to be set explicitly per session/connection before the prefilter query runs. Sizing it to `RETRIEVAL_DENSE_PREFETCH_COUNT` (rather than a separate hardcoded constant) means the two knobs can't drift apart again the same way `LIMIT` and `ef_search` had drifted apart before.

## Prevention

- **Verify GUCs and index behavior empirically, not from config or init scripts alone.** A `LIMIT 1000` in application SQL is not proof that 1000 rows come back from an HNSW index scan — pgvector's `ef_search` (and equivalents in other ANN index types) silently cap results below any SQL `LIMIT`. Use `EXPLAIN (ANALYZE, BUFFERS)` to confirm the actual row count materialized at each stage of a query, not just its final output size.
- **Treat schema-init scripts (`.cypher`, `.sql` DDL bootstrap files) as one-shot unless something reconciles them against the live database on every start.** `memgraph_init.cypher` was correct on disk the whole time; the live instance simply never re-ran it after the `Insight` label was added. If a schema statement matters, either apply it in a reconciliation step that runs on every backend startup (as done here) or add a CI/deploy check that diffs `SHOW INDEX INFO`/`SHOW CONSTRAINT INFO` against the init script.
- **Before shipping any automatic maintenance triggered by app code (vacuum-on-merge, schema-reconcile-on-startup), ask: "what happens to the rest of the system if this specific new I/O call fails?"** The first cut of Memgraph reconciliation answered this implicitly with "the whole backend goes down," which only surfaced via a dedicated multi-reviewer pass, not the original implementation. Wrap new startup-time or per-operation I/O in a failure boundary scoped to what actually depends on it, and add a test that simulates that dependency being unreachable (`MEMGRAPH_URL=bolt://localhost:1` here) before considering the change done.
- **When tuning storage/maintenance parameters live on a running database, immediately also update the init script and add a migration file in the same change.** A live `ALTER TABLE ... SET (...)` that isn't mirrored in `scripts/init/*.sql` or `scripts/migrate/*.sql` only fixes the one database touched during the investigation; it leaves the same bloat-recurrence risk open for every other environment.
- **Check `pg_stat_user_tables.last_autovacuum`/`last_vacuum` periodically for tables with a delete/update-heavy access pattern**, especially ones with large TOASTed columns (embeddings, blobs) where bloat is expensive to carry and expensive to reclaim (`VACUUM FULL` takes an exclusive lock and can run for minutes on a large table).

## Related Issues

- Implementation plan: `docs/plans/2026-07-02-002-fix-storage-vacuum-and-index-fixes-plan.md` — full details, code references, and the Verification Contract/Definition of Done for this change.
- Adjacent (not duplicate) to `docs/solutions/performance-issues/rag-retrieval-vector-prefilter-and-query-fanout.md`, which built the binary-quantized HNSW prefilter that the `ef_search` bug in this fix sits underneath. That prior doc addressed index *existence* (building the prefilter/fanout mechanism); this fix addresses the runtime GUC (`hnsw.ef_search`) that governs how much of that existing index's candidate pool actually gets searched — a distinct failure mode one layer below the one that doc solved, worth distinguishing explicitly since the two will surface as related but are not the same bug.
- `docs/solutions/performance-issues/intake-insight-extraction-vector-prefilter-and-batched-storage.md` — reused the `set_hnsw_ef_search` fix from item 3 above for the intake/ingestion path (insight dedup and linking), which is what prompted extracting it from a `retrieval.py`-local private helper into the shared public helper in `src/rag/db.py` described above.
