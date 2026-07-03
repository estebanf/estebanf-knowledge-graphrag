---
title: RAG Retrieval Vector Prefilter and Query Fanout
date: 2026-07-02
last_refreshed: 2026-07-02
category: performance-issues
module: retrieval
problem_type: performance_issue
component: database
symptoms:
  - "rag retrieve \"insurance triage\" took 1:58.42 wall time before optimization"
  - "rag search \"insurance triage\" took about 29.6 seconds before optimization"
  - "Retrieval latency stayed high after early variant and parallelization changes because dense search and expansion fanout still dominated"
  - "Full Docker backend rebuilds were slow because code changes invalidated the heavy dependency install layer"
root_cause: missing_index
resolution_type: migration
severity: high
related_components:
  - service_object
  - tooling
tags:
  - rag
  - retrieval
  - pgvector
  - hnsw
  - binary-quantization
  - query-fanout
  - performance
  - docker
---

# RAG Retrieval Vector Prefilter and Query Fanout

## Problem

`rag retrieve` was still unusably slow after the first retrieval fanout optimizations. The visible symptom was not a small regression in one stage; the live retrieve path was spending minutes doing dense retrieval, graph expansion, and LLM-assisted expansion work before returning results.

## Symptoms

The user measured:

- `time rag retrieve "insurance triage"` at `1:58.42` wall time after earlier improvements did not land.
- A constrained retrieve run had been `2:40.75` before the insight seed cap and `1:44.32` after that cap.
- `rag search "insurance triage"` was around `29.6s` before the database indexing fix.

Profiling on the live database showed the real bottleneck: the corpus had `125,759` chunk embeddings and `108,282` insight embeddings, both stored as `vector(4096)`, with no valid dense vector indexes. Because standard pgvector HNSW and IVFFlat indexes cannot directly cover `vector(4096)`, dense retrieval fell back to exact corpus scans. Before the fix, `dense_retrieve` took about `14s` per chunk query variant, while `insight_dense_retrieve` took about `29s` per insight variant.

## What Didn't Work

The first optimization pass reduced query fanout but did not address the database-level dense scan. Removing active `step_back`, gating `expanded`, capping decomposed queries, and parallelizing variant generation helped reduce avoidable work, but exact 4096-dimensional vector scans still dominated both search and retrieve.

Lowering `seed_count` also did not initially produce the expected improvement because insight seed expansion used its own default seed count. That meant `--seed-count 1` constrained chunk expansion but still allowed multiple insight seeds to expand.

The first attempt to create binary vector indexes failed with `column does not have dimensions`. The concurrent index commands left invalid index shells behind, so those invalid indexes had to be dropped and recreated with an explicit `bit(4096)` cast on the binary-quantized expression.

Session history was requested, but discovery found only the current Codex session and this run's child research agents. No prior independent session findings were folded into this learning.

## Solution

The durable fix was to add a binary-quantized HNSW prefilter for 4096-dimensional embeddings, then rerank only the prefiltered candidates with full-precision cosine distance.

The migration in `scripts/migrate/009_binary_vector_prefilter_indexes.sql` creates expression indexes for chunks and insights using `binary_quantize(embedding)::bit(4096)`, plus a full-text index for insight content:

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_embedding_binary_hnsw_idx
  ON chunks USING hnsw ((binary_quantize(embedding)::bit(4096)) bit_hamming_ops)
  WHERE deleted_at IS NULL AND embedding IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS insights_embedding_binary_hnsw_idx
  ON insights USING hnsw ((binary_quantize(embedding)::bit(4096)) bit_hamming_ops)
  WHERE embedding IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS insights_content_fts_idx
  ON insights USING gin (to_tsvector('english', coalesce(content, '')))
  WHERE embedding IS NOT NULL;
```

The retrieval SQL now uses a materialized prefilter CTE ordered by binary Hamming distance, then reranks that much smaller candidate set with full-precision cosine:

```sql
WITH dense_prefilter AS MATERIALIZED (
    SELECT c.id
    FROM chunks c
    JOIN sources s ON s.id = c.source_id
    JOIN jobs j ON j.id = c.job_id
    WHERE c.deleted_at IS NULL
      AND s.deleted_at IS NULL
      AND j.status = 'completed'
      AND c.embedding IS NOT NULL
    ORDER BY binary_quantize(c.embedding)::bit(4096)
        <~> binary_quantize(%s::vector)::bit(4096)
    LIMIT %s
)
SELECT c.id,
       c.content,
       s.id,
       s.storage_path,
       s.metadata,
       (1 - (c.embedding <=> %s::vector)) AS score
FROM dense_prefilter p
JOIN chunks c ON c.id = p.id
JOIN sources s ON s.id = c.source_id
ORDER BY c.embedding <=> %s::vector
LIMIT %s;
```

The same pattern applies to insight dense retrieval. This preserves final ranking quality within the candidate set while avoiding a full exact scan across every 4096-dimensional embedding.

Retrieval fanout was also reduced and parallelized:

- Active/default `step_back` variants were removed.
- `expanded` variants are now gated deterministically.
- The default max decomposed query count is `2`.
- Chunk and insight variant generation run in parallel.
- Insight first-stage variant searches run in parallel.
- Insight subquery generation and first-hop entity query generation run in parallel.
- Embeddings are batched where possible.

Graph expansion was tightened:

- `seed_count` now caps insight seed expansion too.
- Chunk graph entity query generation is deterministic by default.
- Second-hop entity selection is deterministic by default.
- Slower LLM-backed graph behavior is now opt-in:

```env
RETRIEVAL_USE_LLM_ENTITY_QUERIES=false
RETRIEVAL_USE_LLM_SECOND_HOP_SELECTION=false
```

Set either flag to `true` only when the extra LLM-backed expansion is worth the latency cost.

The operational rollout included updates to `README.md`, `AGENTS.md`, `.env.example`, and the active `.env`. The backend was recreated and hot-patched with the updated Python files. A full Docker image rebuild was intentionally avoided during the fix because the current Dockerfile invalidates the heavy dependency layer and redownloads the ingest/Torch dependency stack after source changes.

Verification covered both tests and live behavior:

```bash
PYTHONPATH=src venv/bin/pytest -q \
  tests/test_retrieval.py \
  tests/test_cli_retrieve.py \
  tests/test_cli_search.py \
  tests/test_cli_community.py \
  tests/test_api.py \
  tests/test_api_community.py \
  tests/test_config.py \
  tests/test_prompts.py \
  tests/test_migration_009.py
```

Result: `85 passed in 4.50s`.

Postgres showed the new indexes as valid and ready, and backend/Postgres health checks were green. Live timings after the fix:

- `rag retrieve "insurance triage"`: `30.095s`
- `rag search "insurance triage"`: `6.631s`

## Why This Works

The largest cost was not the number of result rows returned; it was the number of high-dimensional embeddings scanned for every dense query variant. With 4096-dimensional vectors, exact cosine over more than 100k chunks and more than 100k insights is too expensive to repeat across multiple variants.

Binary quantization makes the indexable prefilter possible. Hamming distance over the quantized embedding quickly finds a candidate pool that is likely to contain the nearest neighbors. Full-precision cosine reranking then restores the ranking signal inside that pool, so the final ordering is still based on the original embedding values rather than only the compressed binary representation.

The fanout changes address the second class of latency: redundant or low-yield retrieval work. Removing `step_back`, gating `expanded`, capping decomposed queries, and making graph expansion deterministic by default reduce external LLM calls and repeated searches. Parallelization lowers wall time for independent work, but it only helps once the database scans are no longer dominating every branch.

This does not make retrieve free. The path can still spend time in external LLM calls, embedding generation, insight expansion, and graph expansion. The fix makes the default path usable by removing the worst dense-scan behavior and by turning the slowest expansion choices into explicit opt-ins.

## Prevention

When introducing dense retrieval over large embeddings, verify that the planned index can actually support the vector dimensionality in production. A migration that creates no valid dense index is worse than no migration at all because the query shape can look optimized while the database still performs exact scans.

Add migration tests or smoke checks that verify index validity:

```sql
SELECT indexrelid::regclass AS index_name, indisvalid, indisready
FROM pg_index
WHERE indexrelid::regclass::text IN (
    'chunks_embedding_binary_hnsw_idx',
    'insights_embedding_binary_hnsw_idx'
);
```

Profile the live stack before assuming application fanout is the primary bottleneck. In this case, code-level parallelization helped, but the decisive improvement came from making dense retrieval index-backed.

Keep latency-sensitive expansion knobs explicit in config. The default retrieve path should avoid slow LLM-backed expansion unless the operator has chosen the accuracy/latency tradeoff:

```env
RETRIEVAL_MAX_DECOMPOSED_QUERIES=2
RETRIEVAL_USE_LLM_ENTITY_QUERIES=false
RETRIEVAL_USE_LLM_SECOND_HOP_SELECTION=false
```

For containerized deployments, avoid Dockerfile layer ordering that causes dependency reinstall after source-only changes. In this project, the heavy ingest/Torch dependency layer made full rebuilds impractical during a performance incident, so future Dockerfile work should separate dependency installation from application source copies.

A valid dense index is necessary but not sufficient: pgvector's `hnsw.ef_search` GUC defaults to 40 and silently caps how many candidates an HNSW index scan returns, *regardless of the SQL `LIMIT` requested*. A follow-up fix found `dense_retrieve`'s prefilter CTE materializing only 40 rows despite `LIMIT 1000` (`RETRIEVAL_DENSE_PREFETCH_COUNT`) — the index in this doc was valid and in use, but the runtime candidate pool still didn't match what the config implied. See `docs/solutions/database-issues/postgres-toast-bloat-memgraph-schema-drift-and-hnsw-ef-search-cap.md` for the fix (a `SET hnsw.ef_search` call sized to the prefetch count) and confirm any future prefetch-count tuning also verifies `ef_search` via `EXPLAIN (ANALYZE, BUFFERS)`, not just `pg_index.indisvalid`.

## Related Issues

- `docs/plans/2026-07-02-001-refactor-retrieval-performance-plan.md` was the direct implementation plan, but it is a historical plan rather than a durable solution record.
- `docs/plans/2026-04-27-cross-source-communities-pgvector.md` is a related pgvector performance plan for community detection, not this retrieval/search path.
- `docs/plans/2026-05-05-insight-retrieval.md` explains why retrieve contains insight second-hop LLM subqueries; current defaults preserve the feature but make slow graph-expansion behavior explicit.
- `docs/solutions/database-issues/postgres-toast-bloat-memgraph-schema-drift-and-hnsw-ef-search-cap.md` — adjacent, not a duplicate: that doc addresses the `hnsw.ef_search` runtime parameter governing how much of this doc's HNSW prefilter actually gets searched, one layer below index existence.
- `vector(4096)` embeddings cannot use standard pgvector HNSW/IVFFlat indexes directly, so future embedding model changes must be checked against index support.
- Concurrent index failures can leave invalid index shells; remediation should include dropping invalid indexes before recreating them.
- Retrieval performance depends on both database indexing and expansion fanout. Treat them as separate axes during profiling.
- The Dockerfile currently makes source-only backend changes expensive to rebuild because heavy dependency installation is invalidated too easily.
