---
title: Intake Pipeline Regression from an Unindexed 4096-Dim Vector Ordering
date: 2026-07-03
category: performance-issues
module: insight_extraction
problem_type: performance_issue
component: database
symptoms:
  - "Median intake time for a source rose from ~55s to ~86 minutes over roughly two months as corpus size grew, with no error thrown -- the pipeline still completed, just far too slowly"
  - "EXPLAIN ANALYZE on the upsert_insight dedup query showed a full parallel sequential scan over the insights table, ~9,830ms per call, with no index in the plan"
  - "link_related_insights issued up to INSIGHT_LINK_TOP_K+1 full-table scans per newly inserted insight, each ~9.8s"
  - "14-day stage averages showed insight_extraction at ~7,301s versus entity/graph extraction at only ~173s, making insight extraction the dominant cost"
  - "A live 12-chunk source measured 210s (over the 90s target) before an unrelated .env concurrency override was found and corrected"
root_cause: missing_index
resolution_type: code_fix
severity: high
related_components:
  - service_object
  - background_job
  - tooling
tags: [intake-pipeline, pgvector, hnsw, vector-prefilter, insight-extraction, batching, ingestion-performance]
---

# Intake Pipeline Regression from an Unindexed 4096-Dim Vector Ordering

## Problem

Intake pipeline processing time for a single source (chunking through insight extraction/linking) degraded from a ~55s median (week of 2026-04-27) to an ~86-minute median (week of 2026-06-29) as the corpus grew to roughly 108,841 insights, 125,979 chunks, 113,878 entities, and 7,524 sources. The root cause was a per-insight dedup query in `upsert_insight` (`src/rag/insight_extraction.py`) ordering by raw 4096-dimension `embedding <=>` distance, which has no usable pgvector index (native HNSW/IVFFlat caps out at 2000 dimensions) and therefore forced a full sequential scan on every call, worsening as the corpus grew.

## Symptoms

- Median intake time for a source rose from ~55s to ~86 minutes over roughly two months as corpus size grew, with no error thrown — the pipeline still completed, just far too slowly.
- `EXPLAIN ANALYZE` on the `upsert_insight` dedup query showed a full parallel sequential scan over the `insights` table, ~9,830ms per call, with no index in the plan.
- `link_related_insights` compounded the cost further: for every newly inserted insight it issued one forward KNN scan plus up to `INSIGHT_LINK_TOP_K` (default 10) reverse KNN scans — up to 11 full-table scans per insight, each ~9.8s.
- 14-day stage averages showed `insight_extraction` at ~7,301s versus entity/graph extraction (serial per-chunk LLM calls) at only ~173s, making insight extraction the dominant, disproportionate cost.
- A live 12-chunk source measurement came in at 210s, over the 90s target, before the root config issue (below) was found.

## What Didn't Work

- **Investigation step: attributed ~173s of per-stage cost to a tie between `graph_linking` and `graph_extraction` timestamps in `stage_log`.** Corrected after reading `graph_linking.py` directly, which showed it was a no-op compatibility stub (`def link_graph(conn, driver, source_id, job_id): pass`). The ~173s belonged entirely to `graph_extraction`; `graph_linking`'s appearance in `stage_log` was a timestamp-adjacency artifact, not real cost. Corrected before committing to a plan, because acting on the misattribution would have optimized the wrong stage.
- **Doc-review caught a draft-plan assumption that entity extraction made two LLM calls per chunk (entities + relationships).** Corrected after reading `graph_extraction.py` directly: `extract_relationships` exists in the file but is dead code — `extract_and_store_graph` calls only `extract_entities` and hardcodes `relationships=[]`. The plan was revised to preserve this no-relationship-extraction behavior exactly, because a performance rewrite is the wrong place to accidentally activate a dormant, untested code path as a side effect.
- **A naive moving-median drift-guardrail design was proposed and rejected during doc-review.** A rolling/moving baseline drifts upward together with a gradual, corpus-coupled regression and never crosses its own threshold — exactly the failure mode that let the April-to-June slowdown go undetected for months. Corrected by switching to a pinned, explicitly-snapshotted baseline (see Prevention below).
- **Live validation initially measured 210s for a 12-chunk source, over the 90s target.** Corrected after discovering the live `.env` still pinned `INSIGHT_EXTRACTION_CONCURRENCY=5`, an old override left over from before this fix, rather than the new code default of 12. Re-measuring at the actual shipped default (12) gave 71.1s, under target. This distinguished a config-drift confound from a real regression in the rewritten code.

## Solution

**(a) Prefiltered insight dedup.** `upsert_insight`'s SELECT was rewritten to mirror the exact SQL shape already proven in `dense_retrieve` (`src/rag/retrieval.py`) for search/retrieval: a binary-quantized HNSW prefilter narrows the candidate pool, then a full-precision rerank runs only over that small pool.

Before (full sequential scan, no usable index):
```sql
SELECT id, 1 - (embedding <=> %s::vector) AS sim
FROM insights
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s::vector
LIMIT 1
```

After (index-assisted prefilter, then exact rerank on a bounded pool):
```sql
WITH insight_prefilter AS MATERIALIZED (
    SELECT id
    FROM insights
    WHERE embedding IS NOT NULL
    ORDER BY binary_quantize(embedding)::bit(4096) <~> binary_quantize(%s::vector)::bit(4096)
    LIMIT %s
)
SELECT i.id, 1 - (i.embedding <=> %s::vector) AS sim
FROM insight_prefilter p
JOIN insights i ON i.id = p.id
ORDER BY i.embedding <=> %s::vector
LIMIT 1
```

The prefilter runs over `insights_embedding_binary_hnsw_idx` (`bit_hamming_ops`, from migration `scripts/migrate/009_binary_vector_prefilter_indexes.sql`) and is bounded by a new `INSIGHT_PREFILTER_CANDIDATES` config knob (default 100). Because pgvector silently caps HNSW results at `hnsw.ef_search` regardless of the SQL `LIMIT`, the previously-duplicated per-callsite `SET hnsw.ef_search = N` logic was extracted into one shared `set_hnsw_ef_search()` helper in `src/rag/db.py`, used by both `retrieval.py` and `insight_extraction.py`. Result, confirmed via `EXPLAIN (ANALYZE, BUFFERS)`: ~9,830ms to ~52ms per dedup lookup, with an index scan and no `Seq Scan` in the plan. The same prefilter+rerank shape was reused for `link_related_insights`'s set-based mutual top-K linking (batched forward/reverse top-K queries with a single `UNWIND` write per batch instead of per-pair Memgraph calls) and for the weekly maintenance script's insight-merge candidate generation.

**(b) Pinned-baseline drift guardrail, not a moving median.** The rejected and accepted designs differ in what they compare a new duration against:

```python
# Rejected: moving/rolling median recomputed from a recent window.
# Drifts upward together with a gradual, corpus-coupled regression,
# so it never crosses its own threshold -- exactly how the April->June
# slowdown went undetected for months.
recent_durations = fetch_recent_stage_durations(stage, window_days=14)
baseline_ms = median(recent_durations)   # itself already degraded
if duration_ms > baseline_ms * FACTOR:
    warn(...)

# Accepted: compare against a baseline frozen at a point in time,
# snapshotted explicitly via `rag jobs stats --set-baseline` into the
# stage_duration_baseline table -- never recomputed from live data.
baseline_ms = fetch_pinned_baseline(stage)   # frozen row, set once
if baseline_ms is None:
    return   # missing baseline is skipped, not treated as drift
if duration_ms > baseline_ms * settings.STAGE_DRIFT_WARN_FACTOR:
    log.warning("stage_duration_drift", stage=stage,
                duration_ms=duration_ms, baseline_ms=baseline_ms)
```

The real implementation lives in `check_stage_drift()` in `src/rag/worker.py`: it reads a job's `stage_log` (each entry now carries `duration_ms`, computed from the stage's own `started_at` read back from the DB so it survives retries/restarts), reads the frozen `stage_duration_baseline` table, and warns (log-only, does not fail the job) when `duration_ms > baseline_ms * STAGE_DRIFT_WARN_FACTOR` (default 3.0). A stage with no pinned baseline yet is skipped, not treated as drift.

Beyond these two, the other units applied the same reusable patterns: batching per-chunk/per-insight Memgraph writes into `UNWIND` calls, fanning out per-chunk LLM extraction across a `ThreadPoolExecutor` with a shared `STAGE_FAILURE_RATE_THRESHOLD` failure-audit policy instead of silently swallowing per-chunk errors, removing the no-op `graph_linking` stage (with a `_LEGACY_STAGE_ALIASES` compatibility shim for old jobs), and adding a weekly maintenance script whose insight-merge phase defaults to a `--since`-scoped 7-day candidate window rather than a full-corpus sweep.

## Why This Works

pgvector's native HNSW/IVFFlat index types cap at 2000 dimensions. The `insights.embedding` column is 4096-dimensional, so any query that does `ORDER BY embedding <=> %s::vector` directly has no index Postgres can use for that ordering — it must sequentially scan and distance-compute every row with a non-null embedding, then sort, regardless of how selective the query "feels" or how large the table gets. This cost scales linearly with corpus size, which is exactly why the regression was gradual and corpus-coupled: the query wasn't broken, it simply never had an index-assisted path, so its cost tracked table growth 1:1.

`binary_quantize(embedding)::bit(4096)` collapses each 4096-dim float vector into a 4096-bit binary representation, which pgvector CAN index natively with HNSW using `bit_hamming_ops` — Hamming distance over bits is cheap to index. Ordering by that binary Hamming distance with a `LIMIT` turns an unindexable full-precision distance computation into an index-assisted approximate nearest-neighbor lookup, at the cost of some ranking precision. The trick is that this approximate ordering is used only to shrink the candidate pool (`INSIGHT_PREFILTER_CANDIDATES`, default 100) — the actual `embedding <=>` full-precision cosine distance is still computed, but only over that bounded pool via the rerank, not the whole table. So the expensive, exact comparison that's needed for correctness runs on ~100 rows instead of ~100,000+, while the coarse-but-indexed comparison that only needs to be "roughly right" runs on the full table cheaply. This is exactly the shape `dense_retrieve` had already proven for retrieval; intake had simply never received the same treatment, since insight dedup and insight linking were written before that pattern existed and were never retrofitted.

## Prevention

- **Primary regression-visibility mechanism: the pinned-baseline drift guardrail**, not a moving/rolling one. Snapshot a baseline explicitly (`rag jobs stats --set-baseline`) and compare each job's actual `duration_ms` against `baseline_ms * STAGE_DRIFT_WARN_FACTOR` (default 3.0). **Lesson to reuse elsewhere:** if you add a regression-drift alert, verify it fires against a FROZEN baseline recorded before the regression started, not a live recomputed window — a moving window drifts upward with the regression and never crosses its own threshold. This is precisely how a ~55s-to-~86-minute median crept by undetected for two months.
- **Concrete regression test for this property** (`tests/test_worker.py::test_check_stage_drift_regression_guard_pinned_baseline_still_fires`): seeds a scenario where the entire "recent window" has already drifted 5x above the original pinned baseline (e.g. pinned baseline 1000ms, new job duration 5000ms) and asserts the drift warning still fires — proving the comparison is against the frozen baseline row, never recomputed from already-drifted recent data. Any future drift/regression alert should carry an equivalent test before being trusted.
- **Never let a periodic maintenance job's cost scale with total corpus size unless it's explicitly opted into via a flag.** The weekly maintenance script's insight-merge phase (`scripts/weekly_maintenance.py`) defaults to `--since 7` days — only recent insights are used as probes against the full corpus (via the same binary-prefilter query shape), keeping a routine weekly job's cost bounded regardless of corpus growth. A `--full` whole-corpus sweep is available but explicitly opt-in and documented as slower, precisely so it can't silently become the default execution path that scales unboundedly with the corpus the way the original `upsert_insight`/`link_related_insights` queries did.
- When retrofitting a proven pattern (like the binary-prefilter + rerank shape) into a new codepath, audit sibling codepaths with the same query shape (dedup lookup, linking lookup, maintenance merge lookup) rather than fixing only the one that was profiled — the same unindexed-ordering trap tends to recur wherever a raw `ORDER BY embedding <=>` was copy-pasted before the prefilter pattern existed.
- **Two bounded-but-not-zero costs this fix did not eliminate, worth monitoring as usage grows:**
  - `extract_and_store_insights`'s within-batch dedup (Phase C, part 1, `src/rag/insight_extraction.py`) is a pairwise `_cosine_similarity` loop over all extracted insights for *one source*: for each new insight it compares against every survivor found so far, so cost is O(n²) in the number of insights extracted from that source's chunks — bounded by per-source chunk/insight count, not total corpus size, so it doesn't reproduce this bug's failure mode, but an unusually large source (e.g. a bulk import with hundreds of chunks and no early near-duplicates) could make this loop the new local hotspot. If that becomes observable, replace it with the same binary-prefilter shape used corpus-side, applied within the batch.
  - `INSIGHT_PREFILTER_CANDIDATES` (default 100) and `weekly_maintenance.py`'s `--since 7` probe window are both fixed constants tuned for the corpus size at fix time (~109k insights). Neither is corpus-size-aware: as the corpus keeps growing, a fixed prefilter width can under-sample true near-duplicates (recall degrades), and as weekly insight-creation *velocity* grows, a fixed 7-day probe window means phase 2's cost grows with ingestion rate even though it no longer grows with total corpus size. Both should be revisited periodically (e.g. alongside the pinned-baseline refresh), not treated as permanently correct.

## Related Issues

- `docs/solutions/performance-issues/rag-retrieval-vector-prefilter-and-query-fanout.md` — the retrieval-side fix for the same pgvector 2000-dimension index cap, using the same binary-quantized-HNSW-prefilter + full-precision-rerank pattern, applied to search/retrieve rather than intake/ingestion. Root cause and solution approach match closely; problem statement, files, and the failure-audit/drift-guardrail/maintenance-script mechanisms here are new.
- `docs/solutions/database-issues/postgres-toast-bloat-memgraph-schema-drift-and-hnsw-ef-search-cap.md` — introduced the `hnsw.ef_search` runtime-GUC fix this work builds on; its description of that helper as private and local to `retrieval.py` is now stale (it has since been extracted into a shared `set_hnsw_ef_search()` in `src/rag/db.py`, used by both `retrieval.py` and `insight_extraction.py`) — flagged as a refresh candidate.
