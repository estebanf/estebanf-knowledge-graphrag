# Restore Cross-Source Community Detection at Scale

Date: 2026-04-27
Owner: handoff plan — implementable by a fresh session
Area: `src/rag/community.py`, `src/rag/config.py`, CLI/API surfaces, docs

---

## 1. Context

`rag community` (CLI and `POST /api/community`) rarely produces cross-source communities, even when the scope clearly contains semantically related content across multiple sources. Investigation traced this to a structural limitation in `src/rag/community.py:_build_igraph`:

- **Chunk co-occurrence edges** (lines ~197–209): connect entities only when they share a chunk. A chunk belongs to exactly one source, so these edges are intra-source by construction. The `cross` factor on line ~206 only upweights an existing edge — it never *creates* a cross-source edge.
- **Same-source weak fallback** (lines ~218–227): also intra-source; additionally capped at sources with ≤60 entities.
- **Semantic cross-source edges** (lines ~231–250): the only mechanism that creates edges between entities from different sources, but gated by `if len(entity_ids) <= 400:`. Above 400 entities the entire block is skipped and the resulting graph contains zero cross-source edges, so Leiden cannot return a cross-source community.

Commit `d162e7c` ("fix: cap O(N²) loops in _build_igraph that caused hangs on large source sets") added the 400 cap because the all-pairs Python cosine loop hung at scale. The cap solved performance but silently disabled the only cross-source mechanism for any non-trivial scope.

This plan replaces the Python all-pairs loop with per-entity ANN queries against the existing pgvector HNSW index on `entities.embedding` (`scripts/init/postgres/02_schema.sql:79`). Cost becomes ~`N · K · log N` instead of `N²/2`, so the cap can be removed and cross-source edges are produced at any scope size.

The same `_load_cross_source_semantic_edges` helper preserves the "don't overwrite a stronger existing edge" semantics, keeps semantic edge weight at `sim * 0.5` to match the original weighting, and adds an explicit budget knob so extreme scopes degrade gracefully instead of hanging.

---

## 2. Files to modify

- `src/rag/community.py` — primary change.
- `src/rag/config.py` — two new env-backed knobs.
- `.env.example` — document the two new knobs.
- `src/rag/api/routes/community.py` and the matching request schema (`src/rag/api/schemas/...` — locate via grep `community_options` / `CommunityOptions`) — expose the new knobs via the API.
- `src/rag/cli.py` — add CLI flags on the three `community` subcommands.
- `README.md` — extend the "Community and worker settings" table and the community CLI/REST options sections.
- `tests/test_community.py` (and possibly `tests/test_cli_community.py`, `tests/test_api_community.py`) — see §6.

`AGENTS.md` only needs an update if any "Current Behavioral Notes" become stale — review after coding.

---

## 3. Existing utilities to reuse

Use these instead of writing new helpers:

- `src/rag/retrieval.py:_vector_literal` — formats a Python `list[float]` as a `pgvector` literal (`'[1.0,2.0,...]'`). Import and reuse; do not duplicate.
- `src/rag/db.py:get_connection` — already used in `community.py`.
- `src/rag/community.py:_load_entity_embeddings` — already loads embeddings from Postgres for a list of entity IDs. Call once up front; pass embeddings into the new helper instead of re-querying per entity.
- pgvector HNSW index `entities_embedding_hnsw_idx` (`scripts/init/postgres/02_schema.sql:79`, cosine_ops) — already in place. No migration is required.
- Pattern for ANN queries: see `src/rag/retrieval.py` lines ~266–278 for the `(1 - (embedding <=> %s::vector)) AS score ... ORDER BY embedding <=> %s::vector LIMIT N` form. Mirror this exactly so HNSW is actually used.

---

## 4. Implementation steps

### 4.1 Add config knobs

In `src/rag/config.py` add two settings (follow the existing `COMMUNITY_*` convention — `Settings` pydantic class or equivalent):

- `COMMUNITY_CROSS_SOURCE_TOP_K: int = 10`
- `COMMUNITY_MAX_CROSS_SOURCE_QUERIES: int = 5000`

In `.env.example`, append both with one-line comments mirroring the existing community block.

### 4.2 New helper in `src/rag/community.py`

Add this helper alongside `_build_igraph`. Signature:

```python
def _load_cross_source_semantic_edges(
    entities: dict[str, EntityNode],
    idx: dict[str, int],
    embeddings: dict[str, list[float]],
    semantic_threshold: float,
    top_k: int,
    max_queries: int,
) -> dict[tuple[int, int], float]:
```

Behavior:

1. Build the ordered list of entity IDs that have an embedding in `embeddings`. If `len(candidates) > max_queries`, sort by `len(entities[eid].chunk_ids)` descending and truncate to `max_queries` so the most-mentioned entities are prioritized.
2. Open one connection via `get_connection()`. For each entity in the (possibly truncated) list, run:
   ```sql
   SELECT id, 1 - (embedding <=> %s::vector) AS sim
   FROM entities
   WHERE id = ANY(%s)
     AND id <> %s
     AND embedding IS NOT NULL
   ORDER BY embedding <=> %s::vector
   LIMIT %s
   ```
   Pass `top_k` as `LIMIT` and `entity_ids` (full scope candidate list, as a Python list of UUID strings) as the `ANY(%s)` argument. Apply the `sim >= semantic_threshold` filter in Python after fetching, so HNSW ordering is preserved.
3. For each `(a_id, b_id, sim)` neighbor pair returned:
   - If `entities[a_id].source_ids & entities[b_id].source_ids` is non-empty → skip (intra-source; already covered by chunk-cooc / same-source fallback).
   - Else compute `key = (min(idx[a_id], idx[b_id]), max(...))` and set `result[key] = sim * 0.5` only if `key not in result` (preserve "don't overwrite a stronger edge" — the caller will additionally not overwrite chunk-cooc edges).
4. Return the dict.

Notes:

- Use `_vector_literal(embeddings[eid])` for the vector parameter.
- The query is per-entity but uses the HNSW index, so each call is roughly milliseconds. Expect ~5k queries to complete in a few seconds at most. Do not parallelize for v1 — keep it simple.
- Do not lower `COMMUNITY_SEMANTIC_THRESHOLD` in this change; it remains tunable per-deployment via env / CLI.

### 4.3 Wire it into `_build_igraph`

In `src/rag/community.py:_build_igraph`:

1. Update the signature to accept `cross_source_top_k: int` and `max_cross_source_queries: int`. Default the call site in `detect_communities` to read `settings.COMMUNITY_CROSS_SOURCE_TOP_K` and `settings.COMMUNITY_MAX_CROSS_SOURCE_QUERIES` when callers do not override.
2. Delete the existing `if len(entity_ids) <= 400:` block (currently lines ~231–250).
3. After the same-source weak-fallback loop completes, call:
   ```python
   embeddings = _load_entity_embeddings(entity_ids)
   semantic_edges = _load_cross_source_semantic_edges(
       entities, idx, embeddings,
       semantic_threshold, cross_source_top_k, max_cross_source_queries,
   )
   for key, weight in semantic_edges.items():
       if key in edge_weights:
           continue  # never overwrite a chunk-cooc / weak-fallback edge
       edge_weights[key] = weight
   ```
4. Leave the chunk co-occurrence loop, `cross` upweight, and the `>60` same-source-cap fallback unchanged. Those control intra-source clustering and are independent of the cross-source fix.

### 4.4 Plumb the new parameters

Mirror the existing `semantic_threshold` plumbing pattern through:

- `detect_communities(...)` in `src/rag/community.py` — add `cross_source_top_k: Optional[int] = None`, `max_cross_source_queries: Optional[int] = None`. Resolve `None` to settings defaults at the top, just like `semantic_threshold` is resolved today. Include both in the returned `metadata.parameters` dict so traces and tests can see the active values.
- `src/rag/cli.py` — add `--cross-source-top-k INT` and `--max-cross-source-queries INT` to `community ids`, `community search`, and `community retrieve`. Forward them into `detect_communities`.
- `src/rag/api/routes/community.py` and the request schema — add the two fields under `community_options`. Default to `None` so omission falls back to settings.

### 4.5 Documentation

In `README.md`:

- Add two rows to the "Community and worker settings" table:
  - `COMMUNITY_CROSS_SOURCE_TOP_K` — default `10` — Max cross-source semantic neighbors fetched per entity via pgvector ANN.
  - `COMMUNITY_MAX_CROSS_SOURCE_QUERIES` — default `5000` — Hard cap on per-entity ANN queries; entities are prioritized by chunk-mention count.
- Add the matching `--cross-source-top-k` and `--max-cross-source-queries` flags under each `rag community` subcommand option list, and the matching `community_options` fields under the REST `POST /api/community` body example.
- Add a short note in the "Community" prose that cross-source detection now uses pgvector ANN and is no longer disabled at large scopes; tuning `--semantic-threshold` lower (e.g. 0.75–0.80) helps surface looser cross-source clusters.

---

## 5. Compatibility and risk

- The HNSW index already exists for fresh installs (`scripts/init/postgres/02_schema.sql:79`). For older deployments without it, `pgvector` will fall back to a sequential scan but still return correct results; performance is the only impact. No migration is added in this change — call it out in the README note.
- Behavior change: scopes >400 entities will now produce cross-source communities where they previously did not. This is the desired outcome but may surprise existing operators relying on the old (effectively single-source) output. The new metadata fields make the active parameters discoverable.
- Default `COMMUNITY_MAX_CROSS_SOURCE_QUERIES=5000` keeps the worst case bounded at ~5k HNSW probes regardless of scope size.

---

## 6. Verification

### Unit / targeted tests

Add coverage in `tests/test_community.py`:

1. **Cross-source edge created**: build a fixture with two sources, two entities each, where entity A1 (source 1) and entity B1 (source 2) have near-identical embeddings above `semantic_threshold`. Assert `_build_igraph` produces an edge between their indices and that the resulting community has `is_cross_source=True`.
2. **Threshold respected**: same fixture but embeddings below threshold → no cross-source edge.
3. **Budget cap deterministic**: scope with `>max_cross_source_queries` entities → assert the helper queries exactly `max_cross_source_queries` entities and that they are the highest-`len(chunk_ids)` ones.
4. **Existing intra-source mechanics unchanged**: regression assertion on chunk co-occurrence edges and the `>60` same-source cap.

Existing tests to re-run:

```bash
pytest -q tests/test_community.py tests/test_cli_community.py tests/test_api_community.py tests/test_config.py
```

### End-to-end smoke

Assumes the local stack is up (`docker compose up -d`, `curl http://localhost:8000/api/health`).

1. Pick a query whose scope you know spans multiple sources:
   ```bash
   venv/bin/rag community search "<broad query>" --limit 20 --top-k 3
   ```
   Confirm at least one community in the response has `is_cross_source: true`. Confirm `metadata.parameters` includes `cross_source_top_k` and `max_cross_source_queries`.
2. Disable the new pass and re-run to confirm the change is the cause:
   ```bash
   venv/bin/rag community search "<broad query>" --cross-source-top-k 0 --top-k 3
   ```
   Expect zero cross-source communities (matching the pre-change baseline).
3. Time a large scope to confirm no regression to the pre-cap hang:
   ```bash
   time venv/bin/rag community search "<broad query>" --limit 200 --top-k 3
   ```
   Should complete in seconds, not minutes.

### Optional DB sanity

```bash
docker compose exec -T postgres psql -U rag -d rag -c "\di entities_embedding_hnsw_idx"
```
Confirm the HNSW index exists. If absent on this deployment, document the recommended `CREATE INDEX` in the README note.

---

## 7. Out of scope

- Lowering the default `COMMUNITY_SEMANTIC_THRESHOLD`. Tuning is per-deployment.
- Reworking entity dedup at ingestion. Cross-source edges via shared deduped entities continue to work as today; this change adds a second, embedding-based path.
- Parallelizing ANN queries. Sequential is fast enough at the default budget; revisit only if profiling shows it dominates wall-clock time.
- Changing the same-source weak-fallback `>60` cap. It is unrelated to the cross-source goal.
