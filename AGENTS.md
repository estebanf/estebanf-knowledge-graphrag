# Agent Notes

Use `docs/prd.md` for product requirements and intended architecture. This file is only for repo-specific continuation notes that are useful in active sessions.

## Current Retrieval State

- Verified CLI command:
  - `venv/bin/rag retrieve "<query>"`
  - `venv/bin/rag retrieve "<query>" --trace`

- Retrieval is CLI-only right now. There is no REST or MCP retrieval surface yet.

- `--trace` prints activity logs to stdout as retrieval runs, then prints the final response JSON as the last block on stdout.

- Retrieval depends on OpenRouter for:
  - query variant generation
  - query embeddings
  - reranking
  - graph-stage entity/query selection

- Retrieval config is env-backed through `src/rag/config.py`. Keep new knobs there and update both `.env.example` and `.env` when changing defaults.

## Live Data Notes

- Read-only inspection against the local databases showed:
  - `55` active sources
  - `1195` active embedded chunks
  - `18` completed jobs
  - `1650` Postgres `entities` rows
  - `1195` Memgraph `Chunk` nodes
  - `1704` Memgraph `Entity` nodes
  - `1704` `MENTIONS` edges
  - `6591` `MENTIONED_IN` edges
  - `677` `RELATED_TO` edges

- All active sources currently have both `metadata` and `markdown_content`.

- Current chunk population in Postgres is:
  - `1054` `markdown-header`
  - `141` `semantic`
  - `0` child chunks with `parent_chunk_id`

## Retrieval Implementation Notes

- Use `MENTIONS` as the authoritative chunk-to-entity edge for retrieval expansion.
- Do not treat `MENTIONED_IN` as precise mention evidence. In current data it is much broader than true per-mention linkage.
- The live corpus currently has no hierarchical parent/child chunks, so retrieval should not assume parent surfacing exists for already ingested data.
- Sparse retrieval uses runtime Postgres full-text search over `chunks.content`; there are no retrieval-specific schema or index changes.
- Final root scoring reranks the root chunk first, then reranks related chunks and aggregates those scores.
- Graph expansion now uses a per-seed wall-clock budget instead of one shared wall-clock budget for the entire retrieval call.
- When `MENTIONS` expansion yields no non-seed chunks, retrieval falls back to same-source neighbor chunks scored against the entity-aware query.

## Local Environment

- Docker services were verified healthy with `docker compose ps`.
- If a database predates the current schema, the migrations that matter most are:
  - `scripts/migrate/001_add_markdown_content.sql`
  - `scripts/migrate/002_update_vector_dimensions.sql`
  - `scripts/migrate/004_job_improvements.sql`

## Verification Commands

- Retrieval-focused:
  - `pytest -q tests/test_retrieval.py tests/test_cli_retrieve.py tests/test_config.py`
  - `venv/bin/rag retrieve "What topics are covered in the ingested reports?" --result-count 1 --seed-count 1 --trace`

- Fast suite used during ingestion work:
  - `pytest -q tests/test_cli_jobs.py tests/test_job_lifecycle.py tests/test_worker.py tests/test_ingestion_submit.py tests/test_parser.py tests/test_storage.py tests/test_cli_ingest.py tests/test_chunking.py tests/test_observability.py tests/test_cli_health.py tests/test_profiling.py tests/test_chunk_validation.py tests/test_embedding.py`

## Known Gaps Worth Remembering

- Some reranked root chunks currently have no linked `MENTIONS` edges, so valid retrieval results may come back with `related: []`.
- `related: []` is less common after the same-source fallback, but it still happens for seeds that have neither useful graph links nor useful local neighbor chunks.
- `src/rag/cli.py` hard delete still must remove `entities` and `chunks` before `jobs`; otherwise the `chunks.job_id` foreign key breaks deletes.
