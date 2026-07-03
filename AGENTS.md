# Agent Notes


# Project Overview
This is a personal, self-hosted RAG (Retrieval-Augmented Generation) system combining vector embeddings with a knowledge graph. 

Read `README.md` first. It is the canonical operator and API reference for this repository.

## Operating Rules

- Do not duplicate README content here when updating this file. Keep `AGENTS.md` focused on agent workflow and codebase navigation.
- Prefer code-backed answers over stale notes. This file can lag the implementation.
- When you do modifications that affect functionally the application, update `README.md` and `AGENTS.md` accordingly.


## Local Stack

This repo runs its local services in Docker:

- `postgres`
- `memgraph`
- `backend`
- `frontend`

Assume the databases are inside containers, not directly on the host.

### Database access

Use container-scoped commands instead:

```bash
docker compose exec -T postgres psql -U rag -d rag
docker compose exec -T memgraph mgconsole
```

For one-off SQL or migrations, keep using `docker compose exec -T postgres ...`.

Examples:

```bash
docker compose exec -T postgres psql -U rag -d rag -f scripts/migrate/004_job_improvements.sql
docker compose exec -T postgres psql -U rag -d rag -c "SELECT count(*) FROM jobs;"
printf "MATCH (n) RETURN count(n);\n" | docker compose exec -T memgraph mgconsole
```

### Service assumptions

- `backend` is the FastAPI app from the root `Dockerfile`
- `frontend` is the built React app served by nginx from `frontend/`
- frontend API traffic is proxied to `backend`
- if you need to validate service state, start with `docker compose ps` and `curl http://localhost:8000/api/health`

## Code Organization

The repo is split into a few main areas:

- `src/rag/cli.py`: Typer CLI entrypoints for ingest, search, retrieve, jobs, sources, and community commands
- `src/rag/api/`: FastAPI app, routes, and request/response schemas
- `src/rag/config.py`: env-backed runtime settings; new knobs belong here and usually also in `.env.example`
- `src/rag/ingestion.py`: ingestion job submission, pipeline orchestration, retry/cancel cleanup, and artifact deletion
- `src/rag/worker.py`: async job polling and execution loop
- `src/rag/parser.py`: document-to-markdown parsing and inline image description replacement
- `src/rag/chunking.py`, `src/rag/chunk_validation.py`, `src/rag/profiling.py`, `src/rag/embedding.py`: chunk pipeline stages
- `src/rag/graph_extraction.py`, `src/rag/graph_linking.py`: graph creation and linking
- `src/rag/insight_extraction.py`: OpenCode API call, per-chunk insight extraction, pgvector dedup against `insights`, and Memgraph `Insight` node plus `CONTAINS`/`RELATED_TO` edge management
- `src/rag/retrieval.py`: hybrid chunk and insight search, retrieval expansion for both chunks and insights (entity MENTIONS for chunks, RELATED_TO + LLM sub-queries for insights), reranking, and trace behavior
- `src/rag/community.py`: entity-community detection and optional summarization
- `src/rag/answering.py`: answer generation over retrieval output
- `src/rag/prompts/__init__.py`: shared prompt templates; this is the canonical prompt maintenance location
- `src/rag/storage.py`, `src/rag/sources.py`: stored-file and source-detail helpers
- `tests/`: CLI, API, ingestion, retrieval, prompt, and community coverage
- `scripts/`: local environment startup, backups, migrations, and utility entrypoints
- `frontend/`: React UI, Vite config, and nginx assets for the containerized frontend
- `docs/solutions/`: documented solutions to past problems, organized by category with YAML frontmatter (`module`, `tags`, `problem_type`); relevant when implementing or debugging in documented areas
- `CONCEPTS.md`: shared project vocabulary for domain concepts used across retrieval, ingestion, graph, and storage work

## Packaging Note

- Base package installs support API/search/retrieval/community flows **and the backend worker** (which parses markdown/text only). The backend Docker image installs the base package only — no Docling, Torch, or CUDA.
- Converting binary documents (PDF/DOCX/PPTX) to markdown happens on the **CLI** and requires the optional `prepare` extra (Docling): `pip install -e .[prepare]`.
- Ingestion responsibility split: `rag ingest`/`rag prepare` convert binaries locally (`src/rag/prepare.py`), then submit self-contained markdown. Image descriptions are backend-owned via `POST /api/prepare/describe-image` (ingest scope) so the CLI never holds the OpenRouter key. Direct PDF/DOCX/PPTX upload to `/api/ingest` is rejected; markdown/text upload remains supported. Prepared submissions preserve original-binary provenance (filename, extension, md5) — dedup keys on the original binary hash stored in `sources.md5`.

## Auth and server invariants

- Every `/api/*` route except `GET /api/health` and `POST /api/auth/login` requires a `Principal` resolved by `rag.api.auth.require_principal`. The dep accepts either an `Authorization: Bearer <key>` (validated against `data/api_keys.toml`) or a session cookie set by `POST /api/auth/login` (validated against the `user_sessions` table).
- API key file is hot-reloaded by mtime. Editing `data/api_keys.toml` does not require a backend restart.
- The frontend uses username/password → `HttpOnly` session cookie; the CLI and MCP use API keys. The same `require_principal` accepts either.
- Scopes are `read`, `ingest`, `admin`. `admin` implies all. Route declarations use `Depends(require_principal())` and explicit `requires_scope("...")` on top when needed.
- `data/api_keys.toml` lives outside Postgres on purpose so an operator can edit it by hand. Auditing of usage is in the `api_key_audit` table.

## Worker supervisor

- The backend owns worker lifecycle via `rag.worker_supervisor.WorkerSupervisor` (singleton, attached lazily through `get_supervisor()`).
- Invariant: every spawn writes a row to `worker_processes` **before** the subprocess starts; the reaper thread owns subsequent status transitions. Never mutate worker rows outside the supervisor.
- The supervised process is the hidden CLI command `rag _worker-run --worker-id <uuid>`. Do not document it for end users — they use `rag worker launch/stop/list/log`.
- Worker logs are at `data/worker_logs/{worker_id}.log` (append, line-buffered). `tail_log(worker_id, follow=True)` streams them via SSE.
- On startup the supervisor reconciles orphan rows: any `running` row whose PID is not alive is marked `crashed`.

## MCP server

- Mounted at `/mcp` via `rag.mcp_server.build_mcp_app()`. Uses Streamable HTTP transport from the official Python `mcp` SDK.
- The MCP ASGI sub-app is wrapped with `BearerAuthMiddleware`, which reuses the same `KeyStore` as the REST layer. Bypassing it would create a parallel auth surface; do not.
- The FastAPI `lifespan` enters the inner MCP app's lifespan so the streamable-http session manager is initialized. Tests must use `with TestClient(app) as c:` to trigger lifespan.
- Tool surface is intentionally read-only: `search`, `retrieve`, `community`, `list_sources`, `source_insights`. Ingestion, jobs, and worker management are CLI-only.

## CLI

- `rag.cli_config.load_cli_config()` resolves precedence env > `~/.config/rag/config.toml` > none. `rag configure` writes the file (mode `0600`).
- `rag.api_client.RagClient` is the only HTTP entry point. CLI commands check `_use_api()` and dispatch through `_get_client()`; without config they fall back to direct-DB behavior (kept for local-dev convenience). New CLI commands should be API-only unless there's a strong reason.
- CLI tests use `httpx.MockTransport` via `RagClient.with_transport`. The auto-applied `_isolate_cli_config` fixture in `tests/conftest.py` ensures no developer-machine config bleeds into the test run.

## Current Behavioral Notes

- Search, retrieval, and community APIs are implemented under `src/rag/api/routes/`.
- Search now returns both chunks and insights via `hybrid_search()` → `HybridSearchResults`. The `limit` parameter applies per type (e.g. `limit=10` returns up to 10 chunks and up to 10 insights). Insight search uses the same dense+sparse+RRF pattern as chunks, querying `insights.embedding` and `insights.content`, then resolving sources and topics through `chunk_insights`.
- The frontend renders search results in two sections: "Insights" (via `InsightCard`) and "Chunks" (via `ResultCard`).
- The frontend has a Sources tab that calls `/api/sources`, `/api/sources/{source_id}`, and `/api/sources/{source_id}/insights` to browse recent sources, filter by source metadata, copy source insight JSON, copy individual insight text, and switch the selected source detail between rendered markdown and insight topic connections. The left column has a search bar that passes `q` to `GET /api/sources` for live fuzzy metadata search; it combines with exact-match `metadata=` filter chips.
- `GET /api/sources` accepts a `q` query param for fuzzy metadata search: bare value matches any metadata value, `name`, or `file_name` via `jsonb_each_text ILIKE`; `key:value` restricts to that metadata key. Implemented in `list_recent_sources` in `src/rag/sources.py`.
- `rag sources search <query>` CLI command searches sources by metadata value and displays a table. Supports `key:value` for specific-key search.
- `community retrieve` resolves source scope through a lightweight retrieval-stage pass, not full retrieval result expansion.
- `--trace` on retrieval still prints live activity first and the final JSON block last.
- Retrieval config remains env-backed through `src/rag/config.py`.
- Retrieval starts chunk and insight variant generation in parallel. Active first-stage variants are `original`, HyDE, decomposed chunk sub-queries, and gated `expanded`; `step_back` is no longer scheduled by default.
- Dense chunk and insight retrieval over 4096-dimensional embeddings depends on the binary-quantized HNSW prefilter indexes from `scripts/migrate/009_binary_vector_prefilter_indexes.sql`, followed by full-vector reranking in SQL.
- Chunk graph expansion uses deterministic `query + entity` expansion queries by default. `RETRIEVAL_USE_LLM_ENTITY_QUERIES=true` restores the slower LLM-written entity query behavior.
- Second-hop chunk entity selection is deterministic by default. `RETRIEVAL_USE_LLM_SECOND_HOP_SELECTION=true` restores slower LLM-backed second-hop selection.
- Use `MENTIONS` as the authoritative chunk-to-entity edge for retrieval expansion.
- Same-source fallback is part of retrieval when graph expansion yields no non-seed chunk evidence.
- Entity extraction enforces `entity_type` via JSON schema validation at parse time; items with invalid types (e.g. DATE, ROLE, EVENT) are silently dropped before insert.
- `canonical_name` and all aliases are normalised before insert: `html.unescape` → whitespace collapse → leading/trailing punctuation strip (`_normalise_name` in `graph_extraction.py`).
- Exact-match dedup runs at insert time: if a `canonical_name` already exists in `entities`, the existing row is reused and no INSERT is issued. This is a SELECT-first pattern; concurrent ingestion of the same entity may create a duplicate, which is resolved by the offline `merge_semantic_duplicates.py` script.
- The `ENTITY_EXTRACTION` prompt instructs the LLM to correct misspellings and transcript noise to canonical form before extracting (e.g. "Chad GPT" → "ChatGPT").
- CLI functions `submit_ingestion_job`, `retry_job`, `cancel_job`, `get_connection`, and `get_graph_driver` are exposed as module-level names in `rag.cli` (via lazy-import wrappers) so test patches work correctly without eagerly loading heavy modules.
- Insight extraction uses the OpenCode API (`deepseek-v4-flash`) per chunk. Dedup uses pgvector `<=>` cosine distance with `INSIGHT_DEDUP_COSINE_THRESHOLD`.
- Insight extraction parallelizes only OpenCode calls using `INSIGHT_EXTRACTION_CONCURRENCY`; embeddings, dedup, Postgres writes, and Memgraph writes remain serial.
- Mutual top-K for insight `RELATED_TO` edges is computed in Postgres via pgvector and excludes same-source candidates; Memgraph stores the resulting `Insight` nodes and edges.
- `scripts/remediate_insights.py` backfills insights directly, without jobs or workers. It prints source/chunk counts plus extraction and serial storage progress. `--batch-size` caps one run to that many newest eligible sources; `--source-id` targets one source; `--force` cleans that source's existing insight links and rebuilds them.

## Data and Schema Notes

- The live corpus may contain many chunks without hierarchical parent-child structure. Do not assume parent surfacing is available for already ingested data.
- Sparse retrieval uses Postgres full-text search over `chunks.content`; the default `english` config is expected to have the matching GIN index from `scripts/migrate/006_search_performance_indexes.sql`.
- `insights` and `chunk_insights` are added in `scripts/migrate/007_insights.sql`; apply it on any database predating insight extraction.
- Hard delete order matters: remove insight join rows, orphan insights, `entities`, and `chunks` before `jobs`, then `sources`, or foreign keys will break deletes.
- Postgres schema is initialized from `scripts/init/postgres/`.
- `entities` has tightened autovacuum thresholds (`scale_factor = 0.02`, set in `scripts/init/postgres/02_schema.sql` and `scripts/migrate/010_entities_autovacuum_tuning.sql`) because `scripts/merge_semantic_duplicates.py`/`merge_duplicate_entities.py` are update/delete-heavy; both scripts also call `rag.db.vacuum_analyze_entities()` after a run that merges at least one row. Don't remove that call without re-checking `entities` doesn't re-bloat.
- Dense retrieval (`dense_retrieve`/`insight_dense_retrieve` in `src/rag/retrieval.py`) sets Postgres's `hnsw.ef_search` to `RETRIEVAL_DENSE_PREFETCH_COUNT` before the binary-quantized prefilter query. pgvector's HNSW candidate search silently caps returned rows at `hnsw.ef_search` (default `40`) regardless of the SQL `LIMIT`, so without this the prefetch count setting was a no-op past ~40 candidates.
- `src/rag/graph_db.py::reconcile_schema()` re-applies Memgraph's index/constraint statements idempotently on every backend startup (`SCHEMA_STATEMENTS`, mirroring `scripts/init/memgraph_init.cypher`). Add new statements to that list, not just the `.cypher` file, or a live instance won't pick them up.
- The paradedb Postgres image **ignores the `POSTGRES_*` env vars for tuning** (it bakes `shared_buffers=128MB`, `max_connections=100`, etc. into its own `postgresql.conf`). `docker-compose.yml` therefore passes every tuned setting as a `postgres -c` server argument (command-line args override `postgresql.conf`), sourced from those same env vars so overrides still work. When adding a Postgres GUC, add it to the `command:` list, not the `environment:` block, and verify with `SHOW <setting>` — don't trust the compose env value.
- Vector-index prewarming keeps the binary HNSW indexes resident so the first dense query after a restart is fast. Two layers: (1) `docker-compose.yml` preloads `pg_prewarm` with `pg_prewarm.autoprewarm=on` (`shared_preload_libraries` **must keep** ParadeDB's own `pg_search,pg_cron,pg_stat_statements` and only append `pg_prewarm`), so Postgres re-warms its own buffers after any restart; (2) `rag.db.prewarm_vector_indexes()` (index names in `PREWARM_INDEXES`) is called from the FastAPI lifespan (`src/rag/api/main.py::_prewarm_with_retry`) via a non-blocking background retry, tolerant of Postgres-not-yet-ready and missing extension/indexes, mirroring the non-fatal Memgraph reconciliation. Extension is created in `scripts/init/postgres/01_extensions.sql` + `scripts/migrate/011_pg_prewarm.sql`.
- `hybrid_search()` (the `rag search` path in `src/rag/retrieval.py`) runs its chunk dense, chunk sparse, and insight legs concurrently in a `ThreadPoolExecutor`, each on its own connection (psycopg connections aren't thread-safe to share). `insight_hybrid_search` stays a single leg so result shape is unchanged.
- `src/rag/cli.py` lazy-imports the heavy retrieval/graph stack (`get_connection`, `get_graph_driver`, `hybrid_search`, `retrieve` are module-level lazy wrappers, like `submit_ingestion_job`) so API-mode CLI commands don't pull in neo4j/pandas on every invocation. Keep them as module-level names — tests patch `rag.cli.<name>`. Don't convert them back to top-level `from ... import` statements.

## Valuable Carryover From `CLAUDE.md`

These points were still useful and belong here:

- `src/rag/prompts/__init__.py` is the single place to maintain shared prompt text.
- The backend/frontend are containerized and should be treated as first-class local services.
- Postgres schema initialization happens automatically from `scripts/init/postgres/`.

## Verification Shortcuts

Pick verification based on the area you changed:

- retrieval/search/community logic: `pytest -q tests/test_retrieval.py tests/test_cli_retrieve.py tests/test_cli_search.py tests/test_cli_community.py tests/test_api.py tests/test_api_community.py tests/test_config.py`
- ingestion/jobs/parser/storage: `pytest -q tests/test_cli_jobs.py tests/test_job_lifecycle.py tests/test_worker.py tests/test_ingestion_submit.py tests/test_parser.py tests/test_storage.py tests/test_cli_ingest.py`
- insight extraction: `pytest -q tests/test_insight_extraction.py tests/test_config.py tests/test_prompts.py tests/test_cli_sources.py`
- prompts: `pytest -q tests/test_prompts.py`
- auth (api keys + sessions + gating): `pytest -q tests/test_api_auth_apikey.py tests/test_api_auth_session.py tests/test_api_gating.py`
- new server APIs (ingest, jobs, sources delete, workers, mcp): `pytest -q tests/test_api_ingest.py tests/test_api_jobs.py tests/test_api_sources_delete.py tests/test_api_workers.py tests/test_worker_supervisor.py tests/test_mcp_server.py`
- CLI HTTP path: `pytest -q tests/test_cli_api_mode.py tests/test_cli_config.py tests/test_api_client.py`
- end-to-end against the live stack: `scripts/smoke_e2e.sh` (requires `docker compose up -d`)

When unsure, read the README verification section and then narrow to the impacted area.
