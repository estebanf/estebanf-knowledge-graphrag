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
- `src/rag/retrieval.py`: hybrid search, retrieval expansion, reranking, and trace behavior
- `src/rag/community.py`: entity-community detection and optional summarization
- `src/rag/answering.py`: answer generation over retrieval output
- `src/rag/prompts/__init__.py`: shared prompt templates; this is the canonical prompt maintenance location
- `src/rag/storage.py`, `src/rag/sources.py`: stored-file and source-detail helpers
- `tests/`: CLI, API, ingestion, retrieval, prompt, and community coverage
- `scripts/`: local environment startup, backups, migrations, and utility entrypoints
- `frontend/`: React UI, Vite config, and nginx assets for the containerized frontend

## Packaging Note

- Base package installs support API/search/retrieval/community flows.
- Local parsing and ingestion require the optional `ingest` extra: `pip install -e .[ingest]`.

## Current Behavioral Notes

- Search, retrieval, and community APIs are implemented under `src/rag/api/routes/`.
- `community retrieve` resolves source scope through a lightweight retrieval-stage pass, not full retrieval result expansion.
- `--trace` on retrieval still prints live activity first and the final JSON block last.
- Retrieval config remains env-backed through `src/rag/config.py`.
- Use `MENTIONS` as the authoritative chunk-to-entity edge for retrieval expansion.
- Same-source fallback is part of retrieval when graph expansion yields no non-seed chunk evidence.
- Insight extraction uses the OpenCode API (`deepseek-v4-flash`) per chunk. Dedup uses pgvector `<=>` cosine distance with `INSIGHT_DEDUP_COSINE_THRESHOLD`.
- Insight extraction parallelizes only OpenCode calls using `INSIGHT_EXTRACTION_CONCURRENCY`; embeddings, dedup, Postgres writes, and Memgraph writes remain serial.
- Mutual top-K for insight `RELATED_TO` edges is computed in Postgres via pgvector and excludes same-source candidates; Memgraph stores the resulting `Insight` nodes and edges.
- `scripts/remediate_insights.py` backfills insights directly, without jobs or workers. `--source-id` targets one source; `--force` cleans that source's existing insight links and rebuilds them.

## Data and Schema Notes

- The live corpus may contain many chunks without hierarchical parent-child structure. Do not assume parent surfacing is available for already ingested data.
- Sparse retrieval uses Postgres full-text search over `chunks.content`; the default `english` config is expected to have the matching GIN index from `scripts/migrate/006_search_performance_indexes.sql`.
- `insights` and `chunk_insights` are added in `scripts/migrate/007_insights.sql`; apply it on any database predating insight extraction.
- Hard delete order matters: remove insight join rows, orphan insights, `entities`, and `chunks` before `jobs`, then `sources`, or foreign keys will break deletes.
- Postgres schema is initialized from `scripts/init/postgres/`.

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

When unsure, read the README verification section and then narrow to the impacted area.
