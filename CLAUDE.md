# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **personal, self-hosted RAG (Retrieval-Augmented Generation) system** combining vector embeddings with a knowledge graph. The full specification is in `docs/prd.md`.

## Implementation Status

**Phase 3 complete** (`feature04-async`, 2026-04-18): Async ingestion worker, structured logging, audit trail, error handling, batch ingest, PPTX support, graph extraction and linking.

### What's implemented

Full ingestion pipeline: `pending → parsing → profiling → chunking → validation → embedding → graph_extraction → graph_linking → completed`

- **CLI commands**: `rag health`, `rag ingest` (files or folder), `rag worker`, `rag jobs list/status/retry/cancel`, `rag sources list/get/delete`
- **Async worker**: `src/rag/worker.py` — polls for pending jobs, claims atomically with `SELECT FOR UPDATE SKIP LOCKED`, recovers stuck jobs on startup
- **Structured logging**: `src/rag/logging_config.py` — structlog JSON with contextvars; all required fields defaulted
- **Audit trail**: `audit_log` table; events written at job_submitted, job_completed, job_failed, job_retried, job_cancelled, source_soft_deleted, source_hard_deleted
- **Error detail**: `jobs.error_detail JSONB` stores stage, message, and traceback on failure
- **Parsing**: `src/rag/parser.py` — returns `ParseResult(markdown, element_tree)`; PDF/DOCX/PPTX via docling, TXT/MD native; `parse_to_markdown()` kept as compat wrapper
- **Profiling**: `src/rag/profiling.py` — LLM call via `MODEL_DOC_PROFILING`; returns `_DEFAULT_PROFILE` on error
- **Chunking**: `src/rag/chunking.py` — strategies: `markdown-header`, `recursive`, `semantic`; optional hierarchical and proposition layers
- **Validation**: `src/rag/chunk_validation.py` — sample-based LLM check; 10% general / 25% high-stakes / 100% first-of-type
- **Embedding**: `src/rag/embedding.py` — OpenRouter `/api/v1/embeddings` in batches of 32; stored as `vector(4096)`
- **Graph extraction**: `src/rag/graph_extraction.py` — LLM entity/relationship extraction per chunk
- **Graph linking**: `src/rag/graph_linking.py` — entity dedup, MENTIONED_IN edges; Entity embeddings stored at extraction time; dedup uses pgvector `<=>` in SQL (no re-embedding at link time); `MENTIONED_IN` edges created via single `UNWIND` Cypher.
- **Versioned file storage**: `{STORAGE_BASE_PATH}/{source_id}/{version}/original_{filename}`
- **Tests**: 85 tests — all pass

### What's not yet implemented

- REST API and MCP server interfaces
- Retrieval pipeline (hybrid search, graph traversal, reranking, answer generation)
- Authentication / API keys

### Developer setup

```bash
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD and OPENROUTER_API_KEY
./scripts/start.sh
pip install -e .
```

Apply migrations if your database existed before the current code:

```bash
psql $POSTGRES_URL -f scripts/migrate/001_add_markdown_content.sql
psql $POSTGRES_URL -f scripts/migrate/002_update_vector_dimensions.sql
psql $POSTGRES_URL -f scripts/migrate/003_add_graph_tables.sql
psql $POSTGRES_URL -f scripts/migrate/004_job_improvements.sql
psql $POSTGRES_URL -f scripts/migrate/005_entity_embeddings.sql
```

Or use the inline form for the latest migration only:

```bash
docker compose exec -T postgres psql -U rag -d rag -c "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS error_detail JSONB; DROP INDEX IF EXISTS sources_md5_idx; CREATE UNIQUE INDEX IF NOT EXISTS sources_md5_unique_idx ON sources(md5) WHERE deleted_at IS NULL; CREATE INDEX IF NOT EXISTS jobs_status_updated_idx ON jobs(status, updated_at) WHERE status LIKE 'processing:%';"
```

---

## Infrastructure & Operations

### Data Storage

All persistent data lives under `./data/` (git-ignored). Subdirectories:

| Path | Contents |
|---|---|
| `data/postgres/` | PostgreSQL data files |
| `data/memgraph/` | Memgraph snapshots + WAL |
| `data/documents/` | Ingested source files (`STORAGE_BASE_PATH`) |
| `data/backups/` | Timestamped backups from `scripts/backup.sh` |

`scripts/start.sh` creates these directories on first run.

### CLI Scripts

All scripts resolve paths relative to the script location — run from any directory.

| Command | Effect |
|---|---|
| `./scripts/start.sh` | Start Postgres + Memgraph, init Memgraph schema on first run |
| `./scripts/stop.sh` | Stop both containers (data preserved) |
| `./scripts/reset.sh` | Stop + wipe `data/postgres/` and `data/memgraph/` (interactive confirmation required) |
| `./scripts/backup.sh` | Dump Postgres SQL + copy Memgraph data to `data/backups/<timestamp>/` |

### Connection Details (defaults)

| Service | Address | Notes |
|---|---|---|
| PostgreSQL | `localhost:5432` | db: `rag`, user: `rag` |
| Memgraph Bolt | `bolt://localhost:7687` | Used by the application |
| Memgraph Lab UI | `http://localhost:3000` | Browser-based graph explorer |

### Docker Images

| Service | Image | Key features |
|---|---|---|
| PostgreSQL | `paradedb/paradedb:latest` | PostgreSQL 17 + pgvector (HNSW) + pg_search (BM25) |
| Memgraph | `memgraph/memgraph-platform:latest` | Memgraph DB + Lab UI + MAGE algorithms |

### Schema Notes

- Postgres schema is initialized automatically on first container start via `scripts/init/postgres/`.
- Memgraph constraints and indexes are applied by `scripts/start.sh` on first run (flagged by `data/memgraph/.initialized`).
- Vector embedding dimension is `4096` (`qwen/qwen3-embedding-8b`). pgvector 0.8.1 caps HNSW at 2000 dims — no HNSW index on embedding columns; queries use exact sequential scan.
- The BM25 index (`CALL paradedb.create_bm25(...)`) is **not** created during schema init — create it after the first bulk ingestion for optimal index quality.
