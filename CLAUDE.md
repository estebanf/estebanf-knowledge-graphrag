# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **personal, self-hosted RAG (Retrieval-Augmented Generation) system** combining vector embeddings with a knowledge graph. The full specification is in `docs/prd.md`.

## Implementation Status

**Phase 2 complete** (`feature02-chunking` branch, 2026-04-17): Adaptive chunking, chunk validation, and embeddings.

**Phase 1 complete** (`feature01-collection` branch, 2026-04-17): Ingestion pipeline — document storage, metadata capture, and file parsing.

### What's implemented (Phase 2)

- **Profiling**: `src/rag/profiling.py` — LLM call via `MODEL_DOC_PROFILING`; classifies structure_type, heading_consistency, content_density, primary_content_type, avg_section_length, has_tables, has_code_blocks, domain. Returns `_DEFAULT_PROFILE` on any error.
- **Chunking**: `src/rag/chunking.py` — deterministic strategy selection (`markdown-header` for well-structured+consistent, `recursive` otherwise) with optional hierarchical (avg_section_length=long) and proposition layers (legal/financial/medical via `MODEL_PROPOSITION_CHUNKING`). Uses LangChain text splitters + tiktoken. `ChunkData.parent_chunk_id` stores parent chunk_index as a digit string; resolved to DB UUID by `_insert_chunks`.
- **Validation**: `src/rag/chunk_validation.py` — sample-based LLM quality check via `MODEL_CHUNK_VALIDATION`; 10% general / 25% high-stakes / 100% first-of-type. >20% failures → job fails.
- **Embedding**: `src/rag/embedding.py` — OpenRouter `/api/v1/embeddings` in batches of 32 via `MODEL_EMBEDDING`; stored in `chunks.embedding vector(4096)`.
- **Schema migration**: `scripts/migrate/002_update_vector_dimensions.sql` — updates vector columns from 1536 → 4096 dims. Note: pgvector 0.8.1 caps HNSW at 2000 dims; HNSW indexes omitted, queries use exact scan.
- **New env vars**: `MODEL_EMBEDDING=qwen/qwen3-embedding-8b`, `EMBEDDING_DIMENSIONS=4096`
- **Tests**: 26 tests across `test_ingestion.py`, `test_profiling.py`, `test_chunking.py`, `test_chunk_validation.py`, `test_embedding.py` — all pass.

### What's implemented (Phase 1)

- **Python package**: `src/rag/` installed as `pip install -e .`; entrypoint: `venv/bin/rag`
- **CLI commands**: `rag ingest <file>`, `rag sources list/get/delete`
- **Ingestion flow** (synchronous): MD5 dedup → file storage → Postgres record → parse → LLM metadata extraction → profiling → chunking → validation → embedding → completed
- **Supported formats**: PDF, DOCX, MD, TXT
- **Data model addition**: `sources.markdown_content TEXT` (migration: `scripts/migrate/001_add_markdown_content.sql`)

### What's not yet implemented

- Graph extraction and graph linking stages
- REST API and MCP server interfaces
- `rag jobs` commands
- Authentication / API keys

### Developer setup (Phase 1 + 2)

```bash
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD and OPENROUTER_API_KEY
./scripts/start.sh
psql $POSTGRES_URL -f scripts/migrate/001_add_markdown_content.sql  # if DB already existed before Phase 1
psql $POSTGRES_URL -f scripts/migrate/002_update_vector_dimensions.sql  # if DB already existed before Phase 2
pip install -e .
rag ingest test_documents/Play\ 2.md
```

---

## Architecture

### Infrastructure (Docker Compose, single-host)

- **PostgreSQL + pgvector + pg_bm25 (ParadeDB)** — all relational data: sources, chunks, embeddings, jobs, API keys, audit log, BM25 index
- **Memgraph** — graph topology only (entity nodes, relationship edges, chunk nodes); `chunk_id`/`entity_id` in Memgraph are foreign keys back to Postgres. Graph traversal returns IDs → chunk text is hydrated from Postgres in a single batch query
- **pgqueuer** — job queue built on Postgres; no external broker
- **Observability stack** — Loki + Grafana (logs), Prometheus + Grafana (metrics), OpenTelemetry → Tempo (tracing)

### Interfaces

Three interfaces expose the same capabilities:
1. **REST API** — `Authorization: Bearer <key>` header auth
2. **MCP Server** — agent-facing tools (see PRD §2.6 for full tool list)
3. **CLI** — `rag` command; authenticates via local config file

### LLM Access

All LLM calls go through **OpenRouter** via a single `OPENROUTER_API_KEY`. **No model names are hardcoded** — every model role is configured via environment variable (see PRD §3.4 for the full variable list and recommended defaults).

---

## Key Design Constraints

### Ingestion Pipeline

Pipeline stages in order: `pending → parsing → profiling → chunking → validation → embedding → graph_extraction → graph_linking → completed`

- All ingestion is **async** — job submission returns a `job_id` immediately
- A failure at any stage transitions to `failed:<stage>`; no partial content from a failed job is visible to retrieval
- **Deduplication**: MD5 hash checked before pipeline entry; duplicate returns `409 Conflict`
- **Chunking strategy is adaptive** — selected per document based on a profiling step, not globally configured
- **Chunk quality validation** runs on a sample (10% default; 25–30% for legal/financial/medical; 100% on first ingestion of a new type)
- Job retry can resume from the failed stage or any earlier stage; retrying from an earlier stage deletes artifacts from all later stages first

### Retrieval Pipeline

- **Hybrid retrieval**: BM25 (sparse) + vector search (dense), merged via Reciprocal Rank Fusion (RRF)
- **Graph traversal** runs in parallel; pre-filtered by edge confidence threshold + cosine similarity (≥ 0.40) before LLM scoring
- **Query preprocessing**: HyDE, query expansion, step-back prompting, and decomposition — all query variants execute in parallel through the full retrieval pipeline
- **Reranking is mandatory** — cross-encoder or LLM-based, applied after fusion
- Answer generation is strictly grounded: the system prompt forbids using outside knowledge (see PRD §2.3.6 for exact prompt)

### Authentication & API Keys

- Raw key shown **once** at creation; only SHA-256 hash stored
- `api_key_name` is injected into the logging context for every request — all log lines carry it
- Keys are soft-revoked, never deleted

### Data Lifecycle

- **Sources are soft-deleted** (hard delete is CLI-only)
- Document versioning: re-submitting a changed file creates a new version; previous version's chunks/embeddings/graph nodes are soft-deleted
- Only the latest active version participates in retrieval and graph traversal

### Observability

Structured JSON logging via `structlog`. Every log line must include: `timestamp`, `level`, `job_id`, `api_key_name`, `stage`, `action`, `duration_ms`, `model_used`, `token_count`, `status`, `error`.

Health endpoints: `GET /health/live` and `GET /health/ready` (checks Postgres + Memgraph).

---

## Data Model Quick Reference

**Postgres tables:** `api_keys`, `sources`, `jobs`, `chunks`, `entities`, `audit_log`

**Memgraph node labels:** `:Source`, `:Chunk`, `:Entity`

**Memgraph edge types:** `INCLUDES`, `MENTIONS` (with confidence), `RELATED_TO` (with type/confidence/chunk_id), `MENTIONED_IN`

See PRD §3.2 and §3.3 for full schema DDL and graph model.

---

## Environment Variables

Key variables (full list in PRD §3.6):

```bash
STORAGE_BASE_PATH=/data/documents
POSTGRES_URL=postgresql://...
MEMGRAPH_URL=bolt://...
OPENROUTER_API_KEY=...
# One env var per model role — see PRD §3.4
DEFAULT_RETRIEVAL_K=10
DEFAULT_CONFIDENCE_THRESHOLD=0.75
GRAPH_TRAVERSAL_MAX_DEPTH=2
LOG_LEVEL=INFO
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

### First-Time Setup

```bash
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD at minimum
./scripts/start.sh
```

### Connection Details (defaults)

| Service | Address | Notes |
|---|---|---|
| PostgreSQL | `localhost:5432` | db: `rag`, user: `rag` |
| Memgraph Bolt | `bolt://localhost:7687` | Used by the application |
| Memgraph Lab UI | `http://localhost:3000` | Browser-based graph explorer |

### Environment

Copy `.env.example` to `.env` before first run. The `.env` file is git-ignored. Set `POSTGRES_PASSWORD` at minimum. All model roles and retrieval defaults have sensible defaults in `.env.example`.

### Docker Images

| Service | Image | Key features |
|---|---|---|
| PostgreSQL | `paradedb/paradedb:latest` | PostgreSQL 17 + pgvector (HNSW) + pg_search (BM25) |
| Memgraph | `memgraph/memgraph-platform:latest` | Memgraph DB + Lab UI + MAGE algorithms |

### Schema Notes

- Postgres schema is initialized automatically on first container start via `scripts/init/postgres/`.
- Memgraph constraints and indexes are applied by `scripts/start.sh` on first run (flagged by `data/memgraph/.initialized`).
- Vector embedding dimension is `4096` (updated by `002_update_vector_dimensions.sql` for `qwen/qwen3-embedding-8b`). pgvector 0.8.1 caps HNSW at 2000 dims — no HNSW index on embedding columns; queries use exact sequential scan.
- The BM25 index (`CALL paradedb.create_bm25(...)`) is **not** created during schema init — create it after the first bulk ingestion for optimal index quality.
