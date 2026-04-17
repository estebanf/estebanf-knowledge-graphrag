# Ingestion Pipeline — Phase 1: Document Storage, Metadata & Parsing

**Date:** 2026-04-17
**Branch:** feature01-collection
**Status:** Approved, in implementation

## Context

First implementation phase of the personal self-hosted RAG system. No application code existed before this phase; infrastructure (Postgres + Memgraph via Docker Compose) and schema DDL were already in place. This phase wires up the foundational ingestion path: accept a file via CLI, store it on disk, record its metadata in Postgres, parse it to clean markdown, and persist the markdown in the database.

Supported formats: PDF, DOCX, MD, TXT. Out of scope: images, chunking, graph extraction.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Execution model | Synchronous | Simpler for this phase; no async queue needed yet |
| Parser | `markitdown` (Microsoft) | Single dep handles all four target formats → markdown |
| Project layout | `src/rag/` + `pyproject.toml` | Standard installable Python package; `rag` CLI via entrypoint |
| Markdown storage | `markdown_content TEXT` on `sources` | User requirement: markdown lives in the database |
| Metadata extraction | LLM via OpenRouter (`MODEL_METADATA_EXTRACTION`) | Extracts kind/author/source/domain from first 2000 chars of markdown; user-supplied values take precedence |
| Job tracking | Internal only | `jobs` table used but no `rag jobs` CLI commands this phase |

## Module Map

```
src/rag/
  __init__.py
  config.py       # pydantic-settings: POSTGRES_URL, STORAGE_BASE_PATH, LOG_LEVEL
  db.py           # psycopg connection context manager
  storage.py      # copy file to STORAGE_BASE_PATH/<source_id>/<filename>
  parser.py       # markitdown → markdown string (no LLM/image processing)
  ingestion.py    # orchestrates: dedup → store → DB record → parse → persist
  cli.py          # Typer: rag ingest, rag sources list/get/delete
pyproject.toml
```

## Data Model Change

`markdown_content TEXT` added to `sources` table.

- `scripts/init/postgres/02_schema.sql` — updated for fresh installs
- `scripts/migrate/001_add_markdown_content.sql` — ALTER TABLE for existing instances

## Ingestion Flow

```
rag ingest <file>
  1. Compute MD5 → check sources for duplicate → 409-style error if found
  2. Generate source_id (uuid4)
  3. Copy file to STORAGE_BASE_PATH/<source_id>/<filename>
  4. INSERT INTO sources (metadata, version=1)
  5. INSERT INTO jobs (status='processing:parsing')
  6. markitdown.convert(file) → markdown string
  7. UPDATE sources SET markdown_content = markdown
  8. UPDATE jobs SET status='completed'
  9. Print source_id to stdout
```

On parse failure: job → `failed:parsing`; source record kept for retry.
On pre-DB failure: nothing written to DB.

## CLI Interface

```bash
rag ingest <file> [--name TEXT] [--metadata key=value ...]
rag sources list
rag sources get <id>
rag sources delete <id> [--hard]
```

## Testing

One test per format (pdf, docx, md, txt) using `test_documents/`. Each test uses a pytest fixture with a `finally` block that hard-deletes source + job rows and removes the stored file from `STORAGE_BASE_PATH`. No leftover data after any test run.
