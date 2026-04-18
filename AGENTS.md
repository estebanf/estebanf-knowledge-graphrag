# Agent Notes

Use `docs/prd.md` for product requirements and architecture. This file is only for repo-specific continuation notes that matter in active sessions.

## Current Ingestion State

- Verified CLI commands:
  - `venv/bin/rag health`
  - `venv/bin/rag ingest <file>`
  - `venv/bin/rag worker --poll-interval 1 --stuck-minutes 30`
  - `venv/bin/rag jobs list`
  - `venv/bin/rag jobs status <job_id>`
  - `venv/bin/rag jobs retry <job_id> [--from-stage <stage>]`
  - `venv/bin/rag jobs cancel <job_id>`
  - `venv/bin/rag sources list|get|delete`

- Supported ingestion formats currently exercised in tests:
  - `PDF`, `DOCX`, `PPTX`, `MD`, `TXT`

- Storage layout is now:
  - `data/documents/<source_id>/<version>/original_<filename>`

- Parser contract:
  - `src/rag/parser.py` exports `ParseResult`
  - `parse_document()` returns `markdown` and `element_tree`
  - plain text and markdown use a synthetic element tree
  - `docling` handles `PDF`, `DOCX`, and `PPTX`

- Chunking contract:
  - `markdown-header` for well-structured + consistent headings
  - `semantic` for transcript/Q&A and fallback cases
  - `recursive` for uniform unstructured prose
  - hierarchical children carry `metadata["base_strategy"]`

## Local Environment

- Docker services were verified healthy with `docker compose ps`.
- The local database needed `scripts/migrate/004_job_improvements.sql` applied before `rag jobs status` worked.
- If a database predates the current schema, apply:
  - `scripts/migrate/001_add_markdown_content.sql`
  - `scripts/migrate/002_update_vector_dimensions.sql`
  - `scripts/migrate/004_job_improvements.sql`

## Verified Commands

- Fast test suite used during development:
  - `pytest -q tests/test_cli_jobs.py tests/test_job_lifecycle.py tests/test_worker.py tests/test_ingestion_submit.py tests/test_parser.py tests/test_storage.py tests/test_cli_ingest.py tests/test_chunking.py tests/test_observability.py tests/test_cli_health.py tests/test_profiling.py tests/test_chunk_validation.py tests/test_embedding.py`

- Full verification:
  - `pytest -q`

- Live integration:
  - `pytest -q tests/test_ingestion.py`

## Test and Smoke Notes

- `tests/test_ingestion.py` now cleans up stale rows by file MD5 before ingesting. This prevents duplicate failures after interrupted runs.
- A real CLI smoke run was verified with a temporary markdown copy:
  - submit through `rag ingest`
  - process through `rag worker`
  - inspect through `rag jobs status`
  - cleanup through `rag sources delete --hard`

## Known Gaps Worth Remembering

- `src/rag/cli.py` hard delete now removes `entities` and `chunks` before `jobs`; keep that order or the `chunks.job_id` foreign key will break deletes.
