# Knowledge Graph RAG

Self-hosted RAG with PostgreSQL, pgvector, and Memgraph. This README documents the ingestion capabilities that are implemented and verified in this repo today.

## Implemented Capabilities

- Async ingestion through a Postgres-backed job queue
- CLI health check for Postgres and Memgraph
- Job lifecycle commands:
  - submit
  - list
  - inspect status
  - retry from the failed stage or an earlier stage
  - cancel pending or in-flight jobs
- Source management commands:
  - list
  - inspect
  - soft delete
  - hard delete
- Supported input formats:
  - `PDF`
  - `DOCX`
  - `PPTX`
  - `MD`
  - `TXT`
- Versioned file storage:
  - `data/documents/<source_id>/<version>/original_<filename>`
- Parsed document output:
  - markdown content
  - element tree metadata for downstream structure-aware processing
- Adaptive chunking:
  - `markdown-header`
  - `recursive`
  - `semantic`
  - hierarchical parent/child overlay
  - proposition expansion for legal, financial, and medical content
- Sample-based chunk validation
- Embedding generation and storage
- Structured stage logs on jobs

## Setup

```bash
cp .env.example .env
./scripts/start.sh
pip install -e .
```

If your database existed before the current ingestion code, apply the migrations for the current schema:

```bash
docker compose exec -T postgres psql -U rag -d rag -c "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS error_detail JSONB; DROP INDEX IF EXISTS sources_md5_idx; CREATE UNIQUE INDEX IF NOT EXISTS sources_md5_unique_idx ON sources(md5) WHERE deleted_at IS NULL; CREATE INDEX IF NOT EXISTS jobs_status_updated_idx ON jobs(status, updated_at) WHERE status LIKE 'processing:%';"
```

Older databases may also need the earlier migrations in `scripts/migrate/`.

## CLI Usage

Check service health:

```bash
venv/bin/rag health
```

Submit a document:

```bash
venv/bin/rag ingest test_documents/Play\ 2.md
```

Start the worker:

```bash
venv/bin/rag worker --poll-interval 1 --stuck-minutes 30
```

Inspect jobs:

```bash
venv/bin/rag jobs list
venv/bin/rag jobs list --status failed
venv/bin/rag jobs status <job_id>
venv/bin/rag jobs retry <job_id>
venv/bin/rag jobs retry <job_id> --from-stage chunking
venv/bin/rag jobs cancel <job_id>
```

Inspect and remove sources:

```bash
venv/bin/rag sources list
venv/bin/rag sources get <source_id>
venv/bin/rag sources delete <source_id>
venv/bin/rag sources delete <source_id> --hard
```

## Verification

Full test suite:

```bash
pytest -q
```

Live ingestion integration tests:

```bash
pytest -q tests/test_ingestion.py
```

## Notes

- `rag ingest` submits jobs only. The worker performs the pipeline.
- Duplicate detection is based on the file MD5 for active sources.
- Hard deletes remove dependent rows in the correct order: entities, chunks, jobs, then sources.
- The stage log is structured JSON: each stage records status, timestamps (`started_at`, `completed_at` or `failed_at`), and output summary. Failed stages also populate `jobs.error_detail` with the exception stage, message, and traceback.
