# Ingestion PRD Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring the CLI ingestion subsystem into compliance with PRD section 2.2 and the ingestion-related parts of section 3.5, then verify the behavior end to end against local Postgres and Memgraph.

**Architecture:** Keep the worker-driven async job model as the canonical ingestion path. Separate the work into three layers: job lifecycle and artifact visibility, parser/storage/chunking behavior, and observability/documentation. Drive each change from failing tests first, then implement the minimal production changes needed to pass, then refactor for clarity.

**Tech Stack:** Python, Typer, psycopg, structlog, Docker Compose, Postgres, Memgraph, pytest

### Task 1: Baseline and Environment

**Files:**
- Modify: `docs/plans/2026-04-18-ingestion-prd-remediation.md`
- Verify: `docker-compose.yml`
- Verify: `scripts/start.sh`

**Step 1: Verify local services can run**

Run: `docker compose ps`
Expected: service status is visible for Postgres and Memgraph.

**Step 2: Start services if needed**

Run: `./scripts/start.sh`
Expected: Postgres and Memgraph are reachable on their configured ports.

**Step 3: Run baseline ingestion tests**

Run: `pytest -q tests/test_ingestion_submit.py tests/test_cli_ingest.py tests/test_cli_jobs.py tests/test_worker.py tests/test_ingestion.py`
Expected: current failures capture the lifecycle and integration gaps before production code changes.

### Task 2: Job Lifecycle and Visibility

**Files:**
- Modify: `src/rag/ingestion.py`
- Modify: `src/rag/worker.py`
- Modify: `src/rag/cli.py`
- Test: `tests/test_cli_jobs.py`
- Test: `tests/test_worker.py`
- Test: `tests/test_ingestion.py`

**Step 1: Write failing tests for lifecycle semantics**

Add tests for:
- failed jobs are not retrieval-visible
- `jobs list --status failed` returns all failed jobs
- cancellation is only allowed from pending or processing states
- cancellation cleans up partial artifacts
- retry from an earlier stage deletes later-stage artifacts first

**Step 2: Run the targeted tests to verify they fail for the expected reasons**

Run: `pytest -q tests/test_cli_jobs.py tests/test_worker.py tests/test_ingestion.py -k "failed or cancel or retry or visibility"`
Expected: failures point to missing cleanup, bad filtering, or invalid status handling.

**Step 3: Implement the minimal lifecycle fixes**

Make job execution, cleanup, and status reporting conform to the PRD.

**Step 4: Re-run the targeted tests**

Run: `pytest -q tests/test_cli_jobs.py tests/test_worker.py tests/test_ingestion.py -k "failed or cancel or retry or visibility"`
Expected: all targeted lifecycle tests pass.

### Task 3: Parser, Storage, and Format Support

**Files:**
- Modify: `src/rag/parser.py`
- Modify: `src/rag/storage.py`
- Modify: `src/rag/cli.py`
- Test: `tests/test_ingestion.py`
- Test: `tests/test_cli_ingest.py`
- Create or Modify: parser-specific tests as needed under `tests/`

**Step 1: Write failing tests for storage layout and supported formats**

Add tests for:
- stored file path uses `{source_id}/{version}/original_{filename}`
- `.pptx` is accepted by the CLI and parser path
- parsing output preserves structured table and image-derived content needed by chunking

**Step 2: Run the targeted tests to verify they fail**

Run: `pytest -q tests/test_cli_ingest.py tests/test_ingestion.py -k "pptx or storage or table or image"`
Expected: failures show current parser/storage limitations.

**Step 3: Implement the parser/storage changes**

Add a storage abstraction, update path layout, enable supported formats, and return structured parse data needed downstream.

**Step 4: Re-run the targeted tests**

Run: `pytest -q tests/test_cli_ingest.py tests/test_ingestion.py -k "pptx or storage or table or image"`
Expected: all targeted parser/storage tests pass.

### Task 4: Chunking and Stage Logging

**Files:**
- Modify: `src/rag/chunking.py`
- Modify: `src/rag/chunk_validation.py`
- Modify: `src/rag/logging_config.py`
- Modify: `src/rag/ingestion.py`
- Test: `tests/test_chunking.py`
- Test: add logging/stage-log coverage under `tests/`

**Step 1: Write failing tests for chunking matrix and stage-log shape**

Add tests for:
- semantic strategy selection for transcript/fallback cases
- hierarchical overlay on top of the selected base strategy
- structured `stage_log` entries for success and failure
- required logging fields exist with defaults

**Step 2: Run the targeted tests to verify they fail**

Run: `pytest -q tests/test_chunking.py tests/test_profiling.py tests/test_worker.py -k "semantic or hierarchical or stage_log or logging"`
Expected: failures point to the missing strategies and log structure.

**Step 3: Implement the minimal chunking and observability fixes**

Update the chunking strategy matrix and log/stage-log behavior.

**Step 4: Re-run the targeted tests**

Run: `pytest -q tests/test_chunking.py tests/test_profiling.py tests/test_worker.py -k "semantic or hierarchical or stage_log or logging"`
Expected: all targeted tests pass.

### Task 5: Documentation and Final Verification

**Files:**
- Modify: `AGENTS.md`
- Create: `README.md`

**Step 1: Update `AGENTS.md`**

Keep only repo-specific continuation notes that are not already stated in `docs/prd.md`.

**Step 2: Write `README.md`**

Document implemented ingestion capabilities, operating commands, verification commands, and current limitations.

**Step 3: Run the full verification suite**

Run: `pytest -q`
Expected: entire test suite passes.

**Step 4: Run an end-to-end smoke test**

Run the CLI and worker against real files in `test_documents/`, then inspect job and source state.
Expected: representative ingestion jobs complete successfully and produce the expected persisted artifacts.
