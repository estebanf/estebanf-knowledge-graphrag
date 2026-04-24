# YouTube Source Purge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a safe repo script that dry-runs by default and deletes all active `kind=youtube` sources only when `--execute` is provided.

**Architecture:** Put the reusable logic in a small `src/rag/` module that queries matching sources, prints a dry-run preview, and on execution deletes each source through the existing hard-delete helpers. Keep `scripts/delete_youtube_sources.py` as a thin argparse wrapper so the behavior is easy to test and easy to run.

**Tech Stack:** Python, argparse, psycopg, pytest, unittest.mock

### Task 1: Add the failing tests

**Files:**
- Create: `tests/test_delete_youtube_sources.py`
- Test: `tests/test_delete_youtube_sources.py`

**Step 1: Write the failing tests**

Add tests that:
- verify the selector query includes active-source and `kind=youtube` filters
- verify dry run returns matches without calling deletion helpers
- verify execute deletes each matched source and stored file
- verify the script entrypoint refuses to delete without `--execute`

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_delete_youtube_sources.py`
Expected: FAIL because the module and script do not exist yet.

### Task 2: Add the reusable purge module

**Files:**
- Create: `src/rag/youtube_cleanup.py`
- Test: `tests/test_delete_youtube_sources.py`

**Step 1: Write minimal implementation**

Add:
- a query helper that returns matching source ids and names
- an execution function that dry-runs by default
- sequential deletion through `delete_source_artifacts(...)` and `delete_stored_file(...)`

**Step 2: Run focused tests**

Run: `pytest -q tests/test_delete_youtube_sources.py`
Expected: PASS

### Task 3: Add the script wrapper

**Files:**
- Create: `scripts/delete_youtube_sources.py`
- Test: `tests/test_delete_youtube_sources.py`

**Step 1: Write minimal implementation**

Add an argparse script that:
- accepts `--execute`, `--source-id`, and `--limit`
- calls the reusable purge function
- exits `0` on success

**Step 2: Run focused tests again**

Run: `pytest -q tests/test_delete_youtube_sources.py`
Expected: PASS

### Task 4: Verify the targeted surface

**Files:**
- Modify: `src/rag/youtube_cleanup.py`
- Modify: `scripts/delete_youtube_sources.py`
- Test: `tests/test_delete_youtube_sources.py`

**Step 1: Run the targeted tests**

Run: `pytest -q tests/test_delete_youtube_sources.py tests/test_cli_sources.py tests/test_ingestion.py`
Expected: PASS

**Step 2: Review behavior**

Confirm:
- dry run is the default
- `--execute` is required to delete
- deletion reuses the current hard-delete path

**Step 3: Commit**

```bash
git add docs/plans/2026-04-23-youtube-source-purge-design.md docs/plans/2026-04-23-youtube-source-purge-implementation.md src/rag/youtube_cleanup.py scripts/delete_youtube_sources.py tests/test_delete_youtube_sources.py
git commit -m "feat: add youtube source purge script"
```
