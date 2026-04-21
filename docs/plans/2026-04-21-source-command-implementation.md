# Source Command Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `rag source <id>` so the CLI prints only the stored markdown for an active source.

**Architecture:** Extend the top-level Typer app with a narrow read-only command that queries `sources.markdown_content` directly. Keep `rag sources get <id>` unchanged for rich inspection, and cover the new command with isolated CLI tests for success, missing sources, and empty markdown.

**Tech Stack:** Python, Typer, Rich console output, pytest, unittest.mock

### Task 1: Add the failing CLI tests

**Files:**
- Modify: `tests/test_cli_sources.py`
- Test: `tests/test_cli_sources.py`

**Step 1: Write the failing tests**

Add tests that:
- invoke `rag source source-1` and expect stdout to equal stored markdown
- invoke `rag source missing` and expect exit code `1`
- invoke `rag source source-1` with null markdown and expect empty stdout

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_cli_sources.py`
Expected: FAIL because the `source` command does not exist yet.

### Task 2: Add the CLI command

**Files:**
- Modify: `src/rag/cli.py`
- Test: `tests/test_cli_sources.py`

**Step 1: Write minimal implementation**

Add a top-level `@app.command("source")` function that:
- accepts `source_id`
- selects `markdown_content` from `sources` where `id = %s` and `deleted_at IS NULL`
- exits `1` with a not-found error when no row exists
- prints only markdown content when found, treating null as `""`

**Step 2: Run the focused tests**

Run: `pytest -q tests/test_cli_sources.py`
Expected: PASS

### Task 3: Verify the targeted CLI surface

**Files:**
- Modify: `src/rag/cli.py`
- Test: `tests/test_cli_sources.py`

**Step 1: Run the broader CLI checks**

Run: `pytest -q tests/test_cli_sources.py tests/test_cli_ingest.py tests/test_cli_jobs.py tests/test_cli_retrieve.py tests/test_cli_search.py`
Expected: PASS

**Step 2: Review command behavior**

Confirm:
- `rag source <id>` prints only markdown
- existing `rag sources get <id>` behavior is unchanged

**Step 3: Commit**

```bash
git add src/rag/cli.py tests/test_cli_sources.py docs/plans/2026-04-21-source-command-design.md docs/plans/2026-04-21-source-command-implementation.md
git commit -m "feat: add source markdown CLI command"
```
