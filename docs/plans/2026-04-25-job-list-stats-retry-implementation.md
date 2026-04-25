# Job List Stats & Retry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `--stats` and `--retry` flags to `rag jobs list` for quick status counts and bulk retry of all failed jobs.

**Architecture:** Both flags are new options on the existing `jobs_list` command in `cli.py`. `--stats` runs a single aggregate SQL query and renders a count table. `--retry` fetches all failed job IDs and loops over the existing `retry_job()` function, catching per-job errors without aborting.

**Tech Stack:** Python, Typer (CLI), Rich (tables/output), psycopg2 (postgres), pytest + unittest.mock (tests)

---

### Task 1: Add `--stats` flag

**Files:**
- Modify: `src/rag/cli.py:363-398` (the `jobs_list` function)
- Test: `tests/test_cli_jobs.py`

**Step 1: Write the failing tests**

Add to `tests/test_cli_jobs.py`:

```python
def test_jobs_list_stats_shows_counts():
    with patch("rag.cli.get_connection") as mock_conn:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = [
            ("completed", 42),
            ("failed", 5),
            ("pending", 3),
            ("processing", 1),
        ]
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--stats"])
    assert result.exit_code == 0
    assert "pending" in result.output
    assert "failed" in result.output
    assert "processing" in result.output
    assert "42" in result.output


def test_jobs_list_stats_uses_aggregate_sql():
    with patch("rag.cli.get_connection") as mock_conn:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = []
        from rag.cli import app
        runner.invoke(app, ["jobs", "list", "--stats"])
    sql = conn.execute.call_args[0][0]
    assert "GROUP BY" in sql
    assert "COUNT" in sql


def test_jobs_list_stats_empty_db():
    with patch("rag.cli.get_connection") as mock_conn:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = []
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--stats"])
    assert result.exit_code == 0
    assert "No jobs" in result.output
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/estebanf/development/knowledge-graphrag
pytest tests/test_cli_jobs.py::test_jobs_list_stats_shows_counts tests/test_cli_jobs.py::test_jobs_list_stats_uses_aggregate_sql tests/test_cli_jobs.py::test_jobs_list_stats_empty_db -v
```

Expected: FAIL — `--stats` option does not exist yet.

**Step 3: Implement `--stats` in `cli.py`**

Update the `jobs_list` function signature to add the `stats` parameter and handle it before the existing list logic:

```python
@jobs_app.command("list")
def jobs_list(
    status: Annotated[Optional[str], typer.Option("--status", help="Filter by status")] = None,
    stats: Annotated[bool, typer.Option("--stats", help="Show job counts by status")] = False,
    retry: Annotated[bool, typer.Option("--retry", help="Retry all failed jobs")] = False,
) -> None:
    """List ingestion jobs."""
    if stats:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT
                     CASE
                       WHEN status LIKE 'failed:%' THEN 'failed'
                       WHEN status LIKE 'processing:%' THEN 'processing'
                       ELSE status
                     END AS status_group,
                     COUNT(*) AS cnt
                   FROM jobs
                   GROUP BY status_group
                   ORDER BY status_group"""
            ).fetchall()
        if not rows:
            console.print("[dim]No jobs found.[/dim]")
            return
        table = Table(title="Job Stats")
        table.add_column("Status")
        table.add_column("Count", justify="right")
        for status_group, cnt in rows:
            color = "green" if status_group == "completed" else ("red" if status_group == "failed" else "yellow")
            table.add_row(f"[{color}]{status_group}[/{color}]", str(cnt))
        console.print(table)
        return
    # ... rest of existing list + new retry logic (Task 2)
```

Note: also add `retry` parameter here (it will be used in Task 2). For now, just add it to the signature without handling it yet — or add it in Task 2.

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cli_jobs.py::test_jobs_list_stats_shows_counts tests/test_cli_jobs.py::test_jobs_list_stats_uses_aggregate_sql tests/test_cli_jobs.py::test_jobs_list_stats_empty_db -v
```

Expected: PASS

**Step 5: Run the full test suite to catch regressions**

```bash
pytest tests/test_cli_jobs.py -v
```

Expected: all existing tests still pass.

**Step 6: Commit**

```bash
git add src/rag/cli.py tests/test_cli_jobs.py
git commit -m "feat: add --stats flag to rag jobs list"
```

---

### Task 2: Add `--retry` flag

**Files:**
- Modify: `src/rag/cli.py` (same `jobs_list` function, after `--stats` block)
- Test: `tests/test_cli_jobs.py`

**Step 1: Write the failing tests**

Add to `tests/test_cli_jobs.py`:

```python
def test_jobs_list_retry_retries_all_failed():
    with patch("rag.cli.get_connection") as mock_conn, \
         patch("rag.cli.retry_job") as mock_retry:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = [
            ("job-1",), ("job-2",),
        ]
        mock_retry.return_value = {"job_id": "x", "status": "pending", "retry_from_stage": "chunking"}
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--retry"])
    assert result.exit_code == 0
    assert "2 jobs submitted for retry" in result.output
    assert mock_retry.call_count == 2


def test_jobs_list_retry_no_failed_jobs():
    with patch("rag.cli.get_connection") as mock_conn, \
         patch("rag.cli.retry_job") as mock_retry:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = []
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--retry"])
    assert result.exit_code == 0
    assert "No failed jobs" in result.output
    mock_retry.assert_not_called()


def test_jobs_list_retry_continues_on_per_job_error():
    with patch("rag.cli.get_connection") as mock_conn, \
         patch("rag.cli.retry_job") as mock_retry:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = [
            ("job-1",), ("job-2",),
        ]
        mock_retry.side_effect = [
            Exception("graph error"),
            {"job_id": "job-2", "status": "pending", "retry_from_stage": "chunking"},
        ]
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--retry"])
    assert result.exit_code == 0
    assert mock_retry.call_count == 2
    assert "1 jobs submitted for retry" in result.output
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli_jobs.py::test_jobs_list_retry_retries_all_failed tests/test_cli_jobs.py::test_jobs_list_retry_no_failed_jobs tests/test_cli_jobs.py::test_jobs_list_retry_continues_on_per_job_error -v
```

Expected: FAIL — `--retry` option does not exist yet.

**Step 3: Implement `--retry` in `cli.py`**

Add the retry block after the stats block and before the regular list logic in `jobs_list`:

```python
    if retry:
        with get_connection() as conn:
            failed_rows = conn.execute(
                "SELECT id FROM jobs WHERE status LIKE 'failed:%'"
            ).fetchall()
        if not failed_rows:
            console.print("[dim]No failed jobs found.[/dim]")
            return
        retried = 0
        for (job_id,) in failed_rows:
            try:
                retry_job(str(job_id))
                retried += 1
            except Exception as e:
                console.print(f"[yellow]Could not retry {job_id}: {e}[/yellow]")
        console.print(f"[green]{retried} jobs submitted for retry.[/green]")
        return
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cli_jobs.py::test_jobs_list_retry_retries_all_failed tests/test_cli_jobs.py::test_jobs_list_retry_no_failed_jobs tests/test_cli_jobs.py::test_jobs_list_retry_continues_on_per_job_error -v
```

Expected: PASS

**Step 5: Run the full test suite**

```bash
pytest tests/test_cli_jobs.py -v
```

Expected: all tests pass.

**Step 6: Run the full project test suite**

```bash
pytest --tb=short -q
```

Expected: all 85+ tests pass, no regressions.

**Step 7: Commit**

```bash
git add src/rag/cli.py tests/test_cli_jobs.py
git commit -m "feat: add --retry flag to rag jobs list for bulk retry of failed jobs"
```
