# Design: `rag jobs list --stats` and `rag jobs list --retry`

**Date:** 2026-04-25

## Overview

Two new flags on the existing `rag jobs list` command:

- `--stats` — display a summary count of jobs by status group
- `--retry` — bulk retry all failed jobs

## `--stats` flag

Adds a `--stats: bool = False` option to `jobs_list` in `cli.py`. When set, runs a single aggregate SQL query:

```sql
SELECT
  CASE
    WHEN status LIKE 'failed:%' THEN 'failed'
    WHEN status LIKE 'processing:%' THEN 'processing'
    ELSE status
  END AS status_group,
  COUNT(*) AS cnt
FROM jobs
GROUP BY status_group
ORDER BY status_group
```

Renders a two-column Rich table (Status | Count). The regular job list is skipped entirely.

## `--retry` flag

Adds a `--retry: bool = False` option to `jobs_list` in `cli.py`. When set:

1. Queries `SELECT id FROM jobs WHERE status LIKE 'failed:%'`
2. Calls the existing `retry_job(job_id)` for each result (reuses graph cleanup + audit logging)
3. Errors on individual jobs are caught and reported per-job without aborting the loop
4. Prints `"N jobs submitted for retry."` or `"No failed jobs found."` if none

## Files changed

- `src/rag/cli.py` — add `--stats` and `--retry` flags to `jobs_list`
- `tests/test_cli_jobs.py` — new tests for both flags

## Non-goals

- No confirmation prompt before `--retry`
- `--stats` and `--retry` are independent flags; combining them is not prevented but undefined
- No new functions in `ingestion.py` (reuses `retry_job`)
