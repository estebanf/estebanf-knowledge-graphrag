# Chunking Remediation Plan: Heading-Only Chunks

**Date:** 2026-04-23  
**Branch:** chunking  
**Status:** Ready for implementation

---

## Problem

A search for "ai agents red teaming" returned chunks that contain only a single markdown heading — no body text. Example:

```
## Cybersecurity AI Agents Reason & Act as Security Analysts
```

This happens because:

1. Docling converts PDFs to markdown, producing sections headed by `##` markers.
2. The LLM profiler classifies well-formatted reports as `structure_type="well-structured"` + `heading_consistency="consistent"`, which triggers the `markdown-header` strategy in `select_strategy` (`src/rag/chunking.py:35`).
3. `MarkdownHeaderTextSplitter` splits at every `##` boundary. When a heading is immediately followed by another heading (common in slide-style PDFs and Gartner reports), the first heading becomes its own standalone chunk with no body.

The fix is in `src/rag/chunking.py`. Re-processing existing data requires a one-off script because `retry_job` only operates on `failed:` jobs, and all affected sources completed successfully.

---

## Safety Precautions

- **Do not run the remediation script until the code fix is applied and confirmed passing tests.**
- **Run against one source first** to validate end-to-end before batching.
- **Do not modify `sources` rows** — original files, metadata, and `markdown_content` remain intact.
- `cleanup_from_stage` does a **hard DELETE** on `chunks` (not a soft delete). This is intentional and expected — old bad chunks are replaced by new correct ones.
- Memgraph cleanup removes `Chunk` nodes, `INCLUDES` edges, extracted `Entity` nodes for the source, and `MENTIONED_IN` edges. Entities are per-source in this system, so there is no cross-source contamination risk.
- All database operations must be wrapped in a transaction and committed only after both Postgres and Memgraph cleanup succeed.
- Write an `audit_log` entry for each job reset so the action is traceable.

---

## Part 1: Code Fix

**File:** `src/rag/chunking.py`

Add a helper `_is_heading_only` and post-process the output of `_split_markdown_header` to merge heading-only chunks into the next chunk.

### Change

Replace the existing `_split_markdown_header` function (lines 54–67) with:

```python
def _is_heading_only(text: str) -> bool:
    lines = [l for l in text.strip().split("\n") if l.strip()]
    return bool(lines) and all(re.match(r"^#{1,6}\s+", line) for line in lines)


def _split_markdown_header(text: str) -> list[str]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    docs = splitter.split_text(text)
    secondary = _recursive_splitter()
    result = []
    for doc in docs:
        if _token_count(doc.page_content) > _CHUNK_SIZE:
            result.extend(secondary.split_text(doc.page_content))
        else:
            result.append(doc.page_content)
    result = [t for t in result if t.strip()]

    # Merge heading-only chunks (no body text) into the next chunk to avoid
    # single-line heading fragments — common in slide decks where titles and
    # content appear under separate heading markers.
    merged: list[str] = []
    pending_prefix = ""
    for chunk in result:
        if _is_heading_only(chunk):
            pending_prefix = (pending_prefix + "\n\n" + chunk).strip() if pending_prefix else chunk
        else:
            merged.append((pending_prefix + "\n\n" + chunk).strip() if pending_prefix else chunk)
            pending_prefix = ""
    if pending_prefix:
        merged.append(pending_prefix)
    return merged
```

### Behavior

- If a chunk is heading-only (every non-empty line matches `^#{1,6}\s+`), it is accumulated as a `pending_prefix`.
- The next non-heading chunk is prepended with the accumulated prefix, so the heading appears naturally at the top of the body chunk.
- Multiple consecutive headings are also handled (they accumulate).
- A heading-only chunk at the very end of a document (no following body) is kept as-is rather than discarded.

### Tests to verify

Run the existing test suite and confirm all 85 tests pass:

```bash
pytest tests/ -q
```

If tests exist specifically for `_split_markdown_header`, add a test case:

```python
def test_heading_only_chunks_are_merged():
    md = "## Title One\n## Company Name\nContent about the company."
    result = _split_markdown_header(md)
    assert len(result) == 1
    assert "Title One" in result[0]
    assert "Content about the company" in result[0]
```

---

## Part 2: Identify Affected Sources

Run this query against Postgres to find sources with heading-only chunks:

```sql
SELECT
    s.id            AS source_id,
    s.file_name,
    s.file_type,
    j.id            AS job_id,
    COUNT(*)        AS heading_only_chunks,
    COUNT(*) * 100.0 / NULLIF(total.total_chunks, 0) AS pct_heading_only
FROM chunks c
JOIN sources s ON s.id = c.source_id
JOIN jobs j ON j.source_id = s.id AND j.status = 'completed'
JOIN (
    SELECT source_id, COUNT(*) AS total_chunks
    FROM chunks
    WHERE deleted_at IS NULL
    GROUP BY source_id
) total ON total.source_id = s.id
WHERE c.deleted_at IS NULL
  AND c.content ~ '^#{1,6}\s+\S'
  AND c.content NOT LIKE '%' || E'\n' || '%'
GROUP BY s.id, s.file_name, s.file_type, j.id, total.total_chunks
HAVING COUNT(*) > 0
ORDER BY heading_only_chunks DESC;
```

The `c.content NOT LIKE '%\n%'` condition selects chunks that are a single line (no newline), which will catch heading-only chunks without false-positives from multi-paragraph chunks that happen to start with a heading.

Save this result — you will iterate over `(source_id, job_id)` pairs in the script.

---

## Part 3: Remediation Script

Create `scripts/remediate_heading_chunks.py`.

### What it must do

For each `(source_id, job_id)` from the query above:

1. Open a Postgres connection and graph driver.
2. Call `cleanup_from_stage(conn, driver, job_id, source_id, "chunking")` — this:
   - Hard-deletes all `chunks` rows for the job
   - Deletes all `entities` rows for the source from Postgres
   - Removes `Chunk` nodes, `Source→Chunk INCLUDES` edges, `Entity` nodes, and `MENTIONED_IN` edges from Memgraph
3. Update the job row to re-queue it from `profiling` (not `chunking`, to ensure the LLM re-profiles and selects the correct strategy with the fixed chunker):
   ```sql
   UPDATE jobs
   SET status = 'pending',
       current_stage = NULL,
       retry_from_stage = 'profiling',
       error_detail = NULL,
       updated_at = now()
   WHERE id = $job_id;
   ```
4. Write an audit log entry:
   ```python
   _write_audit_log(conn, "job_retried", "job", job_id,
                    {"from_stage": "profiling", "reason": "heading_chunk_remediation"})
   ```
5. Commit.

### Script structure

```python
#!/usr/bin/env python3
"""
Remediation script: re-queues completed jobs that produced heading-only chunks.

Usage:
    python scripts/remediate_heading_chunks.py [--dry-run] [--source-id UUID]

Options:
    --dry-run       Print affected sources without making changes.
    --source-id     Limit to a single source (for testing).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psycopg
import psycopg.types.json

from rag.db import get_connection
from rag.graph_db import get_graph_driver
from rag.ingestion import cleanup_from_stage, _write_audit_log

IDENTIFY_SQL = """
    SELECT DISTINCT s.id AS source_id, j.id AS job_id, s.file_name,
           COUNT(*) OVER (PARTITION BY s.id) AS heading_only_chunks
    FROM chunks c
    JOIN sources s ON s.id = c.source_id
    JOIN jobs j ON j.source_id = s.id AND j.status = 'completed'
    WHERE c.deleted_at IS NULL
      AND c.content ~ '^#{1,6}\\s+\\S'
      AND c.content NOT LIKE '%' || E'\\n' || '%'
    ORDER BY heading_only_chunks DESC
"""

def remediate(dry_run: bool, only_source_id: str | None) -> None:
    with get_connection() as conn:
        rows = conn.execute(IDENTIFY_SQL).fetchall()

    if only_source_id:
        rows = [r for r in rows if str(r[0]) == only_source_id]

    if not rows:
        print("No affected sources found.")
        return

    print(f"{'DRY RUN — ' if dry_run else ''}Found {len(rows)} affected source(s):")
    for source_id, job_id, file_name, count in rows:
        print(f"  {file_name}  source={source_id}  job={job_id}  heading_only={count}")

    if dry_run:
        return

    with get_graph_driver() as driver:
        for source_id, job_id, file_name, count in rows:
            source_id, job_id = str(source_id), str(job_id)
            print(f"Processing {file_name} ({source_id})...")
            with get_connection() as conn:
                cleanup_from_stage(conn, driver, job_id, source_id, "chunking")
                conn.execute(
                    """UPDATE jobs
                       SET status = 'pending', current_stage = NULL,
                           retry_from_stage = 'profiling',
                           error_detail = NULL, updated_at = now()
                       WHERE id = %s""",
                    (job_id,),
                )
                _write_audit_log(
                    conn, "job_retried", "job", job_id,
                    {"from_stage": "profiling", "reason": "heading_chunk_remediation"},
                )
                conn.commit()
            print(f"  Done. Job {job_id} re-queued from profiling.")

    print("Remediation complete. Start the worker to process the jobs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-id", default=None)
    args = parser.parse_args()
    remediate(dry_run=args.dry_run, only_source_id=args.source_id)
```

---

## Part 4: Execution Order

1. **Apply the code fix** (Part 1). Run `pytest tests/ -q` — all tests must pass.
2. **Run the identification query** (Part 2). Note the count and the source names.
3. **Dry run** the script to confirm it identifies the same sources:
   ```bash
   python scripts/remediate_heading_chunks.py --dry-run
   ```
4. **Single-source test**: pick one source from the list and run:
   ```bash
   python scripts/remediate_heading_chunks.py --source-id <uuid>
   rag worker  # or start it if not running
   ```
   Wait for the job to complete, then verify:
   - The new chunks for that source have body text alongside headings.
   - No heading-only chunks remain in the DB for that source.
   - Memgraph has fresh `Chunk` and `Entity` nodes (old ones are gone).
5. **Full run** once the single-source test passes:
   ```bash
   python scripts/remediate_heading_chunks.py
   rag worker
   ```
6. **Post-run validation**: re-run the identification query — it should return zero rows.

---

## Key Files Referenced

| File | Role |
|---|---|
| `src/rag/chunking.py` | Code fix lives here (`_split_markdown_header`, new `_is_heading_only`) |
| `src/rag/ingestion.py` | `cleanup_from_stage` (line 247), `_write_audit_log` (line 108), `STAGE_ORDER` (line 23) |
| `src/rag/worker.py` | Picks up `pending` jobs and calls `execute_ingestion_pipeline` |
| `scripts/remediate_heading_chunks.py` | One-off remediation script (to be created) |

---

## Why `profiling`, not `chunking`

Retrying from `chunking` skips profiling and falls back to `_DEFAULT_PROFILE` (`src/rag/ingestion.py:392`). With a default profile, `select_strategy` may pick a different strategy than the original run. Starting from `profiling` re-runs the LLM profiler (the markdown is already in `sources.markdown_content`, so parsing is skipped), ensuring the strategy is properly determined before the fixed chunker runs.
