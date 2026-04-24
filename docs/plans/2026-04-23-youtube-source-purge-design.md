# YouTube Source Purge Design

## Goal

Add a repo script that deletes every active source whose `metadata.kind` is `youtube`, together with its chunks, jobs, source-owned entities, Memgraph nodes, and stored files.

## Scope

- Default to a non-destructive dry run.
- Require an explicit `--execute` flag before deleting anything.
- Reuse the existing hard-delete path instead of duplicating SQL and Cypher logic.
- Support optional narrowing by source id and limit for safer operation.

## Behavior

- Command shape: `scripts/delete_youtube_sources.py [--execute] [--source-id <uuid>] [--limit N]`
- Match only active sources: `deleted_at IS NULL`
- Match only YouTube sources: `metadata->>'kind' = 'youtube'`
- Dry run prints the matching sources and a count.
- Execute deletes matches one at a time and prints progress plus a final summary.

## Approach

Put the query and deletion loop in a small Python module under `src/rag/` and keep the script as a thin wrapper.

For deletion, call `delete_source_artifacts(...)` and `delete_stored_file(...)` so the script stays aligned with the app's current hard-delete semantics in Postgres and Memgraph.

## Testing

Add isolated unit tests for:

- source selection query inputs
- dry-run behavior not deleting anything
- execute behavior deleting matched sources and stored files
- explicit `--execute` gating in the script entrypoint
