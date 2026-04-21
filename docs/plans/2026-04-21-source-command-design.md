# `rag source <id>` Design

## Goal

Add a top-level CLI command, `rag source <id>`, that prints only the stored markdown for the given source.

## Scope

- Keep the existing `rag sources get <id>` command unchanged.
- Add a new read-only command for scripting and direct content access.
- Read from `sources.markdown_content`.

## Behavior

- Command shape: `rag source <source-id>`
- Lookup only active sources: `deleted_at IS NULL`
- On success, write only the markdown content to stdout
- If the source is missing, exit non-zero with a CLI error
- If the source exists but `markdown_content` is null, print an empty string

## Approach

Use a top-level command in `src/rag/cli.py`.

This is the smallest change and matches the requested UX exactly. It also avoids changing the richer `sources` command group, which already serves a different purpose.

## Testing

Add CLI tests for:

- successful markdown retrieval
- missing source failure
- null markdown returning an empty string
