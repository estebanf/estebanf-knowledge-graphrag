# Agent Notes


# Project Overview
This is a personal, self-hosted RAG (Retrieval-Augmented Generation) system combining vector embeddings with a knowledge graph. 

Read `README.md` first. It is the canonical operator and API reference for this repository.

## Operating Rules

- Do not duplicate README content here when updating this file. Keep `AGENTS.md` focused on agent workflow and codebase navigation.
- Prefer code-backed answers over stale notes. This file can lag the implementation.
- When you do modifications that affect functionally the application, update `README.md` and `AGENTS.md` accordingly.


## Local Stack

This repo runs its local services in Docker:

- `postgres`
- `memgraph`
- `backend`
- `frontend`

Assume the databases are inside containers, not directly on the host.

### Database access

Use container-scoped commands instead:

```bash
docker compose exec -T postgres psql -U rag -d rag
docker compose exec -T memgraph mgconsole
```

For one-off SQL or migrations, keep using `docker compose exec -T postgres ...`. Only `scripts/init/postgres` is mounted into the Postgres container, so migration files must be fed on stdin from the host (`psql ... -f scripts/migrate/...` inside the container fails).

Examples:

```bash
docker compose exec -T postgres psql -U rag -d rag < scripts/migrate/004_job_improvements.sql
docker compose exec -T postgres psql -U rag -d rag -c "SELECT count(*) FROM jobs;"
printf "MATCH (n) RETURN count(n);\n" | docker compose exec -T memgraph mgconsole
```

### Service assumptions

- `backend` is the FastAPI app from the root `Dockerfile`
- `frontend` is the built React app served by nginx from `frontend/`
- frontend API traffic is proxied to `backend`
- if you need to validate service state, start with `docker compose ps` and `curl http://localhost:8000/api/health`

## Code Organization

The repo is split into a few main areas:

- `src/rag/cli.py`: Typer CLI entrypoints for ingest, search, retrieve, jobs, sources, and community commands
- `src/rag/api/`: FastAPI app, routes, and request/response schemas
- `src/rag/config.py`: env-backed runtime settings; new knobs belong here and usually also in `.env.example`
- `src/rag/ingestion.py`: ingestion job submission, pipeline orchestration, retry/cancel cleanup, and artifact deletion
- `src/rag/worker.py`: async job polling and execution loop
- `src/rag/parser.py`: document-to-markdown parsing and inline image description replacement
- `src/rag/chunking.py`, `src/rag/chunk_validation.py`, `src/rag/profiling.py`, `src/rag/embedding.py`: chunk pipeline stages
- `src/rag/graph_extraction.py`: parallel entity extraction (OpenCode Go via `rag.opencode.json_call`, `MODEL_ENTITY_EXTRACTION` default `gpt-6-luna`) and graph creation (relationship extraction was removed; the former `src/rag/graph_linking.py` stub and `graph_linking` pipeline stage were removed; see Current Behavioral Notes)
- `src/rag/insight_extraction.py`: OpenCode API call, per-chunk insight extraction, pgvector dedup against `insights`, and Memgraph `Insight` node plus `CONTAINS`/`RELATED_TO` edge management
- `src/rag/jev_profiling.py`: `PROFILING_BACKEND=jev` helpers (question table, A/B reversed-order averaging, table/code detection, profile persistence). Pilots: `scripts/pilot_jev_profiling.py`, `scripts/pilot_jev_chunk_validation.py` (read-only, SELECT only; do not run them from agent sessions, they call live OpenRouter/Jev)
- `src/rag/jev.py`: Jev (TypeSafe System One) client via OpenRouter `POST /api/v1/systemone` using `OPENROUTER_API_KEY`. `system_one()` returns the answers dict, `system_one_response()` adds usage, `system_one_many()` fans out concurrently (order preserved, per-item exception captured). Retries 429/500/502/503/529 and transport errors (3 attempts, backoff capped at 4s, `Retry-After` honoured); other 4xx are never retried; 402 raises `JevError` mentioning OpenRouter credits; missing key raises `ValueError`. HTTP goes through `_new_client`/`_sleep` (tests inject `httpx.MockTransport`). Put all questions about one state into one call: the state is billed once, each question adds ~20 tokens
- `src/rag/entity_names.py`: Jev canonical-name correction for hub entities (weekly maintenance phase 1b, opt-in); `src/rag/jev_dedup.py`: Jev confirmation/routing of entity and insight merge candidates plus the `entity_merge_review`/`insight_pair_review` helpers. Both take `conn`/`session`/merge callables as parameters and are tested offline (`tests/test_entity_names.py`, `tests/test_jev_dedup.py`, `tests/test_weekly_maintenance_jev.py`). Do not run them from agent sessions: they call live Jev and mutate the corpus. See "Entity merge survivor rule" in Current Behavioral Notes
- `src/rag/entity_audit.py`: Jev audit of chunk->entity `MENTIONS` edges, driven by `scripts/weekly_maintenance.py` phase 0 (opt-in). Pure logic taking `conn` (Postgres) and `session` (Memgraph) parameters, so it is unit-tested with `tests/audit_fakes.py` doubles. Do not run it from agent sessions: it calls live OpenRouter/Jev and mutates the corpus. See "Entity audit" in Current Behavioral Notes
- `src/rag/retrieval.py`: hybrid chunk and insight search, retrieval expansion for both chunks and insights (entity MENTIONS for chunks, RELATED_TO + LLM sub-queries for insights), reranking, and trace behavior
- `src/rag/community.py`: entity-community detection and optional summarization
- `src/rag/opencode.py`: the shared OpenCode Go client (endpoint dispatch, session header, retry, quota fallback); `src/rag/answering.py`: SSE answer generation over retrieval output; `src/rag/answer_models.py`: supported answer models (single `default`); `src/rag/themes.py`: theme-report generation (OpenCode Go, default model `deepseek-v4-pro` when `THEME_REPORT_MODEL` is empty)
- `src/rag/prompts/__init__.py`: shared prompt templates; this is the canonical prompt maintenance location
- `src/rag/storage.py`, `src/rag/sources.py`: stored-file and source-detail helpers
- `tests/`: CLI, API, ingestion, retrieval, prompt, and community coverage
- `scripts/`: local environment startup, backups, migrations, and utility entrypoints
- `frontend/`: React UI, Vite config, and nginx assets for the containerized frontend
- `docs/solutions/`: documented solutions to past problems, organized by category with YAML frontmatter (`module`, `tags`, `problem_type`); relevant when implementing or debugging in documented areas
- `CONCEPTS.md`: shared project vocabulary for domain concepts used across retrieval, ingestion, graph, and storage work

## Packaging Note

- Base package installs support API/search/retrieval/community flows **and the backend worker** (which parses markdown/text only). The backend Docker image installs the base package only — no Docling, Torch, or CUDA.
- Converting binary documents (PDF/DOCX/PPTX) to markdown happens on the **CLI** and requires the optional `prepare` extra (Docling): `pip install -e .[prepare]`.
- Ingestion responsibility split: `rag ingest`/`rag prepare` convert binaries locally (`src/rag/prepare.py`), then submit self-contained markdown. Image descriptions are backend-owned via `POST /api/prepare/describe-image` (ingest scope) so the CLI never holds the OpenRouter key. Direct PDF/DOCX/PPTX upload to `/api/ingest` is rejected; markdown/text upload remains supported. Prepared submissions preserve original-binary provenance (filename, extension, md5) — dedup keys on the original binary hash stored in `sources.md5`.

## Auth and server invariants

- Every `/api/*` route except `GET /api/health` and `POST /api/auth/login` requires a `Principal` resolved by `rag.api.auth.require_principal`. The dep accepts either an `Authorization: Bearer <key>` (validated against `data/api_keys.toml`) or a session cookie set by `POST /api/auth/login` (validated against the `user_sessions` table).
- API key file is hot-reloaded by mtime. Editing `data/api_keys.toml` does not require a backend restart.
- The frontend uses username/password → `HttpOnly` session cookie; the CLI and MCP use API keys. The same `require_principal` accepts either.
- Scopes are `read`, `ingest`, `admin`. `admin` implies all. Route declarations use `Depends(require_principal())` and explicit `requires_scope("...")` on top when needed. Today only `POST /api/prepare/describe-image` actually calls `requires_scope("ingest")`; every other gated route (and the MCP mount) accepts any valid key regardless of scopes. Do not document scope-based restrictions that the code does not enforce; add the `requires_scope` dependency when a route should be restricted.
- `data/api_keys.toml` lives outside Postgres on purpose so an operator can edit it by hand. The `api_key_audit` table exists (migration 008) but nothing in `src/rag` writes to it; do not claim usage is audited.

## Worker supervisor

- The backend owns worker lifecycle via `rag.worker_supervisor.WorkerSupervisor` (singleton, attached lazily through `get_supervisor()`).
- Invariant: every spawn writes a row to `worker_processes` **before** the subprocess starts; the reaper thread owns subsequent status transitions. Never mutate worker rows outside the supervisor.
- The supervised process is the hidden CLI command `rag _worker-run --worker-id <uuid>`. Do not document it for end users — they use `rag worker launch/stop/list/log`.
- Worker logs are at `data/worker_logs/{worker_id}.log` (append, line-buffered). `tail_log(worker_id, follow=True)` streams them via SSE.
- On startup the supervisor reconciles orphan rows: any `running` row whose PID is not alive is marked `crashed`.

## MCP server

- Mounted at `/mcp` via `rag.mcp_server.build_mcp_app()`. Uses Streamable HTTP transport from the official Python `mcp` SDK.
- The MCP ASGI sub-app is wrapped with `BearerAuthMiddleware`, which reuses the same `KeyStore` as the REST layer. Bypassing it would create a parallel auth surface; do not.
- The FastAPI `lifespan` enters the inner MCP app's lifespan so the streamable-http session manager is initialized. Tests must use `with TestClient(app) as c:` to trigger lifespan.
- Tool surface is intentionally read-only (14 tools, listed in the README MCP table; the source of truth is the `@mcp.tool()` functions in `src/rag/mcp_server.py`): search/retrieve/community, source listing and insights, community runs, theme reports, saved answers, working sets and metadata facets. There is no ingest tool; ingestion, jobs, and worker management are CLI-only. Update the README table when adding a tool.

## CLI

- `rag.cli_config.load_cli_config()` resolves precedence env > `~/.config/rag/config.toml` > none. `rag configure` writes the file (mode `0600`).
- `rag.api_client.RagClient` is the only HTTP entry point. CLI commands check `_use_api()` and dispatch through `_get_client()`; without config they fall back to direct-DB behavior (kept for local-dev convenience). New CLI commands should be API-only unless there's a strong reason.
- CLI tests use `httpx.MockTransport` via `RagClient.with_transport`. The auto-applied `_isolate_cli_config` fixture in `tests/conftest.py` ensures no developer-machine config bleeds into the test run.

## Current Behavioral Notes

- Search, retrieval, and community APIs are implemented under `src/rag/api/routes/`.
- Search now returns both chunks and insights via `hybrid_search()` → `HybridSearchResults`. The `limit` parameter applies per type (e.g. `limit=10` returns up to 10 chunks and up to 10 insights). Insight search uses the same dense+sparse+RRF pattern as chunks, querying `insights.embedding` and `insights.content`, then resolving sources and topics through `chunk_insights`.
- The frontend renders search results in two sections: "Insights" (via `InsightCard`) and "Chunks" (via `ResultCard`).
- The frontend has a Sources tab that calls `/api/sources`, `/api/sources/{source_id}`, and `/api/sources/{source_id}/insights` to browse recent sources, filter by source metadata, copy source insight JSON, copy individual insight text, and switch the selected source detail between rendered markdown and insight topic connections. The left column has a search bar that passes `q` to `GET /api/sources` for live fuzzy metadata search; it combines with exact-match `metadata=` filter chips.
- `GET /api/sources` accepts a `q` query param for fuzzy metadata search: bare value matches any metadata value, `name`, or `file_name` via `jsonb_each_text ILIKE`; `key:value` restricts to that metadata key. Implemented in `list_recent_sources` in `src/rag/sources.py`.
- `rag sources search <query>` CLI command searches sources by metadata value and displays a table. Supports `key:value` for specific-key search.
- `community retrieve` resolves source scope through a lightweight retrieval-stage pass, not full retrieval result expansion.
- `--trace` on retrieval still prints live activity first and the final JSON block last.
- Retrieval config remains env-backed through `src/rag/config.py`.
- Retrieval starts chunk and insight variant generation in parallel. Active first-stage variants are `original`, HyDE, decomposed chunk sub-queries, and gated `expanded`; `step_back` variant support (and its weight setting) has been removed.
- Dense chunk and insight retrieval over 4096-dimensional embeddings depends on the binary-quantized HNSW prefilter indexes from `scripts/migrate/009_binary_vector_prefilter_indexes.sql`, followed by full-vector reranking in SQL.
- Chunk graph expansion uses deterministic `query + entity` expansion queries by default. `RETRIEVAL_USE_LLM_ENTITY_QUERIES=true` restores the slower LLM-written entity query behavior.
- Second-hop chunk entity selection is deterministic by default. `RETRIEVAL_USE_LLM_SECOND_HOP_SELECTION=true` restores slower LLM-backed second-hop selection.
- Backend-switch pattern for Jev: `*_BACKEND` settings (currently `ENTITY_SELECTION_BACKEND`: `llm` | `jev` | `off`, `PROFILING_BACKEND`: `llm` | `jev`, `CHUNK_VALIDATION_BACKEND`: `llm` | `jev` | `off`) default in code to the legacy behavior; the operator flips `.env` after a pilot. Jev failures always fall back to a deterministic path and log a warning — never fail the pipeline. First-hop entity selection (`_select_entity_names` in `src/rag/retrieval.py`) with `jev` makes one call per seed, one Noul per candidate entity keyed `e0..eN` (names are mapped back by index, capped at 60 candidates), keeping those ≥ `ENTITY_SELECTION_JEV_THRESHOLD`. Second-hop selection and entity query generation are unaffected.
- Jev profiling (`PROFILING_BACKEND=jev`, default `llm`): the `parsing` stage makes ONE Jev call over the first ~6000 chars (`jev_profiling.extract_profile_and_metadata`) for all `DocumentProfile` choices plus metadata `kind`/`domain`, concurrently with a small LLM call (`metadata_extraction.extract_author_source`, `MODEL_METADATA_EXTRACTION`) for `author`/`source`, which Jev cannot write. Every Choice is asked twice in the same call with reversed option order (`<field>__a`/`__b`) and the probability distributions are averaged (Jev is biased toward the first option). Metadata `domain` is a separate question restricted to the 5 facet values (`legal|financial|technical|medical|general`); the profile `domain` additionally allows `policy`. `has_tables`/`has_code_blocks` are computed in code. The profile is persisted in `stage_log.parsing.output.profile`; the `profiling` stage reuses it (`profile_source` in its output: `jev`, `jev_cached` when retrying from profiling, `llm_fallback`), and makes a profile-only Jev call if it is absent. Any Jev failure falls back to the legacy `extract_metadata`/`profile_document` calls. Stage names/order are unchanged. User-submitted metadata still overrides extracted keys.
- Jev chunk validation (`CHUNK_VALIDATION_BACKEND=jev`, default `llm`; `off` disables): the `validation` stage scores EVERY chunk (`chunk_validation.jev_validate_chunks`, one Jev call per chunk, three Nouls `complete`/`coherent`/`substantive`; low quality = min probability < `CHUNK_VALIDATION_JEV_THRESHOLD`). Low-quality chunks are SKIPPED, not failed: soft-deleted (`deleted_at`, never hard-deleted) and filtered out of `chunk_rows` before embedding/graph/insight. Parent chunks (anything referenced by `parent_chunk_id`, queried from the DB so it also works on retry) are never skipped. If the low-quality fraction of scored chunks exceeds `CHUNK_VALIDATION_MAX_SKIP_FRACTION` (or all chunks would be skipped) nothing is skipped (`cap_triggered`). Fail-open: errored/malformed calls pass; all-errors skips nothing. Stage output holds `validated_chunks`, `skipped_chunks` (ids), `low_quality_chunks`, `jev_errors`, `cap_triggered`. A retry from `validation` scores the DB rows (not re-created chunks); earlier skips stay deleted. `MODEL_DOC_PROFILING`, `MODEL_CHUNK_VALIDATION` and `CHUNK_VALIDATION_SAMPLE_RATE*` are only used by the `llm` backends.
- Use `MENTIONS` as the authoritative chunk-to-entity edge for retrieval expansion.
- Same-source fallback is part of retrieval when graph expansion yields no non-seed chunk evidence.
- Entity extraction (gpt-6-luna on OpenCode `/responses`, JSON object `{"entities": [...]}`) normalises `entity_type` (strip, uppercase, spaces/hyphens to underscores) then enforces it via JSON schema validation at parse time; items with off-list types (e.g. DATE, ROLE, software) are silently dropped before insert. Normalisation is essential: gpt-6-luna emits lowercase types. Entity and insight extraction send `EXTRACTION_REASONING_EFFORT` (default `none`); query variants send `RETRIEVAL_QUERY_VARIANTS_REASONING_EFFORT` (default `none`, DeepSeek is 6-14s without it). The graceful no-key skip for both stages keys off `OPENCODE_API_KEY`; entity-name embeddings still use OpenRouter.
- `canonical_name` and all aliases are normalised before insert: `html.unescape` → whitespace collapse → leading/trailing punctuation strip (`_normalise_name` in `graph_extraction.py`).
- Exact-match dedup runs at insert time: if a `canonical_name` already exists in `entities`, the existing row is reused and no INSERT is issued. This is a SELECT-first pattern; concurrent ingestion of the same entity may create a duplicate, which is resolved by the offline `merge_semantic_duplicates.py` script.
- The `ENTITY_EXTRACTION` prompt instructs the LLM to correct misspellings and transcript noise to canonical form before extracting (e.g. "Chad GPT" → "ChatGPT").
- CLI functions `submit_ingestion_job`, `retry_job`, `cancel_job`, `get_connection`, and `get_graph_driver` are exposed as module-level names in `rag.cli` (via lazy-import wrappers) so test patches work correctly without eagerly loading heavy modules.
- All OpenCode Go calls (answers, retrieval query variants, entity extraction, insight extraction, theme reports) go through the shared client in `src/rag/opencode.py` (`json_call`, `text_call`, `stream_text`); do not hand-roll `requests`/`httpx` calls to `opencode.ai`. It uses the single `OPENCODE_API_KEY`, always sends the `x-opencode-session` header, and dispatches by model family via `endpoint_for_model`: `gpt-*` -> `/responses` (reasoning via `reasoning: {effort}`; **never send `temperature` to `gpt-*`**, the API rejects it), `minimax-*` -> `/messages`, everything else -> `/chat/completions` (reasoning via `reasoning_effort`). It retries 429/500/502/503/529 and transport errors with bounded backoff, and on quota exhaustion (429, or a 4xx mentioning quota/limit/usage/exceeded) retries once on `OPENCODE_FALLBACK_MODEL` (streaming only before the first delta). Tests inject an `httpx.MockTransport` by monkeypatching `rag.opencode._new_client` (and `rag.opencode._sleep` to skip backoff).
- The default answer model is `gpt-6-luna` (`src/rag/answer_models.py`, exactly one entry flagged `default`; the frontend picks that one). `ANSWER_REASONING_EFFORT` controls its reasoning effort.
- Insight extraction uses the OpenCode API per chunk with `MODEL_INSIGHT_EXTRACTION` (default `gpt-6-luna`). Dedup uses pgvector `<=>` cosine distance with `INSIGHT_DEDUP_COSINE_THRESHOLD`, but never scans the full `insights` table: dedup and `RELATED_TO` linking both go through the binary-quantized HNSW prefilter (`INSIGHT_PREFILTER_CANDIDATES` candidates) + full-precision rerank pattern, mirroring retrieval's existing pattern (`src/rag/retrieval.py`, `scripts/migrate/009_binary_vector_prefilter_indexes.sql`). This is what fixed intake's per-insight full-table KNN scan regression (raw `ORDER BY embedding <=>` over 100k+ rows, ~9.8s/scan).
- Insight extraction parallelizes OpenCode extraction calls across a source's chunks via `INSIGHT_EXTRACTION_CONCURRENCY` (default `12`). Storage is no longer per-insight/per-chunk: `extract_and_store_insights` runs phased, set-based batch passes per job — batch-embed all newly extracted insights, batch-dedup them against the existing corpus **and against each other within the same batch** (same-source insights dedup against each other, not only against prior corpus state), then batch-compute mutual top-K `RELATED_TO` edges and batch-write to Memgraph. No per-insight loop remains in the storage path.
- Per-chunk LLM extraction failures (in both `graph_extraction` and `insight_extraction`) are recorded rather than silently dropped: failed chunk ids land in `stage_log.<stage>.output.failed_chunks`, and the stage completes with a warning as long as the failure rate stays at or below `STAGE_FAILURE_RATE_THRESHOLD` (default `0.25`); above it, the stage fails outright instead of losing insights/entities silently. This shared failure-audit policy lives in both extraction modules, gated by the same config knob.
- The `graph_linking` pipeline stage was removed (it was a no-op compatibility stub, `def link_graph(...): pass`). `STAGE_ORDER` in `src/rag/ingestion.py` no longer includes it. Old jobs whose `retry_from_stage`/`current_stage` still names `graph_linking` are remapped to `insight_extraction` via `_LEGACY_STAGE_ALIASES`/`_canonical_stage()`, applied at every stage-name validation point (retry, cleanup, pipeline start, cancel) — so already-queued/retried jobs keep working.
- Stage-timing drift is guarded by a **pinned baseline**, not a rolling one: `stage_duration_baseline` (Postgres table, `scripts/migrate/012_stage_duration_baseline.sql`) stores a frozen per-stage `baseline_ms`, set explicitly via `rag jobs stats --set-baseline` (never auto-updated). At the end of every job the worker compares each stage's actual `duration_ms` against `baseline_ms * STAGE_DRIFT_WARN_FACTOR` (default `3.0`) and logs a warning if exceeded — log-only, does not fail the job. A rolling baseline was deliberately rejected: it would drift upward with a gradual regression and never fire, which is exactly how intake degraded silently from ~55s to ~86 minutes before this was measured.
- `rag jobs stats [--days N] [--set-baseline]` reports per-stage p50/p90/max `duration_ms` and job count from `stage_log`, over both the direct-DB and API-mode CLI paths (`client.job_stage_stats` / `client.set_stage_stats_baseline`). Jobs recorded before duration_ms tracking existed are skipped in the aggregation, not zero-filled.
- `scripts/weekly_maintenance.py` (U7) is the recommended weekly-cadence corpus maintenance sweep: entity duplicate merge, insight duplicate merge (scoped to `--since` days by default, using the same prefilter pattern as intake so it stays index-assisted even at 100k+ insights; `--full` for an occasional whole-corpus pass), an orphan/consistency sweep, and index/stats health (`VACUUM ANALYZE` + vector-index prewarm). `--dry-run` is the safe default (no writes); `--execute` applies changes, refuses to run while any job is non-terminal, and holds a Postgres advisory lock for the run's duration so it can't race in-flight intake or a concurrent `--execute` run. `_invalidate_entity_edge_cache` commits its own deletes (`get_connection()` is not autocommit and closes without committing; an earlier version silently rolled back).
- Entity audit (`src/rag/entity_audit.py`, weekly maintenance phase 0, off unless `MAINTENANCE_JEV_ENABLED=true` or `--jev`; `--no-jev` forces off, `--skip-entity-audit` skips): tables `entity_mention_audit` (PK `(chunk_id, entity_id)`: `is_entity` noul p, averaged-choice `entity_type` + `type_confidence`, `salience` = Score/3, `model`, `scored_at`), plus `entity_merge_review` and `insight_pair_review` (created by `scripts/migrate/015_entity_audit.sql`, populated by the Jev-confirmed merge phases, see the survivor/Jev note below). No foreign keys on purpose (entities/chunks are hard-deleted by merges and source deletes); queries join to `entities`/`chunks` and `prune_stale_audit_rows` cleans leftovers. **Scoring vs mutation split:** `score_mentions` (one Jev call per chunk, <=40 entities per call, type Choice asked twice in reversed order via `jev_profiling._choice_questions`/`average_choice`, resumable, commit per batch, per-call errors counted, stops after 3 consecutive fully-failed batches) writes ONLY `entity_mention_audit`, so `weekly_maintenance.main()` runs it BEFORE the active-job guard and outside the lock; default is score under `--execute`, not under `--dry-run` (`--score/--no-score` overrides; `--dry-run --score` writes only the audit table and spends Jev credits). `apply_audit` mutates the corpus and runs only inside the guard + lock: (a) delete MENTIONS edges with `is_entity < ENTITY_AUDIT_INVALID_THRESHOLD`, delete entities left with zero MENTIONS in Memgraph and Postgres (then the script calls `_invalidate_entity_edge_cache` and vacuums `entities`), (b) retype ONLY off-schema entities when the winning type's mean `type_confidence >= ENTITY_AUDIT_RETYPE_CONFIDENCE`, (c) write salience into `MENTIONS.confidence` for kept audited mentions. Audit rows of dropped mentions are deleted after the orphan entities are gone (crash-safe re-run). Dry-run (`dry_run_report`) only reads. `jev.system_one_many(..., with_usage=True)` returns full `JevResponse`s for cost accounting.
- Entity merge survivor rule + Jev maintenance phases (all Jev parts off unless `--jev`/`MAINTENANCE_JEV_ENABLED`): `pick_survivor` in `merge_semantic_duplicates.py` / `merge_duplicate_entities.py` picks the row with the most MENTIONS edges (`entity_audit.attach_mention_counts`, one batched UNWIND, filled before picking, also in dry-run), then dominant type (tie-break only, no longer a filter), has_embedding, oldest. Always on. With Jev: `run_entity_merge(use_jev=True)` confirms each fuzzy+cosine pair via `src/rag/jev_dedup.py` (3-level alignment Score + `same_name_variant`/`same_type` Nouls; `>=1.5` merge, `0.5-1.5` -> `entity_merge_review`, `<0.5` dropped, errors/missing key -> unconfirmed = never merged; pairs with any existing review row are skipped); `src/rag/entity_names.py` (phase 1b, `--name-limit`, `--skip-entity-names`) corrects hub canonical names (Choice over `[canonical_name, *aliases]` as `n0..nK`, asked in both orders, both argmaxes must agree, avg prob >= `ENTITY_NAME_MIN_CONFIDENCE`; rename in place or merge into an existing exact-name row, injected `merge_fn` = the existing `merge_memgraph`/`merge_postgres`; incompatible-type collisions are blocked; renames are grouped by winning name, so several hubs corrected to the same name plus any existing row of that name become ONE merge, and a second rename to that name with a conflicting type is blocked; apply skips overlapping/changed entities); phase 1c (`merge_duplicate_entities.run_exact_merge`, no Jev, runs with every entity phase unless `--skip-entities`) merges exact `canonical_name`+type duplicates, which the semantic merge skips; phase 2 `run_insight_merge(use_jev=True)` does ONE prefilter pass at `min(INSIGHT_NEAR_DUP_COSINE_LOW, threshold)`, splits client-side, Jev-confirms the band (dedup by normalized pair, skip `insight_pair_review`, cap `--near-dup-limit`, highest cosine first). Review rows are inserted only under `--execute` (normalized order, `ON CONFLICT DO NOTHING`, explicit commit); dry-run calls Jev (spends credits) but writes nothing. Known gaps: a renamed entity keeps its old-name embedding; insert-time dedup still matches `canonical_name` exactly.
- `RETRIEVAL_ENTITY_CONFIDENCE_THRESHOLD` (default `0.2`, also the frontend Retrieve view's initial value) / `entity_confidence_threshold` is live: `_load_seed_entities` and `_load_entities_for_chunks` (retrieval expansion only, not community detection) take `min_confidence` and filter `coalesce(r.confidence, 1.0) >= $min_confidence` on the MENTIONS edge, threaded from `expand_seed_candidate` (seed entities and second-hop candidate entities). `_load_chunk_ids_for_entity` is deliberately unfiltered. All edges are `1.0` until an audit writes salience, so behavior is unchanged until then; salience is `score/3` (levels 0, .33, .67, 1), so the `0.2` default drops only mentions scored as a passing mention (level 0) and keeps minor and up (it was `0.75`, which hid nearly every audited mention; a developer `.env` may still carry the old value, so tests read `Settings.model_fields[...].default`, not `settings`). An explicit `0.0` is honored (`is None` check, not `or`). Entity merges re-point MENTIONS with `MERGE (c)-[m:MENTIONS]->(s) ON CREATE SET m.confidence = old.confidence` (endpoints only) so differing confidences never create parallel edges.
- `scripts/remediate_insights.py` backfills insights directly, without jobs or workers. It calls the same `extract_and_store_insights` path as the pipeline (batched storage, not per-insight), and prints source/chunk counts plus extraction and storage progress. `--batch-size` caps one run to that many newest eligible sources; `--source-id` targets one source; `--force` cleans that source's existing insight links and rebuilds them.

## Data and Schema Notes

- The live corpus may contain many chunks without hierarchical parent-child structure. Do not assume parent surfacing is available for already ingested data.
- Sparse retrieval uses Postgres full-text search over `chunks.content`; the default `english` config is expected to have the matching GIN index from `scripts/migrate/006_search_performance_indexes.sql`.
- `insights` and `chunk_insights` are added in `scripts/migrate/007_insights.sql`; apply it on any database predating insight extraction.
- Hard delete order matters: remove insight join rows, orphan insights, `entities`, and `chunks` before `jobs`, then `sources`, or foreign keys will break deletes.
- Postgres schema is initialized from `scripts/init/postgres/`, but that is not the whole schema: a fresh install also needs migrations 003, 004, 007, 008, 009, 012 and 014 (README, Server installation). When you add schema, update `02_schema.sql` and add a numbered migration (015 did both) and keep both idempotent.
- Memgraph data lives in the named Docker volume `knowledge-graphrag_memgraph_data`, not `data/memgraph/` (a stale leftover directory may still exist on older checkouts; nothing reads it). `scripts/backup.sh` snapshots Memgraph and copies the newest snapshot + WAL out of the container with `docker compose cp`; the archive layout and restore steps are in README "Backup and restore". `scripts/start.sh` no longer does a Memgraph first-run schema init; the backend's `reconcile_schema()` applies it on every startup.
- Graph model: `Source -INCLUDES-> Chunk`, `Chunk -MENTIONS {confidence}-> Entity`, `Chunk -CONTAINS {topics}-> Insight`, `Insight -RELATED_TO {similarity}- Insight`. There are no entity-to-entity edges.
- Supervised workers read `WORKER_POLL_INTERVAL`/`WORKER_STUCK_JOB_MINUTES`: the hidden `_worker-run` command passes `None` unless options are given, so `run_worker` falls back to settings.
- `entities` has tightened autovacuum thresholds (`scale_factor = 0.02`, set in `scripts/init/postgres/02_schema.sql` and `scripts/migrate/010_entities_autovacuum_tuning.sql`) because `scripts/merge_semantic_duplicates.py`/`merge_duplicate_entities.py` are update/delete-heavy; both scripts also call `rag.db.vacuum_analyze_entities()` after a run that merges at least one row. Don't remove that call without re-checking `entities` doesn't re-bloat.
- Dense retrieval (`dense_retrieve`/`insight_dense_retrieve` in `src/rag/retrieval.py`) sets Postgres's `hnsw.ef_search` to `RETRIEVAL_DENSE_PREFETCH_COUNT` before the binary-quantized prefilter query. pgvector's HNSW candidate search silently caps returned rows at `hnsw.ef_search` (default `40`) regardless of the SQL `LIMIT`, so without this the prefetch count setting was a no-op past ~40 candidates.
- `src/rag/graph_db.py::reconcile_schema()` re-applies Memgraph's index/constraint statements idempotently on every backend startup (`SCHEMA_STATEMENTS`, mirroring `scripts/init/memgraph_init.cypher`). Add new statements to that list, not just the `.cypher` file, or a live instance won't pick them up.
- The paradedb Postgres image **ignores the `POSTGRES_*` env vars for tuning** (it bakes `shared_buffers=128MB`, `max_connections=100`, etc. into its own `postgresql.conf`). `docker-compose.yml` therefore passes every tuned setting as a `postgres -c` server argument (command-line args override `postgresql.conf`), sourced from those same env vars so overrides still work. When adding a Postgres GUC, add it to the `command:` list, not the `environment:` block, and verify with `SHOW <setting>` — don't trust the compose env value.
- Vector-index prewarming keeps the binary HNSW indexes resident so the first dense query after a restart is fast. Two layers: (1) `docker-compose.yml` preloads `pg_prewarm` with `pg_prewarm.autoprewarm=on` (`shared_preload_libraries` **must keep** ParadeDB's own `pg_search,pg_cron,pg_stat_statements` and only append `pg_prewarm`), so Postgres re-warms its own buffers after any restart; (2) `rag.db.prewarm_vector_indexes()` (index names in `PREWARM_INDEXES`) is called from the FastAPI lifespan (`src/rag/api/main.py::_prewarm_with_retry`) via a non-blocking background retry, tolerant of Postgres-not-yet-ready and missing extension/indexes, mirroring the non-fatal Memgraph reconciliation. Extension is created in `scripts/init/postgres/01_extensions.sql` + `scripts/migrate/011_pg_prewarm.sql`.
- `hybrid_search()` (the `rag search` path in `src/rag/retrieval.py`) runs its chunk dense, chunk sparse, and insight legs concurrently in a `ThreadPoolExecutor`, each on its own connection (psycopg connections aren't thread-safe to share). `insight_hybrid_search` stays a single leg so result shape is unchanged.
- `src/rag/cli.py` lazy-imports the heavy retrieval/graph stack (`get_connection`, `get_graph_driver`, `hybrid_search`, `retrieve` are module-level lazy wrappers, like `submit_ingestion_job`) so API-mode CLI commands don't pull in neo4j/pandas on every invocation. Keep them as module-level names — tests patch `rag.cli.<name>`. Don't convert them back to top-level `from ... import` statements.

## Valuable Carryover From `CLAUDE.md`

These points were still useful and belong here:

- `src/rag/prompts/__init__.py` is the single place to maintain shared prompt text.
- The backend/frontend are containerized and should be treated as first-class local services.
- Postgres schema initialization happens automatically from `scripts/init/postgres/`.

## Verification Shortcuts

Pick verification based on the area you changed:

- retrieval/search/community logic: `pytest -q tests/test_retrieval.py tests/test_cli_retrieve.py tests/test_cli_search.py tests/test_cli_community.py tests/test_api.py tests/test_api_community.py tests/test_config.py`
- ingestion/jobs/parser/storage: `pytest -q tests/test_cli_jobs.py tests/test_job_lifecycle.py tests/test_worker.py tests/test_ingestion_submit.py tests/test_parser.py tests/test_storage.py tests/test_cli_ingest.py`
- insight extraction: `pytest -q tests/test_insight_extraction.py tests/test_config.py tests/test_prompts.py tests/test_cli_sources.py`
- OpenCode client / answers / theme reports / entity extraction: `pytest -q tests/test_opencode_client.py tests/test_answering.py tests/test_themes.py tests/test_graph_extraction.py tests/test_insight_extraction.py tests/test_config.py`
- Jev client and switchable backends: `pytest -q tests/test_jev.py tests/test_jev_profiling.py tests/test_jev_ingestion_stages.py tests/test_chunk_validation.py tests/test_pilot_jev_scripts.py tests/test_retrieval.py tests/test_config.py`
- migrations and operator scripts: `pytest -q tests/test_migration_008.py tests/test_migration_009.py tests/test_migration_015.py tests/test_remove_api_key_script.py tests/test_remediate_insights_script.py`
- full offline suite: `pytest -q --ignore=tests/test_ingestion.py` (`tests/test_ingestion.py` is a live integration file)
- entity audit / weekly maintenance: `pytest -q tests/test_entity_audit.py tests/test_weekly_maintenance.py tests/test_weekly_maintenance_jev.py tests/test_entity_names.py tests/test_jev_dedup.py tests/test_entity_survivor.py tests/test_merge_vacuum.py tests/test_migration_015.py tests/test_jev.py tests/test_config.py`
- prompts: `pytest -q tests/test_prompts.py`
- auth (api keys + sessions + gating): `pytest -q tests/test_api_auth_apikey.py tests/test_api_auth_session.py tests/test_api_gating.py`
- new server APIs (ingest, jobs, sources delete, workers, mcp): `pytest -q tests/test_api_ingest.py tests/test_api_jobs.py tests/test_api_sources_delete.py tests/test_api_workers.py tests/test_worker_supervisor.py tests/test_mcp_server.py`
- CLI HTTP path: `pytest -q tests/test_cli_api_mode.py tests/test_cli_config.py tests/test_api_client.py`
- end-to-end against the live stack: `scripts/smoke_e2e.sh` (requires `docker compose up -d` and `API_KEY`/`ADMIN_PASSWORD` env vars; cleans up its own key and user — see README Verification)

When unsure, read the README verification section and then narrow to the impacted area.
