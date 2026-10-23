# Knowledge Graph RAG

Self-hosted ingestion, search, retrieval, and community analysis over PostgreSQL/pgvector and Memgraph. The project ships with:

- a FastAPI backend (REST + Streamable HTTP MCP server)
- a background worker supervisor managed through the API
- a React frontend served by nginx in Docker, gated by username/password
- a Python CLI that wraps the API for operators on laptops

Two roles to keep straight:

- **Server** — the host running `docker compose up`. Stores data, runs the API, supervises workers, exposes `/mcp`.
- **CLI** — the laptop client. Talks to the server over HTTP, authenticates with an API key.

## Requirements

- Server: Docker Desktop or a compatible Docker engine; Python `3.11+` with the project installed in `venv` (`pip install -e .`) for host-run operator scripts such as `scripts/create_user.py`, `scripts/weekly_maintenance.py` and `scripts/smoke_e2e.sh`.
- Laptop CLI: Python `3.11+`.
- Two server-side API keys, both required (see [Model selection](#model-selection) for which task uses which):
  - `OPENROUTER_API_KEY`: embeddings, reranking, image descriptions, document metadata/profiling/chunk-validation LLMs, the retrieval graph LLM, community summaries, and **Jev** (the TypeSafe System One model, also reached through OpenRouter). The OpenRouter account needs credits for Jev as well; a `402` from Jev means it is out of credits.
  - `OPENCODE_API_KEY`: an OpenCode Go key for answer generation, retrieval query variants, entity extraction, insight extraction and theme reports. Without it, `graph_extraction` and `insight_extraction` are skipped (the stage completes with `skip_reason`), and `rag retrieve` and answers fail. `rag search` needs only OpenRouter.

## Server installation

1. Clone the repository on the server and copy the env template:

   ```bash
   cp .env.example .env
   ```

2. Fill in at minimum:

   ```bash
   POSTGRES_PASSWORD=changeme
   OPENROUTER_API_KEY=...                        # embeddings, rerank, Jev, auxiliary LLMs (needs credits)
   OPENCODE_API_KEY=...                          # answers, query variants, entity/insight extraction, themes
   RAG_FRONTEND_ORIGIN=https://rag.example.com   # public origin the browser will hit
   RAG_COOKIE_SECURE=true                        # set to true behind HTTPS
   ```

3. Bring the stack up:

   ```bash
   ./scripts/start.sh
   ```

4. Apply the migrations the init scripts do not cover. `scripts/init/postgres/` creates the base schema on first start, but it does not include `entities.source_id` (003), `jobs.error_detail` and the unique md5 index (004), the insight tables (007), the auth/worker tables (008), the insights indexes (009), the stage-duration baseline table (012) or the async theme-report status (014). Apply these from the host. The migration files are not mounted into the Postgres container, so feed them on stdin (each is idempotent):

   ```bash
   for n in 003_add_entity_source_tracking 004_job_improvements 007_insights 008_auth_and_workers \
            009_binary_vector_prefilter_indexes 012_stage_duration_baseline 014_theme_reports_async; do
     docker compose exec -T postgres psql -U rag -d rag < scripts/migrate/$n.sql
   done
   ```

   `008` creates `users`, `user_sessions`, `worker_processes` and `api_key_audit`. See [Migrations and Existing Databases](#migrations-and-existing-databases) for what each migration does and for upgrading an older database.

5. Create at least one frontend user:

   ```bash
   venv/bin/python scripts/create_user.py --username admin
   # (prompts for password)
   ```

   Run this **from the host**, inside the repo checkout with the project installed in `venv` (`pip install -e .`). The backend image only contains `src/`, so `docker compose exec backend python scripts/create_user.py` does not work. The script connects through `POSTGRES_URL` from the repo `.env`, which must point at the published Postgres port (the default `localhost:5432`).

6. Create at least one API key. Edit `data/api_keys.toml` — the backend reloads on mtime change:

   ```toml
   [[keys]]
   id = "laptop-cli"
   token = "use-an-actual-random-secret-here"
   scopes = ["read", "ingest", "admin"]
   ```

   Scopes are `read`, `ingest` and `admin`; `admin` implies all others. See [Authentication](#authentication) for which routes currently enforce them.

7. Sanity check:

   ```bash
   curl http://localhost:8000/api/health
   curl -H "Authorization: Bearer <token>" http://localhost:8000/api/auth/me  # 200, "kind": "apikey", with the key's scopes
   ```

The frontend is now reachable on the host port mapped by `FRONTEND_PORT` (default `80`). Browse to it, sign in, and start ingesting via the UI or the CLI.

## CLI installation (laptop)

1. Install the package. The base install covers search, retrieve, sources, and
   ingesting markdown/text. To ingest **binary documents** (PDF/DOCX/PPTX), add
   the `prepare` extra, which pulls in Docling for local conversion:

   ```bash
   pip install -e .            # base: markdown/text ingestion + search/retrieve
   pip install -e .[prepare]   # add binary (PDF/DOCX/PPTX) conversion via Docling
   # or, when published:
   # pipx install knowledge-graphrag
   ```

   The heavy Docling/Torch dependency lives only in the `prepare` extra on the
   CLI machine — the backend server image never installs it.

2. Point the CLI at your server:

   ```bash
   rag configure
   # Server URL: https://rag.example.com
   # API key:    use-an-actual-random-secret-here
   ```

   The values land in `~/.config/rag/config.toml` (mode `0600`). Override per-shell with `RAG_SERVER_URL` and `RAG_API_KEY`.

3. Verify:

   ```bash
   rag health   # → "Ready: server is reachable."
   ```

Without `RAG_SERVER_URL`+`RAG_API_KEY` (and no config file), the CLI falls back to talking directly to Postgres/Memgraph on `localhost`, which is convenient for local development but not how a deployed server is used.

## Authentication

- **Frontend** — username/password POSTed to `/api/auth/login`. The backend issues an opaque session token, stored in an `HttpOnly; SameSite=Lax` cookie (`Secure` when `RAG_COOKIE_SECURE=true`). Sessions are persisted in `user_sessions` and revoked by `/api/auth/logout`.
- **CLI and MCP** — API keys loaded from `data/api_keys.toml`. The file is the source of truth; edit it directly and the backend reloads on mtime change (no restart). The `api_key_audit` table is created by migration 008 but nothing writes to it yet, so key usage is not audited.
- **Scopes** — `read`, `ingest`, `admin`; `admin` implies all. Current enforcement is narrow: only `POST /api/prepare/describe-image` checks a scope (`ingest`). Every other `/api/*` route, and every MCP tool, accepts any valid API key or session regardless of its scopes, so treat all keys as full-access until per-route scope checks are added. Sessions always carry all three scopes.
- A request can present a Bearer token *or* a session cookie. Routes that accept either use the `require_principal` dependency. The MCP mount accepts Bearer tokens only.

## Environment

All runtime settings are env-backed. The CLI, backend, worker, and Docker services read configuration from `.env`.

### Core services and paths

| Variable | Default | Purpose |
| --- | --- | --- |
| `POSTGRES_PASSWORD` | `changeme` | Password used by the Postgres container. Required for local Docker startup. |
| `POSTGRES_USER` | `rag` | Postgres username created by the container. |
| `POSTGRES_DB` | `rag` | Postgres database name created by the container. |
| `POSTGRES_PORT` | `5432` | Host port mapped to the Postgres container. |
| `POSTGRES_URL` | `postgresql://rag:${POSTGRES_PASSWORD}@localhost:5432/rag` | Connection string used by the Python app outside Docker. |
| `POSTGRES_SHARED_BUFFERS` | `1GB` | Postgres shared buffers. Sized to keep the binary-quantized HNSW vector indexes (~186MB) resident so dense retrieval stays fast; raise it if you have more RAM. Applied as a `postgres -c` server arg in `docker-compose.yml` — the paradedb image ignores this env var for tuning (it bakes `shared_buffers=128MB` into its own `postgresql.conf`), so the value is passed on the command line where it actually takes effect. |
| `POSTGRES_EFFECTIVE_CACHE_SIZE` | `3GB` | Planner hint for total cache (shared buffers + OS). Also applied via `-c`. |
| `POSTGRES_WORK_MEM` | `64MB` | Postgres per-operation working memory (applied via `-c`). |
| `POSTGRES_MAINTENANCE_WORK_MEM` | `256MB` | Postgres maintenance memory for operations like indexing (applied via `-c`). |
| `POSTGRES_MAX_CONNECTIONS` | `100` | Postgres connection limit (applied via `-c`; matches the paradedb image default, which previously ignored the compose-declared `50`). |
| `MEMGRAPH_URL` | `bolt://localhost:7687` | Bolt connection string used by the Python app outside Docker. |
| `MEMGRAPH_BOLT_PORT` | `7687` | Host port mapped to Memgraph Bolt. |
| `MEMGRAPH_LAB_PORT` | `3000` | Host port mapped to Memgraph Lab. |
| `MEMGRAPH_MEMORY_LIMIT` | `2048` | Memgraph container memory limit passed to the server. |
| `STORAGE_BASE_PATH` | `./data/documents` | Base directory where original source files and copied markdown images are stored. |
| `BACKEND_PORT` | `8000` | Host port mapped to the FastAPI backend container. |
| `FRONTEND_PORT` | `80` | Host port mapped to the nginx-served frontend container. |
| `OPENROUTER_API_KEY` | empty | Required. OpenRouter key for embeddings, reranking, image descriptions, the OpenRouter-hosted LLM roles in [Model selection](#model-selection), community summaries and Jev. The account needs credits. The OpenCode Go key is the separate `OPENCODE_API_KEY`. |
| `LOG_LEVEL` | `INFO` | Application log verbosity. |

### Server / auth

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_API_KEYS_PATH` | `./data/api_keys.toml` | File-backed API keys store. Reloaded on mtime change. |
| `RAG_WORKER_LOG_DIR` | `./data/worker_logs` | Where supervised worker subprocesses write their logs. |
| `RAG_FRONTEND_ORIGIN` | `http://localhost` | Origin allowed by CORS for credentialed requests. Set to your real domain in production. |
| `RAG_COOKIE_SECURE` | `false` | Set to `true` behind HTTPS so session cookies carry the `Secure` flag. |
| `RAG_SESSION_TTL_HOURS` | `168` | Session lifetime. |
| `RAG_SESSION_COOKIE_NAME` | `rag_session` | Cookie name used for the frontend session. |
| `RAG_DISABLE_MCP` | unset | Set to `1` to skip mounting the `/mcp` server (useful in unit tests). |

### CLI (laptop)

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_SERVER_URL` | unset | Server base URL the CLI hits. Overrides `~/.config/rag/config.toml`. |
| `RAG_API_KEY` | unset | Bearer token the CLI sends. Overrides the config file. |

### Model selection

One row per LLM task. **Provider** is where the call goes: *OpenRouter* uses `OPENROUTER_API_KEY`; *OpenCode Go* uses `OPENCODE_API_KEY` through the shared client in `src/rag/opencode.py`; *Jev* is reached through OpenRouter (`POST /api/v1/systemone`) with `OPENROUTER_API_KEY`. OpenCode model names have no vendor prefix (`gpt-6-luna`, `deepseek-v4.1-flash`) and OpenRouter names do (`google/...`, `qwen/...`), so a pinned OpenCode-style value on an OpenRouter role (or the reverse) fails at request time.

| Task | Setting | Default | Provider | Notes |
| --- | --- | --- | --- | --- |
| Document metadata extraction | `MODEL_METADATA_EXTRACTION` | `google/gemma-3-4b-it` | OpenRouter | `parsing` stage. With `PROFILING_BACKEND=jev` it only extracts `author` and `source`. |
| Document profiling | `MODEL_DOC_PROFILING` | `google/gemma-3-4b-it` | OpenRouter | `profiling` stage. Used only with `PROFILING_BACKEND=llm` (or as the Jev fallback). |
| Chunk validation | `MODEL_CHUNK_VALIDATION` | `qwen/qwen2.5-7b-instruct` | OpenRouter | `validation` stage. Used only with `CHUNK_VALIDATION_BACKEND=llm`. |
| Proposition chunking | `MODEL_PROPOSITION_CHUNKING` | `qwen/qwen2.5-14b-instruct` | OpenRouter | Decomposes text into propositions when chunking requires it. |
| Embeddings | `MODEL_EMBEDDING` | `qwen/qwen3-embedding-8b` | OpenRouter | Chunks, insights, entity names and search queries. 4096 dimensions, fixed by the schema. |
| Image description | `MODEL_IMAGE_DESCRIPTION` | `google/gemini-2.0-flash-lite-001` | OpenRouter | Runs on the backend for markdown images and for `rag ingest`/`rag prepare` binary conversions. |
| Retrieval graph LLM | `MODEL_RETRIEVAL_GRAPH` | `google/gemini-2.5-flash-lite` | OpenRouter | First-hop entity selection when `ENTITY_SELECTION_BACKEND=llm`; optional LLM entity queries and second-hop selection. |
| Reranker | `MODEL_RETRIEVAL_RERANKER` | `cohere/rerank-v3.5` | OpenRouter | Reranks first-stage and final retrieval candidates. |
| Entity extraction | `MODEL_ENTITY_EXTRACTION` | `gpt-6-luna` | OpenCode Go | `graph_extraction` stage, `/responses`. Reasoning effort: `EXTRACTION_REASONING_EFFORT`. Set to e.g. `deepseek-v4.1-flash` to move off the gpt-6-luna Go quota. |
| Insight extraction | `MODEL_INSIGHT_EXTRACTION` | `gpt-6-luna` | OpenCode Go | `insight_extraction` stage. Reasoning effort: `EXTRACTION_REASONING_EFFORT`. For large backfills (`scripts/remediate_insights.py`) set `deepseek-v4.1-flash`. |
| Retrieval query variants | `MODEL_RETRIEVAL_QUERY_VARIANTS` | `deepseek-v4.1-flash` | OpenCode Go | Chunk and insight query variants, HyDE, decomposition, insight sub-queries. Reasoning effort: `RETRIEVAL_QUERY_VARIANTS_REASONING_EFFORT`. |
| Answer generation | `model` field of each request | `gpt-6-luna` | OpenCode Go | Chosen per request from the list in [Answer generation](#answer-generation). Reasoning effort for `gpt-*` models: `ANSWER_REASONING_EFFORT`. |
| Theme reports | `THEME_REPORT_MODEL` (or the `model` field of each request) | empty, which resolves to `deepseek-v4-pro` | OpenCode Go | The code falls back to OpenRouter with the same model name only when `OPENCODE_API_KEY` is unset, which will fail for an OpenCode-only model name. `ANSWER_REASONING_EFFORT` applies to `gpt-*` models. |
| Community summaries | `--summarize` / `summarize_model` | none (summaries are off unless set) | OpenRouter | Per-request model, for example `google/gemini-2.5-flash-lite`. |
| Jev (calibrated yes/no, choice and score questions) | `JEV_MODEL` | `typesafe/jev-1.13` | Jev via OpenRouter | Entity selection, profiling/metadata, chunk validation and the maintenance phases; see [Jev](#jev-typesafe-system-one-via-openrouter). |

When an OpenCode Go call hits its quota or spend cap (notably `gpt-6-luna`), the shared client retries once on `OPENCODE_FALLBACK_MODEL`; see [OpenCode Go and insight extraction](#opencode-go-and-insight-extraction).

### Jev (TypeSafe System One) via OpenRouter

Jev is a question-answering model that returns calibrated probabilities instead of text: you send one `state` plus several typed questions (yes/no, choice, score) and get every answer back in a single ~0.3s call. It is reached through OpenRouter (`POST /api/v1/systemone`) using the existing `OPENROUTER_API_KEY`; the client lives in `src/rag/jev.py`. Calls retry on 429/500/502/503/529 and transport errors (max 3 attempts, short capped backoff, `Retry-After` honoured); a 402 means the OpenRouter account is out of credits.

| Variable | Default | Purpose |
| --- | --- | --- |
| `JEV_MODEL` | `typesafe/jev-1.13` | Jev model slug on OpenRouter. |
| `JEV_TIMEOUT_SECONDS` | `20` | Per-call HTTP timeout. |
| `JEV_CONCURRENCY` | `8` | Parallel Jev calls for batch callers (`system_one_many`). |
| `ENTITY_SELECTION_BACKEND` | `llm` | How first-hop graph entities are chosen per retrieval seed: `llm` (graph LLM, legacy behavior), `jev` (one Jev call per seed scoring every candidate entity), or `off` (first `RETRIEVAL_ENTITY_SELECTION_COUNT` entities, no network call). Any other value fails at startup. |
| `ENTITY_SELECTION_JEV_THRESHOLD` | `0.5` | With `jev`, minimum yes-probability for an entity to be followed; the best `RETRIEVAL_ENTITY_SELECTION_COUNT` above it are kept, highest first. At most 60 candidate entities are sent per seed. |
| `MAINTENANCE_JEV_ENABLED` | `false` | Master switch for the Jev phases of `scripts/weekly_maintenance.py` (entity audit, Jev-confirmed entity merges, entity name correction, Jev-confirmed near-duplicate insights). Off keeps weekly maintenance exactly as before; `--jev` / `--no-jev` override it per run. See [Jev entity audit](#jev-entity-audit-phase-0-opt-in). |
| `ENTITY_AUDIT_INVALID_THRESHOLD` | `0.2` | Entity audit apply: a `MENTIONS` edge whose "is a specific named entity" probability is below this is deleted (and entities left with no mention are removed). |
| `ENTITY_AUDIT_RETYPE_CONFIDENCE` | `0.8` | Entity audit apply: an off-schema entity is retyped only when the mean type probability of the mentions voting for the winning type reaches this value. |
| `ENTITY_NAME_REVIEW_MIN_ALIASES` | `3` | Weekly maintenance entity-name correction: only entities with at least this many aliases (the hubs) are reviewed. |
| `ENTITY_NAME_MIN_CONFIDENCE` | `0.8` | Entity-name correction: a rename is applied only when Jev's winning spelling has at least this averaged probability (and both option orders agree). |

If a Jev call fails for any reason (no credits, rate limit, timeout, missing key), entity selection logs a warning and falls back to the first `RETRIEVAL_ENTITY_SELECTION_COUNT` entities; retrieval never fails because of Jev. The default stays `llm` until you flip `ENTITY_SELECTION_BACKEND` in `.env` after a pilot.

#### Jev for ingestion: profiling/metadata and chunk validation

Two more switches move ingestion quality checks onto Jev. Both default to the legacy `llm` behavior and both can be flipped in `.env` after running the matching pilot script.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROFILING_BACKEND` | `llm` | `llm` (legacy: one OpenRouter call for metadata in `parsing`, one for the profile in `profiling`) or `jev` (see below). Any other value fails at startup. |
| `CHUNK_VALIDATION_BACKEND` | `llm` | `llm` (legacy: sample 10%/25% of chunks, fail the job if more than 20% of the sample fails), `jev` (score every chunk and skip the low-quality ones, never fails the job) or `off` (no validation, every chunk passes). Any other value fails at startup. |
| `CHUNK_VALIDATION_JEV_THRESHOLD` | `0.3` | With `jev`, a chunk is low quality when the minimum of its `complete` / `coherent` / `substantive` probabilities is below this value. |
| `CHUNK_VALIDATION_MAX_SKIP_FRACTION` | `0.5` | With `jev`, safety cap: if more than this fraction of a job's scored chunks is low quality, **no** chunk is skipped (that is more likely a scoring problem than a bad document) and a warning is logged. |

`MODEL_DOC_PROFILING`, `MODEL_CHUNK_VALIDATION`, `CHUNK_VALIDATION_SAMPLE_RATE` and `CHUNK_VALIDATION_SAMPLE_RATE_HIGH_STAKES` are used only by the `llm` backends. `MODEL_METADATA_EXTRACTION` is still used with `PROFILING_BACKEND=jev`, but only to extract `author` and `source`.

**`PROFILING_BACKEND=jev`.** The `parsing` stage makes ONE Jev call on the first ~6000 characters of the markdown. It answers every `DocumentProfile` choice (`structure_type`, `heading_consistency`, `content_density`, `primary_content_type`, `avg_section_length`, profile `domain`) and the metadata `kind` and `domain` (metadata `domain` stays one of `legal | financial | technical | medical | general`, so `/api/sources/facets` and the frontend filters are unaffected). Jev favours the first option of a choice, so every choice is asked twice in the same call with the option order reversed, and the two probability distributions are averaged before taking the winner. `has_tables` and `has_code_blocks` are computed in code from the markdown (table separator rows, fenced code blocks), not by Jev. Jev cannot write free text, so one small LLM call (`MODEL_METADATA_EXTRACTION`) asks only for `author` and `source` and runs concurrently with the Jev call. User-submitted metadata still overrides extracted keys.

The profile is stored in the parsing stage's `stage_log` output (`{"markdown_chars": ..., "profile": {...}, "profile_source": "jev"}`); the `profiling` stage reads it from there (no second call). If you retry from `profiling` the stored profile is reused; if it is absent (for example the job ran with `llm` before) the stage makes a profile-only Jev call. The profiling stage output keeps `domain` and adds `profile_source` (`jev`, `jev_cached` or `llm_fallback`). Any Jev failure logs a warning and falls back to the legacy `extract_metadata` / `profile_document` LLM calls; the job never fails because of Jev. Stage names and order are unchanged.

**`CHUNK_VALIDATION_BACKEND=jev`.** The `validation` stage scores **every** chunk with one Jev call per chunk (`JEV_CONCURRENCY` in parallel) and three yes/no questions with explicit criteria: `complete` (at least one complete, self-contained thought), `coherent` (not garbled or fragmentary) and `substantive` (useful for answering questions, not boilerplate such as navigation, legal footers, tables of contents or sponsor reads). Chunks below `CHUNK_VALIDATION_JEV_THRESHOLD` are **skipped, not failed**: the job continues. Skipped chunks are soft-deleted (`chunks.deleted_at`, never hard-deleted) and removed from the embedding, graph and insight stages.

- A chunk that is the parent of another chunk (hierarchical or proposition parents) is never skipped.
- If the low-quality fraction (of successfully scored chunks) exceeds `CHUNK_VALIDATION_MAX_SKIP_FRACTION`, or would skip every chunk, nothing is skipped.
- Fail-open: a chunk whose Jev call errored passes; if every call errors nothing is skipped and a warning is logged.
- The `validation` stage output in `rag jobs status <job_id>` (stage log) records `validated_chunks`, `skipped_chunks` (the chunk ids), `low_quality_chunks`, `jev_errors` and `cap_triggered`.
- Retrying from `validation` scores the job's current (non-deleted) chunks from the database. Chunks skipped earlier stay soft-deleted and are not re-scored.

Both switches have read-only pilot scripts (SELECT only; no Postgres writes, no Memgraph calls; the only side effect is outbound OpenRouter/Jev calls):

```bash
venv/bin/python scripts/pilot_jev_profiling.py --limit 100
venv/bin/python scripts/pilot_jev_chunk_validation.py --sources 200
```

`pilot_jev_profiling.py` runs both the legacy LLM profiler/metadata extractor and the Jev path over the most recent sources and compares what drives chunking (`select_strategy(profile)`, `use_hierarchical`, `use_propositions`), kind and domain agreement, Jev cost and per-path latency. `pilot_jev_chunk_validation.py` samples chunks from the most recent sources (`--chunks-per-source`, default 20), scores them with the same questions, and reports the min-probability distribution, how many chunks and sources would be skipped at thresholds 0.2/0.3/0.4/0.5, how many sources would hit the cap, example low-scoring chunk texts and the Jev cost.

### Embedding, chunking, and graph extraction

The embedding dimension is fixed at 4096 by the Postgres schema (`vector(4096)` columns and `bit(4096)` prefilter indexes); it is not configurable, and switching `MODEL_EMBEDDING` to a model with a different output size requires a schema migration.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHUNK_VALIDATION_SAMPLE_RATE` | `0.10` | Fraction of chunks sampled for validation on standard documents. Used only by `CHUNK_VALIDATION_BACKEND=llm`. |
| `CHUNK_VALIDATION_SAMPLE_RATE_HIGH_STAKES` | `0.25` | Higher validation sampling rate for high-stakes domains. Used only by `CHUNK_VALIDATION_BACKEND=llm`. |
| `GRAPH_EXTRACTION_CONCURRENCY` | `8` | Number of concurrent entity-extraction LLM calls per ingestion job; Memgraph/Postgres writes are batched afterward, not threaded. |
| `STAGE_FAILURE_RATE_THRESHOLD` | `0.25` | Fraction of per-chunk LLM extraction failures (graph or insight extraction) tolerated before the stage fails outright instead of completing with recorded `failed_chunks`. |
| `STAGE_DRIFT_WARN_FACTOR` | `3.0` | Multiplier over the pinned `stage_duration_baseline` (see `rag jobs stats --set-baseline`) above which a stage's duration triggers a drift warning log. |

**Entity quality at insert time:** The extraction prompt instructs the LLM to correct misspellings and transcript noise before extracting. The model returns a JSON object `{"entities": [...]}`. `entity_type` is normalised (trimmed, uppercased, spaces/hyphens to underscores) and then validated against a JSON schema — items whose type is still outside the 7 defined types are dropped. `canonical_name` and aliases are normalised (HTML unescaping, whitespace collapse, punctuation trim) before insert. If a `canonical_name` already exists in the `entities` table, the existing row is reused rather than creating a duplicate. Residual semantic duplicates across ingestion runs are resolved offline using `scripts/merge_semantic_duplicates.py`.

### OpenCode Go and insight extraction

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENCODE_API_KEY` | empty | The single OpenCode Go API key, used for answer generation, retrieval query variants, entity and insight extraction and theme reports. Only this one variable is read; a key left over under any other name in `.env` is ignored. Without it, `graph_extraction` and `insight_extraction` skip (with `skip_reason`), and retrieval and answers fail. |
| `OPENCODE_FALLBACK_MODEL` | `deepseek-v4.1-flash` | Model retried once (with reasoning off) when a call hits its OpenCode Go quota/spend cap: HTTP 429 after retries, or a 4xx whose body mentions quota/limit/usage/exceeded. Empty disables the fallback. Streaming answers fall back only if the failure occurs before any text was streamed. |
| `EXTRACTION_REASONING_EFFORT` | `none` | Reasoning effort (`none`/`low`/`medium`/`high`) for entity and insight extraction. `none` keeps gpt-6-luna cheap and avoids multi-second DeepSeek reasoning latency. |
| `RETRIEVAL_QUERY_VARIANTS_REASONING_EFFORT` | `none` | Reasoning effort for query-variant generation. DeepSeek v4.1 flash takes ~2.5s with `none` vs 6-14s otherwise. |
| `ANSWER_REASONING_EFFORT` | `low` | Reasoning effort (`none`/`low`/`medium`/`high`) sent for `gpt-*` answer models such as the default `gpt-6-luna`; also applied to `gpt-*` theme-report models. Other model families ignore it. |
| `INSIGHT_DEDUP_COSINE_THRESHOLD` | `0.95` | Minimum cosine similarity to reuse an existing insight instead of creating a new one. |
| `INSIGHT_NEAR_DUP_COSINE_LOW` | `0.88` | Weekly maintenance, Jev only: insight pairs with cosine in `[this, INSIGHT_DEDUP_COSINE_THRESHOLD)` are confirmed by Jev before merging (see [Jev-confirmed merges](#jev-confirmed-merges-names-and-near-duplicates-opt-in)). |
| `INSIGHT_LINK_TOP_K` | `10` | Number of nearest insight neighbors used for mutual top-K `RELATED_TO` edge creation. |
| `INSIGHT_EXTRACTION_CONCURRENCY` | `12` | Number of concurrent OpenCode chunk extraction calls per ingestion job. Batched dedup/storage (below) means the LLM fan-out is the remaining serial bottleneck, so this can run wider than when storage was per-insight. |
| `INSIGHT_PREFILTER_CANDIDATES` | `100` | Candidate pool size pulled from the binary-quantized HNSW prefilter index before full-precision rerank, for both dedup and `RELATED_TO` linking. |

**Insight dedup and linking at scale:** Insight storage no longer loops per-insight. It runs as phased, set-based batch passes per ingestion job: batch-embed all newly extracted insights, dedup them against both the existing corpus and each other within the same batch using the same binary-quantized-HNSW-prefilter + full-precision-rerank pattern retrieval already used for search (`scripts/migrate/009_binary_vector_prefilter_indexes.sql`), then compute mutual top-K `RELATED_TO` edges over the prefiltered candidate pool and write them to Memgraph in batches. This keeps every stage's cost bounded by the size of the incoming source rather than the total corpus size — the earlier per-insight full-precision KNN scan over the whole `insights` table (the root cause of intake regressing from ~55s to tens of minutes as the corpus grew) is gone.

### Search defaults

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEARCH_DEFAULT_LIMIT` | `10` | Default `rag search --limit` value. |
| `SEARCH_MIN_SCORE` | `0.7` | Default minimum score for `rag search`. |

### Retrieval defaults and tunables

| Variable | Default | Purpose |
| --- | --- | --- |
| `RETRIEVAL_RRF_K` | `60` | Reciprocal rank fusion constant used to merge candidate lists. |
| `RETRIEVAL_RRF_SCORE_FLOOR` | `0.0` | Minimum fused score retained after RRF. |
| `RETRIEVAL_SEED_COUNT` | `10` | Number of top reranked seed chunks expanded in the graph stage. |
| `RETRIEVAL_INSIGHT_SEED_COUNT` | `5` | Maximum number of top insights expanded through `RELATED_TO` and sub-queries (capped by the request's seed count). |
| `RETRIEVAL_EXPANSION_MAX_WORKERS` | `5` | Thread-pool size for parallel graph and insight expansion work inside one retrieval request. |
| `RETRIEVAL_RESULT_COUNT` | `5` | Number of final root results returned. |
| `RETRIEVAL_MAX_DECOMPOSED_QUERIES` | `2` | Max number of decomposed variants generated from the query. |
| `RETRIEVAL_DENSE_PREFETCH_COUNT` | `250` | Binary-quantized ANN candidate count for dense chunk/insight search before full-vector reranking. Set to `0` to use exact full-corpus dense scans. Dense retrieval sets Postgres's `hnsw.ef_search` to this value before running the prefilter query — pgvector's HNSW candidate search otherwise caps returned rows at the `hnsw.ef_search` default (`40`) regardless of this setting. Raising this value widens the exact-rerank candidate pool but costs roughly linear extra query latency (measured locally: ~0.25s at 40 candidates, ~0.74s at 250, ~1.6s at 1000 for a single dense query). `250` was chosen to trade most of that latency back for still-far-wider-than-40 recall; live `rag retrieve "insurance triage"` measured ~30.1s at the previous silent ~40-candidate cap, ~40.1s at `1000`, and ~32.2s at `250` — raise this only if recall quality is confirmed to need it. |
| `RETRIEVAL_USE_LLM_ENTITY_QUERIES` | `false` | Use the graph LLM to rewrite entity expansion queries. Disabled by default because deterministic `query + entity` queries avoid slow expansion calls. |
| `RETRIEVAL_USE_LLM_SECOND_HOP_SELECTION` | `false` | Use the graph LLM to select second-hop entities. Disabled by default so chunk graph expansion stays within the time budget. |
| `RETRIEVAL_FIRST_STAGE_TOP_N` | `20` | Top candidates kept per first-stage search path before fusion. |
| `RETRIEVAL_FUSED_CANDIDATE_COUNT` | `50` | Max fused first-stage candidates retained before reranking. |
| `RETRIEVAL_ENTITY_SELECTION_COUNT` | `5` | Max entities selected from each seed for first-hop expansion. |
| `RETRIEVAL_SECOND_HOP_SELECTION_COUNT` | `5` | Max entities selected for second-hop expansion. |
| `RETRIEVAL_FIRST_HOP_CHUNK_COUNT` | `5` | Max chunk candidates pulled per first-hop entity query. |
| `RETRIEVAL_SECOND_HOP_CHUNK_COUNT` | `5` | Max chunk candidates pulled per second-hop entity query. |
| `RETRIEVAL_FIRST_HOP_SIMILARITY_THRESHOLD` | `0.5` | Minimum similarity for first-hop entity-linked chunks. |
| `RETRIEVAL_SECOND_HOP_SIMILARITY_THRESHOLD` | `0.5` | Minimum similarity for second-hop entity-linked chunks. |
| `RETRIEVAL_ENTITY_CONFIDENCE_THRESHOLD` | `0.2` | Minimum `MENTIONS` edge confidence (salience written by the [Jev entity audit](#jev-entity-audit-phase-0-opt-in); `1.0` for mentions never audited) for an entity to be used in graph expansion. It filters the seed chunk's entities and the second-hop candidate entities (`coalesce(confidence, 1.0) >= threshold`), not the chunks reached through a followed entity. **Calibration:** salience is the probability-weighted Jev score divided by 3 (levels 0 passing mention, 0.33 minor, 0.67 important, 1.0 central, with values in between), so `0.2` drops only mentions the audit scored as a passing mention (level 0, e.g. salience 0.05) and keeps everything from minor up; unaudited edges are `1.0` and always pass, so nothing is filtered before the first audit. The previous default `0.75` would have hidden almost every audited mention. Raise it (`0.33` also drops minor mentions, `0.67` keeps only important and central ones) from the salience distribution the audit dry-run prints; an explicit `0` keeps every mention. An existing `.env` that still sets `0.75` overrides this default. |
| `RETRIEVAL_MAX_GRAPH_LLM_CALLS` | `100` | Shared safety cap on graph-stage LLM calls across a retrieval request. |
| `RETRIEVAL_MAX_GRAPH_EXPANSION_MS_PER_SEED` | `4000` | Wall-clock budget applied to each seed during graph expansion. |
| `RETRIEVAL_TEXT_SEARCH_CONFIG` | `english` | Postgres full-text search configuration for sparse retrieval. |
| `RETRIEVAL_WEIGHT_ORIGINAL` | `1.0` | Weight applied to the original query path in first-stage fusion. |
| `RETRIEVAL_WEIGHT_DECOMPOSED` | `1.0` | Weight applied to decomposed query paths. |
| `RETRIEVAL_WEIGHT_EXPANDED` | `0.85` | Weight applied to expanded query variants when the deterministic gate includes them. |
| `RETRIEVAL_WEIGHT_HYDE` | `0.65` | Weight applied to HyDE-style variants. |
| `RETRIEVAL_FINAL_ROOT_WEIGHT` | `0.60` | Weight of the reranked root chunk in final aggregation. |
| `RETRIEVAL_FINAL_FIRST_HOP_WEIGHT` | `0.25` | Weight of first-hop evidence in final aggregation. |
| `RETRIEVAL_FINAL_SECOND_HOP_WEIGHT` | `0.15` | Weight of second-hop evidence in final aggregation. |
| `RETRIEVAL_MULTI_PATH_BONUS` | `0.05` | Bonus for evidence found through multiple graph paths. |
| `RETRIEVAL_SAME_SOURCE_NEIGHBOR_WINDOW` | `2` | Window used when falling back to same-source neighboring chunks. |
| `RETRIEVAL_SAME_SOURCE_NEIGHBOR_COUNT` | `3` | Max neighboring chunks returned by same-source fallback. |
| `RETRIEVAL_EXPANSION_MIN_TOKENS` | `200` | Lower bound when trimming related chunk text used in expansion. |
| `RETRIEVAL_EXPANSION_MAX_TOKENS` | `600` | Upper bound when trimming related chunk text used in expansion. |
| `RETRIEVAL_TRACE_MAX_CANDIDATES` | `5` | Max candidates shown per trace step. |
| `RETRIEVAL_TRACE_MAX_ENTITIES` | `5` | Max entities shown per trace step. |

### Search and retrieve latency

Search and retrieve latency depends on three things beyond the tuning knobs above:

- **Warm vector-index buffers.** Every dense query walks the binary-quantized HNSW indexes (`chunks_embedding_binary_hnsw_idx`, `insights_embedding_binary_hnsw_idx`, ~186MB combined). They must stay resident in `shared_buffers` (see `POSTGRES_SHARED_BUFFERS`, sized for this). To survive restarts, `docker-compose.yml` preloads `pg_prewarm` with `pg_prewarm.autoprewarm=on` (Postgres re-warms its own buffers after any restart, including a Postgres-only restart or a post-`VACUUM FULL` flush), and the backend's FastAPI lifespan also calls `pg_prewarm()` on startup via a non-blocking retry (immediate warmth on backend start / fresh install). A cold, un-prewarmed index page-in adds ~20s to the first query — this is what the prewarm layer eliminates.
- **Parallel first-stage queries.** `rag search` runs its chunk dense, chunk sparse, and insight legs concurrently (one connection each) rather than serially.
- **Fast CLI startup.** The `rag` CLI lazy-imports the heavy retrieval/graph stack, so API-mode commands start in ~0.2s instead of ~0.75s.

Live measurements on the reference corpus (~126k chunks / ~112k insights, `rag search`/`rag retrieve "insurance triage"`, median of repeated runs):

| Metric | Before this work | After |
|---|---|---|
| First `rag search` after a full restart (cold) | ~22.6s | ~1.9s |
| Warm `rag search` (CLI wall time) | ~3.3s (server) + ~0.75s CLI | ~1.9s |
| First `rag search` after a Postgres-only restart | (cold page-in) | ~1.6s |
| `rag retrieve` (CLI wall time) | ~30–80s | ~24s |

`rag search` wall time has ~0.5s of inherent remote embedding-API latency with no tail bound, so an occasional single run lands slightly above 2s; the median stays under. If you need more margin without reducing retrieval depth elsewhere, a lower `RETRIEVAL_DENSE_PREFETCH_COUNT` for search would trade some rerank-pool width for speed.

### Community and worker settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMMUNITY_SEMANTIC_THRESHOLD` | `0.85` | Similarity threshold for semantic edges between entities. |
| `COMMUNITY_SOURCE_COOC_WEIGHT` | `0.1` | Extra weight added when entities co-occur in the same source. |
| `COMMUNITY_CUTOFF` | `0.5` | Minimum chunk score kept when selecting representative chunks. |
| `COMMUNITY_MIN_COMMUNITY_SIZE` | `3` | Minimum entity count required for a community. |
| `COMMUNITY_TOP_K_CHUNKS` | `5` | Max representative chunks returned per community. |
| `COMMUNITY_SUMMARIZATION_PROMPT` | empty | Optional prompt override for community summarization. |
| `COMMUNITY_CROSS_SOURCE_TOP_K` | `10` | Max cross-source semantic neighbors fetched per entity via pgvector ANN. |
| `COMMUNITY_MAX_CROSS_SOURCE_QUERIES` | `5000` | Hard cap on per-entity ANN queries; entities are prioritized by chunk-mention count. |
| `COMMUNITY_RESOLUTION` | `1.0` | Leiden resolution parameter. 1.0 is balanced; below 1.0 merges into fewer, larger communities; above 1.0 splits into more, smaller ones. |
| `COMMUNITY_VIRTUAL_MERGE_THRESHOLD` | `0.90` | Cosine similarity threshold for run-local virtual entity merging. Set to `0` to disable. |
| `COMMUNITY_SUMMARY_MAX_WORKERS` | `4` | Max concurrent LLM calls during community summarization. |
| `COMMUNITY_EDGE_CACHE_PREFETCH` | `250` | BQ HNSW prefetch count for cross-source edge ANN queries. |
| `COMMUNITY_MAX_CONCURRENT_RUNS` | `3` | Max concurrent non-terminal community runs allowed before `create_run` rejects. |
| `COMMUNITY_RUN_STALE_SECONDS` | `600` | Seconds before a `running` run row is marked `failed` as stale. |
| `WORKER_POLL_INTERVAL` | `5` | Idle poll interval (seconds) for workers, including supervised workers started with `rag worker launch`. |
| `WORKER_STUCK_JOB_MINUTES` | `30` | Age after which a processing job is considered stuck (applies to supervised workers too). |

### Theme report settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `THEME_REPORT_MODEL` | empty | Model for theme report LLM calls (OpenCode Go). Empty resolves to the built-in default `deepseek-v4-pro`. A per-request `model` overrides it. |
| `THEME_REPORT_PROMPT` | empty | Optional prompt override for per-community theme analysis. |

## Services and Data

Start or restart the full local stack:

```bash
docker compose up -d
docker compose ps
```

The Docker stack includes:

- `postgres`: PostgreSQL plus pgvector/ParadeDB storage for sources, chunks, jobs, metadata, and embeddings
- `memgraph`: graph store. Nodes: `Source`, `Chunk`, `Entity` and `Insight`. Edges: `(Source)-[:INCLUDES]->(Chunk)`, `(Chunk)-[:MENTIONS {confidence}]->(Entity)`, `(Chunk)-[:CONTAINS {topics}]->(Insight)` and `(Insight)-[:RELATED_TO {similarity}]-(Insight)`. `RELATED_TO` links insights only; there are no entity-to-entity relationship edges (relationship extraction was removed, and entity similarity is computed on demand and cached in Postgres `entity_semantic_edges`). `MENTIONS.confidence` is `1.0` until the Jev entity audit writes a salience score.
- `backend`: FastAPI API server on `http://localhost:${BACKEND_PORT:-8000}`
- `frontend`: nginx-served React app on `http://localhost:${FRONTEND_PORT:-80}`

Persistent local data lives in:

- `data/postgres`: Postgres data directory (bind mount)
- the named Docker volume `knowledge-graphrag_memgraph_data` (Memgraph snapshots and WAL; not a directory under `data/`)
- `data/documents`: stored original files and copied markdown images
- `data/api_keys.toml`, `data/worker_logs`, `data/uploads`: API keys, worker logs and ingest uploads, bind-mounted into the backend
- `data/backups`: output of `scripts/backup.sh`

See [Backup and restore](#backup-and-restore) for how to back all of this up.

### Backup and restore

`scripts/backup.sh` writes one self-contained archive, `data/backups/knowledge_graph_<timestamp>.tar.gz` (mode `600`):

| Inside `<timestamp>/` | What it is |
| --- | --- |
| `postgres.dump` | `pg_dump --format=custom` (compressed); restore with `pg_restore` |
| `memgraph/snapshots/` | a fresh Memgraph snapshot (`CREATE SNAPSHOT`), copied out of the live `knowledge-graphrag_memgraph_data` volume with `docker compose cp` |
| `memgraph/wal/` | the Memgraph write-ahead log at backup time |
| `documents/` | `data/documents` (stored original files; cannot be regenerated) |
| `manifest.txt` | git commit, snapshot name, and row/node/edge counts to compare after a restore |
| `secrets/` | `data/api_keys.toml` and `.env`, only with `--with-secrets` |

```bash
scripts/backup.sh                  # refuses while intake jobs are pending/processing
scripts/backup.sh --force          # run anyway (Postgres and Memgraph are captured seconds apart)
scripts/backup.sh --with-secrets   # also include api keys and .env
scripts/backup.sh --out-dir /Volumes/External/rag-backups
```

The script checks free disk space first (it needs roughly 1.5x the database size plus documents while it runs), verifies the dump header and the archive listing, and removes its staging directory. Copy archives off the machine; a disk failure takes local backups with it.

**Restore** (into the same compose project; stop intake first):

```bash
tar -xzf data/backups/knowledge_graph_<ts>.tar.gz -C /tmp        # creates /tmp/<ts>/
venv/bin/rag worker stop --all
docker compose stop backend frontend

# Postgres: replaces the contents of the rag database
docker compose cp /tmp/<ts>/postgres.dump postgres:/tmp/restore.dump
docker compose exec -T postgres pg_restore -U rag -d rag --clean --if-exists --no-owner -j 4 /tmp/restore.dump
docker compose exec -T postgres rm /tmp/restore.dump

# Memgraph: swap the snapshot/WAL in the named volume, then restart
docker compose stop memgraph
docker run --rm -v knowledge-graphrag_memgraph_data:/var/lib/memgraph -v /tmp/<ts>/memgraph:/restore:ro \
  --entrypoint bash memgraph/memgraph-platform:latest -c \
  'rm -f /var/lib/memgraph/snapshots/* /var/lib/memgraph/wal/*; cp /restore/snapshots/* /var/lib/memgraph/snapshots/; cp /restore/wal/* /var/lib/memgraph/wal/ 2>/dev/null; chown -R memgraph:memgraph /var/lib/memgraph/snapshots /var/lib/memgraph/wal'
docker compose up -d memgraph

# Documents
rsync -a /tmp/<ts>/documents/ data/documents/

docker compose up -d backend frontend
venv/bin/rag worker launch 5
```

Afterwards compare the counts in `/tmp/<ts>/manifest.txt` with `SELECT count(*) FROM ...` and `MATCH (n) RETURN labels(n)[0], count(*)`.

Useful service checks:

```bash
curl http://localhost:8000/api/health
docker compose exec -T postgres psql -U rag -d rag
printf "MATCH (n) RETURN count(n);\n" | docker compose exec -T memgraph mgconsole
```

## Worker management

Workers run on the server, supervised by the backend. They are created, listed, stopped, and tailed through the API (or, equivalently, via the CLI which wraps it):

```bash
rag worker launch 4              # spawn 4 workers; prints their IDs (default 1)
rag worker list                  # active workers; add --all for stopped/crashed ones
rag worker stop <worker-id>      # SIGTERM, then SIGKILL after 5s
rag worker stop --all            # stop every active worker
rag worker log <worker-id> -f    # print the log; -f/--follow streams it over SSE
```

REST surface:

```text
POST /api/workers/launch?n=4
GET  /api/workers
POST /api/workers/{id}/stop
POST /api/workers/stop-all
GET  /api/workers/{id}/log?follow=true
```

`rag ingest` only queues jobs; nothing is processed until at least one worker is running. Workers are not restarted automatically after a backend rebuild, so relaunch them after `docker compose up -d`.

Implementation notes:

- The supervisor spawns `python -m rag.cli _worker-run --worker-id <id>` as a subprocess and tracks PID, status, exit code, and log path in the `worker_processes` table.
- Logs are written to `data/worker_logs/<worker-id>.log` (appended, line-buffered) and tailed via Server-Sent Events.
- On backend startup, any rows in `running` state with dead PIDs are reconciled to `crashed` so the operator can see the failure.
- Worker endpoints require an authenticated principal (API key or session). They are meant for `admin` keys, but scopes are not enforced on these routes yet (see [Authentication](#authentication)), so any valid key can launch or stop workers.

## MCP server

The backend mounts an MCP (Model Context Protocol) server at `/mcp` using Streamable HTTP transport. Tools wrap the read-only REST endpoints:

| Tool              | Wraps                       |
| ----------------- | --------------------------- |
| `search`          | `POST /api/search`          |
| `retrieve`        | `POST /api/retrieve`        |
| `community`       | `POST /api/community`       |
| `list_sources`    | `GET  /api/sources`         |
| `source_insights` | `GET  /api/sources/{id}/insights` |
| `list_community_runs` | `GET  /api/community/runs` |
| `get_community_run`   | `GET  /api/community/runs/{id}` |
| `list_theme_reports`  | `GET  /api/themes`         |
| `get_theme_report`    | `GET  /api/themes/{id}`    |
| `list_answers`        | `GET  /api/answers`        |
| `get_answer`          | `GET  /api/answers/{id}`   |
| `list_working_sets`   | `GET  /api/working-sets`   |
| `get_working_set`     | `GET  /api/working-sets/{id}` |
| `list_metadata_facets` | `GET  /api/sources/facets` |

Connect from a compatible client (e.g. Claude Desktop, Claude Code) with the URL `https://your-server/mcp/` and `Authorization: Bearer <token>`. Quick probe:

```bash
venv/bin/python scripts/mcp_probe.py --server https://your-server --api-key TOKEN
```

The MCP server reuses the same `data/api_keys.toml` file as the REST API; any valid key is accepted and scopes are not checked. All 14 tools are read-only: ingest, jobs, and worker management are deliberately not exposed via MCP — use the CLI for those.

## Ingestion

The CLI submits ingestion jobs. The worker processes them asynchronously through these stages:

1. `parsing`
2. `profiling`
3. `chunking`
4. `validation`
5. `embedding`
6. `graph_extraction`
7. `insight_extraction`

(The former `graph_linking` stage was a no-op compatibility stub and has been removed from the pipeline. Already-queued or retried jobs that still reference it by name are transparently remapped to `insight_extraction`.)

### What `rag ingest` does

- **markdown/text** (`.md`, `.markdown`, `.txt`): submits the file as-is; the worker parses it into markdown, extracts metadata, chunks, embeds, builds graph artifacts, and extracts insights
- **binary documents** (`.pdf`, `.docx`, `.pptx`): converts the document to self-contained markdown **on the CLI** (via the `prepare` extra), asks the backend to describe each embedded image, then submits the prepared markdown — so the backend worker only ever parses markdown/text and never needs Docling
- creates a `sources` row and a queued `jobs` row; for prepared binaries, the source records the original filename, extension, and MD5 (duplicate detection keys on the original binary hash)

Binary preparation happens on the CLI machine, which must have the `prepare`
extra installed (`pip install -e .[prepare]`). Image descriptions are made by the
backend through `POST /api/prepare/describe-image` (requires the `ingest` scope),
so the CLI never holds the `OPENROUTER_API_KEY`. Direct upload of a PDF/DOCX/PPTX
to `/api/ingest` is rejected with a message pointing to `rag ingest`/`rag prepare`.

### Supported file types

- `PDF`, `DOCX`, `PPTX` — converted to markdown on the CLI (needs the `prepare` extra)
- `MD`, `MARKDOWN`, `TXT` — submitted directly

### Preparing without ingesting

`rag prepare` converts a binary document to markdown and writes it to a file
without queuing a job. The output is ingestible with `rag ingest <out>`:

```bash
venv/bin/rag prepare deck.pptx --out deck.md
venv/bin/rag ingest deck.md
```

Documents that contain images require API configuration (`rag configure`) so the
backend can describe them.

### CLI syntax

```bash
venv/bin/rag ingest [OPTIONS] PATH...
```

Arguments:

- `PATH...`: one or more files, or one folder

Options:

- `--name TEXT`: display name stored on the source when ingesting a single file
- `--metadata key=value`: attach one or more metadata pairs at submission time

Behavior notes:

- if you pass one folder, ingest scans only that folder's immediate files and queues every supported extension it finds
- `--name` is ignored for folder ingestion and for multi-file ingestion
- submitted metadata overrides extracted metadata when keys collide
- duplicate files are rejected by MD5 against active sources

Examples:

```bash
venv/bin/rag ingest test_documents/Play\ 2.md
venv/bin/rag ingest test_documents --metadata kind=report --metadata domain=technical
venv/bin/rag ingest report.pdf --name "Quarterly Report" --metadata team=platform
```

### Inline image handling

Image handling depends on the file type:

- Markdown:
  - local relative image references such as `![diagram](images/flow.png)` are copied into source storage at submission time
  - during parsing, those local image references are replaced with an LLM-generated text description
  - remote URLs and `data:` URLs are left unchanged
  - missing local files are left unchanged
- PDF, DOCX, and PPTX:
  - the CLI converts the document to markdown (Docling) during preparation
  - embedded pictures are extracted and each `<!-- image -->` placeholder is replaced with a description the **backend** generates via `POST /api/prepare/describe-image`
  - preparation hard-fails (no job is queued) if any image cannot be described, so submitted markdown never contains unresolved placeholders
- TXT:
  - treated as plain text, with no image handling

Image descriptions use the backend's `MODEL_IMAGE_DESCRIPTION` and
`OPENROUTER_API_KEY` — the CLI never calls the image model directly.

### Worker

Workers run on the server under the backend's supervisor; see [Worker management](#worker-management):

```bash
venv/bin/rag worker launch 2
venv/bin/rag worker list
```

`rag worker launch` takes only the number of workers. Supervised workers read their poll interval and stuck-job age from `WORKER_POLL_INTERVAL` and `WORKER_STUCK_JOB_MINUTES`.

## Job Operations

### List jobs

```bash
venv/bin/rag jobs list
venv/bin/rag jobs list --status failed
venv/bin/rag jobs list --status completed
```

Options:

- `--status TEXT`: filter by status; `failed` and `processing` match their stage-qualified forms such as `failed:chunking`
- `--stats`: show counts grouped by status instead of listing jobs
- `--retry`: retry every failed job instead of listing jobs

Examples:

```bash
venv/bin/rag jobs list --stats
venv/bin/rag jobs list --retry
```

### Inspect a job

```bash
venv/bin/rag jobs status <job_id>
```

This prints the current stage, timestamps, stage log, and error detail when present.

### Retry a failed job

```bash
venv/bin/rag jobs retry <job_id>
venv/bin/rag jobs retry <job_id> --from-stage chunking
```

Options:

- `--from-stage TEXT`: restart from one of `parsing`, `profiling`, `chunking`, `validation`, `embedding`, `graph_extraction`, or `insight_extraction`. `graph_linking` (the former no-op stage, removed) is still accepted on old jobs and remapped transparently to `insight_extraction`.

Retry cleanup is stage-aware. Earlier stage retries remove downstream artifacts before the job is re-queued.

### Cancel a queued or running job

```bash
venv/bin/rag jobs cancel <job_id>
```

### Stage timing stats and drift baseline

```bash
venv/bin/rag jobs stats
venv/bin/rag jobs stats --days 30
venv/bin/rag jobs stats --set-baseline
```

`rag jobs stats` reports per-stage p50/p90/max `duration_ms` and job count over a time window (`--days`, default 14, max 365), computed from completed jobs' `stage_log.<stage>.duration_ms`. Jobs recorded before stage-duration telemetry existed have no `duration_ms` in `stage_log` and are skipped, not zero-filled.

`--set-baseline` snapshots the current window's per-stage p50 as the pinned drift-guardrail baseline, stored in the `stage_duration_baseline` table (migration `scripts/migrate/012_stage_duration_baseline.sql`). This baseline is a frozen snapshot, not a rolling average — a rolling baseline would drift upward together with a gradual regression and never fire, which is exactly the failure mode that let intake degrade silently from ~55s to ~86 minutes before this was measured. Re-run `--set-baseline` explicitly after any deliberate performance change; otherwise the baseline stays pinned until you update it.

At the end of every job, the worker compares each stage's actual `duration_ms` against `baseline_ms * STAGE_DRIFT_WARN_FACTOR` (default `3.0`) and logs a warning if exceeded. This is a log-only guardrail — it does not fail the job — intended to surface a regression long before it silently compounds across weeks.

### Per-chunk extraction failure auditing

`graph_extraction` and `insight_extraction` each run per-chunk LLM calls concurrently. A failure on one chunk no longer silently drops that chunk's entities/insights: failed chunk ids are recorded under `stage_log.<stage>.output.failed_chunks`, and the stage still completes as long as the failure rate stays at or below `STAGE_FAILURE_RATE_THRESHOLD` (default `0.25`). If the failure rate exceeds the threshold, the stage fails outright instead of completing with silent data loss. Check `rag jobs status <job_id>` (stage log panel) to see whether a completed job had any recorded chunk failures.

## Source Operations

List active sources:

```bash
venv/bin/rag sources list
```

Inspect one source:

```bash
venv/bin/rag sources get <source_id>
venv/bin/rag source <source_id>
```

- `sources get` prints metadata plus a markdown preview
- `source` prints the full stored markdown only

List insights for a source:

```bash
venv/bin/rag sources insights <source_id>
```

Search sources by metadata value (fuzzy substring match, no embeddings):

```bash
venv/bin/rag sources search "climate"
venv/bin/rag sources search "title:quarterly"
venv/bin/rag sources search "report" --limit 5
```

- a bare value matches any metadata value, the source `name`, or `file_name`
- `key:value` restricts the match to that specific metadata key
- `--limit N` caps the result count (default 20)

The frontend includes a Sources tab for browsing the latest ingested sources. The left column has a search bar that triggers a live fuzzy search as you type; when empty it shows the newest 20 sources. The search combines with metadata filter chips (which remain exact-match OR filters). It renders the selected source markdown, lists linked insights with their `chunk_insights.topics` values as connection labels, and supports copying either the selected source's insight JSON or one insight's text.

### REST API

List recent sources:

```text
GET /api/sources?limit=20&offset=0
```

The response includes `sources`, `total`, `limit`, and `offset` for frontend pagination. Add repeated `metadata=key:value` query parameters to filter by metadata attributes; multiple metadata filters are treated as OR.

Add `q` for fuzzy metadata search:

```text
GET /api/sources?q=climate
GET /api/sources?q=title:quarterly&limit=5
```

- `q=value` matches any metadata value, `name`, or `file_name` using `ILIKE '%value%'`
- `q=key:value` restricts the search to the given metadata key
- `q` is AND-ed with any `metadata=` exact filters

Fetch one source's markdown:

```text
GET /api/sources/{source_id}
```

Fetch insights linked to one source's chunks:

```text
GET /api/sources/{source_id}/insights
```

Return source IDs for the last N sources, or for sources since a date:

```bash
venv/bin/rag sources last 5
venv/bin/rag sources last 2026-01-01
```

Delete a source:

```bash
venv/bin/rag sources delete <source_id>
venv/bin/rag sources delete <source_id> --hard
```

- soft delete marks the source deleted
- hard delete removes insight links, orphan insights, `entities`, `chunks`, `jobs`, `sources`, graph nodes, and the stored file tree

## Remediation

Backfill insight extraction for the newest sources that already have chunks but no `chunk_insights` rows. `--batch-size` is a hard cap for one run; for example, `--batch-size 10` processes the newest 10 eligible sources and exits.

```bash
venv/bin/python scripts/remediate_insights.py --batch-size 10
venv/bin/python scripts/remediate_insights.py --source-id <source_id>
venv/bin/python scripts/remediate_insights.py --source-id <source_id> --force
```

The remediation script runs insight extraction directly in the script process; it does not create or rerun ingestion jobs and does not require `rag worker`. With `--source-id`, the script skips the source if it already has insight links. Add `--force` to delete that source's existing insight links, remove orphan insights, and rebuild insights from its chunks. The script prints source counts, chunk counts, cleanup steps, extraction progress, and storage progress as it runs. It calls the same `extract_and_store_insights` path the ingestion pipeline uses, so storage runs as the same phased, set-based batch passes (batch dedup, then batch mutual top-K linking), not per-insight.

## Search

Search is the lightweight retrieval surface. It runs hybrid dense+sparse search over both chunks and insights and returns two ranked result lists. The `limit` parameter applies per type (e.g. `limit=10` returns up to 10 chunks and up to 10 insights). Search does not perform graph expansion, related-chunk aggregation, or answer synthesis.

On the default `english` text-search config, sparse search uses a Postgres GIN full-text index on `chunks.content`. Insight sparse search runs against `insights.content`. If you change `RETRIEVAL_TEXT_SEARCH_CONFIG`, recreate the sparse-search index to match the new config or Postgres will fall back to slower scans.

### CLI

Syntax:

```bash
venv/bin/rag search QUERY [OPTIONS]
```

Arguments:

- `QUERY`: free-text search query

Options:

- `--limit, -n INTEGER`: maximum number of results per type (chunks and insights)
- `--min-score FLOAT`: minimum score threshold applied after ranking

Examples:

```bash
venv/bin/rag search "graph storage" 
venv/bin/rag search "customer rollout" --limit 5
venv/bin/rag search "quarterly roadmap" --limit 20 --min-score 0.8
```

Response shape:

```json
{
  "chunks": [
    {
      "score": 0.93,
      "chunk": "Chunk text...",
      "chunk_id": "chunk-uuid",
      "source_id": "source-uuid",
      "source_path": "data/documents/source-uuid/1/original_report.pdf",
      "source_metadata": {
        "kind": "report"
      }
    }
  ],
  "insights": [
    {
      "score": 0.91,
      "insight": "Insight text...",
      "insight_id": "insight-uuid",
      "topics": ["economics", "strategy"],
      "sources": [
        {
          "source_id": "source-uuid",
          "source_path": "data/documents/source-uuid/1/original_report.pdf",
          "source_metadata": {
            "kind": "report"
          }
        }
      ]
    }
  ]
}
```

### REST API

Endpoint:

```text
POST /api/search
```

Request body:

```json
{
  "query": "quarterly roadmap",
  "limit": 10,
  "min_score": 0.7
}
```

Example:

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"query":"quarterly roadmap","limit":5,"min_score":0.75}'
```

Every `/api/*` route except `GET /api/health` and `POST /api/auth/login` needs `Authorization: Bearer <key>` (or a session cookie); the examples below assume `RAG_API_KEY` holds an API key from `data/api_keys.toml`.

The response contains a `results` object with `chunks` and `insights` arrays matching the CLI shape above.

## Retrieval

Retrieval is the graph-aware query pipeline. It:

1. generates chunk and insight query variants in parallel
2. runs first-stage hybrid search for chunks and insights
3. fuses and reranks candidates
4. expands selected chunk seeds through graph evidence (entity MENTIONS), and insight seeds through RELATED_TO + LLM-generated sub-queries
5. falls back to same-source neighbors when chunk graph expansion has no non-seed evidence
6. reranks and returns root chunk results plus related supporting chunks, and a parallel list of ranked insights

Active variant fanout is `original`, HyDE, up to two decomposed chunk sub-queries, and `expanded` only when the deterministic gate treats the query as short or under-specified. Step-back variants have been removed from retrieval. Dense chunk and insight search use binary-quantized pgvector HNSW indexes as a candidate prefilter for 4096-dimensional embeddings, then rerank those candidates with the original full-precision cosine score. Chunk graph expansion uses deterministic entity queries and second-hop entity selection by default; set `RETRIEVAL_USE_LLM_ENTITY_QUERIES=true` or `RETRIEVAL_USE_LLM_SECOND_HOP_SELECTION=true` to restore those slower LLM-backed graph steps.

To profile a live retrieval run, use:

```bash
PYTHONPATH=src venv/bin/python scripts/profile_retrieve.py "insurance triage"
```

### CLI

Syntax:

```bash
venv/bin/rag retrieve QUERY [OPTIONS]
```

Arguments:

- `QUERY`: natural-language query

Options:

- `--source-id TEXT`: restrict retrieval to one or more source IDs
- `--filter key=value`: restrict retrieval by source metadata; repeatable
- `--seed-count INTEGER`: override the number of seeds expanded
- `--result-count INTEGER`: override the number of final root results
- `--rrf-k INTEGER`: override reciprocal rank fusion `k`
- `--entity-confidence-threshold FLOAT`: override the minimum `MENTIONS` salience/confidence (0-1) an entity needs to be used in graph expansion (see `RETRIEVAL_ENTITY_CONFIDENCE_THRESHOLD`)
- `--first-hop-similarity-threshold FLOAT`: override first-hop chunk similarity threshold
- `--second-hop-similarity-threshold FLOAT`: override second-hop chunk similarity threshold
- `--trace`: print live trace lines to stdout before the final JSON block

Examples:

```bash
venv/bin/rag retrieve "What topics are covered in the ingested reports?"
venv/bin/rag retrieve "What changed?" --source-id <source_uuid> --source-id <source_uuid>
venv/bin/rag retrieve "What topics are covered?" --filter kind=report --filter domain=technical
venv/bin/rag retrieve "What topics are covered?" \
  --seed-count 3 \
  --result-count 2 \
  --rrf-k 40 \
  --entity-confidence-threshold 0.8 \
  --first-hop-similarity-threshold 0.6 \
  --second-hop-similarity-threshold 0.6
venv/bin/rag retrieve "What topics are covered in the ingested reports?" --trace
```

Trace note:

- `--trace` prints live activity first and the final response JSON last
- omit `--trace` if you need machine-only stdout

Response shape:

```json
{
  "retrieval_results": [
    {
      "score": 0.0,
      "chunk": "Root chunk text...",
      "chunk_id": "chunk-uuid",
      "source_id": "source-uuid",
      "source_path": "data/documents/source-uuid/1/original_report.pdf",
      "source_metadata": {
        "kind": "report"
      },
      "related": [
        {
          "entity": "Entity Name",
          "chunks": [
            {
              "score": 0.0,
              "chunk": "Supporting chunk text...",
              "chunk_id": "related-chunk-uuid",
              "source_id": "source-uuid",
              "source_path": "data/documents/source-uuid/1/original_report.pdf",
              "source_metadata": {
                "kind": "report"
              }
            }
          ],
          "second_level_related": []
        }
      ]
    }
  ],
  "insights": [
    {
      "score": 0.88,
      "insight": "Seed insight text...",
      "insight_id": "insight-uuid",
      "related": [
        {
          "type": "first_hop",
          "insights": [
            {
              "score": 0.92,
              "insight": "Related insight text...",
              "insight_id": "related-insight-uuid"
            }
          ]
        },
        {
          "type": "second_hop",
          "sub_query": "LLM-generated sub-query text...",
          "insights": [
            {
              "score": 0.85,
              "insight": "Semantically related insight...",
              "insight_id": "found-insight-uuid"
            }
          ]
        }
      ]
    }
  ]
}
```

### REST API

Endpoint:

```text
POST /api/retrieve
```

Request body:

```json
{
  "query": "What changed?",
  "source_ids": ["source-uuid-1", "source-uuid-2"],
  "filters": {
    "kind": "report"
  },
  "seed_count": 3,
  "result_count": 2,
  "rrf_k": 40,
  "entity_confidence_threshold": 0.8,
  "first_hop_similarity_threshold": 0.6,
  "second_hop_similarity_threshold": 0.6,
  "trace": false
}
```

`entity_confidence_threshold` is the minimum `MENTIONS` salience/confidence for an entity to be used in graph expansion. It has no effect until the entity audit has written salience (every edge is `1.0` before that); after an audit, `0.8` keeps only central mentions. An explicit `0` keeps every mention.

Example:

```bash
curl -X POST http://localhost:8000/api/retrieve \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H 'content-type: application/json' \
  -d '{
    "query": "What changed?",
    "filters": {"kind": "report"},
    "seed_count": 3,
    "result_count": 2
  }'
```

The REST API returns the same JSON structure as the CLI, without trace lines.

## Answer generation

Answer generation runs the full retrieval pipeline for a question, then asks an OpenCode Go model to write an answer from the retrieved chunks and insights (prompt: `ANSWER_GENERATION`). It is exposed through REST only (and the frontend's Answer view); there is no `rag` command that generates an answer. Saved answers are managed separately, see [Saved Answers](#saved-answers).

```text
GET  /api/answer/models   → {"models": [{"id", "label", "default"}, ...]}
POST /api/answer/stream   → text/event-stream
```

`POST /api/answer/stream` takes every field of the `/api/retrieve` request body (see [Retrieval](#retrieval); `query`, `source_ids`, `filters`, `seed_count`, `result_count`, `rrf_k`, `entity_confidence_threshold`, `first_hop_similarity_threshold`, `second_hop_similarity_threshold`) plus a required `model` that must be one of the ids from `GET /api/answer/models`:

```bash
curl -N -X POST http://localhost:8000/api/answer/stream \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"query":"What changed?","model":"gpt-6-luna","result_count":5}'
```

Server-sent events, in order:

| Event | Payload | Meaning |
| --- | --- | --- |
| `answer_delta` | `{"delta": "text..."}` | One chunk of generated answer text; sent repeatedly. |
| `results` | the `/api/retrieve` response (`retrieval_results`, `insights`) | The evidence the answer was written from; sent once, last. |

Retrieval runs to completion first, so the first `answer_delta` arrives only after retrieval finishes (typically tens of seconds on a large corpus). An unsupported `model`, or a missing `OPENCODE_API_KEY`, raises inside the stream: the connection ends without an `answer_delta` or `results` event and the error is in the backend log.

Supported models (all served by OpenCode Go; the list lives in `src/rag/answer_models.py`, and exactly one entry is flagged `default`):

| Id | Label | Default |
| --- | --- | --- |
| `gpt-6-luna` | GPT 6 Luna | yes |
| `deepseek-v4-pro` | DeepSeek V4 Pro | |
| `deepseek-v4-flash` | DeepSeek V4 Flash | |
| `glm-5`, `glm-5.1` | GLM-5, GLM-5.1 | |
| `kimi-k2.5`, `kimi-k2.6` | Kimi K2.5, Kimi K2.6 | |
| `mimo-v2-pro`, `mimo-v2-omni`, `mimo-v2.5-pro`, `mimo-v2.5` | MiMo-V2-Pro, MiMo-V2-Omni, MiMo-V2.5-Pro, MiMo-V2.5 | |
| `minimax-m2.5`, `minimax-m2.7` | MiniMax M2.5, MiniMax M2.7 | |
| `qwen3.5-plus`, `qwen3.6-plus` | Qwen3.5 Plus, Qwen3.6 Plus | |

The shared client picks the endpoint by model family: `gpt-*` use `/responses` with `reasoning.effort` set from `ANSWER_REASONING_EFFORT` (default `low`; a larger output cap leaves room for reasoning tokens), `minimax-*` use `/messages`, and everything else uses `/chat/completions` at temperature `0.2`. If the chosen model hits its OpenCode Go quota or spend cap (a hard cap applies to `gpt-6-luna`), the call is retried once on `OPENCODE_FALLBACK_MODEL`, but only if nothing has been streamed yet.

## Research workspace

Working sets, theme reports, saved answers and source facets sit around [Community](#community) analysis. They are available from the CLI, the REST API and (read-only) the MCP server.

### CLI

#### `rag working-set`

```bash
rag working-set create --name "AI infra" --source-id <id> --source-id <id>
rag working-set list
rag working-set show <ws-id>
rag working-set add <ws-id> --source-id <id>
rag working-set remove <ws-id> --source-id <id>
rag working-set rename <ws-id> --name "New name"
rag working-set delete <ws-id>
```

#### `rag themes`

```bash
rag themes generate --run-id <run_id> [--model TEXT]
rag themes list
rag themes show <report_id>
rag themes regenerate <report_id> [--model TEXT]
```

#### `rag answers`

```bash
rag answers list
rag answers show <answer_id>
```

#### `rag sources facets`

```bash
rag sources facets
# Returns distinct values with counts for kind, author, source, domain metadata keys
# Each key includes an explicit (none) bucket for sources missing the key.
```

### REST API

#### Working Sets

```text
GET    /api/working-sets           → list working sets
POST   /api/working-sets           → create {name, source_ids}
GET    /api/working-sets/{id}      → get one
PATCH  /api/working-sets/{id}      → update {name?, source_ids?}
DELETE /api/working-sets/{id}      → delete
```

Working sets are named, persisted collections of source IDs usable as scope for community, search, retrieve, answer, and theme report generation. Scope resolution snapshots member IDs at dispatch; later edits to the set do not change in-flight or completed runs.

#### Theme Reports

```text
GET    /api/themes                 → list theme reports
POST   /api/themes                 → generate {run_id, model?} → {id}
GET    /api/themes/{id}            → get report with analyses, buckets, narrative
POST   /api/themes/{id}/regenerate → retry failed per-community calls {model?} → {id}
```

A Theme Report is an LLM analysis generated from a persisted community run. Parallel per-community calls produce label, community_type, confidence (1-5), summary, key entities/sources; a synthesis pass produces higher-order thematic buckets, cross-community narrative, and cleanup recommendations. If some community calls fail, the report saves with status `partial` and regeneration retries only the failed parts.

#### Saved Answers

```text
GET    /api/answers                → list saved answers
POST   /api/answers                → save {question, answer, model, params, evidence}
GET    /api/answers/{id}           → get one
DELETE /api/answers/{id}           → delete
```

Answers are saved with denormalized evidence snapshots (source_id, source_name, chunk text at save time) so they survive later source deletion. The evidence payload is validated against a bounded Pydantic model.

#### Source Facets

```text
GET /api/sources/facets            → {facets: {kind: [{value, count}], ...}}
```

Returns distinct values with counts for `kind`, `author`, `source`, `domain` metadata keys over non-deleted sources. Each key includes an explicit `(none)` bucket for sources missing the key.

## Community

Community analysis groups connected entities into communities and returns representative chunks for each group. Scope can come from:

- explicit source IDs
- sources matched by search
- sources matched by retrieval
- a named working set (resolved to its member source IDs at dispatch)

`rag community retrieve` resolves source scope with a lightweight retrieval-stage pass before running community graphing. It does not run the full graph-expanded retrieval result assembly just to collect source IDs, which keeps community detection substantially faster on retrieval-scoped runs.

Optional summarization adds an LLM-written summary per community (an OpenRouter model you name per request; see [Model selection](#model-selection)).

Cross-source community detection uses the binary-quantized HNSW index on `entities.embedding` (migration 013) as a fast prefilter, combined with a persistent `entity_semantic_edges` cache for read-through reuse across runs. Virtual entity resolution merges near-duplicate entities within a run's subgraph without mutating stored data. Leiden uses `RBConfigurationVertexPartition` with a configurable `resolution` parameter and a fixed random seed (`42`) for reproducible partitions.

Every community execution is persisted as a run with a stable `run_id`, and progress streams over SSE so the frontend can show live stage counts. A synchronous `POST /api/community` is kept for CLI/MCP compatibility and records a run the same way. Async `POST /api/community/runs` + `GET .../events` are the frontend flow.

### CLI

The community surface has three scope subcommands (`ids`, `search`, `retrieve`) plus run history. Besides the per-command options below, all three accept:

- `--resolution FLOAT`: Leiden resolution; 1.0 balanced, below 1.0 merges into fewer communities, above 1.0 splits into more
- `--source-cooc-weight FLOAT`: extra weight added when entities co-occur in the same source
- `--working-set UUID`: resolve scope from a working set instead of explicit IDs / criteria

Run history:

```bash
rag community runs list
rag community runs show <run_id>
```

#### `rag community ids`

Use explicit source IDs as the scope.

```bash
venv/bin/rag community ids <source_id> <source_id> [OPTIONS]
```

Options:

- `--semantic-threshold FLOAT`: override entity-to-entity semantic edge threshold
- `--cutoff FLOAT`: minimum chunk score kept in each community
- `--min-community-size INTEGER`: minimum entities per community
- `--top-k INTEGER`: max representative chunks per community
- `--summarize TEXT`: model name used to summarize each community
- `--cross-source-top-k INTEGER`: max cross-source ANN neighbors fetched per entity
- `--max-cross-source-queries INTEGER`: hard cap on per-entity ANN queries

Example:

```bash
venv/bin/rag community ids <source_id> <source_id> --top-k 3 --summarize google/gemini-2.5-flash-lite
```

#### `rag community search`

Use search criteria to select sources first.

```bash
venv/bin/rag community search CRITERION... [OPTIONS]
```

Arguments:

- `CRITERION...`: one or more search strings

Options:

- `--filter key=value`: metadata filter; repeatable
- `--limit INTEGER`: max search results per criterion
- `--min-score FLOAT`: minimum search score
- `--semantic-threshold FLOAT`
- `--cutoff FLOAT`
- `--min-community-size INTEGER`
- `--top-k INTEGER`
- `--summarize TEXT`
- `--cross-source-top-k INTEGER`
- `--max-cross-source-queries INTEGER`

Example:

```bash
venv/bin/rag community search "quarterly roadmap" "launch plan" \
  --filter kind=report \
  --limit 5 \
  --min-score 0.75 \
  --top-k 3
```

#### `rag community retrieve`

Use retrieval criteria to select sources first.

```bash
venv/bin/rag community retrieve CRITERION... [OPTIONS]
```

Arguments:

- `CRITERION...`: one or more retrieval query strings

Options:

- `--filter key=value`: metadata filter; repeatable
- `--seed-count INTEGER`
- `--result-count INTEGER`
- `--rrf-k INTEGER`
- `--entity-confidence-threshold FLOAT`
- `--first-hop-similarity-threshold FLOAT`
- `--second-hop-similarity-threshold FLOAT`
- `--trace`
- `--semantic-threshold FLOAT`
- `--cutoff FLOAT`
- `--min-community-size INTEGER`
- `--top-k INTEGER`
- `--summarize TEXT`
- `--cross-source-top-k INTEGER`
- `--max-cross-source-queries INTEGER`

Example:

```bash
venv/bin/rag community retrieve "What changed?" \
  --filter kind=report \
  --seed-count 3 \
  --result-count 2 \
  --top-k 3 \
  --summarize google/gemini-2.5-flash-lite
```

Response shape:

```json
{
  "metadata": {
    "scope_mode": "search",
    "source_count": 2,
    "sources_excluded": [],
    "parameters": {
      "semantic_threshold": 0.85,
      "source_cooc_weight": 0.1,
      "cutoff": 0.5,
      "min_community_size": 3,
      "top_k_chunks": 5,
      "cross_source_top_k": 10,
      "max_cross_source_queries": 5000
    }
  },
  "communities": [
    {
      "community_id": "0",
      "is_cross_source": true,
      "entity_count": 4,
      "entities": [
        {
          "entity_id": "entity-uuid",
          "canonical_name": "OpenRouter",
          "entity_type": "ORG"
        }
      ],
      "contributing_sources": [
        {
          "source_id": "source-uuid",
          "source_name": "Quarterly Report"
        }
      ],
      "chunks": [
        {
          "chunk_id": "chunk-uuid",
          "source_id": "source-uuid",
          "source_name": "Quarterly Report",
          "entity_overlap_count": 2,
          "score": 0.81,
          "content": "Representative chunk text..."
        }
      ],
      "summary": "Optional community summary."
    }
  ]
}
```

### REST API

Endpoint:

```text
POST /api/community
```

Request body:

```json
{
  "scope_mode": "search",
  "source_ids": [],
  "criteria": ["quarterly roadmap"],
  "filters": {
    "kind": "report"
  },
  "search_options": {
    "limit": 5,
    "min_score": 0.75
  },
  "retrieve_options": {
    "seed_count": null,
    "result_count": null,
    "rrf_k": null,
    "entity_confidence_threshold": null,
    "first_hop_similarity_threshold": null,
    "second_hop_similarity_threshold": null,
    "trace": false
  },
  "community_options": {
    "semantic_threshold": 0.85,
    "cutoff": 0.5,
    "min_community_size": 3,
    "top_k_chunks": 5,
    "cross_source_top_k": null,
    "max_cross_source_queries": null
  },
  "summarize_model": null
}
```

Notes:

- `scope_mode` must be one of `ids`, `search`, or `retrieve`
- use `source_ids` only with `scope_mode: "ids"`
- use `criteria` with `search` or `retrieve`
- the response body matches the CLI JSON

Async runs (the frontend flow):

```text
POST /api/community/runs         → {run_id}  (starts background thread)
GET  /api/community/runs          → {runs, total, limit, offset}
GET  /api/community/runs/{id}     → full run row
GET  /api/community/runs/{id}/events  → SSE stream of stage_log events
```

The `events` endpoint polls the run row every 0.5s and emits one SSE event per new `stage_log` entry. On terminal status (`completed` or `failed`) it emits a final `result` event and closes. Reconnecting replays progress from the persisted `stage_log`.

## Prompts

Prompt templates live in [src/rag/prompts/__init__.py](src/rag/prompts/__init__.py).

That module is the canonical place to maintain prompt text for:

- document profiling (`DOCUMENT_PROFILING`) and chunk validation (`CHUNK_VALIDATION`), used by the `llm` backends
- entity extraction (`ENTITY_EXTRACTION`) and insight extraction (`INSIGHT_EXTRACTION`)
- proposition decomposition (`PROPOSITION_DECOMPOSITION`)
- retrieval query variants for chunks (`QUERY_VARIANTS`) and insights (`INSIGHT_QUERY_VARIANTS`), and the insight second-hop sub-query (`INSIGHT_SUB_QUERY`)
- graph-stage entity selection (`ENTITY_SELECTION`, `SECOND_HOP_ENTITY_SELECTION`) and entity query generation (`ENTITY_QUERY_GENERATION`)
- answer generation (`ANSWER_GENERATION`)
- community summarization (`COMMUNITY_SUMMARIZATION`)
- theme reports: per-community analysis (`THEME_COMMUNITY_ANALYSIS`) and cross-community synthesis (`THEME_SYNTHESIS`)

There is no relationship-extraction prompt (relationship extraction was removed). Two prompts live next to their callers instead: metadata extraction (`src/rag/metadata_extraction.py`) and image description (`src/rag/image_description.py`). Jev does not use prompts; its questions are defined in code (`src/rag/jev_profiling.py`, `src/rag/chunk_validation.py`, `src/rag/entity_audit.py`, `src/rag/entity_names.py`, `src/rag/jev_dedup.py` and the entity-selection step in `src/rag/retrieval.py`).

Maintenance rules:

- add or update prompt strings in `src/rag/prompts/__init__.py`
- keep prompt call sites formatting values through `.format(...)` where needed
- if prompt behavior changes, update or add tests in [tests/test_prompts.py](tests/test_prompts.py)
- if you need a deployment-specific community summary prompt without changing code, set `COMMUNITY_SUMMARIZATION_PROMPT`

## Migrations and Existing Databases

If your database predates the current code, apply the repo migrations in `scripts/migrate/` in numeric order. The Postgres container only mounts `scripts/init/postgres`, so `psql -f scripts/migrate/...` inside the container does not work; always feed the file on stdin from the repo root on the host:

```bash
docker compose exec -T postgres psql -U rag -d rag < scripts/migrate/NNN_name.sql
```

All migrations are idempotent except `002`. A fresh install already has the effect of `001`, `002`, `005`, `006`, `010`, `011`, `013` and `015` through `scripts/init/postgres/`; see [Server installation](#server-installation) for the ones it still needs.

| Migration | Purpose |
| --- | --- |
| `001_add_markdown_content.sql` | Adds `sources.markdown_content`. |
| `002_update_vector_dimensions.sql` | Resizes `chunks.embedding`/`entities.embedding` to `vector(4096)`. **Destructive and one-shot**: it drops existing embeddings. Only for databases still on the old 1536-dimension schema. |
| `003_add_entity_source_tracking.sql` | Adds `entities.source_id` (per-source cleanup on retry/delete) and normalizes legacy `stage_log` values. |
| `004_job_improvements.sql` | Adds `jobs.error_detail`, a unique partial index on `sources.md5` for active sources (duplicate detection) and the stuck-job index. |
| `005_entity_embeddings.sql` | Adds `entities.embedding vector(4096)`. |
| `006_search_performance_indexes.sql` | Sparse-search GIN index on `chunks.content` and the `(source_id, chunk_index)` index. |
| `007_insights.sql` | `insights` and `chunk_insights` tables. |
| `008_auth_and_workers.sql` | `users`, `user_sessions`, `worker_processes` and `api_key_audit` (the last is not written to yet). |
| `009_binary_vector_prefilter_indexes.sql` | Binary-quantized HNSW prefilter indexes for chunks and insights, and the insights full-text index. Dense search, insight dedup and linking depend on these. |
| `010_entities_autovacuum_tuning.sql` | Tighter autovacuum thresholds on `entities`. |
| `011_pg_prewarm.sql` | Creates the `pg_prewarm` extension used by startup index prewarming. Also needs the compose `shared_preload_libraries` setting. |
| `012_stage_duration_baseline.sql` | `stage_duration_baseline` table for `rag jobs stats --set-baseline` and the drift warning. |
| `013_research_workspace.sql` | Binary-quantized HNSW index on `entities.embedding` and the `community_runs`, `theme_reports`, `saved_answers`, `working_sets` and `entity_semantic_edges` tables. |
| `014_theme_reports_async.sql` | Adds the `generating` status to `theme_reports` for background generation. |
| `015_entity_audit.sql` | Jev audit tables `entity_mention_audit`, `entity_merge_review` and `insight_pair_review`. Required before the weekly-maintenance entity audit and Jev merge phases run; harmless otherwise. |

Example for an older database, in order:

```bash
for n in 003_add_entity_source_tracking 004_job_improvements 005_entity_embeddings 006_search_performance_indexes \
         007_insights 008_auth_and_workers 009_binary_vector_prefilter_indexes 010_entities_autovacuum_tuning \
         011_pg_prewarm 012_stage_duration_baseline 013_research_workspace 014_theme_reports_async 015_entity_audit; do
  docker compose exec -T postgres psql -U rag -d rag < scripts/migrate/$n.sql
done
```

Skip any migration you already applied (the loop is safe to repeat) and apply `001` and `002` only if your schema needs them. `011` needs the `pg_prewarm` library preloaded, which `docker-compose.yml` already does for a recreated Postgres container.

### Storage maintenance

`entities` accumulates dead TOAST rows from `scripts/merge_semantic_duplicates.py` and `scripts/merge_duplicate_entities.py` (each merge is an `UPDATE` + `DELETE`). Both scripts now run `VACUUM (ANALYZE) entities` automatically after a run that actually merged rows. `entities` also carries lower autovacuum thresholds (`autovacuum_vacuum_scale_factor = 0.02`, `autovacuum_vacuum_threshold = 500`, matching for analyze) so any other update/delete activity on the table gets reclaimed without operator intervention too — this is set in `scripts/init/postgres/02_schema.sql` for fresh installs and in `scripts/migrate/010_entities_autovacuum_tuning.sql` for existing databases (see Migrations above).

If `entities` is already bloated on an existing database (check with the query below), reclaim it once with a maintenance-window `VACUUM FULL`, which takes an exclusive lock for its duration:

```bash
docker compose exec -T postgres psql -U rag -d rag -c "
SELECT pg_size_pretty(pg_total_relation_size('entities')) AS entities_total;
"
docker compose exec -T postgres psql -U rag -d rag -c "VACUUM FULL VERBOSE entities;"
```

Prefer `pg_repack -t entities` over `VACUUM FULL` if the exclusive lock is too disruptive for your deployment; it avoids the lock at the cost of needing roughly 1x extra disk headroom during the run.

### Weekly corpus maintenance

`scripts/weekly_maintenance.py` runs the full maintenance sweep in one command: entity duplicate merge (reuses `scripts/merge_semantic_duplicates.py`), insight duplicate merge, an orphan/consistency sweep, and index/stats health (`VACUUM ANALYZE` + vector-index prewarm). Run it weekly from the repo root on the Docker host (it connects through `POSTGRES_URL`/`MEMGRAPH_URL` from `.env`; exactly one of `--dry-run` or `--execute` is required):

```bash
venv/bin/python scripts/weekly_maintenance.py --dry-run   # safe default: reports counts, writes nothing
venv/bin/python scripts/weekly_maintenance.py --execute    # applies merges/cleanup/vacuum
```

Each phase is independently skippable (`--skip-entities`, `--skip-insights`, `--skip-consistency`, `--skip-health`). The insight-merge phase is scoped by default to insights created in the last `--since` days (default 7) probed against the full corpus via the same binary-HNSW prefilter + full-precision rerank pattern intake uses (U1) — this keeps a weekly run index-assisted and fast even at 100k+ insights. Pass `--full` for an occasional whole-corpus sweep; it is slower and not meant for the routine weekly cadence.

**Concurrency guard:** `--execute` refuses to run while any `jobs` row is non-terminal (`status = 'pending'` or `status LIKE 'processing:%'`) and holds a fixed Postgres advisory lock for the duration of the run, so it can never race an in-flight intake job (which could otherwise hold or re-create an insight id the maintenance run is merging away) or another concurrent `--execute` run. `--dry-run` performs no writes and skips this check entirely. The one thing that runs outside this guard is the Jev entity-audit *scoring* described below, because it writes only the `entity_mention_audit` table.

#### Jev entity audit (phase 0, opt-in)

The graph has known quality problems: most entities are mentioned in a single chunk, generic terms such as "AI" are stored as entities, some entities are mistyped (LinkedIn as `CONCEPT`), and ~1,300 legacy entities carry off-schema types (`EVENT`, `DATE`, `ROLE`, ...). The entity audit uses Jev (see [Jev](#jev-typesafe-system-one-via-openrouter)) to judge every chunk-to-entity `MENTIONS` edge and then cleans the graph from those judgments. It is **off by default**: with it off, weekly maintenance behaves exactly as described above. Enable it per run with `--jev`, or permanently with `MAINTENANCE_JEV_ENABLED=true` (`--no-jev` forces it off for one run). Apply migration `015_entity_audit.sql` first.

**Scoring.** One Jev call per chunk (the chunk text, truncated to 4,000 characters, is the state; a chunk with more than 40 entities uses extra calls). For every entity mentioned in the chunk it asks: `is_entity` (yes/no: is the name a specific real-world named entity as used in the passage, as opposed to a generic term, role, date or phrase), `type` (one of the 7 schema types, asked twice with the options in reversed order and averaged, because Jev is biased toward the first option; the helper is shared with `src/rag/jev_profiling.py`) and `salience` (4 levels: 0 only mentioned in passing, 1 minor supporting role, 2 important to the passage, 3 a central subject; stored as `score / 3`, so 0..1). Results are upserted into `entity_mention_audit` (one row per chunk/entity pair, with the model and timestamp), committed after every batch of chunks. Scoring is **resumable** (pairs already in the table are skipped unless `--rescore`), runs `JEV_CONCURRENCY` calls in parallel, logs and counts a failed call without aborting (it only stops after three consecutive batches in which every call failed, for example out of credits), and prints progress and the cost reported by Jev. It writes **only** `entity_mention_audit`: never `entities`, `chunks` or Memgraph. That is why it also runs while intake jobs are active, ahead of the active-job check.

Cost and time: about $6 and about 2 hours for `--full` over ~160k chunks at `JEV_CONCURRENCY=8` (state is billed once per call, output is free); a weekly `--since 7` run is a small fraction of that.

**Apply** (mutates the corpus; only under `--execute`, after the active-job guard, under the advisory lock), using the audit rows in scope:

- a. Delete `MENTIONS` edges whose `is_entity` probability is below `ENTITY_AUDIT_INVALID_THRESHOLD` (default `0.2`). Entities left with no `MENTIONS` edge at all are deleted from Memgraph (`DETACH DELETE`) and Postgres, their `entity_semantic_edges` cache rows are invalidated, and `entities` is vacuumed. The audit rows of dropped mentions are removed afterwards, so the table describes live mentions.
- b. Retype only entities whose *current* `entity_type` is **not** one of the 7 schema types: the majority audited type across their (kept) mentions wins when the mean type probability of the mentions voting for it is at least `ENTITY_AUDIT_RETYPE_CONFIDENCE` (default `0.8`). In-schema entities are never retyped.
- c. Write `salience` into `MENTIONS.confidence` for every audited, kept mention. Unaudited mentions keep `1.0`. This is what `entity_confidence_threshold` filters on at retrieval time (see below).

**Modes and flags.**

| Flag | Meaning |
| --- | --- |
| `--jev` / `--no-jev` | Enable / force off the Jev phase for this run (default: `MAINTENANCE_JEV_ENABLED`). |
| `--skip-entity-audit` | Skip phase 0 even when Jev maintenance is enabled. |
| `--score` / `--no-score` | Whether to Jev-score unscored mentions. Default: score under `--execute`, do **not** score under `--dry-run`. `--dry-run --score` is allowed: it spends Jev credits and writes only to `entity_mention_audit`. |
| `--rescore` | Re-score mentions that already have an audit row. |
| `--since N` / `--full` | Scope: chunks created in the last N days (default 7), or every chunk. Shared with the insight-merge phase. |

`--dry-run` (the default-safe mode) prints what apply *would* do from the existing audit rows, without writing anything: mentions that would be dropped (with examples), entities that would be deleted, off-schema entities that would be retyped (with examples), the salience distribution that would be written, and how many in-scope mentions are still unscored.

**Recommended rollout.**

1. Apply migration 015 and set `OPENROUTER_API_KEY`.
2. Pilot on a small window: `venv/bin/python scripts/weekly_maintenance.py --dry-run --jev --score --since 1 --skip-entities --skip-insights --skip-consistency --skip-health`. This scores yesterday's chunks and prints the plan; nothing but `entity_mention_audit` is written.
3. Inspect the plan and the table (`SELECT * FROM entity_mention_audit WHERE is_entity < 0.2 LIMIT 50`). Adjust `ENTITY_AUDIT_INVALID_THRESHOLD` / `ENTITY_AUDIT_RETYPE_CONFIDENCE` if the dropped examples look wrong.
4. **Check `RETRIEVAL_ENTITY_CONFIDENCE_THRESHOLD` before the first `--execute`** against the salience distribution the dry-run prints (see the threshold note in the settings table). `--execute --jev` changes retrieval behavior: as soon as apply writes salience into `MENTIONS.confidence`, mentions below the threshold stop being used in graph expansion. The default `0.2` drops only passing mentions (level 0). If your `.env` still sets the old `0.75`, change it (a value that high would hide almost every audited mention) and restart the backend. The Retrieve view in the frontend starts at `0.2` and always sends its own value, which overrides the environment setting for UI requests; change it there too if you calibrate to another value.
5. Run the first full backfill when no jobs are running: `venv/bin/python scripts/weekly_maintenance.py --execute --jev --full` (scores everything, then applies). It is resumable: if interrupted, rerun the same command. Then set `MAINTENANCE_JEV_ENABLED=true` so the weekly `--execute` run keeps new chunks audited.

#### Jev-confirmed merges, names and near-duplicates (opt-in)

Three more phases use Jev and share the same gate as the entity audit (`--jev` / `MAINTENANCE_JEV_ENABLED`; **off by default**, and with them off the merge phases behave as before except for the survivor rule below). Merges are irreversible, so only the high-confidence band is applied automatically and the middle band goes to review tables for a human.

**Survivor rule (always on, not Jev-gated).** When a cluster of duplicate entities is merged (`merge_semantic_duplicates.py`, `merge_duplicate_entities.py`, and phase 1), the survivor is the row with the **most `MENTIONS` edges** (counted in Memgraph in one batched query, also in dry-run), so the best-connected row keeps its id and the fewest edges move. Ties fall back to the cluster's dominant `entity_type`, then has-embedding, then oldest. The dominant type used to be a hard filter, which let a one-mention row beat a hub just because its type was the majority. Absorbed names still go into the survivor's `aliases`; the survivor's own `canonical_name` is unchanged by the merge.

**Phase 1: Jev-confirmed entity merges.** Candidate pairs still come from today's fuzzy-name + embedding-cosine filters; with Jev on, each pair gets one Jev call (state `{"entity_a": {name, type, aliases[:10]}, "entity_b": {...}}`) with a 3-level alignment Score (0 two different things, 1 closely related and may or may not be the same, 2 one and the same) plus informational Nouls `same_name_variant` and `same_type`. Routing by the probability-weighted score: **>= 1.5 merge** (clustering runs over confirmed pairs only), **0.5 to 1.5 `entity_merge_review`**, **< 0.5 dropped**. A Jev error, or a missing `OPENROUTER_API_KEY`, leaves the pair unconfirmed: it is never merged. Pairs that already have a review row (any status) are neither re-judged nor merged, so a pair you rejected is not auto-merged later. The standalone `merge_semantic_duplicates.py` takes `--jev` for the same behavior.

**Phase 1b: entity name correction** (`src/rag/entity_names.py`; skip with `--skip-entity-names`, bound with `--name-limit`, default 500). The biggest hubs in a long-lived corpus can be misspellings ("Open AI", "Antropic", "Googles") while correctly spelled rows exist separately, because insert-time dedup matches `canonical_name` exactly. For the entities with at least `ENTITY_NAME_REVIEW_MIN_ALIASES` aliases (most aliases first), one Jev call gets `{"entity_type", "names": [canonical_name, *aliases]}` (deduplicated, at most 120) and a Choice "which is the correct, standard real-world spelling?" over the candidates (keys `n0..nK`, asked twice in reversed order and averaged). A rename is proposed only if both orders pick the same spelling, the averaged probability is at least `ENTITY_NAME_MIN_CONFIDENCE`, and it differs from the current name. Under `--execute`: if another entity already has exactly that name (and a compatible type) the rows are **merged** (survivor = more mentions, ends up with the winning name, every other name becomes an alias, `MENTIONS` are repointed by the same code as every other merge); otherwise `canonical_name` is updated in Postgres and on the Memgraph node and the old name becomes an alias. A same-name row of an incompatible type blocks the rename (reported, never applied). `entity_semantic_edges` is invalidated for every touched entity. Not handled: the renamed entity keeps the embedding of its old name, and new extractions of the old misspelling create a fresh row until the next run.

**Phase 2: Jev-confirmed near-duplicate insights** (`--near-dup-limit`, default 2000). With Jev on, the same single prefilter pass also keeps pairs with cosine in `[INSIGHT_NEAR_DUP_COSINE_LOW, --insight-cosine-threshold)` (0.88 to 0.95). Pairs are deduplicated by normalized order, pairs already in `insight_pair_review` are skipped, and the highest-cosine `--near-dup-limit` pairs are judged. One Jev call per pair (state `{"insight_a": text, "insight_b": text}`, 3-level Score: 0 different claims, 1 related claims that differ in substance or scope, 2 the same claim stated differently). **>= 1.5 merge** through the usual insight merge path (oldest survives), **0.5 to 1.5 `insight_pair_review`**, otherwise dropped; errors never merge. Pairs at or above the normal threshold merge as before without a Jev call.

**Dry-run versus execute.** `--dry-run --jev` still calls Jev for all three phases (that is how it can report proposals), so it spends Jev credits, but it writes nothing: no merges, renames or review rows. `--execute` (behind the active-job check and advisory lock) applies confirmed merges and renames and inserts review rows (`ON CONFLICT DO NOTHING`, pair stored in normalized order).

| Flag | Meaning |
| --- | --- |
| `--skip-entity-names` | Skip phase 1b even when Jev maintenance is enabled. |
| `--name-limit N` | Phase 1b: review at most N hub entities per run (default 500, most aliases first). |
| `--near-dup-limit N` | Phase 2: judge at most N near-duplicate insight pairs per run (default 2000, highest cosine first). |

**Inspecting the review tables.** Nothing in the review tables is applied automatically; they are queues to read (`status` defaults to `pending`):

```sql
-- entity pairs Jev found closely related but not clearly the same
SELECT r.id, r.score, a.canonical_name AS a, b.canonical_name AS b, r.reason
FROM entity_merge_review r
JOIN entities a ON a.id = r.entity_a
JOIN entities b ON b.id = r.entity_b
WHERE r.status = 'pending'
ORDER BY r.score DESC LIMIT 50;

-- near-duplicate insight pairs
SELECT r.id, r.cosine, r.score, left(a.content, 120) AS a, left(b.content, 120) AS b
FROM insight_pair_review r
JOIN insights a ON a.id = r.insight_a
JOIN insights b ON b.id = r.insight_b
WHERE r.status = 'pending'
ORDER BY r.score DESC LIMIT 50;
```

Set a row's `status` (for example `'rejected'`) to record your decision; any row, whatever its status, keeps that pair out of future Jev judging and auto-merging.

**Recommended rollout.** (1) Finish the entity audit rollout above (the `0.2` retrieval threshold is part of it). (2) `venv/bin/python scripts/weekly_maintenance.py --dry-run --jev --skip-consistency --skip-health` and read the printed proposals: Jev-confirmed merges (with mention counts), proposed renames and merges into existing rows, blocked renames, review-band counts, and the cost line. (3) Query the review tables as above. (4) When the proposals look right, run `venv/bin/python scripts/weekly_maintenance.py --execute --jev` with no jobs running; then set `MAINTENANCE_JEV_ENABLED=true` for the weekly run. Only review-band pairs are remembered (in the review tables); pairs Jev dropped are judged again on later runs, which costs a few Jev calls per pair.

Two notes on side effects. Scoring runs before the advisory lock (it must, so it can overlap intake), so two overlapping `--execute --jev` runs would both spend Jev credits (they cannot corrupt data: only the apply phase mutates, and it is serialized by the lock). And `_invalidate_entity_edge_cache` now commits its deletes (it previously did not, so the entity-merge phase left stale `entity_semantic_edges` rows behind); the first `--execute` after upgrading clears whatever orphaned cache rows built up.

### Memgraph schema reconciliation

The backend re-applies `src/rag/graph_db.py`'s `SCHEMA_STATEMENTS` (the same index/constraint statements as `scripts/init/memgraph_init.cypher`) idempotently on every startup, so a live instance can't silently drift out of sync with statements added after its initial setup (this is how the `Insight(insight_id)` index/constraint gap was found and fixed). No manual migration step is needed for future schema additions to that list — just add the statement and restart the backend.

## Rolling out this release

Ordered checklist for deploying this branch (shared OpenCode client, GPT 6 Luna defaults, Jev switches and Jev maintenance) onto an existing server. Run it from the repo checkout on the Docker host, with the CLI configured against the server for the `rag` commands. Nothing here runs by itself.

1. **Apply migration 015** (idempotent; harmless until the Jev maintenance phases are enabled):

   ```bash
   docker compose exec -T postgres psql -U rag -d rag < scripts/migrate/015_entity_audit.sql
   ```

2. **Update `.env`** so the new defaults actually apply. A value pinned in `.env` wins over the code default:
   - Remove `MODEL_ENTITY_EXTRACTION` if it pins an OpenRouter model (for example `qwen/qwen-2.5-7b-instruct`). Entity extraction now goes to OpenCode Go, which does not know that name, so every chunk would fail. The default is `gpt-6-luna`.
   - Remove `MODEL_RETRIEVAL_QUERY_VARIANTS` if it pins `deepseek-v4-flash`; the default is `deepseek-v4.1-flash`.
   - Remove `RETRIEVAL_ENTITY_CONFIDENCE_THRESHOLD` if it is set to `0.75`. The default is now `0.2`, and `0.75` would hide almost every audited mention once the entity audit has run.
   - Make sure `OPENCODE_API_KEY` is set (it is the only OpenCode key variable) and delete any other keys that are no longer in `.env.example`; unknown keys are ignored.
   - Add `OPENCODE_FALLBACK_MODEL=deepseek-v4.1-flash` (or leave it unset to use that default; set it empty to disable the quota fallback).
   - Keep the Jev switches at their defaults for now: `ENTITY_SELECTION_BACKEND=llm`, `PROFILING_BACKEND=llm`, `CHUNK_VALIDATION_BACKEND=llm`, `MAINTENANCE_JEV_ENABLED=false`.

3. **Rebuild and restart.** Stop workers first so no job is mid-flight on old code, then recreate the containers (recreation is what makes the backend re-read `.env`), then relaunch workers:

   ```bash
   rag worker stop --all
   docker compose build backend frontend
   docker compose up -d backend frontend
   rag worker launch 2
   ```

4. **Run the smoke script** with freshly generated secrets (see [Verification](#verification)). It mutates the live stack and cleans up after itself.

5. **Ingest a few test documents** and read their stage logs with `rag jobs status <job_id>`: `graph_extraction` should show `entities_stored` > 0, `insight_extraction` should show `insights_extracted` / `insights_reused`, and `failed_chunks` should be empty or small. A `skipped: true` with `skip_reason` means the OpenCode key is missing in the backend container.

6. **Flip the Jev switches one at a time**, after running the matching read-only pilot (`scripts/pilot_jev_profiling.py`, `scripts/pilot_jev_chunk_validation.py`; they only SELECT, but spend a little Jev credit). For each switch: set it in `.env`, recreate the backend (`docker compose up -d backend`), relaunch workers (they inherit settings at start, so the two ingestion switches only take effect on newly launched workers), and ingest a few documents before moving on. Suggested order: `ENTITY_SELECTION_BACKEND=jev`, then `PROFILING_BACKEND=jev`, then `CHUNK_VALIDATION_BACKEND=jev`. Pilot results on the reference corpus: chunk validation at the `0.3` threshold skipped about 1.2% of chunks (mostly outros, share links and disclaimers), and Jev profiling agreed with the LLM profiler on 98% of chunking decisions at roughly 3x lower latency.

7. **Weekly maintenance with Jev.** Start with `venv/bin/python scripts/weekly_maintenance.py --dry-run --jev`: it spends a small amount of Jev credit and writes nothing. Review the printed proposals and the review tables, then `venv/bin/python scripts/weekly_maintenance.py --execute --jev` with no jobs running. Run the first full audit with `--execute --jev --full --score` (about 2 hours and about $6 at the reference corpus size; resumable). Read [Jev entity audit](#jev-entity-audit-phase-0-opt-in) before the first `--execute`: it changes which entities retrieval uses. Set `MAINTENANCE_JEV_ENABLED=true` afterwards for the routine weekly run.

8. **After intake looks stable**, pin the stage-timing baseline: `rag jobs stats --set-baseline`.

9. **Rollback.** Revert the `.env` model and backend values (the three `*_BACKEND` switches back to `llm`, any model pins you changed) and run `docker compose up -d backend`, then relaunch workers. For a full code rollback, `git checkout main`, `docker compose build backend frontend`, `docker compose up -d backend frontend`, and relaunch workers. Migration 015 only adds tables, so it does not need reverting. Data written by the Jev maintenance phases (renames, merges, deleted mentions, `MENTIONS.confidence` salience) is not undone by a code rollback.

## Verification

Targeted retrieval and CLI tests:

```bash
pytest -q tests/test_retrieval.py tests/test_cli_retrieve.py tests/test_config.py
```

Search, community, and API coverage:

```bash
pytest -q tests/test_cli_search.py tests/test_cli_community.py tests/test_api.py tests/test_api_community.py
```

Fast ingestion-oriented suite:

```bash
pytest -q tests/test_cli_jobs.py tests/test_job_lifecycle.py tests/test_worker.py tests/test_ingestion_submit.py tests/test_parser.py tests/test_storage.py tests/test_cli_ingest.py tests/test_chunking.py tests/test_observability.py tests/test_cli_health.py tests/test_profiling.py tests/test_chunk_validation.py tests/test_embedding.py tests/test_jev_profiling.py tests/test_jev_ingestion_stages.py tests/test_pilot_jev_scripts.py
```

Prompt regression coverage:

```bash
pytest -q tests/test_prompts.py
```

Insight and entity extraction coverage:

```bash
pytest -q tests/test_insight_extraction.py tests/test_graph_extraction.py tests/test_config.py tests/test_prompts.py tests/test_cli_sources.py
```

OpenCode client, answers and theme reports (endpoint dispatch, quota fallback, answer models and streaming):

```bash
pytest -q tests/test_opencode_client.py tests/test_answering.py tests/test_themes.py tests/test_config.py
```

Jev client and switchable backends (retrieval entity selection is in `tests/test_retrieval.py`):

```bash
pytest -q tests/test_jev.py tests/test_jev_profiling.py tests/test_jev_ingestion_stages.py tests/test_chunk_validation.py tests/test_pilot_jev_scripts.py tests/test_retrieval.py tests/test_config.py
```

Entity audit and weekly maintenance, including the Jev phases, entity names, merge survivor rule and migration 015:

```bash
pytest -q tests/test_entity_audit.py tests/test_weekly_maintenance.py tests/test_weekly_maintenance_jev.py tests/test_entity_names.py tests/test_jev_dedup.py tests/test_entity_survivor.py tests/test_merge_vacuum.py tests/test_migration_015.py
```

Auth, server APIs, migrations and operator scripts:

```bash
pytest -q tests/test_api_auth_apikey.py tests/test_api_auth_session.py tests/test_api_gating.py tests/test_api_ingest.py tests/test_api_jobs.py tests/test_api_workers.py tests/test_worker_supervisor.py tests/test_mcp_server.py tests/test_migration_008.py tests/test_migration_009.py tests/test_remove_api_key_script.py
```

Full offline suite (everything except the live integration file `tests/test_ingestion.py`, which needs the running stack and real documents):

```bash
pytest -q --ignore=tests/test_ingestion.py
```

Live end-to-end smoke (mutates the running stack, so use a non-production server; needs `curl`, `jq`, `docker compose` and the host `venv`). `API_KEY` and `ADMIN_PASSWORD` are required and have no defaults; generate fresh random values for each run:

```bash
API_KEY=$(python3 -c 'import secrets;print("smoke-"+secrets.token_urlsafe(24))') \
ADMIN_PASSWORD=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))') \
scripts/smoke_e2e.sh
```

Optional variables: `ADMIN_USERNAME` (default `smoke`) and `BASE_URL` (default `http://localhost:8000`). `API_KEY` must be at least 16 characters of `[A-Za-z0-9._~-]`. Run it from the repo root on the Docker host (it creates the test user with the host-run `scripts/create_user.py`). It applies migration 008, ingests a tiny markdown file, launches a worker, waits for the job, searches, and runs the MCP probe. It refuses to run if `ADMIN_USERNAME` already exists. On exit, pass or fail, it hard-deletes the smoke source (found through the job's `Source ID` in `rag jobs status`), stops its worker, removes its `smoke-cli` block from `data/api_keys.toml` in place (`scripts/remove_api_key.py`, which preserves the inode because the file is bind-mounted into the backend container), and deletes the test user.

Live integration:

```bash
pytest -q tests/test_ingestion.py
venv/bin/rag search "What topics are covered in the ingested reports?" --limit 1
venv/bin/rag retrieve "What topics are covered in the ingested reports?" --result-count 1 --seed-count 1 --trace
```

## Troubleshooting

- `rag health` fails:
  - confirm `docker compose ps`
  - confirm `POSTGRES_URL` and `MEMGRAPH_URL`
- `rag search` fails before ranking:
  - confirm `OPENROUTER_API_KEY` (query embeddings) and that the OpenRouter account has credits
  - confirm `MODEL_EMBEDDING` is an OpenRouter embedding model
- `rag retrieve` or an answer fails:
  - retrieval also needs `OPENCODE_API_KEY` (query variants run on OpenCode Go), plus `OPENROUTER_API_KEY` for embeddings and reranking; answers need the key too
  - a model name pinned in `.env` must belong to the right provider (`MODEL_RETRIEVAL_QUERY_VARIANTS` is an OpenCode name such as `deepseek-v4.1-flash`; `MODEL_RETRIEVAL_GRAPH` and `MODEL_RETRIEVAL_RERANKER` are OpenRouter names)
- Entities or insights are missing after ingestion:
  - a completed job with `skipped: true` and `skip_reason: OPENCODE_API_KEY not configured` in its stage log means the backend container has no key; set it and recreate the backend and workers
  - check `failed_chunks` in the stage log; the stage fails outright above `STAGE_FAILURE_RATE_THRESHOLD`. A pinned `MODEL_ENTITY_EXTRACTION` or `MODEL_INSIGHT_EXTRACTION` that OpenCode does not serve fails every chunk
- OpenCode quota errors (429, or "quota"/"limit" in the message) on `gpt-6-luna`:
  - `gpt-6-luna` has a hard spend cap on the Go plan. The shared client retries once on `OPENCODE_FALLBACK_MODEL` (default `deepseek-v4.1-flash`); if both are exhausted the call fails
  - to stay off the cap, set `MODEL_ENTITY_EXTRACTION` / `MODEL_INSIGHT_EXTRACTION` to `deepseek-v4.1-flash` (large backfills especially) and choose a non-`gpt-*` answer model
  - an empty `OPENCODE_FALLBACK_MODEL` disables the fallback
- Jev-related problems (`ENTITY_SELECTION_BACKEND`, `PROFILING_BACKEND`, `CHUNK_VALIDATION_BACKEND`, `--jev` maintenance):
  - HTTP `402` from Jev means the OpenRouter account is out of credits; top up and rerun
  - Jev failures never fail retrieval or ingestion: entity selection falls back to the first `RETRIEVAL_ENTITY_SELECTION_COUNT` entities, profiling falls back to the LLM calls, chunk validation passes the affected chunks. Look for `Jev` warnings in the backend and worker logs (`rag worker log <id>`) and for `profile_source: llm_fallback` or `jev_errors` in stage logs
  - weekly maintenance never merges, renames or deletes on a Jev error; it stops scoring after three consecutive fully failed batches
  - an unrecognized `*_BACKEND` value stops the backend at startup
  - `typesafe/jev-1.13` must be available to your OpenRouter account; override `JEV_MODEL` if the slug changes
- Retrieval returns roots with empty `related`:
  - some chunks still have no useful linked graph evidence
  - retrieval may still return the reranked root chunk alone
  - if you ran the entity audit, `RETRIEVAL_ENTITY_CONFIDENCE_THRESHOLD` (or a UI/CLI override) may be filtering mentions; lower it or set `0`
- Jobs stay `pending`: no worker is running. `rag worker list`, then `rag worker launch`. Workers do not survive a backend container recreation.
- Markdown images are not described:
  - only local relative image paths are copied and described
  - remote URLs, `data:` URLs, and missing files are left untouched
- Hard delete breaks:
  - dependent rows must be removed in this order: insight links (`chunk_insights`) and orphan insights, `entities`, `chunks`, `jobs`, then `sources`. `rag sources delete <id> --hard` does this for you
