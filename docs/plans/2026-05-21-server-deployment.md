# Server Deployment, Auth, Workers, and MCP

## Context

Today the stack is local-only. The Typer CLI talks directly to Postgres and Memgraph; the FastAPI backend has no auth; the React frontend ships unauthenticated; the worker is a single foreground process started by `rag worker`; and there is no MCP surface. We want to move the backend to a server, keep the CLI as the operator's primary interface from a laptop, expose the same capabilities through MCP for AI tools, and put authentication in front of everything reachable on the internet.

Outcome:

- CLI commands are thin HTTP wrappers around the API; identical flags and output.
- `rag ingest` uploads files to the server over HTTP.
- `rag worker` becomes a small fleet manager: `launch N`, `stop <id>`, `list`, `log <id> -f`. The backend supervises worker subprocesses.
- API keys (file-based) authenticate CLI and MCP. Frontend uses username/password → session cookie.
- MCP server is mounted on the same FastAPI app at `/mcp` (Streamable HTTP), sharing auth.
- One `docker compose up` brings up a deployable server; `pipx install` + `rag configure` sets up the CLI on a laptop.

Decisions already locked (do not revisit):

- Backend supervises worker subprocesses.
- Frontend uses username/password → httpOnly session cookie.
- MCP transport: Streamable HTTP mounted on the FastAPI app.
- CLI keeps every existing flag and output shape; only the implementation changes.
- API keys stored in `data/api_keys.toml` (operator-edited, reloaded on mtime change).
- CLI config at `~/.config/rag/config.toml`; `RAG_SERVER_URL` / `RAG_API_KEY` env override.
- Ingest uses multipart POST per file; folder ingest loops client-side.
- Worker log tailing via SSE.

## Architecture summary

- **Auth layer**: `src/rag/api/auth.py` exports a single `require_principal()` dependency that accepts either `Authorization: Bearer <key>` (validated against `data/api_keys.toml`) or a session cookie (validated against a `user_sessions` table). Scope check factory `requires("admin"|"ingest"|"read")`. Mounted on every `/api/*` route except `/api/health` and `/api/auth/login`.
- **Worker supervisor**: `src/rag/worker_supervisor.py` is a singleton attached to `app.state.workers`. `launch(n)` spawns `python -m rag.cli _worker-run --worker-id <uuid>` via `subprocess.Popen`, redirecting stdout/stderr to `data/worker_logs/{id}.log` (append, line-buffered). A row in `worker_processes` is written before spawn. An asyncio `reap_loop()` polls `Popen.poll()` and transitions rows. On startup, orphan reconciliation marks dead PIDs as `crashed`.
- **CLI**: `src/rag/api_client.py` (built on `httpx`) is the only thing CLI commands call. `src/rag/cli_config.py` loads server URL + key with env-override.
- **MCP**: `src/rag/mcp_server.py` builds tools via the official `mcp` Python SDK; the app is mounted at `/mcp`. Tools wrap REST endpoints and reuse `auth.py` to validate bearer tokens.

## Public vs gated routes

- Public: `GET /api/health`, `POST /api/auth/login`, frontend static files.
- Gated: everything else under `/api/*` and all `/mcp` tool calls.

## DB additions (`scripts/migrate/008_auth_and_workers.sql`)

- `users(id uuid pk, username text unique, password_hash text, is_active bool, created_at, last_login_at)`
- `user_sessions(id uuid pk, user_id uuid fk, created_at, expires_at, revoked_at)`
- `worker_processes(id uuid pk, pid int, status text, started_at, stopped_at, exit_code int, log_path text, host text, launched_by uuid null)`
- `api_key_audit(id bigserial pk, key_id text, route text, ts timestamptz, ip text, user_agent text)`

## Phased plan (TDD per phase; code review before moving on)

Each phase: (1) write failing tests, (2) implement, (3) run the targeted test slice + full suite, (4) request code review, (5) address feedback, (6) commit. Do not start the next phase until review is closed.

### Phase 1 — Scaffolding & dependencies
- Branch `feat/server-deployment`. Stub: `src/rag/api/auth.py`, `src/rag/api/routes/auth.py`, `src/rag/api/routes/ingest.py`, `src/rag/api/routes/jobs.py`, `src/rag/api/routes/workers.py`, `src/rag/worker_supervisor.py`, `src/rag/mcp_server.py`, `src/rag/cli_config.py`, `src/rag/api_client.py`.
- Add deps to `pyproject.toml`: `bcrypt`, `python-multipart`, `mcp`, `sse-starlette`, `httpx`, `httpx-sse`, `tomli-w`.
- Frontend: install nothing yet; just create empty `frontend/src/auth/` folder.
- Tests: existing suite stays green.
- **Review checkpoint.**

### Phase 2 — Migration 008
- TDD: `tests/test_migration_008.py` applies the SQL to an ephemeral DB and asserts tables/columns/indexes.
- Implement `scripts/migrate/008_auth_and_workers.sql` and add it to the migrate runner. Idempotent.
- **Review checkpoint.**

### Phase 3 — API key auth dependency
- TDD: `tests/test_api_auth_apikey.py` covers TOML hot-reload (mtime), missing key → 401, valid key → 200, audit row inserted, scope enforcement.
- Implement `src/rag/api/auth.py`: `load_api_keys()`, `require_api_key()`, `requires(scope)`, `Principal`.
- TOML format documented in README:
  ```toml
  [[keys]]
  id = "cli-esteban"
  token = "..."
  scopes = ["read", "ingest", "admin"]
  ```
- **Review checkpoint.**

### Phase 4 — Session auth + login routes
- TDD: `tests/test_api_auth_session.py` for create-user helper, login success/failure, cookie attributes (`HttpOnly`, `SameSite=Lax`, `Secure` when `RAG_COOKIE_SECURE=true`), logout revokes, `/api/auth/me` returns user. Brute-force rate-limit hook in place.
- Implement `src/rag/api/routes/auth.py`: `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me`. Server-side DB sessions (opaque tokens). Combined `require_principal()` accepts either bearer or session.
- CORS: `allow_credentials=True`, `allow_origins=[settings.frontend_origin]`.
- `scripts/create_user.py` interactive bcrypt helper.
- **Review checkpoint.**

### Phase 5 — Gate existing routes + frontend login
- TDD: update `tests/test_api.py` and `tests/test_api_community.py` to inject a fake principal via dependency override. Add `tests/test_api_gating.py` asserting 401 without creds on every router.
- Add `dependencies=[Depends(require_principal)]` on existing routers in `src/rag/api/main.py`.
- Frontend: `frontend/src/auth/AuthContext.tsx`, `Login.tsx` page, route guard, `fetch` wrapper with `credentials: "include"` and 401→login redirect.
- Manual: log in from the browser end-to-end.
- **Review checkpoint.**

### Phase 6 — Ingest, jobs, source-delete API endpoints
- TDD: `tests/test_api_ingest.py`, `tests/test_api_jobs.py`, `tests/test_api_sources_delete.py`.
- Implement:
  - `POST /api/ingest` (multipart: file + JSON metadata) — saves under `data/uploads/{job_id}/`, calls existing `submit_ingestion_job()`. Requires `ingest` scope.
  - `POST /api/ingest/text` (JSON body) for MCP.
  - `GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/retry`, `POST /api/jobs/{id}/cancel` — move the current CLI job-detail logic (audit_log join, stage log) into these endpoints.
  - `DELETE /api/sources/{id}?hard=true|false` — extend `routes/sources.py`.
- **Review checkpoint.**

### Phase 7 — Worker supervisor + worker API
- TDD: `tests/test_worker_supervisor.py` — spawn a fake long-running subprocess via `WorkerSupervisor.launch()`, assert DB row + status transitions on `stop()`/exit, `list()` accuracy, orphan reconciliation on startup. `tests/test_api_workers.py` — `launch/stop/list`, SSE log tailing returns recent lines + follows new ones.
- Implement `src/rag/worker_supervisor.py` and `src/rag/api/routes/workers.py`:
  - `POST /api/workers/launch?n=1`, `POST /api/workers/{id}/stop`, `GET /api/workers`, `GET /api/workers/{id}/log?follow=true` (SSE via `sse-starlette`).
  - Requires `admin` scope.
- CLI: add hidden `rag _worker-run --worker-id` (Typer `hidden=True`) — this is the actual long-running process the supervisor spawns. It wraps the current worker loop.
- Manual: `docker compose exec backend` → curl launch/stop/list/log; verify orphan recovery after `kill -9` + container restart.
- **Review checkpoint.**

### Phase 8 — API client + CLI refactor
- TDD: convert `tests/test_cli_*.py` from DB-mock fixtures to `httpx.MockTransport` injected into `RagClient`. Add `tests/test_api_client.py` and `tests/test_cli_config.py` (env > file > default precedence). Snapshot existing CLI output for a fixture corpus and lock it.
- Implement:
  - `src/rag/cli_config.py`: load/merge config; `rag configure` writes `~/.config/rag/config.toml`.
  - `src/rag/api_client.py`: one method per endpoint; SSE methods (`stream_answer`, `tail_worker_log`) yield events via `httpx-sse`.
  - Rewrite each command in `src/rag/cli.py` to call `RagClient`. Preserve flags, table rendering, JSON output byte-for-byte.
  - Replace the old `rag worker` (no subcommand) with `rag worker launch | stop | list | log` subcommands. The hidden `_worker-run` from Phase 7 remains as the supervised process.
- **Review checkpoint.**

### Phase 9 — MCP server
- TDD: `tests/test_mcp_server.py` — in-process MCP client lists tools; calls `search` with valid bearer; asserts 401 without.
- Implement `src/rag/mcp_server.py`: `mcp.server.Server` with exactly these tools, each a thin wrapper over the matching REST endpoint:
  - `search` → `POST /api/search`
  - `retrieve` → `POST /api/retrieve`
  - `community` → `POST /api/community` (supports all three scope modes: `ids`, `search`, `retrieve`)
  - `list_sources` → `GET /api/sources` (supports `q`, `metadata`, `limit`, `offset`)
  - `source_insights` → `GET /api/sources/{id}/insights`
- Mount streamable-http app at `/mcp` in `main.py`. Auth shim validates Bearer via the same `auth.py` (read scope required).
- Out of scope for MCP: ingest, jobs, workers, source detail/markdown/download, source delete, answer streaming, stats. Operators use the CLI for those.
- **Review checkpoint.**

### Phase 10 — Docker, frontend build, deployment
- Update `Dockerfile`: multi-stage with a Node build that produces `frontend/dist`, copied into the backend image; `StaticFiles` mount at `/`.
- Update `docker-compose.yml`: backend mounts a named volume containing `data/uploads`, `data/worker_logs`, `data/api_keys.toml`. New env vars: `RAG_COOKIE_SECURE`, `RAG_FRONTEND_ORIGIN`, `RAG_DATA_DIR`, `RAG_API_KEYS_PATH`. Healthcheck on `/api/health`.
- The standalone `frontend` service in `docker-compose.yml` is removed (now served by backend) — or kept behind nginx if we prefer separation. Choose the merged path for simplicity.
- **Review checkpoint.**

### Phase 11 — End-to-end smoke + cleanup
- `scripts/smoke_e2e.sh`:
  1. `docker compose up -d`; wait for `/api/health`.
  2. `python scripts/create_user.py --username demo --password demo`.
  3. Login via curl; capture cookie; `GET /api/auth/me`.
  4. Append a test entry to `data/api_keys.toml`; assert hot reload.
  5. `rag configure` (or env vars in CI).
  6. `rag ingest tests/fixtures/smoke.md` — assert job id.
  7. `rag worker launch 1`; poll `rag jobs status` until complete.
  8. `rag worker log <wid>` for 2 s, then break.
  9. `rag search "smoke"` returns hits.
  10. MCP probe (`scripts/mcp_probe.py`) calls `search` tool.
  11. **Cleanup**: `rag sources delete <id> --hard`; `rag worker stop <wid>`; remove the test API key entry; `docker compose down -v`.
- **Review checkpoint.**

### Phase 12 — Documentation
- `README.md` rewrite into two installation tracks:
  - **Server**: `docker compose up`, run migration 008, `scripts/create_user.py`, edit `data/api_keys.toml`, set `RAG_FRONTEND_ORIGIN` and `RAG_COOKIE_SECURE`.
  - **CLI**: `pipx install` (or `pip install`), `rag configure` for `RAG_SERVER_URL` + `RAG_API_KEY`.
  - New sections: **Authentication** (sessions vs keys, scopes), **Worker management** (`launch/stop/list/log`), **MCP setup** (URL, header, tool list, Claude Desktop snippet), expanded **Environment variables** table.
- `AGENTS.md` additions: auth model + `Principal`, worker supervisor invariants ("row before spawn; reap loop owns transitions"), hidden `_worker-run` is internal, MCP tools never bypass auth, CLI tests use `httpx.MockTransport`.

## Critical files

- `src/rag/api/main.py` — wire CORS, gating dependency, mount `/mcp`, mount static frontend.
- `src/rag/api/auth.py` — keys loader, session validator, `require_principal`, `requires(scope)`.
- `src/rag/api/routes/auth.py`, `routes/ingest.py`, `routes/jobs.py`, `routes/workers.py`.
- `src/rag/worker_supervisor.py` — singleton + reap loop + orphan recovery.
- `src/rag/api_client.py`, `src/rag/cli_config.py`, `src/rag/cli.py`.
- `src/rag/mcp_server.py`.
- `scripts/migrate/008_auth_and_workers.sql`, `scripts/create_user.py`, `scripts/smoke_e2e.sh`, `scripts/mcp_probe.py`.
- `frontend/src/auth/AuthContext.tsx`, `frontend/src/pages/Login.tsx`, fetch wrapper.
- `Dockerfile`, `docker-compose.yml`.

## Reused code (do not reinvent)

- `submit_ingestion_job()` in `src/rag/ingestion.py` — called from `POST /api/ingest`.
- Existing job retry/cancel cleanup in `src/rag/ingestion.py` — wrapped by `routes/jobs.py`.
- Existing answer SSE pattern in `routes/answer.py` — reuse `sse-starlette` style for worker logs.
- Existing CLI table rendering and JSON output — kept verbatim; only data source changes.

## Verification

- `pytest -q` (full suite) after each phase.
- Targeted slices listed in `AGENTS.md`'s verification section, plus new files (`test_api_auth_*.py`, `test_worker_supervisor.py`, `test_api_workers.py`, `test_api_ingest.py`, `test_api_jobs.py`, `test_api_client.py`, `test_cli_config.py`, `test_mcp_server.py`, `test_migration_008.py`).
- `scripts/smoke_e2e.sh` runs the end-to-end flow against a real docker stack and cleans up after itself.
- Manual: log in from the browser; tail a worker log from a second terminal; ingest a doc from a laptop CLI against the server URL; call the `search` MCP tool from an MCP-capable client.
