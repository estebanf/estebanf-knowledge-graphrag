#!/usr/bin/env bash
# End-to-end smoke for the server deployment.
#
# Assumes the docker stack is already up (`docker compose up -d`) and the
# 008 migration has been applied. Creates a throwaway test user and API key,
# ingests a tiny markdown file, runs a worker until the job completes,
# searches, and removes everything it created (source, worker, API key block,
# test user) on exit -- whether the run passes or fails.
#
# Secrets are REQUIRED from the environment; the script has no defaults for
# them and never stores them anywhere beyond the temporary key entry that the
# cleanup removes. Generate fresh random values for every run:
#
#   API_KEY=$(python3 -c 'import secrets;print("smoke-"+secrets.token_urlsafe(24))') \
#   ADMIN_PASSWORD=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))') \
#   scripts/smoke_e2e.sh
#
# Must run from a repo checkout on the same host as the docker stack: user
# creation runs on the host via venv/bin/python scripts/create_user.py, which
# connects through POSTGRES_URL from the repo .env (published Postgres port).
#
# Env:
#   API_KEY=...            (required) token for the temporary API key
#   ADMIN_PASSWORD=...     (required) password for the temporary test user
#   ADMIN_USERNAME=smoke   (optional) must not already exist; it is deleted at the end
#   BASE_URL=http://localhost:8000

set -euo pipefail

# --- Required secrets: checked first, before any side effect ----------------
: "${API_KEY:?Set API_KEY to a freshly generated random token}"
: "${ADMIN_PASSWORD:?Set ADMIN_PASSWORD to a freshly generated random password}"

BASE_URL=${BASE_URL:-http://localhost:8000}
ADMIN_USERNAME=${ADMIN_USERNAME:-smoke}
KEY_ID="smoke-cli"

# Values are interpolated into TOML and JSON below; keep them to a safe alphabet.
if [[ ! "$API_KEY" =~ ^[A-Za-z0-9._~-]{16,}$ ]]; then
  echo "API_KEY must be at least 16 characters of [A-Za-z0-9._~-]" >&2
  exit 2
fi
if [[ ! "$ADMIN_USERNAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ADMIN_USERNAME must match [A-Za-z0-9._-]+" >&2
  exit 2
fi
if [[ "$ADMIN_PASSWORD" == *[\"\\]* ]]; then
  echo 'ADMIN_PASSWORD must not contain " or \' >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
KEYS_FILE=data/api_keys.toml

note() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }

# --- Cleanup: runs on every exit path, preserves the real exit code ---------
worker_id=""
job_payload=""
tmp=""
cookie_jar=""
key_added=0
user_created=0

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  note "Cleanup"

  local source_id
  source_id=$(echo "$job_payload" | grep -oE 'source-[0-9a-f-]+' | head -n1)
  if [[ -n "$source_id" ]]; then
    venv/bin/rag sources delete "$source_id" --hard
  fi
  if [[ -n "$worker_id" ]]; then
    venv/bin/rag worker stop "$worker_id"
  fi

  if (( key_added )); then
    # In-place rewrite: the file is bind-mounted into the backend container as a
    # single file, so the inode must not change.
    venv/bin/python scripts/remove_api_key.py --id "$KEY_ID" --path "$KEYS_FILE" \
      || echo "WARNING: could not remove key '$KEY_ID' from $KEYS_FILE; remove it by hand" >&2
  fi
  if (( user_created )); then
    printf '%s\n' \
      "UPDATE worker_processes SET launched_by = NULL WHERE launched_by IN (SELECT id FROM users WHERE username = :'u');" \
      "DELETE FROM users WHERE username = :'u';" \
      | docker compose exec -T postgres psql -U rag -d rag -v ON_ERROR_STOP=1 -v u="$ADMIN_USERNAME" >/dev/null \
      || echo "WARNING: could not delete user '$ADMIN_USERNAME'; delete it by hand" >&2
  fi

  [[ -n "$tmp" ]] && rm -rf "$tmp"
  [[ -n "$cookie_jar" ]] && rm -f "$cookie_jar"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

note "1) Apply migration 008 (idempotent)"
docker compose exec -T postgres psql -U rag -d rag -f - < scripts/migrate/008_auth_and_workers.sql >/dev/null

note "2) Create test user (host-run; reads POSTGRES_URL from the repo .env)"
existing=$(printf '%s\n' "SELECT 1 FROM users WHERE username = :'u';" \
  | docker compose exec -T postgres psql -U rag -d rag -tA -v u="$ADMIN_USERNAME")
if [[ -n "$existing" ]]; then
  echo "user '$ADMIN_USERNAME' already exists; refusing to reset and later delete it. Set ADMIN_USERNAME to an unused name." >&2
  exit 1
fi
user_created=1
venv/bin/python scripts/create_user.py --username "$ADMIN_USERNAME" --password "$ADMIN_PASSWORD"

note "3) Add temporary API key entry"
touch "$KEYS_FILE"
# Drop any stale block from an aborted earlier run, then append (append keeps the inode).
venv/bin/python scripts/remove_api_key.py --id "$KEY_ID" --path "$KEYS_FILE"
key_added=1
cat >> "$KEYS_FILE" <<TOML

[[keys]]
id = "$KEY_ID"
token = "$API_KEY"
scopes = ["read", "ingest", "admin"]
TOML

note "4) Health check"
curl -s "$BASE_URL/api/health" | jq

note "5) Login (session) round-trip"
cookie_jar=$(mktemp)
curl -s -c "$cookie_jar" -X POST "$BASE_URL/api/auth/login" \
  -H 'content-type: application/json' \
  -d "{\"username\":\"$ADMIN_USERNAME\",\"password\":\"$ADMIN_PASSWORD\"}" | jq
curl -s -b "$cookie_jar" "$BASE_URL/api/auth/me" | jq

note "6) Configure CLI to use this server"
export RAG_SERVER_URL="$BASE_URL"
export RAG_API_KEY="$API_KEY"
venv/bin/rag health

note "7) Ingest a tiny markdown file"
tmp=$(mktemp -d)
fixture="$tmp/smoke.md"
cat > "$fixture" <<'MARKDOWN'
# Smoke Document

A smoke-test source for end-to-end verification.

It mentions Postgres and Memgraph and FastAPI so retrieval has something to chew on.
MARKDOWN
job_payload=$(venv/bin/rag ingest "$fixture" 2>&1)
echo "$job_payload"

note "8) Launch a worker"
worker_id=$(venv/bin/rag worker launch 1 | tail -n1)
echo "worker: $worker_id"

note "9) Wait for the job to complete"
deadline=$((SECONDS + 180))
job_id=$(echo "$job_payload" | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -n1)
while true; do
  status=$(venv/bin/rag jobs status "$job_id" 2>&1 | grep -oE 'completed|failed:[a-z_]+' | head -n1 || true)
  if [[ "$status" == "completed" ]]; then
    echo "job complete"
    break
  fi
  if [[ "$status" == failed:* ]]; then
    echo "job failed: $status" >&2
    venv/bin/rag jobs status "$job_id" || true
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "timeout waiting for job" >&2
    exit 1
  fi
  sleep 3
done

note "10) Search through the CLI (HTTP-backed)"
venv/bin/rag search "smoke document" --limit 3 | jq

note "11) MCP probe"
venv/bin/python scripts/mcp_probe.py --server "$BASE_URL" --api-key "$API_KEY"

note "OK — server smoke passed (cleanup runs on exit)"
