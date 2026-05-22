#!/usr/bin/env bash
# End-to-end smoke for the server deployment.
#
# Assumes the docker stack is already up (`docker compose up -d`) and the
# 008 migration has been applied. Creates a test user, an API key, ingests
# a tiny markdown file, runs a worker until the job completes, searches,
# and cleans up after itself.
#
# Usage:
#   scripts/smoke_e2e.sh
#
# Env (optional):
#   BASE_URL=http://localhost:8000
#   ADMIN_USERNAME=smoke
#   ADMIN_PASSWORD=smoke
#   API_KEY=smoke-token-please-rotate

set -euo pipefail

BASE_URL=${BASE_URL:-http://localhost:8000}
ADMIN_USERNAME=${ADMIN_USERNAME:-smoke}
ADMIN_PASSWORD=${ADMIN_PASSWORD:-smoke}
API_KEY=${API_KEY:-smoke-token-please-rotate}
KEY_ID="smoke-cli"
KEYS_FILE=data/api_keys.toml

note() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }

note "1) Apply migration 008 (idempotent)"
docker compose exec -T postgres psql -U rag -d rag -f - < scripts/migrate/008_auth_and_workers.sql >/dev/null

note "2) Create / refresh test user"
docker compose exec -T backend python scripts/create_user.py --username "$ADMIN_USERNAME" --password "$ADMIN_PASSWORD"

note "3) Ensure API key entry exists"
if ! grep -q "id = \"$KEY_ID\"" "$KEYS_FILE" 2>/dev/null; then
  cat >> "$KEYS_FILE" <<EOF

[[keys]]
id = "$KEY_ID"
token = "$API_KEY"
scopes = ["read", "ingest", "admin"]
EOF
fi

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
cat > "$fixture" <<'EOF'
# Smoke Document

A smoke-test source for end-to-end verification.

It mentions Postgres and Memgraph and FastAPI so retrieval has something to chew on.
EOF
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
    venv/bin/rag worker stop "$worker_id" || true
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "timeout waiting for job" >&2
    venv/bin/rag worker stop "$worker_id" || true
    exit 1
  fi
  sleep 3
done

note "10) Search through the CLI (HTTP-backed)"
venv/bin/rag search "smoke document" --limit 3 | jq

note "11) MCP probe"
python scripts/mcp_probe.py --server "$BASE_URL" --api-key "$API_KEY"

note "12) Cleanup"
source_id=$(echo "$job_payload" | grep -oE 'source-[0-9a-f-]+' | head -n1 || true)
if [[ -n "$source_id" ]]; then
  venv/bin/rag sources delete "$source_id" --hard || true
fi
venv/bin/rag worker stop "$worker_id" || true
rm -rf "$tmp" "$cookie_jar"

note "OK — server smoke passed"
