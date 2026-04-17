#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Load .env if present (makes POSTGRES_USER etc. available for healthcheck polling)
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

POSTGRES_USER="${POSTGRES_USER:-rag}"
POSTGRES_DB="${POSTGRES_DB:-rag}"
MEMGRAPH_BOLT_PORT="${MEMGRAPH_BOLT_PORT:-7687}"
MEMGRAPH_LAB_PORT="${MEMGRAPH_LAB_PORT:-3000}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

# Create data subdirectories (data/ itself is .gitignored)
mkdir -p \
  "$PROJECT_ROOT/data/postgres" \
  "$PROJECT_ROOT/data/memgraph" \
  "$PROJECT_ROOT/data/documents" \
  "$PROJECT_ROOT/data/backups"

echo "Starting services..."
docker compose up -d

# Wait for PostgreSQL
echo "Waiting for PostgreSQL to be ready..."
until docker compose exec -T postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" &>/dev/null; do
  sleep 2
done
echo "  PostgreSQL is ready."

# Wait for Memgraph (bolt port accepting connections)
echo "Waiting for Memgraph to be ready..."
until docker compose exec -T memgraph bash -c "nc -z localhost 7687" &>/dev/null 2>&1; do
  sleep 2
done
# Extra grace period — Memgraph may accept TCP before it processes Cypher
sleep 3
echo "  Memgraph is ready."

# Initialize Memgraph schema on first run
MEMGRAPH_INIT_FLAG="$PROJECT_ROOT/data/memgraph/.initialized"
if [[ ! -f "$MEMGRAPH_INIT_FLAG" ]]; then
  echo "Initializing Memgraph schema..."
  docker compose exec -T memgraph bash -c "mgconsole --no-history" \
    < "$SCRIPT_DIR/init/memgraph_init.cypher"
  touch "$MEMGRAPH_INIT_FLAG"
  echo "  Memgraph schema initialized."
fi

echo ""
echo "All services are up."
echo "  PostgreSQL : localhost:${POSTGRES_PORT}  (db: ${POSTGRES_DB}, user: ${POSTGRES_USER})"
echo "  Memgraph   : bolt://localhost:${MEMGRAPH_BOLT_PORT}"
echo "  Memgraph Lab: http://localhost:${MEMGRAPH_LAB_PORT}"
