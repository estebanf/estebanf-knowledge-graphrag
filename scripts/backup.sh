#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Load .env if present
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

POSTGRES_USER="${POSTGRES_USER:-rag}"
POSTGRES_DB="${POSTGRES_DB:-rag}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$PROJECT_ROOT/data/backups/$TIMESTAMP"
mkdir -p "$BACKUP_DIR"

echo "Backing up to $BACKUP_DIR..."

# PostgreSQL: plain-SQL dump (human-readable, easy to inspect)
echo "  Dumping PostgreSQL..."
docker compose exec -T postgres pg_dump \
  -U "$POSTGRES_USER" \
  -d "$POSTGRES_DB" \
  --format=plain \
  > "$BACKUP_DIR/postgres.sql"
echo "  PostgreSQL dump: $BACKUP_DIR/postgres.sql"

# Memgraph: trigger a manual snapshot, then copy the data directory
echo "  Snapshotting Memgraph..."
docker compose exec -T memgraph bash -c \
  "echo 'CREATE SNAPSHOT;' | mgconsole --no-history" \
  > /dev/null 2>&1 || true
# Wait for snapshot to flush to disk
sleep 3
echo "  Copying Memgraph data..."
cp -r "$PROJECT_ROOT/data/memgraph" "$BACKUP_DIR/memgraph"
echo "  Memgraph data: $BACKUP_DIR/memgraph/"

echo ""
echo "Backup complete: $BACKUP_DIR"
echo ""
echo "To restore PostgreSQL:"
echo "  docker compose exec -T postgres psql -U $POSTGRES_USER -d $POSTGRES_DB < $BACKUP_DIR/postgres.sql"
