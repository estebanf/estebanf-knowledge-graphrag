#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "WARNING: This will permanently delete all data in data/postgres/ and data/memgraph/."
echo "         data/documents/ and data/backups/ are preserved."
echo ""
read -r -p "Type 'yes' to confirm: " confirm
if [[ "$confirm" != "yes" ]]; then
  echo "Aborted."
  exit 0
fi

echo "Stopping services..."
docker compose down

echo "Clearing database data..."
rm -rf "$PROJECT_ROOT/data/postgres/"
rm -rf "$PROJECT_ROOT/data/memgraph/"

echo ""
echo "Data cleared. Run scripts/start.sh to reinitialize from scratch."
