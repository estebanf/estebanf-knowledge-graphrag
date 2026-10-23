#!/usr/bin/env bash
# Full backup of the RAG stack into ONE compressed archive:
#
#   data/backups/knowledge_graph_<timestamp>.tar.gz
#     <timestamp>/postgres.dump        pg_dump custom format (compressed; restore with pg_restore)
#     <timestamp>/memgraph/snapshots/  newest Memgraph snapshot, taken from the live named volume
#     <timestamp>/memgraph/wal/        Memgraph write-ahead log at backup time
#     <timestamp>/documents/           stored original files (data/documents)
#     <timestamp>/manifest.txt         counts, sizes, git commit (for verifying a restore)
#     <timestamp>/secrets/             api_keys.toml + .env  (only with --with-secrets)
#
# Memgraph lives in the Docker named volume `knowledge-graphrag_memgraph_data`
# (see docker-compose.yml), NOT in data/memgraph/ — the snapshot is copied out of
# the running container with `docker compose cp`.
#
# Usage: scripts/backup.sh [--force] [--with-secrets] [--keep-staging] [--out-dir DIR]
#   --force         run even if intake jobs are pending/processing (the Postgres
#                   dump and the Memgraph snapshot are taken seconds apart, so
#                   active ingestion can make them disagree slightly)
#   --with-secrets  also archive data/api_keys.toml and .env (archive is chmod 600
#                   either way; think twice before copying it off-machine)
#   --keep-staging  keep the uncompressed staging directory next to the archive
#   --out-dir DIR   where to write the archive (default: data/backups)
#
# Restore steps: README.md, "Backup and restore".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

FORCE=0
WITH_SECRETS=0
KEEP_STAGING=0
OUT_DIR="$PROJECT_ROOT/data/backups"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --with-secrets) WITH_SECRETS=1 ;;
    --keep-staging) KEEP_STAGING=1 ;;
    --out-dir) OUT_DIR="$2"; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
POSTGRES_USER="${POSTGRES_USER:-rag}"
POSTGRES_DB="${POSTGRES_DB:-rag}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STAGING_PARENT="$OUT_DIR/.staging"
STAGING="$STAGING_PARENT/$TIMESTAMP"
ARCHIVE="$OUT_DIR/knowledge_graph_$TIMESTAMP.tar.gz"

note() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

psql_q() {
  docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "$1"
}

cleanup() {
  local rc=$?
  if (( rc != 0 )); then
    rm -f "$ARCHIVE"
    echo "Backup FAILED (exit $rc); partial archive removed." >&2
  fi
  if (( rc != 0 || KEEP_STAGING == 0 )); then
    rm -rf "$STAGING"
    rmdir "$STAGING_PARENT" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT

# --- Preflight ---------------------------------------------------------------
for svc in postgres memgraph; do
  docker compose ps --status running --services 2>/dev/null | grep -qx "$svc" \
    || die "service '$svc' is not running (docker compose up -d)"
done

if (( FORCE == 0 )); then
  active=$(psql_q "SELECT count(*) FROM jobs WHERE status = 'pending' OR status LIKE 'processing%'")
  if (( active > 0 )); then
    die "$active intake job(s) pending/processing; wait for them, or pass --force"
  fi
fi

mkdir -p "$OUT_DIR"
db_bytes=$(psql_q "SELECT pg_database_size('$POSTGRES_DB')")
docs_kb=$(du -sk data/documents 2>/dev/null | awk '{print $1}')
snap_bytes=$(docker compose exec -T memgraph bash -c \
  'f=$(ls -t /var/lib/memgraph/snapshots | head -1); [ -n "$f" ] && stat -c %s "/var/lib/memgraph/snapshots/$f" || echo 0')
# Peak usage: staging dump + snapshot, then the archive alongside them.
need_kb=$(( (db_bytes / 1024) * 3 / 2 + docs_kb + 2 * (snap_bytes / 1024) + 2 * 1024 * 1024 ))
free_kb=$(df -Pk "$OUT_DIR" | awk 'NR==2 {print $4}')
if (( free_kb < need_kb )); then
  die "not enough disk space in $OUT_DIR: need ~$(( need_kb / 1024 / 1024 )) GB, have $(( free_kb / 1024 / 1024 )) GB"
fi

mkdir -p "$STAGING/memgraph/snapshots" "$STAGING/memgraph/wal"
umask 077
note "Backing up to $ARCHIVE"
note "  database $(( db_bytes / 1024 / 1024 / 1024 )) GB, documents $(( docs_kb / 1024 )) MB, free space $(( free_kb / 1024 / 1024 )) GB"

# --- PostgreSQL --------------------------------------------------------------
note "  [1/5] Dumping PostgreSQL (custom format, compressed)..."
docker compose exec -T postgres pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  --format=custom --compress=6 > "$STAGING/postgres.dump"
# pg_dump's exit status is checked by `set -e`; also confirm the custom-format header.
[[ "$(head -c 5 "$STAGING/postgres.dump")" == "PGDMP" ]] || die "postgres.dump is not a pg_dump custom-format file"
note "        $(du -h "$STAGING/postgres.dump" | cut -f1)"

# --- Memgraph ----------------------------------------------------------------
note "  [2/5] Snapshotting Memgraph..."
before=$(docker compose exec -T memgraph bash -c 'ls -t /var/lib/memgraph/snapshots | head -1')
echo "CREATE SNAPSHOT;" | docker compose exec -T memgraph mgconsole > /dev/null \
  || die "CREATE SNAPSHOT failed"
latest=$(docker compose exec -T memgraph bash -c 'ls -t /var/lib/memgraph/snapshots | head -1')
[[ -n "$latest" ]] || die "no Memgraph snapshot found"
[[ "$latest" != "$before" ]] || note "        (no new snapshot file; graph unchanged since $latest)"
docker compose cp "memgraph:/var/lib/memgraph/snapshots/$latest" "$STAGING/memgraph/snapshots/" > /dev/null
docker compose cp "memgraph:/var/lib/memgraph/wal/." "$STAGING/memgraph/wal/" > /dev/null
note "        $latest ($(du -h "$STAGING/memgraph/snapshots/$latest" | cut -f1))"

# --- Manifest ----------------------------------------------------------------
note "  [3/5] Writing manifest..."
{
  echo "created_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo ?))"
  echo "postgres_db: $POSTGRES_DB ($db_bytes bytes)"
  echo "memgraph_snapshot: $latest"
  echo "with_secrets: $WITH_SECRETS"
  echo "postgres_counts:"
  for t in sources chunks entities insights chunk_insights jobs; do
    echo "  $t: $(psql_q "SELECT count(*) FROM $t")"
  done
  echo "memgraph_counts:"
  printf 'MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label;\n' \
    | docker compose exec -T memgraph mgconsole --output-format=csv \
    | tail -n +2 | tr -d '"' | awk -F, '{print "  " $1 ": " $2}'
  printf 'MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY rel;\n' \
    | docker compose exec -T memgraph mgconsole --output-format=csv \
    | tail -n +2 | tr -d '"' | awk -F, '{print "  " $1 ": " $2}'
} > "$STAGING/manifest.txt"

if (( WITH_SECRETS == 1 )); then
  mkdir -p "$STAGING/secrets"
  [[ -f data/api_keys.toml ]] && cp -p data/api_keys.toml "$STAGING/secrets/"
  [[ -f .env ]] && cp -p .env "$STAGING/secrets/env"
fi

# --- Archive -----------------------------------------------------------------
# data/documents is added straight from its location (no staging copy) and
# renamed into <timestamp>/documents inside the archive.
note "  [4/5] Creating archive..."
if tar --version 2>/dev/null | grep -qi bsdtar; then
  rename=(-s ",^documents,$TIMESTAMP/documents,")
else
  rename=(--transform "s,^documents,$TIMESTAMP/documents,")
fi
tar -czf "$ARCHIVE" "${rename[@]}" -C "$STAGING_PARENT" "$TIMESTAMP" -C "$PROJECT_ROOT/data" documents
chmod 600 "$ARCHIVE"

# Reads the whole archive once (gzip CRC is checked on the way) and confirms
# the key members are present.
note "  [5/5] Verifying archive..."
listing="$STAGING_PARENT/listing.$TIMESTAMP"
tar -tzf "$ARCHIVE" > "$listing"
grep -q "^$TIMESTAMP/postgres.dump$" "$listing" || die "archive is missing postgres.dump"
grep -q "^$TIMESTAMP/memgraph/snapshots/$latest$" "$listing" || die "archive is missing the Memgraph snapshot"
grep -q "^$TIMESTAMP/documents/" "$listing" || die "archive is missing documents/"
note "        $(wc -l < "$listing" | tr -d ' ') entries"
rm -f "$listing"

note ""
note "Backup complete: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
(( KEEP_STAGING == 1 )) && note "Staging kept at: $STAGING"
note "Restore steps: README.md -> Backup and restore"
